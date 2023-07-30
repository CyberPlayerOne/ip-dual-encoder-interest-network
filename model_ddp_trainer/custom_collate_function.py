import ast

import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

MAX_LEN = 512


def _pad_job_id_sequence(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Does dynamic padding on the column `job_id_sequence` and then make it to tensors
    """
    # var_job_id_sequences = [torch.tensor(ast.literal_eval(x)) for x in batch['job_id_sequence']]
    job_id_sequences_ragged = [sample['job_id_sequence'] for sample in batch]
    job_id_sequences_ragged = [ast.literal_eval(x) for x in job_id_sequences_ragged]
    job_id_sequences_ragged = [torch.tensor(x) for x in job_id_sequences_ragged]

    # Pad the sequences in the batch to the same length
    job_id_sequences_padded = torch.nn.utils.rnn.pad_sequence(job_id_sequences_ragged, batch_first=True, padding_value=0)
    # Create a boolean mask indicating the padded positions
    job_id_sequences_padding_mask = job_id_sequences_padded.eq(0)

    return {
        'job_id_sequence': job_id_sequences_padded,
        'job_id_sequence_padding_mask': job_id_sequences_padding_mask
    }


class CustomCollateFunc:
    def __init__(
            self,
            tokenizer_checkpoint,
            tokenizer_model_max_length=MAX_LEN
    ):
        pass

        # # https://huggingface.co/jjzha/jobbert-base-cased
        # checkpoint = 'jjzha/jobbert-base-cased'
        checkpoint = tokenizer_checkpoint
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.model_max_length = tokenizer_model_max_length
        # tokenizer.model_max_length

        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self._job_text_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')  # Default return_tensors is pt

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """custom_collate_function
        1. Does Dynamic padding on ('input_ids', 'token_type_ids', 'attention_mask'), and on 'job_id_sequence', and creates the corresponding 'job_id_sequence_padding_mask'.
        2. Wrap the data in PyTorch tensors.
        :param batch:
        :return:
        """
        # step 1: job text padding: pad the encoded input columns ('input_ids', 'token_type_ids', 'attention_mask') with DataCollatorWithPadding
        # column `job_id_sequence` is of variable length thus it cannot be directly made to tensors, therefore it needs to be excluded in step 1
        batch_job_text_cols = self._job_text_data_collator(
            [{key: sample[key] for key in sample.keys() if key != 'job_id_sequence'} for sample in batch]
        )  # list
        # step 2: pad_job_id_sequence
        jobid_cols = _pad_job_id_sequence(
            [{'job_id_sequence': sample['job_id_sequence']} for sample in batch]
        )
        batch_job_text_cols.update(jobid_cols)
        return batch_job_text_cols
