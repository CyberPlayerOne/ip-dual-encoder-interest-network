import pandas as pd
import numpy as np
import math
from pprint import pprint
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import load_dataset, Dataset, DatasetDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import shutil


# running_on_aws = False
# running_on_aws = True

def tokenize_text_columns(running_on_aws=True, checkpoint='bert-base-cased', col1="title", col2="abstract"):
    # Load the processed data from step 2 ################################
    proj_root_path = os.path.dirname(os.path.dirname(os.getcwd())) if running_on_aws else '/root/projects/PythonProjects/ip-dual-encoder-factorization-machine/'

    data_path = 's3://tyler-s3-bucket/other/interview-projects/seek/input_processed_output/' if running_on_aws else '/root/projects/PythonProjects/ip-dual-encoder-factorization-machine/data/input_processed_output/'
    print(data_path)

    df_dtypes = {
        'event_datetime': str,
        'resume_id_encoded': np.int32,
        'job_id_sequence': str,
        'job_id_next': np.int32,
        'class': np.int8,
        'title': str,
        'abstract': str,
        'location': str,
        'classification': str
    }

    dataframe_dict = {
        ds: pd.read_json(os.path.join(data_path, f'{ds}_data.json'), dtype=df_dtypes) for ds in ['train', 'valid', 'test']
    }

    dataset_dict = DatasetDict({k: Dataset.from_pandas(v, split=k, preserve_index=False) for k, v in dataframe_dict.items()})
    dataset_dict = dataset_dict.remove_columns(['event_datetime'])

    # Load the Tokenizer object ############################
    MAX_LEN = 512

    # https://huggingface.co/jjzha/jobbert-base-cased
    # checkpoint = 'bert-base-cased'  # 'jjzha/jobbert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = MAX_LEN

    def tokenize(batch):
        """
        Tokenizer function for job text columns. Postponed the padding part to batching by DataLoader.
        :param batch:
        :return:
        """
        # return tokenizer(batch["title"], batch["abstract"], return_tensors='pt', truncation=True, max_length=MAX_LEN, padding='max_length')
        return tokenizer(batch[col1], batch[col2], truncation=True, max_length=MAX_LEN, padding=False)

    # Delete data dir if exists ############################
    if running_on_aws:
        # !rm -r "/home/ec2-user/projects/ip-dual-encoder-factorization-machine/data/tmp_data/"
        dir_data = '/home/ubuntu/projects/ip-dual-encoder-factorization-machine/data/tmp_data/'
    else:
        dir_data = "/root/projects/PythonProjects/ip-dual-encoder-factorization-machine/data/tmp_data/"

    if os.path.exists(dir_data):
        shutil.rmtree(dir_data)

    # cache_file_path = f'{proj_root_path}/data/tmp_data'
    cache_paths = [
        f'{proj_root_path}/data',
        f'{proj_root_path}/data/tmp_data',
        f'{proj_root_path}/data/tmp_data/tokenize',
        # f'{proj_root_path}/tmp_data/pad_job_id_sequence'
    ]
    for path in cache_paths:
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
    # Tokenization in parallel #######################################

    map_batch_size = 15  # 6
    num_proc = 15  # 6
    writer_batch_size = 1000  # 500

    dataset_dict = dataset_dict.map(
        tokenize,
        batched=True,
        batch_size=map_batch_size,
        num_proc=num_proc,
        cache_file_names={k: f'{cache_paths[-1]}/{k}_cache.arrow' for k in dataset_dict.keys()},
        writer_batch_size=writer_batch_size,
        desc='tokenize'
    )

    print('Map execution completed.')

    # Keep a subset of columns #######################################

    keeping_column_names = [
        'input_ids',
        'attention_mask',
        'token_type_ids',
        'job_id_next',
        'resume_id_encoded',
        'job_id_sequence',
        # 'padded_job_id_sequences',
        # 'padded_job_id_sequences_padding_mask',
        'class'
    ]
    removing_column_names = [x for x in dataset_dict['train'].column_names if x not in keeping_column_names]

    dataset_dict = dataset_dict.remove_columns(removing_column_names)
    dataset_dict = dataset_dict.rename_column('class', 'labels')

    # # Save the DatasetDict to S3 ######################################
    dataset_dict_s3_path = 's3://tyler-s3-bucket/other/interview-projects/seek/final_dataset_dict/'
    # dataset_dict_s3_path = f'{proj_root_path}/data/tmp_data/final_dataset_dict'

    dataset_dict.save_to_disk(dataset_dict_path=dataset_dict_s3_path)


if __name__ == '__main__':
    tokenize_text_columns()
