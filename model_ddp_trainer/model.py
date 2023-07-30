import copy
from typing import Optional, Any, Union, Callable
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pandas as pd
import numpy as np


# from model_ddp_trainer.model_layers import BipolarScaledDotProductAttention


class JobFeatureEmbeddings(nn.Module):
    def __init__(self,
                 d_model,
                 metadata_file='./model_ddp_trainer/jobs_metadata.csv',
                 using_location: bool = False,
                 using_classification: bool = False,
                 using_sub_classification: bool = False,
                 using_work_type: bool = False,
                 ):
        """
        LocationEmbedding: mapping job_ids to location embeddings (job_ids -> location_ids -> location embeddings)
        :param d_model:
        :param metadata_file:
        """
        super().__init__()
        self.using_location = using_location
        self.using_classification = using_classification
        self.using_sub_classification = using_sub_classification
        self.using_work_type = using_work_type

        metadata_dataframe = pd.read_csv(metadata_file, dtype=np.int32).set_index(keys='job_id_encoded', verify_integrity=True)
        metadata_dataframe.loc[0] = [0] * 4  # Add: 0th jobid (padding) ---> 0th location (padding), etc
        metadata_dataframe = metadata_dataframe.sort_index(axis=0, ascending=True)
        self.metadata_lookup_table = nn.Embedding.from_pretrained(
            embeddings=torch.tensor(metadata_dataframe.to_numpy(), dtype=torch.int32),
            freeze=True,
            padding_idx=0)

        self.metadata_embedding_layers = nn.ModuleList()
        self.padding_token_index = 0  # Index of the padding token
        for feature in ['location_encoded', 'classification_encoded', 'sub_classification_encoded', 'work_type_encoded']:
            nunique = metadata_dataframe[feature].nunique()  # with padding index
            self.metadata_embedding_layers.append(
                nn.Embedding(nunique, d_model, padding_idx=self.padding_token_index)  # 0th id is for padding
            )

        self.init_weights()

    def init_weights(self):
        init_range = 0.02
        for layer in self.metadata_embedding_layers:
            layer.weight.data.normal_(mean=0, std=init_range)
            nn.init.constant_(layer.weight[self.padding_token_index], 0)  # padding's embedding

    def forward(self, job_ids: torch.Tensor):
        """

        :param job_ids:
        :return: tuple[location_embeddings, classification_embeddings, sub_classification_embeddings, work_type_embeddings]
        """
        output = [None] * 4
        for i, using_feature in enumerate([self.using_location, self.using_classification, self.using_sub_classification, self.using_work_type]):
            if using_feature:
                feature_ids = self.metadata_lookup_table(job_ids)[..., i].int()
                feature_embeddings = self.metadata_embedding_layers[i](feature_ids)
                output[i] = feature_embeddings
        return tuple(output)


class JobEmbedding(nn.Module):
    def __init__(self,
                 nitem: int,
                 d_model: int,
                 dropout: float = 0.1,
                 job_feature_embedding_layer: Optional[JobFeatureEmbeddings] = None,
                 metadata_file: Optional[str] = './model_ddp_trainer/jobs_metadata.csv'
                 ):
        """
        item id embedding + location embedding
        :param nitem:
        :param d_model:
        :param dropout:
        :param job_feature_embedding_layer: if not None, use the passed `job_feature_embedding_layer`, otherwise create a new `JobFeatureEmbeddings` object internally with all features turned off.
        :param metadata_file: if `job_feature_embedding_layer` is None, use metadata_file to create a new `JobFeatureEmbeddings` object internally with all features turned off.
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Item ID Embeddings (job ids)
        self.padding_token_index = 0  # Index of the padding token
        self.job_id_embedding_layer = nn.Embedding(nitem, d_model, padding_idx=self.padding_token_index)

        if job_feature_embedding_layer:
            self.job_feature_embedding_layer = job_feature_embedding_layer
        else:
            print('Creating JobFeatureEmbeddings internally!')
            self.job_feature_embedding_layer = JobFeatureEmbeddings(
                d_model=d_model,
                metadata_file=metadata_file,
                using_location=False,
                using_classification=False,
                using_sub_classification=False,
                using_work_type=False
            )

        self.init_weights()

    def init_weights(self):
        # init_range = 0.1
        # self.job_id_embedding_layer.weight.data.uniform_(-init_range, init_range)
        init_range = 0.02
        self.job_id_embedding_layer.weight.data.normal_(mean=0, std=init_range)
        nn.init.constant_(self.job_id_embedding_layer.weight[self.padding_token_index], 0)  # padding token's embedding is all 0s

    def forward(self, job_id):
        item_embedding = self.job_id_embedding_layer(job_id)

        feature_embedding_tuple = self.job_feature_embedding_layer(job_id)
        for feature_embedding in feature_embedding_tuple:
            if feature_embedding is not None:
                item_embedding += feature_embedding

        item_embedding = self.layer_norm(item_embedding)
        item_embedding = self.dropout(item_embedding)
        return item_embedding


class ItemEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float,
                 checkpoint: str,
                 job_embedding_layer: JobEmbedding,
                 job_feature_embedding_layer: JobFeatureEmbeddings
                 ):
        """
        item encoder
        :param d_model:
        :param dropout:
        :param checkpoint: 'jjzha/jobbert-base-cased'
        :param job_embedding_layer: used by job ID embeddings only
        :param job_feature_embedding_layer: used by Bert embeddings only
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.job_embedding_layer = job_embedding_layer
        self.job_feature_embedding_layer = job_feature_embedding_layer

        # model with given checkpoint and extract its body
        bert_config = AutoConfig.from_pretrained(checkpoint, output_attentions=False, output_hidden_states=True)
        # bert_config.attention_probs_dropout_prob = dropout
        # bert_config.hidden_dropout_prob = dropout
        self.bert_encoder = AutoModel.from_pretrained(checkpoint, config=bert_config)

    def forward(self, input_ids, attention_mask, token_type_ids, item_id):
        """
        https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel.forward
        :param input_ids: token id sequence of the target item
        :param attention_mask: padding mask sequence of the target item
        :param token_type_ids: token type id sequence of the target item
        :param item_id: target item id (job id)
        :return:
        """
        # BERT embedding ################################
        # Extract outputs from the body
        bert_outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # sequence_output = self.dropout(bert_outputs[0])  #outputs[0]=last hidden state
        last_hidden_state = bert_outputs['last_hidden_state']
        cls_embedding = last_hidden_state[:, 0, :]

        feature_embedding_tuple = self.job_feature_embedding_layer(item_id)
        for feature_embedding in feature_embedding_tuple:
            if feature_embedding is not None:
                cls_embedding += feature_embedding

        cls_embedding = self.layer_norm(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = torch.unsqueeze(cls_embedding, dim=1)

        # output = cls_embedding
        #################################################

        # Item ID embedding #############################
        item_embedding = self.job_embedding_layer(item_id)
        item_embedding = torch.unsqueeze(item_embedding, dim=1)

        # output = item_embedding
        #################################################

        output = torch.concat([cls_embedding, item_embedding], dim=1)

        return output


class UserEmbedding(nn.Module):
    def __init__(self,
                 nuser: int,
                 d_model: int,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # User ID Embeddings
        self.padding_token_index = 0  # Index of the padding token
        self.user_ids_embedding_layer = nn.Embedding(nuser, d_model, padding_idx=self.padding_token_index)
        self.init_weights()

    def init_weights(self):
        # initrange = 0.1
        # self.user_ids_embedding_layer.weight.data.uniform_(-initrange, initrange)
        init_range = 0.02
        self.user_ids_embedding_layer.weight.data.normal_(mean=0, std=init_range)
        nn.init.constant_(self.user_ids_embedding_layer.weight[self.padding_token_index], 0)  # padding's embedding

    def forward(self, user_id):
        user_embedding = self.user_ids_embedding_layer(user_id)
        return self.dropout(user_embedding)


class UserEncoder(nn.Module):
    def __init__(self,
                 nuser: int,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 dropout: float,
                 job_embedding_layer: JobEmbedding
                 ):
        """
        :param nuser: NUM_USERS, total number of users in the dataset
        :param d_model: d_model
        :param nhead: nhead
        :param d_hid: dim_feedforward
        :param nlayers: nlayers
        :param dropout: dropout
        :param job_embedding_layer: item embedding
        """
        super().__init__()

        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)

        # User ID Embeddings
        self.user_embedding_layer = UserEmbedding(nuser=nuser, d_model=d_model, dropout=dropout)

        self.job_embedding_layer = job_embedding_layer

    def forward(self, src_user, src, src_key_padding_mask) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param src_user: torch.Tensor, shape ``[batch_size]`` - user's id
        :param src: torch.Tensor, shape ``[batch_size, seq_len]`` - sequence of the user's clicked jobs
        :param src_key_padding_mask: torch.Tensor, shape ``[batch_size, seq_len]`` - padding mask
        :return: output torch.Tensor of shape ``[batch_size, seq_len+1, ntoken]``, ntoken is the size of item_embeddings: item_embeddings.weight.size()[0]
        """
        src = torch.concat([
            torch.unsqueeze(self.user_embedding_layer(src_user), 1),
            self.job_embedding_layer(src),
        ], dim=1)

        # src_key_padding_mask should include and unmask src_user
        user_encoder_padding_mask = F.pad(src_key_padding_mask, pad=(1, 0), value=False)

        user_encoder_embeddings = self.transformer_encoder(src, src_key_padding_mask=user_encoder_padding_mask)

        return user_encoder_embeddings, user_encoder_padding_mask


class DualEncoderInterestNetwork(nn.Module):
    def __init__(self,
                 nitem: int,
                 nuser: int,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 dropout: float,
                 checkpoint: str,
                 using_cross_attention: bool = False
                 ):
        """
        :param nitem: NUM_ITEMS
        :param nuser: NUM_USERS
        :param d_model: d_model
        :param nhead: nhead
        :param d_hid: dim_feedforward
        :param nlayers: nlayers
        :param dropout: dropout
        :param checkpoint: 'jjzha/jobbert-base-cased'
        :param using_cross_attention:
        """
        super().__init__()

        self.job_feature_embedding_layer = JobFeatureEmbeddings(
            d_model=d_model,
            using_location=True,  # False,
            using_classification=True,  # False,
            using_sub_classification=False,  # False,
            using_work_type=False  # False
        )

        self.job_embedding_layer = JobEmbedding(nitem=nitem, d_model=d_model, dropout=dropout, job_feature_embedding_layer=self.job_feature_embedding_layer, metadata_file=None)

        self.item_encoder = ItemEncoder(d_model=d_model, dropout=dropout, checkpoint=checkpoint, job_embedding_layer=self.job_embedding_layer, job_feature_embedding_layer=self.job_feature_embedding_layer)

        self.user_encoder = UserEncoder(nuser=nuser, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=dropout, job_embedding_layer=self.job_embedding_layer)

        # self.using_cross_attention = using_cross_attention
        # self.register_buffer('item_encoder_padding_mask', torch.tensor([False, False]))
        # self.bipolar_attention = BipolarScaledDotProductAttention(dim=d_model, activation=F.tanh, masked_fill=0.)

        self.fully_connected_block = nn.Sequential(
            nn.Linear(768 * 2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self,
                input_ids, attention_mask, token_type_ids, item_id,
                src_user, src, src_key_padding_mask,  # labels
                ):
        item_encoder_embeddings = self.item_encoder(
            input_ids=input_ids,  # dataset_dict['train']['input_ids'],
            attention_mask=attention_mask,  # dataset_dict['train']['attention_mask'],
            token_type_ids=token_type_ids,  # dataset_dict['train']['token_type_ids'],
            item_id=item_id  # dataset_dict['train']['job_id_next']
        )
        user_encoder_embeddings, user_encoder_padding_mask = self.user_encoder(
            src_user=src_user,  # dataset_dict['train']['resume_id_encoded'],
            src=src,  # dataset_dict['train']['padded_job_id_sequences'],
            src_key_padding_mask=src_key_padding_mask,  # dataset_dict['train']['padded_job_id_sequences_padding_mask']
        )

        user_id_index = 0

        # # Interaction: cross attention ########################
        # if self.using_cross_attention:
        #     #     item_encoder_embeddings = self.item_decoder(
        #     #         tgt=item_encoder_embeddings,
        #     #         # memory=user_encoder_embeddings[:, user_id_index + 1:, :],
        #     #         memory=user_encoder_embeddings,
        #     #         # memory_key_padding_mask=user_encoder_padding_mask[:, user_id_index + 1:])
        #     #         memory_key_padding_mask=user_encoder_padding_mask)
        #
        #     key = value = user_encoder_embeddings
        #     query = item_encoder_embeddings
        #
        #     key_padding_mask = user_encoder_padding_mask
        #     batch_size = key_padding_mask.size()[0]
        #     query_padding_mask = self.item_encoder_padding_mask[None, :].repeat(batch_size, 1)
        #
        #     item_encoder_embeddings, _ = self.bipolar_attention(query=query, key=key, value=value, key_padding_mask=key_padding_mask, query_padding_mask=query_padding_mask)
        #####################################################

        # # L2 norm: not helpful with L2 norm alone
        # item_encoder_embeddings = F.normalize(item_encoder_embeddings, p=2, dim=2, )
        # user_encoder_embeddings = F.normalize(user_encoder_embeddings, p=2, dim=2, )
        #####################################################

        # Interaction layer #################################
        logits1 = logits2 = 0
        # ----------------------------------------------------
        # # (1) inner product layer
        interactions = torch.bmm(
            item_encoder_embeddings,
            user_encoder_embeddings[:, user_id_index, :].unsqueeze(1).transpose(1, 2)
        )
        logits1 = torch.sum(interactions, dim=(1), keepdim=False)  # .shape
        # ----------------------------------------------------
        # # (2) fully connected layers (Alibaba BST)
        for i in range(2):
            features = torch.concat(
                [
                    item_encoder_embeddings[:, i, :].unsqueeze(dim=1),
                    user_encoder_embeddings[:, user_id_index, :].unsqueeze(dim=1)
                ], dim=1)
            logits2 += self.fully_connected_block(features.view(features.size()[0], -1))

        ####################################################
        # # Encouraging job id embedding and job text embedding to be close
        # logits_sim = torch.bmm(
        #     F.normalize(item_encoder_embeddings[:,0,:].unsqueeze(1), p=2, dim=2, ), 
        #     F.normalize(item_encoder_embeddings[:,1,:].unsqueeze(1), p=2, dim=2, ).transpose(1,2)
        #     ).view(-1,1)

        ####################################################
        ####################################################
        logits = logits1 + logits2

        return logits
        # return logits, logits_sim
