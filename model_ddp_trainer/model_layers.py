# import copy
# from typing import Optional, Any, Union, Callable
#
# from transformers import AutoTokenizer, AutoModel, AutoConfig
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# import pandas as pd
# import numpy as np
#
#
# class MHAInteractionLayer(nn.Module):
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False,  # norm_first: bool = False,
#                  device=None, dtype=None):
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#         #                                     **factory_kwargs)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                                     **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
#         self.dropout3 = nn.Dropout(dropout)
#
#         # self.norm_first = norm_first
#         # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#
#         # self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.activation = activation
#
#     def forward(
#             self,
#             tgt: Tensor,
#             memory: Tensor,
#             # tgt_mask: Optional[Tensor] = None,
#             memory_mask: Optional[Tensor] = None,
#             # tgt_key_padding_mask: Optional[Tensor] = None,
#             memory_key_padding_mask: Optional[Tensor] = None,
#             # tgt_is_causal: bool = False,
#             memory_is_causal: bool = False,
#     ):
#         # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
#         x = tgt
#
#         x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
#         x = self.norm3(x + self._ff_block(x))
#
#         return x
#
#     # multihead attention block
#     def _mha_block(self, x: Tensor, mem: Tensor,
#                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
#         x = self.multihead_attn(x, mem, mem,
#                                 attn_mask=attn_mask,
#                                 key_padding_mask=key_padding_mask,
#                                 is_causal=is_causal,
#                                 need_weights=False)[0]
#         return self.dropout2(x)
#
#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout3(x)
#
#
# class BipolarScaledDotProductAttention(nn.Module):
#     """
#     Scaled Dot-Product Attention proposed in "Attention Is All You Need"
#     Compute the dot products of the query with all keys, divide each by sqrt(dim),
#     and apply a softmax function to obtain the weights on the values
#
#     Args: dim, mask
#         dim (int): dimention of attention
#         mask (torch.Tensor): tensor containing indices to be masked
#
#     Inputs: query, key, value, mask
#         - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
#         - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
#         - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
#         - **mask** (-): tensor containing indices to be masked
#
#     Returns: context, attn
#         - **context**: tensor containing the context vector from attention mechanism.
#         - **attn**: tensor containing the attention (alignment) from the encoder outputs.
#     """
#
#     def __init__(self, dim: int, activation: Callable[[Tensor], Tensor] = F.tanh, masked_fill=0.):
#         super(BipolarScaledDotProductAttention, self).__init__()
#         self.sqrt_dim = np.sqrt(dim)
#         self.activation = activation
#         self.masked_fill = masked_fill
#
#     def _make_cross_attention_mask(self, key_padding_mask: Tensor, query_padding_mask: Tensor) -> Tensor:
#         """
#
#         :param key_padding_mask: shape (batch_size, key_seq_len)
#         :param query_padding_mask: shape (batch_size, query_seq_len)
#         :return: A batch of Squared attention mask, shape (batch_size, query_seq_len, key_seq_len)
#         """
#         for mask in [key_padding_mask, query_padding_mask]:
#             if mask is None:
#                 raise Exception('mask is None')
#             if mask.dim() != 2:
#                 raise Exception('mask.dim() != 2')
#         key_padding_mask = key_padding_mask.unsqueeze(-1)
#         query_padding_mask = query_padding_mask.unsqueeze(-1)
#         return torch.bitwise_not(
#             torch.bitwise_not(query_padding_mask) * torch.bitwise_not(key_padding_mask).transpose(-2, -1)
#         )
#
#     def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None, query_padding_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
#         score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
#
#         if key_padding_mask is not None and query_padding_mask is not None:
#             attention_mask = self._make_cross_attention_mask(key_padding_mask=key_padding_mask, query_padding_mask=query_padding_mask)
#             score.masked_fill_(attention_mask.view(score.size()), self.masked_fill)  # -float('Inf') for Softmax / Sigmoid; 0 for Tanh
#
#         # attn = F.softmax(score, -1)
#         div = torch.count_nonzero(torch.bitwise_not(key_padding_mask), dim=-1)[:, None, None]
#         attn = torch.div(self.activation(score), div)  # divided by num of False mask, as a form of normalization like Softmax
#         context = torch.bmm(attn, value)
#         # return {'output': context, 'attention_weights': attn, 'attention_mask': attention_mask}
#         return context, attn
