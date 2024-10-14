# Copyright (c) Sudeep Dasari, 2023
# Heavy inspiration taken from ACT by Tony Zhao: https://github.com/tonyzhaozh/act
# and DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data4robotics.agent import BaseAgent


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class _TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, pos=None, query_pos=None):
        q = k = _with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, _ = self.multihead_attn(
            query=_with_pos_embed(tgt, query_pos),
            key=_with_pos_embed(memory, pos),
            value=memory,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class _TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        return output


class _TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos, query_pos, return_intermediate=False):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
            if return_intermediate:
                intermediate.append(self.norm(output))

        if return_intermediate:
            return torch.stack(intermediate)
        return output


class _ACT(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        encoder_layer = _TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.encoder = _TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = _TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.decoder = _TransformerDecoder(decoder_layer, num_decoder_layers)

        self._reset_parameters()

        self.pos_helper = _PositionalEncoding(d_model)
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_tokens, query_enc):
        input_tokens = input_tokens.transpose(0, 1)
        input_pos = self.pos_helper(input_tokens)
        memory = self.encoder(input_tokens, input_pos)

        query_enc = query_enc[:, None].repeat((1, input_tokens.shape[1], 1))
        tgt = torch.zeros_like(query_enc)
        acs_tokens = self.decoder(tgt, memory, input_pos, query_enc)
        return acs_tokens.transpose(0, 1)


class TransformerAgent(BaseAgent):
    def __init__(
        self,
        features,
        odim,
        n_cams,
        ac_dim,
        ac_chunk,
        use_obs="add_token",
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        early_fusion=False,
        feat_norm=False,
        token_dim=512,
        transformer_kwargs=dict(),
    ):

        # initialize obs and img tokenizers
        super().__init__(
            odim=odim,
            features=features,
            n_cams=n_cams,
            imgs_per_cam=imgs_per_cam,
            use_obs=use_obs,
            share_cam_features=share_cam_features,
            early_fusion=early_fusion,
            dropout=dropout,
            feat_norm=feat_norm,
            token_dim=token_dim,
        )

        self.transformer = _ACT(**transformer_kwargs)
        self.ac_query = nn.Embedding(ac_chunk, self.transformer.d_model)
        self.ac_proj = nn.Linear(self.transformer.d_model, ac_dim)
        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk

    def forward(self, imgs, obs, ac_flat, mask_flat):
        tokens = self.tokenize_obs(imgs, obs)
        action_tokens = self.transformer(tokens, self.ac_query.weight)
        actions = self.ac_proj(action_tokens)

        ac_flat_hat = actions.reshape((actions.shape[0], -1))
        all_l1 = F.l1_loss(ac_flat_hat, ac_flat, reduction="none")
        l1 = (all_l1 * mask_flat).mean()
        return l1

    def get_actions(self, imgs, obs):
        tokens = self.tokenize_obs(imgs, obs)
        action_tokens = self.transformer(tokens, self.ac_query.weight)
        return self.ac_proj(action_tokens)

    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim
