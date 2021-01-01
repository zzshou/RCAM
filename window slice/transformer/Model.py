# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:55:27 2020

@author: 31906
"""

''' Define the encoder part of transformer '''

import torch.nn as nn
import torch
import numpy as np
from transformer.Modules import MultiHeadAttention, PositionwiseFeedForward


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
    
    
class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, device, n_choice, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_slice=10):

        super().__init__()
        
        self.device = device
        self.d_hid = d_model
        self.n_choice = n_choice
        self.position_enc = PositionalEncoding(d_hid=self.d_hid, n_position=n_slice+1)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, cls_embeddings, src_mask=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        CLS = torch.randn(self.n_choice, 1, self.d_hid).to(self.device)    
        cls_embeddings = torch.cat((CLS,cls_embeddings), 1)
        
        enc_output = self.dropout(self.position_enc(cls_embeddings))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output, # (n_choice, n_slice+1, d_hid)