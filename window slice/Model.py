# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:38:34 2021

@author: 31906
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Model import TransformerEncoder


class SelfAttention(nn.Module):
    """对每个slice的CLS做self attention，得到document的CLS"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        # (B, L, H) -> (B, L, 1)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, 1) -> (B, L)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        return outputs, weights
        
    
class MultiChoiceModel(nn.Module):
    """有三种模型可以选择：max, atten, transformer"""
    
    def __init__(self, model, args, is_requires_grad=True):
        super(MultiChoiceModel, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = is_requires_grad
        self.d_hid = model.config.hidden_size
        self.attention = SelfAttention(self.d_hid)
        self.args = args
        self.transformer = TransformerEncoder(self.args.device, self.args.n_choice, self.args.n_layer, \
                            self.args.n_head, self.d_hid, self.d_hid, self.d_hid, self.args.d_inner, self.args.dropout)
        self.classifier = nn.Linear(self.d_hid, 1)

    def forward(
        self,
        input_ids=None,  #输入的id,模型会帮你把id转成embedding
        attention_mask=None,   #attention里的mask
        token_type_ids=None,    # [CLS]A[SEP]B[SEP] 就这个A还是B, 有的话就全1, 没有就0
        position_ids=None,     # 位置id
        head_mask=None,       # 哪个head需要被mask掉
        inputs_embeds=None,   # 可以选择不输入id,直接输入embedding
        labels=None,          # 做分类时需要的label
    ):

        input_ids = input_ids.view(-1, self.args.max_seq_len) #(n_choice*n_slice, max_seq_len)
        attention_mask = attention_mask.view(-1, self.args.max_seq_len) #(n_choice*n_slice, max_seq_len)
        token_type_ids = token_type_ids.view(-1, self.args.max_seq_len) #(n_choice*n_slice, max_seq_len)

        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
        )

        pooled_output = outputs[1] #(n_choice*n_slice, d_hid)
        pooled_output = pooled_output.view(self.args.n_choice, -1, self.d_hid) #(n_choice, n_slice, d_hid)

        if self.args.method == 'max': 
            pooled_output = pooled_output.permute(0,2,1)
            MaxPooling = torch.nn.MaxPool1d(pooled_output.shape[-1])
            document_cls = MaxPooling(pooled_output).squeeze(-1) #(n_choice, d_hid)

        elif self.args.method == 'atten': 
            document_cls, _, = self.attention(pooled_output) #(n_choice, d_hid)

        else: 
            enc_output = self.transformer(pooled_output) #(n_choice, n_slice+1, d_hid)
            document_cls = enc_output[0][:, 0, :] #(n_choice, d_hid)

        logits = self.classifier(document_cls) # (n_choice, 1)
        loss = -torch.log(torch.exp(logits[labels][0])/torch.exp(logits).sum())
        
        return loss, logits
    
