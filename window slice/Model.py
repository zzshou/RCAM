# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:38:34 2021

@author: 31906
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Model import TransformerEncoder


class Max(nn.Module):
    """对每个slice的CLS做max pooling，得到document的CLS"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def create_max_mask(self, padding, n_slice, n_choice):
        """ 对padding的slice进行mask """
        mask = []
        for i in padding:
            a = [1] * i + [0] * (n_slice - i)     
            mask.extend([a]*n_choice)
        return torch.tensor(mask, dtype=torch.long)
    
    def forward(self, pooled_output, padding):
        mask = self.create_max_mask(padding, self.args.n_slice, self.args.n_choice).to(self.args.device) #(batch*n_choice, n_slice)
        mask = mask.unsqueeze(-1) #(batch*n_choice, n_slice, 1)
        
        pooled_output = pooled_output.masked_fill(mask == 0, -1e9) #(batch*n_choice, n_slice, d_hid)
        pooled_output = pooled_output.permute(0,2,1) #(batch*n_choice, d_hid, n_slice)
        
        MaxPooling = torch.nn.MaxPool1d(pooled_output.shape[-1])
        document_cls = MaxPooling(pooled_output).squeeze(-1) # (batch*n_choice, d_hid)
        
        return document_cls
        
        
class SelfAttention(nn.Module):
    """对每个slice的CLS做self attention，得到document的CLS"""
    
    def __init__(self, args, d_hid):
        super().__init__()
        self.args = args
        self.projection = nn.Sequential(
            nn.Linear(d_hid, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
        
    def create_atten_mask(self, padding, n_slice, n_choice):
        """ 对padding的slice进行mask """
        mask = []
        for i in padding:
            a = [1] * i + [0] * (n_slice - i)     
            mask.extend([a]*n_choice)
        return torch.tensor(mask, dtype=torch.long)

    def forward(self, pooled_output, padding):
        mask = self.create_atten_mask(padding, self.args.n_slice, self.args.n_choice).to(self.args.device) #(batch*n_choice, n_slice)
        energy = self.projection(pooled_output) # (B, L, H) -> (B, L, 1)
        energy = energy.squeeze(-1).masked_fill(mask == 0, -1e9) # (B, L, 1) -> (B, L)
        weights = F.softmax(energy, dim=1)
        document_cls = (pooled_output * weights.unsqueeze(-1)).sum(dim=1) # (B, L, H) * (B, L, 1) -> (B, H)
        
        return document_cls
        
        
class Transformer(nn.Module):
    """将每个slice的CLS输入transformer，得到document的CLS"""
    
    def __init__(self, args, d_hid):
        super().__init__()
        self.args = args
        self.d_hid = d_hid
        self.CLS = torch.randn(1, self.d_hid)
        self.transformer = TransformerEncoder(
                                                n_layers = self.args.transformer_n_layer,
                                                n_head = self.args.transformer_n_head, 
                                                d_k = self.d_hid, 
                                                d_v = self.d_hid, 
                                                d_model = self.d_hid, 
                                                d_inner = self.args.transformer_d_inner, 
                                                dropout = self.args.transformer_dropout,
                                                n_slice = self.args.n_slice
                                             )
    
    def create_transformer_mask(self, padding, n_slice, n_choice):
        """ 对padding的slice进行mask """
        mask = []
        for i in padding:
            a = np.zeros((n_slice, n_slice))     
            a[:i.item(), :i.item()] = np.ones((i.item(),i.item()))
            a = [a]*n_choice
            mask.extend(a)
        return torch.tensor(mask, dtype=torch.long)
      
    def forward(self, pooled_output, padding):
        # # 第一种方法：多加一个CLS
        # CLS = self.CLS.unsqueeze(0).expand(pooled_output.size()[0], 1, self.d_hid).to(self.args.device) #(batch*n_choice, 1, d_hid)
        # pooled_output = torch.cat((CLS, pooled_output), 1) #(batch*n_choice, n_slice+1, d_hid)
        # mask = self.create_transformer_mask(padding+1, self.args.n_slice+1, self.args.n_choice).to(self.args.device) #(batch*n_choice, n_slice, n_slice)
        # enc_output = self.transformer(pooled_output, mask) 
        # document_cls = enc_output[0][:, 0, :] #(batch*n_choice, d_hid)
            
        # 第二种方法：对transformer的output做max pooling
        mask = self.create_transformer_mask(padding, self.args.n_slice, self.args.n_choice).to(self.args.device) #(batch*n_choice, n_slice, n_slice)
        enc_output = self.transformer(pooled_output, mask) 
        document_cls = enc_output[0] #(batch*n_choice, n_slice, d_hid)
        document_cls = document_cls.permute(0,2,1) #(batch*n_choice, d_hid, n_slice)
        MaxPooling = nn.MaxPool1d(document_cls.shape[-1])
        document_cls = MaxPooling(document_cls).squeeze(-1) #(batch*n_choice, d_hid)
        
        return document_cls
        
        
class Lstm(nn.Module):
    """将每个slice的CLS输入lstm，得到document的CLS"""
    
    def __init__(self, args, d_hid):
        super().__init__()
        self.args = args
        self.d_hid = d_hid
        self.lstm = nn.LSTM(
                              input_size = self.d_hid, 
                              hidden_size = self.d_hid, 
                              num_layers = self.args.lstm_n_layer,
                              batch_first = False, 
                              dropout = self.args.lstm_dropout if self.args.lstm_n_layer > 1 else 0.0,
                              bidirectional = False, 
                           )
    
    def forward(self, pooled_output, padding):
        padding = padding.detach().cpu().numpy()
        lengths = []
        for i in padding:
            lengths.extend([i] * self.args.n_choice)
        
        pooled_output = nn.utils.rnn.pack_padded_sequence(pooled_output, lengths=lengths, enforce_sorted=False, batch_first=True) #(n_slice, batch*n_choice, d_hid)
        lstm_output, _ = self.lstm(pooled_output) #(n_slice, batch*n_choice, d_hid)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output)
        document_cls = lstm_output[-1] #(batch*n_choice, d_hid)
          
        return document_cls
    
    
class MultiChoiceModel(nn.Module):
    """有四种模型可以选择：max, atten, transformer, lstm"""
    
    def __init__(self, model, args, is_requires_grad=True):
        super(MultiChoiceModel, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = is_requires_grad
        self.d_hid = model.config.hidden_size
        self.args = args
        
        if self.args.method == 'max':
            self.method = Max(self.args)
        elif self.args.method == 'atten':
            self.method = SelfAttention(self.args, self.d_hid)
        elif self.args.method == 'transformer':
            self.method = Transformer(self.args, self.d_hid)
        else:
            self.method = Lstm(self.args, self.d_hid)
        
        self.classifier = nn.Linear(self.d_hid, 1)

    def forward(
        self,
        input_ids=None,  #输入的id,模型会帮你把id转成embedding
        attention_mask=None,   #attention里的mask
        token_type_ids=None,    # [CLS]A[SEP]B[SEP] 就这个A还是B, 有的话就全1, 没有就0
        position_ids=None,     # 位置id
        head_mask=None,       # 哪个head需要被mask掉
        inputs_embeds=None,   # 可以选择不输入id,直接输入embedding
        padding=None,       # 对滑窗后的slice进行padding
        labels=None,          # 做分类时需要的label
    ):

        input_ids = input_ids.view(-1, self.args.max_seq_len) #(batch*n_choice*n_slice, max_seq_len)
        attention_mask = attention_mask.view(-1, self.args.max_seq_len) #(batch*n_choice*n_slice, max_seq_len)
        token_type_ids = token_type_ids.view(-1, self.args.max_seq_len) #(batch*n_choice*n_slice, max_seq_len)

        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
        )

        pooled_output = outputs[1] #(batch*n_choice*n_slice, d_hid)
        pooled_output = pooled_output.view(-1, self.args.n_slice, self.d_hid) #(batch*n_choice, n_slice, d_hid)
        
        document_cls = self.method(pooled_output, padding) #(batch*n_choice, d_hid)
        logits = self.classifier(document_cls) # (batch*n_choice, 1)
        logits = logits.view(-1, self.args.n_choice) # (batch, n_choice)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        
        return loss, logits
    
