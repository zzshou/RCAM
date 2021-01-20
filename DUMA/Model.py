# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:54:58 2021

@author: 31906
"""


import numpy as np
import torch
import torch.nn as nn
from CoAttention import MultiHeadAttention


class MultiChoiceModel(nn.Module):
    def __init__(self, model, args, is_requires_grad):
        super(MultiChoiceModel, self).__init__()
        self.model = model #bert encoder
        for param in self.model.parameters():
            param.requires_grad = is_requires_grad  # 对bert层的每个参数都要求梯度
        self.d_hid = model.config.hidden_size
        self.args = args
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.d_hid, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            MultiHeadAttention(
                                n_head=self.args.n_head, 
                                d_model=self.d_hid, 
                                d_k=self.args.d_k, 
                                d_v=self.args.d_v, 
                                dropout=self.args.dropout
                              )
            for _ in range(self.args.n_layer)])
        self.classifier = nn.Linear(self.d_hid*2, 1)

    def forward(
        self,
        input_ids=None,  #输入的id, bert encoder把id转成embedding
        attention_mask=None,   #attention里的mask
        token_type_ids=None,    # [CLS]A[SEP]B[SEP]
        position_ids=None,     # 位置id
        head_mask=None,       
        inputs_embeds=None,   
        lengths=None,         # 记录每个example的article的长度，做co-attention时用来进行mask
    ):

        input_ids = input_ids.view(-1, self.args.max_seq_len) #(batch*n_choice, max_seq_len)
        attention_mask = attention_mask.view(-1, self.args.max_seq_len) #(batch*n_choice, max_seq_len)
        token_type_ids = token_type_ids.view(-1, self.args.max_seq_len) #(batch*n_choice, max_seq_len)

        # 输入bert进行encode，得到每个token的embedding
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0] #(batch*n_choice, max_seq_len, d_hid)
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.layer_norm(sequence_output)
        
        lengths = lengths.detach().cpu().numpy()
        max_a_len = max(lengths) #article的最大长度
        max_c_len = self.args.max_seq_len-min(lengths)-2 #choice的最大长度
        article_output = sequence_output[:, :max_a_len, :] #根据article的最大长度取出article_output
        choice_output = sequence_output[:, -(max_c_len+1):-1, :] #根据choice的最大长度取出choice_output
        
        def create_mask(lengths, n_choice):
            """ 根据每个example的article的长度生成attention mask """
            mask = []
            for i in lengths:
                a = np.zeros((max_c_len, max_a_len))
                a[max_c_len-(self.args.max_seq_len-2-i):, :i] = np.ones((self.args.max_seq_len-2-i, i))
                a = [a] * n_choice
                mask.extend(a)
            return torch.tensor(mask, dtype=torch.long)
        
        a_to_c_mask = create_mask(lengths, n_choice=self.args.n_choice).to(self.args.device) #(batch*n_choice, max_c_len, max_a_len)
        c_to_a_mask = a_to_c_mask.permute(0, 2, 1) #(batch*n_choice, max_a_len， max_c_len)
        
        # 对article和choice做co-attention
        for co_attention_layer in self.layer_stack: 
            choice_output, _ = co_attention_layer(q=choice_output, k=article_output, v=article_output, mask=a_to_c_mask)
            article_output, _ = co_attention_layer(q=article_output, k=choice_output, v=choice_output, mask=c_to_a_mask)
        
        # use mean pooling to pool the sequence output of co-attention
        AvgPooling1 = nn.AvgPool1d(choice_output.shape[1])
        choice_to_article = AvgPooling1(choice_output.permute(0,2,1)).squeeze(-1) #(batch*n_choice, d_hid)
        
        AvgPooling2 = nn.AvgPool1d(article_output.shape[1])
        article_to_choice = AvgPooling2(article_output.permute(0,2,1)).squeeze(-1) #(batch*n_choice, d_hid)
        
        # concatenate the two pooled output
        fuze = torch.cat((choice_to_article, article_to_choice), 1) #(batch*n_choice, d_hid*2)
        
        # compute the logits
        logits = self.classifier(fuze) # (batch*n_choice, 1)
        logits = logits.view(-1, self.args.n_choice) # (batch, n_choice)
        
        return logits
