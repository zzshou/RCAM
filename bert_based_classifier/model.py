import logging
from typing import Any, Dict, List

import torch
from allennlp.nn import InitializerApplicator
from torch import nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from torch.nn import Dropout
from bert_based_classifier.metrics import Metric
from transformers import AlbertModel, BertModel, RobertaModel, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
import re
import os.path as path
from bert_based_classifier.CoAttention import MultiHeadAttention

logger = logging.getLogger(__name__)


@Model.register("RCAM_model")
class RCAMModel(Model):
    """
    BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Inputs:
        `article_with_question`: dict, article_with_question['bert_tokens']['token_ids'].shape: batch * num_choices * seq_len
        `labels`: batch * num_choices

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    """

    def __init__(
            self,
            vocab: Vocabulary,
            pretrained_model_type,
            pretrained_model_path,
            embedding_dim: int,
            dropout: float = None,
            num_choices: int = 5,
            mode: str = 'rnn',
            rnn_dim: int = 768,
            attention_n_head: int = 24,
            attention_d_k: int = 64,
            attention_d_v: int = 64,
            attention_dropout: float = 0.3,
            rnn_bidirection: bool = False,
            rnn_layer: int = 2,
            window_slice: bool = False,
            initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        self.pretrain = pretrained_model_type
        if pretrained_model_type == 'bert':
            self.pretrain = 'bert'
            self.embedding = BertModel.from_pretrained(pretrained_model_path)
        elif pretrained_model_type == 'albert':
            self.embedding = AlbertModel.from_pretrained(pretrained_model_path)
        elif pretrained_model_type == 'robert':
            self.embedding = RobertaModel.from_pretrained(pretrained_model_path)
        elif pretrained_model_type == 'fine-tune':
            fine_tune_model_path = '/home/data/zshou/RCAM/model-roberta-question_length_1028_2/best.th'
            self.embedding = RobertaModel(config=BertConfig.from_pretrained(pretrained_model_path))
            fine_tune_state_dict = torch.load(fine_tune_model_path, map_location="cuda:3")
            new_state_dict = OrderedDict()
            for key, value in fine_tune_state_dict.items():
                new_key = re.sub('^embedding\.', '', key)
                new_state_dict[new_key] = value
            info = self.embedding.load_state_dict(new_state_dict, strict=False)
            logger.info(info)
        else:
            print('pretrained model type not included')
        self.embedding_dim = embedding_dim
        self._dropout = None
        if dropout:
            self._dropout = Dropout(dropout)
        self._num_choices = num_choices
        self.scorer = Metric()
        if mode == 'rnn':
            self.encoder_mode = mode
            self.encoder = nn.LSTM(input_size=self.embedding_dim,
                                   hidden_size=rnn_dim,
                                   num_layers=2,
                                   batch_first=True,
                                   dropout=0.33,
                                   bidirectional=True)
        elif mode == 'max':
            self.encoder_mode = mode
            self.encoder = torch.nn.MaxPool1d
        elif mode == 'bert':
            self.encoder_mode = mode
            self.encoder = RobertaModel.from_pretrained(pretrained_model_path)
        elif mode == 'attention':
            self.encoder_mode = mode
            self.encoder = MultiHeadAttention(n_head=attention_n_head,
                                              d_model=embedding_dim,
                                              d_k=attention_d_k,
                                              d_v=attention_d_v,
                                              dropout=attention_dropout)
            # self.rnn = nn.LSTM(input_size=self.embedding_dim,
            #                    hidden_size=rnn_dim,
            #                    num_layers=rnn_layer,
            #                    batch_first=True,
            #                    dropout=0.33,
            #                    bidirectional=False)
        if mode in ('rnn'):
            if not window_slice:
                # self.article_score_layer = nn.Linear(in_features=rnn_dim, out_features=1)
                # self.option_score_layer = nn.Linear(in_features=rnn_dim, out_features=1)
                # self.score_layer = nn.Linear(in_features=rnn_dim, out_features=1)
                # self._linear_layer = nn.Linear(in_features=rnn_dim, out_features=1)
                # self.softmax_layer = nn.Softmax(dim=1)
                # self._linear_layer = nn.Linear(
                #     in_features=rnn_dim * rnn_layer * 2 if rnn_bidirection else rnn_dim * rnn_layer, out_features=1)
                self.lm = BertModel.from_pretrained('/home/data/zshou/corpus/cased_L-12_H-768_A-12')
                self._linear_layer = nn.Linear(in_features=embedding_dim, out_features=1)
            else:
                self._linear_layer = nn.Linear(in_features=rnn_dim * 2, out_features=1)
        elif mode in ('attention') and not window_slice:
            self.lm = BertModel.from_pretrained('/home/data/zshou/corpus/cased_L-12_H-768_A-12')
            self._linear_layer = nn.Linear(in_features=embedding_dim, out_features=1)
        else:
            self._linear_layer = nn.Linear(in_features=embedding_dim, out_features=1)

        initializer(self)

    def forward(
            self,
            paragraph_with_question_field: Dict[str, Dict[str, torch.LongTensor]] = None,
            article_with_question: Dict[str, Dict[str, torch.LongTensor]] = None,
            label: List[torch.Tensor] = None,
            metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        if paragraph_with_question_field is None and article_with_question is None:
            raise ValueError
        elif article_with_question:
            tokens = article_with_question['bert_tokens']['token_ids']
            mask = article_with_question['bert_tokens']['mask']
            type_ids = article_with_question['bert_tokens']['type_ids']
            outputs = self.embedding(
                tokens.view(-1, tokens.size(-1)),
                attention_mask=mask.view(-1, tokens.size(-1)),
                token_type_ids=type_ids.view(-1, tokens.size(-1)),
            )
            encode_output = outputs[1]
            batch_size = tokens.size()[0]
            if self.encoder_mode == 'attention':
                embed_text = outputs[0]
                type_mask = type_ids.view(batch_size * 5, -1)
                # logger.info(embed_text.shape)
                # logger.info(type_mask.shape)
                # logger.info(type_mask.ge(1).squeeze(0).unsqueeze(-1).expand(embed_text.shape).shape)
                option_output = embed_text * (type_mask.ge(1).squeeze(0).unsqueeze(-1).expand(embed_text.shape)) * ()
                article_output = embed_text * (type_mask.eq(0).squeeze(0).unsqueeze(-1).expand(embed_text.shape))
                option_output, _ = self.encoder(q=option_output, k=article_output, v=article_output)
                article_output, _ = self.encoder(q=article_output, k=option_output, v=option_output)

                # encode_output, (h, c) = self.rnn(torch.cat((article_output, option_output), 1))
                encode_output = torch.cat((article_output, option_output), 1)
                '''
                # option_score = self.option_score_layer(option_output)
                # article_score = self.article_score_layer(article_output)
                encode_output_score = self.score_layer(encode_output)
                # option_score = self.softmax_layer(option_score)
                # article_score = self.softmax_layer(article_score)
                encode_output_score = self.softmax_layer(encode_output_score)
                # logger.info(score)
                # article_output = torch.sum(article_output * article_score, 1)
                # option_output = torch.sum(option_output * option_score, 1)
                # encode_output = torch.cat((article_output, option_output), dim=1)
                encode_output = torch.sum(encode_output_score * encode_output, 1)
                '''
                # encode_output = self.lm(
                #     inputs_embeds=encode_output,
                #     attention_mask=mask.view(-1, tokens.size(-1)),
                #     token_type_ids=type_ids.view(-1, tokens.size(-1)),
                # )
                # encode_output = encode_output[1]
                encode_output, _ = torch.max(encode_output, dim=1)

        else:
            tokens = paragraph_with_question_field['bert_tokens']['token_ids']
            batch_size = tokens.size()[0]
            mask = paragraph_with_question_field['bert_tokens']['mask']
            type_ids = paragraph_with_question_field['bert_tokens']['type_ids']
            outputs = self.embedding(tokens.view(-1, tokens.size(-1)),
                                     attention_mask=mask.view(-1, tokens.size(-1)),
                                     token_type_ids=type_ids.view(-1, tokens.size(-1)), )
            embedded_text = outputs[1]
            window_length = embedded_text.size()[0] // batch_size // 5
            if self.encoder_mode == 'max':
                embedded_text = embedded_text.view(batch_size * 5, window_length, self.embedding_dim).transpose(2, 1)
                encode_output = self.encoder(kernel_size=window_length)(embedded_text).transpose(2, 1)
                encode_output = torch.squeeze(encode_output, 1)
            elif self.encoder_mode == 'rnn':
                embedded_text = embedded_text.view(batch_size * 5, window_length, self.embedding_dim)
                encode_output, _ = self.encoder(embedded_text)
                encode_output = torch.mean(encode_output, 1)
                encode_output = torch.squeeze(encode_output, 1)
            elif self.encoder_mode == 'attention':
                choice_output = outputs[0]
                type_ids = type_ids.view(batch_size * window_length * 5, -1)
                choice_output = choice_output * (type_ids.ge(1).squeeze(0).unsqueeze(-1).expand(choice_output.shape))
                # logger.info(choice_output.shape)
                embedded_text = embedded_text.unsqueeze(1)
                # logger.info(embedded_text.shape)
                encode_output, _ = self.encoder(q=embedded_text, k=choice_output, v=choice_output)
                # encode_output, _ = self.encoder(q=choice_output, k=embedded_text, v=embedded_text)
                encode_output = encode_output.view(batch_size * 5, window_length, self.embedding_dim)
                # encode_output, _ = self.rnn(encode_output)
                encode_output, _ = torch.max(encode_output, dim=1, keepdim=True)
                encode_output = encode_output.squeeze(1)
        if self.training:
            if self._dropout:
                encode_output = self._dropout(encode_output)
            logits = self._linear_layer(encode_output)  # batch* num_choices, num_labels
            reshaped_logits = logits.view(-1, self._num_choices)  # batch, num_choices
            # logger.info(reshaped_logits)
            # logger.info(label)
            _loss = self.scorer.update(reshaped_logits, label)
            return {  # "reshape_logits": reshaped_logits,
                "predicted_label": torch.argmax(reshaped_logits, dim=-1),
                "label": label,
                "loss": _loss}
        self.eval()
        with torch.no_grad():
            logits = self._linear_layer(encode_output)  # batch* num_choices, num_labels
            reshaped_logits = logits.view(-1, self._num_choices)  # batch, num_choices

            if label is not None:
                _loss = self.scorer.update(reshaped_logits, label)
                return {  # "reshape_logits": reshaped_logits,
                    "predicted_label": torch.argmax(reshaped_logits, dim=-1),
                    "label": label,
                    "loss": _loss}
            return {"predicted_label": torch.argmax(reshaped_logits, dim=-1),
                    "label": label}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        res = self.scorer.get_metrics(reset=reset)
        return res
