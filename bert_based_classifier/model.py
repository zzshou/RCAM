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
            initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        self.pretrain = pretrained_model_type
        if pretrained_model_type == 'bert':
            self.pretrain = 'bert'
            self.embedding = BertModel.from_pretrained(pretrained_model_path)
        elif pretrained_model_type == 'albert':
            self.embedding = AlbertModel.from_pretrained(pretrained_model_path)
        elif pretrained_model_type == 'albertxxlarge':
            config = BertConfig.from_json_file(path.join(pretrained_model_path, 'config.json'))
            self.embedding = AlbertModel.from_pretrained(pretrained_model_path, from_tf=True, config=config)
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
            self.encoder = nn.LSTM(input_size=768,
                                   hidden_size=768,
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
        if mode == 'rnn':
            self._linear_layer = nn.Linear(in_features=embedding_dim * 2, out_features=1)
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
                # logger.info(embedded_text.shape)
                encode_output, _ = self.encoder(embedded_text)
                # logger.info(encode_output.shape)
                encode_output = torch.mean(encode_output, 1)
                encode_output = torch.squeeze(encode_output, 1)
            elif self.encoder_mode == 'bert':
                pass

        if self.training:
            if self._dropout:
                encode_output = self._dropout(encode_output)
            logits = self._linear_layer(encode_output)  # batch* num_choices, num_labels
            reshaped_logits = logits.view(-1, self._num_choices)  # batch, num_choices

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
