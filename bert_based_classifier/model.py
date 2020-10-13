import logging
from typing import Any, Dict, List

import torch
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator
from torch import nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from torch.nn import Dropout
from bert_based_classifier.metrics import Metric

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
            text_field_embedder: TextFieldEmbedder,
            seq2vec: Seq2VecEncoder,
            dropout: float = None,
            num_choices: int = 5,
            initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._seq2vec = seq2vec
        self._linear_layer = nn.Linear(in_features=self._seq2vec.get_output_dim(), out_features=1)
        if dropout:
            self._dropout = Dropout(dropout)
        self._num_choices = num_choices
        self.scorer = Metric()

        initializer(self)

    def forward(
            self,
            article_with_question: Dict[str, Dict[str, torch.LongTensor]],
            label: List[torch.Tensor] = None,
            metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = article_with_question['bert_tokens']['token_ids']
        tokens = tokens.view(-1, tokens.size(-1))  # batch * num_choices, seq_len

        mask = article_with_question['bert_tokens']['mask']
        mask = mask.view(-1, tokens.size(-1))
        type_ids = article_with_question['bert_tokens']['type_ids']
        type_ids = type_ids.view(-1, tokens.size(-1))

        embedded_text = self._text_field_embedder(
            {"bert_tokens": {"token_ids": tokens, "mask": mask,
                             "type_ids": type_ids}})  # batch * num_choices, seq_len, hidden_dim
        embedded_text = self._seq2vec(embedded_text, mask)  # batch * num_choices, vector_dim

        if self.training:
            if self._dropout:
                embedded_text = self._dropout(embedded_text)
            logits = self._linear_layer(embedded_text)  # batch* num_choices, num_labels
            reshaped_logits = logits.view(-1, self._num_choices)  # batch, num_choices

            _loss = self.scorer.update(reshaped_logits, label)
            return {  # "reshape_logits": reshaped_logits,
                "predicted_label": torch.argmax(reshaped_logits, dim=-1),
                "label": label,
                "loss": _loss}
        self.eval()
        with torch.no_grad():
            logits = self._linear_layer(embedded_text)  # batch* num_choices, num_labels
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
