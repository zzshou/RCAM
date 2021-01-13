import json
import re
import logging
from typing import Dict

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ListField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer, Token
from overrides import overrides
from transformers import RobertaTokenizer
import stanza

logger = logging.getLogger(__name__)


@DatasetReader.register("RCAM_reader")
class RCAMDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 article_length_limit: int = 0,
                 question_length_limit: int = 0,
                 skip_invalid_examples: bool = False,
                 use_window_slice: bool = False,
                 window_slice_length: int = 0,
                 window_slice_overlap: int = 0,
                 max_window_slice_step: int = 0,
                 lazy: bool = False):
        super().__init__(lazy)
        self._tokenizer = tokenizer or RobertaTokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.article_length_limit = article_length_limit
        self.option_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples
        # self.nlp_pipeline = stanza.Pipeline('en', dir='/home/data/zshou/corpus', processors='tokenize,pos,ner',
        #                                     tokenize_pretokenized=True)
        self.use_window_slice = use_window_slice
        self.window_slice_length = window_slice_length
        self.window_slice_overlap = window_slice_overlap
        self.max_window_slice_step = max_window_slice_step

    @overrides
    def _read(self, file_path: str):
        logger.info("Reading RCAM data from: %s", file_path)
        with open(file_path, encoding='utf-8') as f:
            list_data = list(f)
            for line in list_data:
                article, option_list, label_list = parse_sentence(line)
                yield self.text_to_instance(article, option_list, label_list)

    @overrides
    def text_to_instance(self, article: str, option_list: list, label_list: list = None) -> Instance:
        fields: Dict[str, Field] = {}

        option_token_list = []
        combine_token = []
        if self.use_window_slice:
            paragraphs, num_paragraphs = get_split(article, self.window_slice_length, self.window_slice_overlap,
                                                   self.max_window_slice_step)
            for option in option_list:
                option_tokens = self._tokenizer.tokenize(option)
                option_token_list.append(option_tokens)
            for option_token in option_token_list:
                for paragraph in paragraphs:
                    paragraph_token = self._tokenizer.tokenize(paragraph)
                    paragraph_with_question = self._tokenizer.add_special_tokens(paragraph_token, option_token)
                    combine_token.append(paragraph_with_question)
            paragraph_with_question_field = ListField(
                [TextField(combine, self._token_indexers) for combine in combine_token])

            fields['paragraph_with_question_field'] = paragraph_with_question_field
            if label_list is not None:
                label_field = ListField([LabelField(label) for label in label_list])
                fields['label'] = label_field
            return Instance(fields)
        else:
            article_tokens = self._tokenizer.tokenize(article)
            if self.article_length_limit is not None:
                article_tokens = article_tokens[: self.article_length_limit]
            for option in option_list:
                option_tokens = self._tokenizer.tokenize(option)
                if self.option_length_limit is not None:
                    option_tokens = option_tokens[:self.option_length_limit]
                option_token_list.append(option_tokens)

                article_with_question = self._tokenizer.add_special_tokens(article_tokens, option_tokens)
                combine_token.append(article_with_question)

            article_with_question_field = ListField(
                [TextField(combine, self._token_indexers) for combine in combine_token])

            fields['article_with_question'] = article_with_question_field
            if label_list is not None:
                label_field = ListField([LabelField(label) for label in label_list])
                fields['label'] = label_field

            return Instance(fields)


def parse_sentence(line):
    data = json.loads(line, strict=False)
    article = data['article']
    question = data['question']
    option_list = []
    label_list = None
    for i in range(5):
        option_id = "option_" + str(i)
        answer = data[option_id]
        answer_candidate = re.sub('@placeholder', answer, question)
        option_list.append(answer_candidate)
    if 'label' in data:
        label_list = ['false'] * 5
        right_answer = data['label']
        label_list[right_answer] = 'true'
    return article, option_list, label_list


def get_split(text, sep, overlap, max_step):
    """
    Parameters
    ----------
    text : str
        text to be splited.
    sep : int
        length of each slice.
    overlap : int
        length of overlap between two slices.
    Returns
    -------
    l_total : list
        the splited text.

    """
    l_total = []
    n = min(max(len(text.split()) // (sep - overlap), 1), max_step)

    for w in range(n):
        if w == 0:
            l_parcial = text.split()[:sep]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text.split()[w * (sep - overlap): w * (sep - overlap) + sep]
            l_total.append(" ".join(l_parcial))
    return l_total, n


if __name__ == '__main__':
    file = 'data/trail_data/Task_1_Imperceptibility.jsonl'
