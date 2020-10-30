import json
import re
import logging
from typing import Dict

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from overrides import overrides
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


@DatasetReader.register("RCAM_reader")
class RCAMDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 article_length_limit: int = None,
                 question_length_limit: int = None,
                 skip_invalid_examples: bool = False,
                 lazy: bool = False,
                 top_k_relative_sentence: int = None): # could be adjusted according to the article_length_limit
        super().__init__(lazy)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.article_length_limit = article_length_limit
        self.option_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples
        self.top_k_relative_sentence = top_k_relative_sentence
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        
    @overrides
    def _read(self, file_path: str):
        logger.info("Reading RCAM data from: %s", file_path)
        with open(file_path, encoding='utf-8') as f:
            list_data = list(f)
            for line in list_data:
                article, option_list, label_list = parse_sentence(line, self.model, self.top_k_relative_sentence)
                yield self.text_to_instance(article, option_list, label_list)

    @overrides
    def text_to_instance(self, article: str, option_list: list, label_list: list = None) -> Instance:
        fields: Dict[str, Field] = {}
        metadata = {}

        option_token_list = []
        combine_token = []

        article_tokens = self._tokenizer.tokenize(article)
        if self.article_length_limit is not None:
            article_tokens = article_tokens[: self.article_length_limit]
            metadata['article_tokens'] = [token.text for token in article_tokens]

        for option in option_list:
            option_tokens = self._tokenizer.tokenize(option)
            if self.option_length_limit is not None:
                option_tokens = option_tokens[:self.option_length_limit]
            option_token_list.append(option_tokens)

            article_with_question = self._tokenizer.add_special_tokens(article_tokens, option_tokens)
            combine_token.append(article_with_question)

            metadata['option_tokens'] = [[token.text for token in option] for option in option_token_list]

        article_with_question_field = ListField([TextField(combine, self._token_indexers) for combine in combine_token])
        fields['article_with_question'] = article_with_question_field
        if label_list is not None:
            label_field = ListField([LabelField(label) for label in label_list])
            fields['label'] = label_field
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)


def parse_sentence(line, model, k):
    data = json.loads(line, strict=False)
    article = data['article']
    question = data['question']
    article = process_article(article, re.sub('@placeholder', '', question), model, k)
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


def remove_contents(contents,targets):
    for content in contents:
    	while content in targets:
    		targets.remove(content)
    return targets
     
def find_contents(keys,targets):
    results = []
    for k in range(0,len(keys)):
    	for target in targets:
    		if target.find(keys[k]) != -1:
                    results.append(target)
    return results
     
def rmv_by_keys(keys,targets):
    return remove_contents(find_contents(keys,targets),targets)


def process_article(article, question, model, k):
    new_article = re.sub(r'\".*?\.\"','.',article)
    sentences = new_article.split('.')
    if(len(sentences) <= k):
        return ".".join(sentences)
    else:
        # remove sentences containing 'said'
        sentences = rmv_by_keys(['said',': .'], sentences)
        # Encode question and sentences of article
        embeddings1 = model.encode(question)
        embeddings2 = model.encode(sentences)
        # Compute cosine similarity between question and sentences of article
        cos_sim = util.pytorch_cos_sim(embeddings1, embeddings2)
        # return top_k similar sentences of article remaining the order
        top_k_index = sorted(cos_sim.numpy()[0].argsort()[-k:][::-1])
        article_final = []
        for i in top_k_index:
            article_final.append(sentences[i])
        return ".".join(article_final)
