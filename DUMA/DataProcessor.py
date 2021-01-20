# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:21:22 2021

@author: 31906
"""

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from wiktionaryparser import WiktionaryParser
import re


class SemEvalExample(object):
    def __init__(self, article, choice_0, choice_1, choice_2, choice_3, choice_4, label=None):
        self.article = article
        self.choices = ([
            choice_0,
            choice_1,
            choice_2,
            choice_3,
            choice_4,
        ])
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        attributes = [
            "article: {}".format(self.article),
            "choice_0: {}".format(self.choices[0]),
            "choice_1: {}".format(self.choices[1]),
            "choice_2: {}".format(self.choices[2]),
            "choice_3: {}".format(self.choices[3]),
            "choice_4: {}".format(self.choices[4]),
        ]

        if self.label is not None:
            attributes.append("label: {}".format(self.label))

        return ", ".join(attributes)


class InputFeatures(object):
    def __init__(self, choices_features, length, label):
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.length = length
        self.label = label


def read_recam(path, is_labeling, add_wiki=False):
    """
    Parameters
    ----------
    path : str
        data path.
        
    Returns
    -------
    examples : list
        list of object.

    """
    parser = WiktionaryParser()
    with open(path, mode='r', encoding="utf8") as f:
        reader = jsonlines.Reader(f)
        examples = []
        for line in reader:
            example = SemEvalExample(
                article=line['article'],
                choice_0=generate_choice(parser, line['option_0'], line['question'], add_wiki),
                choice_1=generate_choice(parser, line['option_1'], line['question'], add_wiki),
                choice_2=generate_choice(parser, line['option_2'], line['question'], add_wiki),
                choice_3=generate_choice(parser, line['option_3'], line['question'], add_wiki),
                choice_4=generate_choice(parser, line['option_4'], line['question'], add_wiki),
                label = int(line['label']) if is_labeling else None,
            )
            examples.append(example)
        return examples


def convert_examples_to_features(examples, tokenizer, max_seq_len):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example_index, example in tqdm(enumerate(examples)):

        article_tokens = tokenizer.tokenize(example.article)
        choices_features = []

        for choice_index, choice in enumerate(example.choices):
            choice_tokens = tokenizer.tokenize(choice)
            _truncate_seq_pair(article_tokens, choice_tokens, max_seq_len - 2)

            length = len(article_tokens)
            tokens = article_tokens + ["[SEP]"] + choice_tokens + ["[SEP]"]
            segment_ids = [0] * (length + 1) + [1] * (len(choice_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_len - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        features.append(InputFeatures(choices_features=choices_features, length=length, label=label))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def convert_features_to_dataset(features, is_labeling):
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_length = torch.tensor([f.length for f in features], dtype=torch.long)
    if is_labeling:
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_length, all_label)
    else:
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_length)


def generate_choice(parser, option, question, add_wiki):
    choice = question.replace('@placeholder', option)
    if add_wiki:
        syns, ts = get_wiki_definition(parser, option)
        if syns:
            ts = syns + " " + ts
            choice_length = len(choice)
            text = ts[:min(len(ts), choice_length)]
            choice = choice + ' ' + text
        else:
            choice_length = len(choice)
            text = ts[:min(len(ts), choice_length)]
            choice = choice + ' ' + text
    print(choice)
    return choice


def get_wiki_definition(parser, word_str):
    word = parser.fetch(word_str)
    syns = []
    ts = []
    for item in word:
        for definition in item['definitions']:
            relatedWords = definition['relatedWords']
            for word in relatedWords:
                if word['relationshipType'] == 'synonyms':
                    syn = ' '.join(
                        [re.sub('\(.*\):+|see Thesaurus:|and Thesaurus:|See also Thesaurus:|see also Thesaurus:', '',
                                item).strip() for item
                         in word['words']])
                    syns.append(syn)
            text = definition['text']
            for t in text:
                t = re.sub('\(((?!\)).)*\)|\[((?!\]).)*\]', ';', t).strip()
                t = re.sub('^([;\s]+)|([;\s]+)$', '', t).strip()
                t = re.sub(';\.', '.', t).strip()
                if not t == word_str:
                    ts.append(t)
    syns = ';'.join(syns)
    ts = ' '.join(ts)
    return syns, ts


if __name__ == "__main__":
    # train_examples = read_recam('/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_train.jsonl', is_labeling=True, add_wiki=True)
    # tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    # train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_len=100)
    # train_dataset = convert_features_to_dataset(train_features, is_labeling=True)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)
    # pass
    read_recam('/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_train.jsonl', is_labeling=True, add_wiki=True)
    
