# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:21:22 2021

@author: JIANG Yuxin
"""

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from get_wordnet_definition import add_wordnet_definition
import re


class SemEvalExample(object):
    def __init__(self, passage, option_0, option_1, option_2, option_3, option_4, label=None):
        self.passage = passage
        self.options = ([
            option_0,
            option_1,
            option_2,
            option_3,
            option_4,
        ])
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        attributes = [
            "passage: {}".format(self.passage),
            "option_0: {}".format(self.options[0]),
            "option_1: {}".format(self.options[1]),
            "option_2: {}".format(self.options[2]),
            "option_3: {}".format(self.options[3]),
            "option_4: {}".format(self.options[4]),
        ]

        if self.label is not None:
            attributes.append("label: {}".format(self.label))

        return ", ".join(attributes)


class InputFeatures(object):
    def __init__(self, options_features, length, label):
        self.options_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for _, input_ids, input_mask, segment_ids in options_features
        ]
        self.length = length
        self.label = label


def read_recam(path, is_labeling, add_definition=False):
    if add_definition:
        add_wordnet_definition_path = add_wordnet_definition(path)
        with open(add_wordnet_definition_path, mode='r', encoding="utf8") as f:
            reader = jsonlines.Reader(f)
            examples = []
            for line in reader:
                example = SemEvalExample(
                    passage = line['article'],
                    option_0 = line['question'].replace('@placeholder', line['option_0']['ans'] + ' ' + line['option_0']['definitions']),
                    option_1 = line['question'].replace('@placeholder', line['option_1']['ans'] + ' ' + line['option_1']['definitions']),
                    option_2 = line['question'].replace('@placeholder', line['option_2']['ans'] + ' ' + line['option_2']['definitions']),
                    option_3 = line['question'].replace('@placeholder', line['option_3']['ans'] + ' ' + line['option_3']['definitions']),
                    option_4 = line['question'].replace('@placeholder', line['option_4']['ans'] + ' ' + line['option_4']['definitions']),
                    label = int(line['label']) if is_labeling else None,
                )
                examples.append(example)
            return examples
    else:
        with open(path, mode='r', encoding="utf8") as f:
            reader = jsonlines.Reader(f)
            examples = []
            for line in reader:
                example = SemEvalExample(
                    passage = line['article'],
                    option_0 = line['question'].replace('@placeholder', line['option_0']),
                    option_1 = line['question'].replace('@placeholder', line['option_1']),
                    option_2 = line['question'].replace('@placeholder', line['option_2']),
                    option_3 = line['question'].replace('@placeholder', line['option_3']),
                    option_4 = line['question'].replace('@placeholder', line['option_4']),
                    label = int(line['label']) if is_labeling else None,
                )
                examples.append(example)
            return examples


def convert_examples_to_features(examples, tokenizer, max_seq_len):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example_index, example in tqdm(enumerate(examples)):

        passage_tokens = tokenizer.tokenize(example.passage)
        options_features = []

        for option_index, option in enumerate(example.options):
            option_tokens = tokenizer.tokenize(option)
            _truncate_seq_pair(passage_tokens, option_tokens, max_seq_len - 3)

            length = len(passage_tokens)
            tokens = ["[CLS]"] + passage_tokens + ["[SEP]"] + option_tokens + ["[SEP]"]
            segment_ids = [0] * (length + 2) + [1] * (len(choice_tokens) + 1)

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

            options_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        features.append(InputFeatures(options_features=options_features, length=length, label=label))

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
    return [[option[field] for option in feature.options_features] for feature in features]


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

    

if __name__ == "__main__":
    train_examples = read_recam('/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_train.jsonl', is_labeling=True, add_definition=True)
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_len=150)
    train_dataset = convert_features_to_dataset(train_features, is_labeling=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)
    pass
    
