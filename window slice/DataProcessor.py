# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:34:20 2021

@author: 31906
"""

import jsonlines
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset


class SemEvalExample(object):
    def __init__(self, document, padding, choice_0, choice_1, choice_2, choice_3, choice_4, label=None):
        self.document = document
        self.padding = padding
        self.choices = [
            choice_0,
            choice_1,
            choice_2,
            choice_3,
            choice_4,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        attributes = [
            "document: {}".format(self.document),
            "padding: {}".format(self.padding),
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
    def __init__(self, choices_features, padding, label):
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.padding = padding
        self.label = label


def get_split(text, sep, overlap):
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
    l_parcial = [] 
    if len(text.split()) // (sep - overlap) > 0:
        n = len(text.split()) // (sep - overlap)
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text.split()[:sep]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text.split()[w * (sep - overlap) : w * (sep - overlap) + sep]
            l_total.append(" ".join(l_parcial))
    return l_total


def read_recam(path, sep, overlap, n_slice):
    """
    Parameters
    ----------
    path : str
        data path.
    sep : int
        length of each slice.
    overlap : int
        length of overlap between two slices.
    n_slice : int
        the max number of slices.

    Returns
    -------
    examples : list
        list of object.

    """
    with open(path, mode='r', encoding="utf8") as f:
        reader = jsonlines.Reader(f)
        examples = []
        for line in reader:
            document = get_split(line['article'], sep, overlap)[: n_slice]
            
            example = SemEvalExample(
            document = document,
            padding = len(document),
            choice_0 = line['question'].replace('@placeholder', line['option_0']),
            choice_1 = line['question'].replace('@placeholder', line['option_1']),
            choice_2 = line['question'].replace('@placeholder', line['option_2']),
            choice_3 = line['question'].replace('@placeholder', line['option_3']),
            choice_4 = line['question'].replace('@placeholder', line['option_4']),
            label = int(line['label']),
            )
            examples.append(example)
        return examples
    
    
def convert_examples_to_features(examples, n_slice, tokenizer, max_seq_len):
    """Loads a data file into a list of `InputBatch`s."""
    
    features = []
    for example_index, example in enumerate(examples):

        choices_features = []

        for choice_index, choice in enumerate(example.choices):
            choice_tokens = tokenizer.tokenize(choice)

            paragraph_tokens = []
            paragraph_input_ids = []
            paragraph_input_mask = []
            paragraph_segment_ids = []
            
            for i in range(n_slice):
                if i < example.padding:
                    paragraph_tokens = tokenizer.tokenize(example.document[i])
                    _truncate_seq_pair(paragraph_tokens, choice_tokens, max_seq_len - 3)

                    tokens = ["[CLS]"] + paragraph_tokens + ["[SEP]"] + choice_tokens + ["[SEP]"]
                    segment_ids = [0] * (len(paragraph_tokens) + 2) + [1] * (len(choice_tokens) + 1)

                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)

                    # Zero-pad up to the sequence length.
                    padding = [0] * (max_seq_len - len(input_ids))
                    input_ids += padding
                    input_mask += padding
                    segment_ids += padding
                
                else:
                    tokens = ["[PAD]"] * max_seq_len
                    input_ids = [0] * max_seq_len
                    input_mask = [0] * max_seq_len
                    segment_ids = [0] * max_seq_len

                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len

                paragraph_tokens.append(tokens)
                paragraph_input_ids.append(input_ids)
                paragraph_input_mask.append(input_mask)
                paragraph_segment_ids.append(segment_ids)

            choices_features.append((paragraph_tokens, paragraph_input_ids, paragraph_input_mask, paragraph_segment_ids))

        padding = example.padding
        label = example.label
        features.append(InputFeatures(choices_features=choices_features, padding=padding, label=label))

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


def convert_features_to_dataset(features):
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_padding = torch.tensor([f.padding for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_padding, all_label)


if __name__ == "__main__":
    train_examples = read_recam('/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_train.jsonl', sep=80, overlap=50, n_slice=10)
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    train_features = convert_examples_to_features(train_examples, 10, tokenizer, max_seq_len=100)
    train_dataset = convert_features_to_dataset(train_features)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
    pass
