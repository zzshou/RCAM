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
# from wiktionaryparser import WiktionaryParser
import re


###### create the dictionary of options ######

def get_wiki_definition(parser, word_str):
    if word_str == 'export':  # exoprt是包的bug
        return ''
    word = parser.fetch(word_str)
    ts = []
    for item in word:
        for definition in item['definitions']:
            text = definition['text']
            for t in text:
                t = re.sub('\(((?!\)).)*\)|\[((?!\]).)*\]', ';', t).strip()
                t = re.sub('^([;\s]+)|([;\s]+)$', '', t).strip()
                t = re.sub(';\.', '.', t).strip()
                if (not word_str == 'as') and (
                        ('plural of' in t) or ('Third-person singular simple present indicative form of' in t)):
                    return get_wiki_definition(parser, t.split(' ')[-1])
                elif (not t == word_str) and ('present participle' not in t) and ('past tense' not in t) and (
                        'past participle' not in t):
                    ts.append(t)
    ts = ' '.join(ts[:3])  # 只返回前三个definition
    return ts


def create_dic(data_path, save_path):
    with open(data_path, mode='r', encoding="utf8") as f:
        reader = jsonlines.Reader(f)
        options = []
        for line in reader:
            options.append(line['option_0'])
            options.append(line['option_1'])
            options.append(line['option_2'])
            options.append(line['option_3'])
            options.append(line['option_4'])

    options = list(set(options))  # 先去重
    options.sort()  # 按照单词首字母排序

    dic = {}
    for _, option in tqdm(enumerate(options)):
        definition = get_wiki_definition(parser, option)
        print(definition)
        dic[option] = definition

    # save the dic to txt file.
    f = open(save_path, mode='w', encoding="utf8")
    f.write(str(dic))
    f.close()
    print("save dict successfully.")


###### end of create the dictionary of options ######


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


def generate_choice(option, question, add_wiki, dic):
    length = len(question.split(" "))
    if add_wiki:
        # choice = question.replace('@placeholder', option) + ' ' + " ".join(dic[option].split(" ")[:length]) if len(
        #     dic[option]) > 0 else question.replace('@placeholder', option)
        choice = question.replace('@placeholder', option['ans']) + '##SEP##' + option['definitions']
    else:
        choice = question.replace('@placeholder', option['ans'])
    return choice


def read_recam(path, is_labeling, add_wiki=False, dic=None):
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
    with open(path, mode='r', encoding="utf8") as f:
        reader = jsonlines.Reader(f)
        examples = []
        for line in reader:
            example = SemEvalExample(
                article=line['article'],
                choice_0=generate_choice(line['option_0'], line['question'], add_wiki, dic),
                choice_1=generate_choice(line['option_1'], line['question'], add_wiki, dic),
                choice_2=generate_choice(line['option_2'], line['question'], add_wiki, dic),
                choice_3=generate_choice(line['option_3'], line['question'], add_wiki, dic),
                choice_4=generate_choice(line['option_4'], line['question'], add_wiki, dic),
                label=int(line['label']) if is_labeling else None,
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
            choice, definition = choice.split('##SEP##')
            choice_tokens = tokenizer.tokenize(choice)
            definition_tokens = tokenizer.tokenize(definition)
            _truncate_seq_pair(article_tokens, choice_tokens, definition_tokens, max_seq_len - 2)

            length = len(article_tokens)
            tokens = article_tokens + ["[SEP]"] + choice_tokens + ["[SEP]"] + definition_tokens
            segment_ids = [0] * (length + 1) + [1] * (len(choice_tokens) + len(definition_tokens) + 1)

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


def _truncate_seq_pair(tokens_a, tokens_b, definiton_token, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(definiton_token)

        if total_length <= max_length:
            break
        if len(definiton_token) > len(tokens_b):
            definiton_token.pop()
        elif len(tokens_a) > (len(definiton_token) + len(definiton_token)):
            tokens_a.pop()
        elif len(definiton_token) > 0:
            definiton_token.pop()
        else:
            tokens_a.pop()
            print('option is longer than article')


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


def read_dic(dic_path):
    f = open(dic_path, mode='r', encoding="utf8")
    dic = eval(f.read())
    f.close()
    return dic


if __name__ == "__main__":
    dic = read_dic("/content/drive/My Drive/SemEval2021-task4/data/training_data/task_1_train_dic.txt")
    train_examples = read_recam('/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_train.jsonl',
                                is_labeling=True, add_wiki=True, dic=dic)
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_len=100)
    train_dataset = convert_features_to_dataset(train_features, is_labeling=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)
    pass
