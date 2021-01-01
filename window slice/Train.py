# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:58:29 2021

@author: 31906
"""

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import os
import time
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

from Config import arg_conf
from DataProcessor import read_recam, convert_examples_to_features, select_field
from Model import MultiChoiceModel


def seed_torch(seed=2020):
    """set the random seed"""
    
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_features, evaluate_features, model, optimizer, max_gradient_norm=10, is_evaluate=True):

    total_steps = len(train_features) * args.epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    for epoch in range(1, args.epoch + 1):
        model.train()
        random.shuffle(train_features)
        epoch_loss = 0.0
        correct_preds = 0

        for feature in train_features:
            input_ids = torch.tensor(select_field(feature, "input_ids"), dtype=torch.long).to(args.device)
            input_mask = torch.tensor(select_field(feature, "input_mask"), dtype=torch.long).to(args.device)
            segment_ids = torch.tensor(select_field(feature, "segment_ids"), dtype=torch.long).to(args.device)
            label = feature.label

            optimizer.zero_grad()
            loss, logits = model(input_ids=input_ids, 
                                 attention_mask=input_mask,
                                 token_type_ids=segment_ids,
                                 labels = label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            predicted_label = logits.argmax().item()
            if predicted_label == label:
                correct_preds += 1

        epoch_loss = epoch_loss/len(train_features)
        epoch_accuracy = correct_preds/len(train_features)
        print("==>>> Epoch {:d} training loss: {:.4f}, training accuracy: {:.2f}%"\
              .format(epoch, epoch_loss, epoch_accuracy*100))

        if is_evaluate:
            epoch_loss, epoch_accuracy, true_labels, predictions = evaluate(model, evaluate_features)
            print("      Epoch {:d} evaluating loss: {:.4f}, evaluating accuracy: {:.2f}%"\
                  .format(epoch, epoch_loss, epoch_accuracy*100))
        print('')
    
    # save the model
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    torch.save({"model": model.state_dict(), 
                "method": args.method},
                os.path.join(args.save_path, "model-" + now_time + ".pt"))
    print("save model successfully!")
    
    return pd.DataFrame({"true_labels": true_labels,"predictions": predictions})


def evaluate(model, evaluate_features):
    model.eval()
    epoch_loss = 0.0
    correct_preds = 0
    true_labels = []
    predictions = []
    with torch.no_grad():
        for feature in evaluate_features:
            input_ids = torch.tensor(select_field(feature, "input_ids"), dtype=torch.long).to(args.device)
            input_mask = torch.tensor(select_field(feature, "input_mask"), dtype=torch.long).to(args.device)
            segment_ids = torch.tensor(select_field(feature, "segment_ids"), dtype=torch.long).to(args.device)
            label = feature.label

            loss, logits = model(input_ids=input_ids, 
                                 attention_mask=input_mask,
                                 token_type_ids=segment_ids,
                                 labels = label)

            epoch_loss += loss.item()
            true_labels.append(label)
            predicted_label = logits.argmax().item()
            predictions.append(predicted_label)
            if predicted_label == label:
                correct_preds += 1

        epoch_loss = epoch_loss/len(evaluate_features)
        epoch_accuracy = correct_preds/len(evaluate_features)

    return epoch_loss, epoch_accuracy, true_labels, predictions  




if __name__ == "__main__":
    
    args = arg_conf()
    seed_torch(args.random_seed)
    
    # if use GPU or CPU
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')
        
    # data read and process
    print("* Loading training data and evaluating data...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_examples = read_recam(os.path.join(args.data_path, 'Task_1_train.jsonl'), sep=args.sep, overlap=args.overlap, n_slice=args.n_slice)
    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_len=args.max_seq_len)
    evaluate_examples = read_recam(os.path.join(args.data_path, 'Task_1_dev.jsonl'), sep=args.sep, overlap=args.overlap,  n_slice=args.n_slice)
    evaluate_features = convert_examples_to_features(evaluate_examples, tokenizer, max_seq_len=args.max_seq_len)
    
    print("* Building model...")
    # if use fine-tuned bert_model or pretrained bert_model
    bert_model = AutoModel.from_pretrained(args.bert_model)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        bert_model.load_state_dict(checkpoint["model"])
        multi_choice_model = MultiChoiceModel(bert_model, args, is_requires_grad=False).to(args.device)
    else:
        multi_choice_model = MultiChoiceModel(bert_model, args, is_requires_grad=True).to(args.device)
          
    param_optimizer = list(multi_choice_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
                {
                    'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay':0.01
                    },
                {
                    'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay':0.0
                    }
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_params = sum(p.numel() for p in multi_choice_model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in multi_choice_model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    print("* Training model...")
    results = train(args=args, train_features=train_features, evaluate_features=evaluate_features, \
          model=multi_choice_model, optimizer=optimizer, max_gradient_norm=10, is_evaluate=True)











  