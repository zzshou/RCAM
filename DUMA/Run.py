# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:59:19 2021

@author: 31906
"""

import os
import time
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

from DataProcessor_add_definition import read_recam, convert_examples_to_features, convert_features_to_dataset
from Model import MultiChoiceModel
from utils.params import Params
from utils.checks import ConfigurationError
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial


def seed_torch(seed=2021):
    """set the random seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(out, labels):
    """ compute the number of correct prediction """
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def get_next(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.data_iter = iter(self.data_loader)
            data = next(self.data_iter)
        return data


def train(params, train_datasets, model, eval_dataset):
    """ train the model """

    train_iters = []
    tr_batches = []

    for _, train_dataset in enumerate(train_datasets):
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=params['model']['batch_size'])
        train_iters.append(InfiniteDataLoader(train_dataloader))
        tr_batches.append(len(train_dataloader))

    ## set sampling proportion
    total_n_tr_batches = sum(tr_batches)
    sampling_prob = [float(n_batches) / total_n_tr_batches for n_batches in tr_batches]

    if params['model']['max_train_steps'] > 0:
        total_steps = params['model']['max_train_steps']
        params['model']['n_epoch'] = params['model']['max_train_steps'] // (
                total_n_tr_batches // params['model']['gradient_accumulation_steps']) + 1
    else:
        total_steps = total_n_tr_batches // params['model']['gradient_accumulation_steps'] * params['model']['n_epoch']

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': params['model']['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['model']['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                                num_training_steps=total_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", params['model']['n_epoch'])
    logger.info("  Gradient Accumulation steps = %d", params['model']['gradient_accumulation_steps'])
    logger.info("  Total optfimization steps = %d", total_steps)

    global_step = 0
    tr_loss = 0.0
    best_eval_accuracy = 0
    model.zero_grad()
    train_iterator = trange(int(params['model']['n_epoch']), desc="Epoch")

    if args.do_eval:
        if not os.path.exists(params['environment']['serialization_dir']):
            os.makedirs(params['environment']['serialization_dir'])
        output_eval_file = os.path.join(params['environment']['serialization_dir'], "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write("***** Eval results per %d training steps *****\n" % params['model']['evaluate_steps'])

    # added here for reproductibility
    seed_torch(params['environment']['random_seed'])

    for epoch in train_iterator:
        epoch_iterator = tqdm(range(total_n_tr_batches), desc="Training")
        batch_time_avg = 0.0
        train_accuracy = 0
        nb_train_examples = 0

        for step in epoch_iterator:
            batch_start = time.time()
            model.train()

            # select task id
            task_id = np.argmax(np.random.multinomial(1, sampling_prob))
            batch = train_iters[task_id].get_next()
            batch = tuple(t.to(args.device) for t in batch)
            # batch = tuple(t.cuda() for t in batch)
            logits = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                lengths=batch[3],
            )
            labels = batch[4]
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

            if params['model']['gradient_accumulation_steps'] > 1:
                loss = loss / params['model']['gradient_accumulation_steps']

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['model']['max_grad_norm'])

            tr_loss += loss.item()
            global_step += 1

            if (step + 1) % params['model']['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                batch_time_avg += time.time() - batch_start
                description = "Avg. time per gradient updating: {:.4f}s, loss: {:.4f}" \
                    .format(batch_time_avg / (step + 1), tr_loss / global_step)
                epoch_iterator.set_description(description)

            logits = logits.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            tmp_train_accuracy = accuracy(logits, labels)
            train_accuracy += tmp_train_accuracy
            nb_train_examples += batch[0].size(0)

            if args.do_eval:
                if global_step % params['model']['evaluate_steps'] == 0:
                    result = evaluate(params, eval_dataset, model, output_eval_file)
                    # logger.info(multi_choice_model.weights.state_dict())

                    # save the model having the best accuracy on dev dataset.
                    if result['eval_accuracy'] > best_eval_accuracy:
                        best_eval_accuracy = result['eval_accuracy']
                        now_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
                        torch.save({"model": multi_choice_model.state_dict(),
                                    "name": params['model']['bert_model']},
                                   os.path.join(params['environment']['serialization_dir'],
                                                "model-" + now_time + ".pt"))
                        logger.info("***** Better eval accuracy, save model successfully *****")

            if params['model']['max_train_steps'] > 0 and global_step > params['model']['max_train_steps']:
                epoch_iterator.close()
                break

        train_accuracy = train_accuracy / nb_train_examples
        logger.info("After epoch {:}, train_accuracy = {:.2%}".format(epoch, train_accuracy))

        if params['model']['max_train_steps'] > 0 and global_step > params['model']['max_train_steps']:
            epoch_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(params, eval_dataset, model, output_eval_file):
    """ evaluate the model """
    model_params = params['model']
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, batch_size=model_params['batch_size'])

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", model_params['batch_size'])

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        # batch = tuple(t.cuda() for t in batch)
        with torch.no_grad():
            logits = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                lengths=batch[3],
            )
            labels = batch[4]
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        tmp_eval_accuracy = accuracy(logits, labels)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
        nb_eval_examples += batch[0].size(0)

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}

    # write eval results to txt file.
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        logger.info("eval_loss = %.4f", eval_loss)
        logger.info("eval_accuracy = {:.2%}".format(eval_accuracy))
        writer.write("eval_loss = %s\n" % str(round(eval_loss, 4)))
        writer.write("eval_accuracy = %s\n" % (str(round(eval_accuracy * 100, 2)) + '%'))

    return result


def test(params, test_dataset, model):
    """ test the model """
    model_params = params['model']
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=model_params['batch_size'])

    # Test!
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", model_params['batch_size'])

    predictions = []
    logits_list = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        # batch = tuple(t.cuda() for t in batch)
        with torch.no_grad():
            logits = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                lengths=batch[3],
            )

        logits = logits.detach().cpu().numpy()
        prediction = np.argmax(logits, axis=1)
        predictions.extend(prediction)
        logits_list.extend(logits)

    # write predictions to csv file.
    pd.DataFrame({"predictions": predictions, "logits_list": logits_list}).to_csv(
        os.path.join(params['environment']['serialization_dir'], "task1_predictions.csv"), header=0)


def create_serialization_dir(params: Params) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.
    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    """
    serialization_dir = params['environment']['serialization_dir']
    recover = params['environment']['recover']
    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
                                     f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, "config.json")
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            if params != loaded_params:
                raise ConfigurationError("Training configuration does not match the configuration we're "
                                         "recovering from.")

    else:
        if recover:
            raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
                                     "does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)
        params.to_file(os.path.join(serialization_dir, "config.json"))


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    import argparse

    import os

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser(description='DUMA')

    parser.add_argument('config_file', help='config file.')
    parser.add_argument('--do_train', action='store_true', help='if training, default False')
    parser.add_argument('--do_eval', action='store_true', help='if evaluating, default False')
    parser.add_argument('--do_test', action='store_true', help='if testing, default False')
    parser.add_argument('--checkpoint', default=None,
                        help='if use fine-tuned bert model, please enter the checkpoint path.')
    # parser.add_argument('--batch_size', default=1)
    # parser.add_argument('--n_choice', default=5)
    # parser.add_argument('--max_seq_len', default=150)
    # parser.add_argument('--gradient_accumulation_steps', default=1)
    # parser.add_argument('--weight_decay', default=0.01)
    # parser.add_argument('--lr', default=1e-5)
    # parser.add_argument('--max_grad_norm', default=10)
    # parser.add_argument('--n_head', default=64)
    # parser.add_argument('--d_k', default=64)
    # parser.add_argument('--d_v', default=64)
    # parser.add_argument('--attention_dropout', default=0.1)
    # parser.add_argument('--sequence_dropout', default=0.33)
    # parser.add_argument('--n_layer', default=1)
    # parser.add_argument('--pool', default='max')
    args = parser.parse_args()

    params = Params.from_file(args.config_file)

    # set the random seed
    environment_params = params['environment']
    seed_torch(environment_params['random_seed'])

    serialization_dir = environment_params['serialization_dir']
    if os.path.exists(serialization_dir):
        pass
    else:
        create_serialization_dir(params)

    handler = logging.FileHandler(os.path.join(params['environment']['serialization_dir'], "logfile"))
    logger.addHandler(handler)
    logger.info(params)

    # use GPU or CPU
    if torch.cuda.is_available() and environment_params['cuda'] >= 0:
        args.device = torch.device('cuda', environment_params['cuda'])
    else:
        args.device = torch.device('cpu')
    # args.device = torch.device("cuda")
    logger.info("  Device = %s", args.device)

    # data read and process

    tokenizer = AutoTokenizer.from_pretrained(params['model']['bert_model'])

    data_params = params['data']
    if data_params['train_data_paths'] and args.do_train:
        logger.info("***** Loading training data *****")
        train_datasets = []
        for train_data_path in data_params['train_data_paths']:
            train_examples = read_recam(train_data_path, is_labeling=True, add_wiki=True)
            train_features = convert_examples_to_features(train_examples, tokenizer,
                                                          max_seq_len=data_params['max_seq_len'])
            train_dataset = convert_features_to_dataset(train_features, is_labeling=True)
            train_datasets.append(train_dataset)

    if data_params['dev_data_path'] and args.do_eval:
        logger.info("***** Loading evaluating data *****")
        evaluate_examples = read_recam(data_params['dev_data_path'], is_labeling=True, add_wiki=True)
        evaluate_features = convert_examples_to_features(evaluate_examples, tokenizer,
                                                         max_seq_len=data_params['max_seq_len'])
        evaluate_dataset = convert_features_to_dataset(evaluate_features, is_labeling=True)

    if data_params['test_data_path'] and args.do_test:
        logger.info("***** Loading testing data *****")
        test_examples = read_recam(data_params['test_data_path'], is_labeling=False, add_wiki=True)
        test_features = convert_examples_to_features(test_examples, tokenizer, max_seq_len=data_params['max_seq_len'])
        test_dataset = convert_features_to_dataset(test_features, is_labeling=False)

    # bulid the model
    logger.info("***** Building multi_choice model based on '%s' BERT model *****", params['model']['bert_model'])
    bert_model = AutoModel.from_pretrained(params['model']['bert_model'])
    multi_choice_model = MultiChoiceModel(bert_model, params, is_requires_grad=True).to(args.device)
    # multi_choice_model = nn.parallel.DistributedDataParallel(multi_choice_model, output_device=1)

    multi_choice_model.to(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(1))
        # checkpoint = torch.load(args.checkpoint)
        logger.info("***** Loading saved model based on '%s' *****", args.checkpoint)
        multi_choice_model.load_state_dict(checkpoint["model"])

    # print the number of parameters of the model
    total_params = sum(p.numel() for p in multi_choice_model.parameters())
    logger.info("{:,} total parameters.".format(total_params))
    total_trainable_params = sum(p.numel() for p in multi_choice_model.parameters() if p.requires_grad)
    logger.info("{:,} training parameters.".format(total_trainable_params))

    # train and evaluate
    if args.do_train and args.do_eval:
        global_step, tr_loss = train(params, train_datasets, multi_choice_model, evaluate_dataset)
        logger.info("***** End of training *****")
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # only evaluate
    elif not args.do_train and args.do_eval:
        if not os.path.exists(serialization_dir):
            os.makedirs(serialization_dir)
        output_eval_file = os.path.join(serialization_dir, "evaluate_result.txt")
        result = evaluate(params, evaluate_dataset, multi_choice_model, output_eval_file)
        logger.info("***** End of evaluating *****")

    else:
        pass

    # test
    if args.do_test:
        if not os.path.exists(serialization_dir):
            os.makedirs(serialization_dir)
        test(params, test_dataset, multi_choice_model)
        logger.info("***** End of testing *****")
'''
    space4kge = {
        "rnn_dim": hp.choice("rnn_dim", [32, 64, 128]),
        "eds_dim": hp.choice("eds_dim", [32, 64, 128, 256]),
        "num_units": hp.choice("num_units", [32, 64, 128, 256]),
        "dropout": hp.uniform("dropout", 0.0, 0.4),
        "gcn_layer": hp.choice("gcn_layer", [2]),
        "mode": hp.choice("mode", ['rnn']),
        "use_gate": hp.choice("use_gate", [True, False]),
        "learning_rate": hp.uniform("learning_rate", 0.0005, 0.001),
        # "n_dim": hp.choice("n_dim", [512]),
        # "lamb": hp.uniform("lamb", 0, 0.1)
    }


    def f(args):
        mrr = train_model(params, args)
        return {'loss': -mrr['dev_F1'], 'status': STATUS_OK}


    trials = Trials()
    best = fmin(f, space4kge, algo=partial(tpe.suggest, n_startup_jobs=25), max_evals=500, trials=trials)

    print('best performance:', best)
    '''
