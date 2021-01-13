# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:59:19 2021

@author: 31906
"""


import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

from Config import arg_conf
from DataProcessor import read_recam, convert_examples_to_features, convert_features_to_dataset
from Model import MultiChoiceModel


def seed_torch(seed=2021):
    """set the random seed"""
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(out, labels):
    """ compute the number of correct prediction """ 
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def train(args, train_dataset, model, eval_dataset):
    """ train the model """
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    
    if args.max_train_steps > 0:
        total_steps = args.max_train_steps
        args.n_epoch = args.max_train_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.n_epoch
  
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {
                    'params':[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': args.weight_decay
            },
            {
                    'params':[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay':0.0
            }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.n_epoch)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_steps)

    global_step = 0
    tr_loss = 0.0
    best_eval_accuracy = 0
    model.zero_grad()
    train_iterator = trange(int(args.n_epoch), desc="Epoch")
    
    if args.do_eval:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        output_eval_file = os.path.join(args.save_path, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write("***** Eval results per %d training steps *****\n" % args.evaluate_steps)
                
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Training")
        batch_time_avg = 0.0
        train_accuracy = 0; nb_train_examples = 0
        
        for step, batch in enumerate(epoch_iterator):
            batch_start = time.time()
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            loss, logits = model(
                                    input_ids = batch[0], 
                                    attention_mask = batch[1],
                                    token_type_ids = batch[2],
                                    lengths = batch[3],
                                    labels = batch[4],
                                )
      
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                batch_time_avg += time.time() - batch_start
                description = "Avg. time per gradient updating: {:.4f}s, loss: {:.4f}"\
                                .format(batch_time_avg/(step+1), tr_loss/global_step)
                epoch_iterator.set_description(description)
            
            logits = logits.detach().cpu().numpy()
            labels = batch[4].to("cpu").numpy()
            tmp_train_accuracy = accuracy(logits, labels)
            train_accuracy += tmp_train_accuracy
            nb_train_examples += batch[0].size(0)
            
            if args.do_eval:
                if global_step % args.evaluate_steps == 0:
                    result = evaluate(args, eval_dataset, model, output_eval_file)
                    
                    # save the model having the best accuracy on dev dataset.
                    if result['eval_accuracy'] > best_eval_accuracy:
                        best_eval_accuracy = result['eval_accuracy']
                        now_time = time.strftime('%Y-%m-%d',time.localtime(time.time()))
                        torch.save({"model": multi_choice_model.state_dict(), 
                                    "name": args.bert_model},
                                    os.path.join(args.save_path, "model-" + now_time + ".pt"))
                        logger.info("***** Better eval accuracy, save model successfully *****")
                        
            if args.max_train_steps > 0 and global_step > args.max_train_steps:
                epoch_iterator.close()
                break
        
        train_accuracy = train_accuracy / nb_train_examples
        logger.info("train_accuracy = {:.2%}".format(train_accuracy))
        
        if args.max_train_steps > 0 and global_step > args.max_train_steps:
            epoch_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, output_eval_file):
    """ evaluate the model """
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size)
    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            loss, logits = model(
                                    input_ids = batch[0], 
                                    attention_mask = batch[1],
                                    token_type_ids = batch[2],
                                    lengths = batch[3],
                                    labels = batch[4],
                                )
                                
            eval_loss += loss.item()
    
        logits = logits.detach().cpu().numpy()
        labels = batch[4].to("cpu").numpy()
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
        writer.write("eval_accuracy = %s\n" % (str(round(eval_accuracy*100, 2))+'%'))

    return result




if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = arg_conf()
    
    # set the random seed
    seed_torch(args.random_seed)
    
    # if use GPU or CPU
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')
    logger.info("  Device = %s", args.device)
        
    # data read and process
    logger.info("***** Loading training data and evaluating data *****")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_examples = read_recam(args.train_data_path)
    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_len=args.max_seq_len)
    train_dataset = convert_features_to_dataset(train_features)
    evaluate_examples = read_recam(args.dev_data_path)
    evaluate_features = convert_examples_to_features(evaluate_examples, tokenizer, max_seq_len=args.max_seq_len)
    evaluate_dataset = convert_features_to_dataset(evaluate_features)
    
    # bulid the model
    logger.info("***** Building multi_choice model based on '%s' BERT model *****", args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model)
    multi_choice_model = MultiChoiceModel(bert_model, args, is_requires_grad=True).to(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        if checkpoint["name"] == args.bert_model:
            logger.info("***** Loading saved model based on '%s' *****", checkpoint["name"])
            multi_choice_model.load_state_dict(checkpoint["model"])
        else:
            raise Exception("The loaded model does not match the pre-trained model", checkpoint["name"])
            
    # print the number of parameters of the model
    total_params = sum(p.numel() for p in multi_choice_model.parameters())
    logger.info("{:,} total parameters.".format(total_params))
    total_trainable_params = sum(p.numel() for p in multi_choice_model.parameters() if p.requires_grad)
    logger.info("{:,} training parameters.".format(total_trainable_params))

    # train and evaluate
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, multi_choice_model, evaluate_dataset)
        logger.info("***** End of training *****")
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
        