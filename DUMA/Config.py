# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:34:21 2021

@author: 31906
"""


import argparse


def arg_conf():
    parser = argparse.ArgumentParser(description = 'DUMA')
    
    # parameters of environment
    parser.add_argument('-cuda', type=int, default=0, help="which device, default gpu.")
    parser.add_argument('-random_seed', type=int, default=2021, help='set the random seed so that we can reporduce the result.')
    
    # parameters of data processor
    parser.add_argument('-train_data_paths', nargs='+', default=None, help='data paths of multi train datasets.')
    parser.add_argument('-dev_data_path', default=None, help='data path of eval dataset.')
    parser.add_argument('-test_data_path', default=None, help='data path of test dataset.')
    parser.add_argument('-n_choice', type=int, default=5, help='number of choices.')
    parser.add_argument('-max_seq_len', type=int, default=150, help='max sequence length of article + question.')
    
    # parameters of model
    parser.add_argument('-checkpoint', default=None, help='if use fine-tuned bert model, please enter the checkpoint path.')
    parser.add_argument('-bert_model', default=None, help='model name can be accessed from huggingface')
    parser.add_argument('-n_last_layer', default=1, type=int, help='weighted sum of last layers of bert model')
    # parameters of co-attention
    parser.add_argument('-n_layer', type=int, default=1, help='num of layer in co-attention') 
    parser.add_argument('-n_head', type=int, default=64, help='num of head in co-attention') 
    parser.add_argument('-d_k', type=int, default=64, help='dimension of key and query in co-attention') 
    parser.add_argument('-d_v', type=int, default=64, help='dimension of value in co-attention') 
    parser.add_argument('-dropout', type=float, default=0.1, help='dropout in co-attention') 
 
    # parameters of training
    parser.add_argument('-do_train', action='store_true', help='if training, default False')
    parser.add_argument('-do_eval', action='store_true', help='if evaluating, default False')
    parser.add_argument('-do_test', action='store_true', help='if testing, default False')
    parser.add_argument('-evaluate_steps', type=int, default=200, help='evaluate on the dev set at every xxx evaluate_steps.')
    parser.add_argument('-max_train_steps', type=int, default=-1, help='If > 0: set total number of training steps to perform. Override num_train_epochs.')
    parser.add_argument('-n_epoch', type=int, default=3, help='number of training epochs')
    parser.add_argument('-batch_size', type=int, default=2, help='number of examples per batch')
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1, help='num of gradient_accumulation_steps') 
    parser.add_argument("-max_grad_norm", type=float, default=10.0, help="Max gradient norm.")
    parser.add_argument('-weight_decay', type=float, default=0.01, help='regularize parameters')
    parser.add_argument('-lr', type=float, default=5e-6, help='initial learning rate')
    parser.add_argument('-save_path', default=None, help='model save path') 

    # args = parser.parse_args()
    args = parser.parse_known_args()[0] #类jupyter环境下用这个

    # 打印出对象的属性和方法
    print(vars(args))
    return args


if __name__ == "__main__":
    args = arg_conf()
