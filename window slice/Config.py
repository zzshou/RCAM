# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:33:38 2021

@author: 31906
"""

import argparse


def arg_conf():
    parser = argparse.ArgumentParser(description = 'MultiChoiceModel')
    
    # parameters of environment
    parser.add_argument('-cuda', type=int, default=0, help="which device, default gpu.")
    parser.add_argument('-random_seed', type=int, default=2020, help='set the random seed so that we can reporduce the result.')
    
    # parameters of data processor
    parser.add_argument('-data_path', default=None)
    parser.add_argument('-n_choice', type=int, default=5, help='number of choices.')
    parser.add_argument('-max_seq_len', type=int, default=100, help='max sequence length of article + question.')
    parser.add_argument('--sep', type=int, default=80, help='length of each slice.')
    parser.add_argument('-overlap', type=int, default=40, help='length of overlap between two slices.')
    parser.add_argument('-n_slice', type=int, default=10, help='the max number of slices.')
    
    # parameters of model
    parser.add_argument('-checkpoint', default=None, help='if use fine-tuned bert model, please enter the checkpoint path.')
    parser.add_argument('-bert_model', default=None, help='model name can be accessed from huggingface')
    parser.add_argument('-method', type=str, choices=['max','atten','transformer'], default='transformer', help='choice one method after bert_model.')
    # parameters of transformer
    parser.add_argument('-n_layer', type=int, default=2, help='num of layer in transoformer') 
    parser.add_argument('-n_head', type=int, default=12, help='num of head in transoformer') 
    parser.add_argument('-d_inner', type=int, default=1024, help='dimension of FNN in transoformer') 
    parser.add_argument('-dropout', type=float, default=0.1, help='dropout in transoformer') 
    
    # parameters of training
    parser.add_argument('-epoch', type=int, default=3, help='number of training epochs')
    parser.add_argument('-lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('-save_path', default=None, help='model save path') 

    # args = parser.parse_args()
    args = parser.parse_known_args()[0] #类jupyter环境下用这个

    # 打印出对象的属性和方法
    print(vars(args))
    return args


if __name__ == "__main__":
    args = arg_conf()
    
