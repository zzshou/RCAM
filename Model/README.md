# XRJL-HKUST at SemEval-2021 Task 4: WordNet-Enhanced Dual Multi-head Co-Attention for Reading Comprehension of Abstract Meaning

This repository gives a Pytorch implementation of the method in the system description paper (see in Citation).

Prerequisite: python >=3.6, torch>=1.0, transformers>=3.0.

## 1. Model Architecture

<img src="https://github.com/zzshou/RCAM/blob/master/Model/model%20architecture.png" width="400" height="600">

## 2. Model training and evaluating
You are highly recommanded to run our code on **Google Colab**, where you could get free GPU resources.  
It's convenient to implement, just open a notebook and run the "Run.py" file (Please refer to Config.py for the specific meaning of each parameter):
```
! python Run.py -train_data_paths '/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_train.jsonl' \
          -dev_data_path='/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_dev.jsonl' \
          -n_choice=5 \
          -add_definition \
          -max_seq_len=150 \
          -bert_model='albert-xxlarge-v2' \
          -n_layer=1 \
          -n_head=64 \
          -d_k=64 \
          -d_v=64 \
          -dropout=0.1 \
          -do_train \
          -do_eval \
          -evaluate_steps=200 \
          -max_train_steps=-1 \
          -n_epoch=3 \
          -batch_size=2 \
          -gradient_accumulation_steps=1 \
          -max_grad_norm=10.0 \
          -weight_decay=0.01 \
          -lr=5e-6 \
          -save_path='/content/drive/My Drive/SemEval2021-task4/log/task_1/'
```

## 3. Model predicting
Open a notebook and run the "Run.py" file (Please refer to Config.py for the specific meaning of each parameter):
```
! python Run.py -test_data_path='/content/drive/My Drive/SemEval2021-task4/data/test_data/Task_1_test.jsonl' \
          -n_choice=5 \
          -max_seq_len=150 \
          -add_definition \
          -checkpoint='/content/drive/My Drive/SemEval2021-task4/log/task_1/model-2021-01-20.pt' \
          -bert_model='albert-xxlarge-v2' \
          -n_layer=1 \
          -n_head=64 \
          -d_k=64 \
          -d_v=64 \
          -dropout=0.1 \
          -do_test \
          -batch_size=2 \
          -save_path='/content/drive/My Drive/SemEval2021-task4/log/task_1/'
```

## 4. Citation
If our method is useful for your research, please consider citing:
