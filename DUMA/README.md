# XRJL at SemEval-2020 Task 4: Dual Multi-Head Co-Attention for Abstract Meaning Understanding
## 1. Model Architecture

<img src="https://github.com/zzshou/RCAM/blob/master/DUMA/model%20architecture.png" width="600" height="800">

## 2. Model training and evaluating
Directly run Run.py file on the console (Please refer to Config.py for the specific meaning of each parameter):
```
$ python Run.py -train_data_paths '/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_train.jsonl' \
          -dev_data_path='/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_dev.jsonl' \
          -n_choice=5 \
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
          -save_path='/content/drive/My Drive/SemEval2021-task4/log/task_1_train/'
```

## 3. Model predicting
Directly run Run.py file on the console (Please refer to Config.py for the specific meaning of each parameter):
```
$ python Run.py -test_data_path='/content/drive/My Drive/SemEval2021-task4/data/test_data/Task_1_test.jsonl' \
          -n_choice=5 \
          -max_seq_len=150 \
          -checkpoint='/content/drive/My Drive/SemEval2021-task4/log/task_1_train/model-2021-01-20.pt' \
          -bert_model='albert-xxlarge-v2' \
          -n_layer=1 \
          -n_head=64 \
          -d_k=64 \
          -d_v=64 \
          -dropout=0.1 \
          -do_test \
          -batch_size=2 \
          -save_path='/content/drive/My Drive/SemEval2021-task4/log/task_1_train/'
```

## 4. Ciation
