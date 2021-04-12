# XRJL-HKUST at SemEval-2021 Task 4: WordNet-Enhanced Dual Multi-head Co-Attention for Reading Comprehension of Abstract Meaning

This is the repository for SemEval 2021 Task 4: Reading Comprehension of Abstract Meaning. It includes data and code which a PyTorch implementation of our method in the system description paper (see in Citation).

Prerequisite: python >=3.6, torch>=1.0, transformers>=3.0.

## 1. Data
**Data Format**

Data is stored one-question-per-line in json format. Each instance of the data can be trated as a python dictinoary object. See examples below for further help in reading the data.


**Sample**
```
{
"article": "... observers have even named it after him, ``Abenomics". It is based on three key pillars -- the "three arrows" of monetary policy, fiscal stimulus and structural reforms in order to ensure long-term sustainable growth in the world's third-largest economy. In this weekend's upper house elections, ....",
"question": "Abenomics: The @placeholder and the risks",
"option_0": "chances",
"option_1": "prospective",
"option_2": "security",
"option_3": "objectives",
"option_4": "threats",
"label": 3
}
```
* article : the article that provide the context for the question.
* question : the question models are required to answer.
* options : five answer options for the question. Model are required to select the true answer from 5 options.
* label : index of the answer in options

**Code**

Data can be treated as python dictionary objects. A simple script to read **ReCAM** data is as follows:
```
def read_recam(path):
    with open(path, mode='r') as f:
        reader = jsonlines.Reader(f)
        for instance in reader:
            print(instance)
```


## 2. Model Architecture

<img src="https://github.com/zzshou/RCAM/blob/master/Model/model%20architecture.png" width="400" height="600">

## 3. Model training and evaluating
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

## 4. Model predicting
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

## 5. Citation
If our method is useful for your research, please consider citing:
```
@article{DBLP:journals/corr/abs-2103-16102,
  author    = {Yuxin Jiang and
               Ziyi Shou and
               Qijun Wang and
               Hao Wu and
               Fangzhen Lin},
  title     = {{XRJL-HKUST} at SemEval-2021 Task 4: WordNet-Enhanced Dual Multi-head
               Co-Attention for Reading Comprehension of Abstract Meaning},
  journal   = {CoRR},
  volume    = {abs/2103.16102},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.16102},
  archivePrefix = {arXiv},
  eprint    = {2103.16102},
  timestamp = {Wed, 07 Apr 2021 15:31:46 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-16102.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


