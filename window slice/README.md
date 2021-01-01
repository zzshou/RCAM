# window slice
This is a brief description of the model.

## 1. Data Review
Firstly, let's see an example from the training data of task_1:

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/example.png" width="1000" height="240">

As you can see, an example contains 4 parts:
* article : the article that provide the context for the question. (The article is too long in this example, so we omit the middle part which is marked in yellow)
* question : the question with a placeholder remaining to be filled.
* options : 5 answer options for the question. Model are required to select the true answer from 5 options.
* label : index of the answer in options

## 2. Data Preprocess
We do window slice with overlap on the article of each example, and set the max number of slices for too long articles. 
The placeholder in the question is replaced by 5 choices separately, resulting in 5 questions for every example. Then for each question, we connect it to the back of the slices.

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/data_process.png" width="800" height="180">

## 3. Model Building

This model is highly inspired by [PARADE: Passage Representation Aggregation for Document Reranking](https://arxiv.org/pdf/2008.09093.pdf), which proposes 3 methods (max, atten, transformer) for sloving the limitation of pre-trained models like BERT on the input sequence's length. The architectures for multi choice task are depicted below.

### 3.1 Max

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/max.png" width="1000" height="550">

### 3.2 Atten

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/atten.png" width="1000" height="550">

### 3.3 Transformer

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/transformer.png" width="1000" height="550">

## 4. Model Training & Evaluating
It's convenient to run the code in terminal.:
```
$ python Train.py -data_path='./SemEval2021-task4/data/training_data/' \
          -n_choice=5 \
          -max_seq_len=100 \
          -sep=80 \
          -overlap=30 \
          -n_slice=10 \
          -bert_model='albert-large-v2' \
          -method='transformer' \
          -n_layer=2 \
          -n_head=6 \
          -d_inner=1024 \
          -dropout=0.1 \
          -epoch=3 \
          -lr=1e-5 \
          -save_path='./SemEval2021-task4/model/log/'
```

For the meaning of each parameters, please refer to the file named "Config.py".
