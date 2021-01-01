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
The placeholder in the question is replaced by 5 choices separately, resulting in 5 questions for every example.

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/data_process.png" width="800" height="180">

## 3. Model Building

This model is highly inspired by [PARADE: Passage Representation Aggregation for Document Reranking](https://arxiv.org/pdf/2008.09093.pdf), which proposes 3 methods (max, atten, transformer) for sloving the limitation of pre-trained models like BERT on the input sequence's length. The architectures for multi choice task are depicted below.

### 3.1 Max
we concatenate a pair of slice_j and question_i with a [SEP] token in between and another [SEP] token at the end. Then we put the pairs through BERT model and regard the output embedding of CLS_j_i as the representation of slice_j_question_i. For different j, we use max pooling to get CLS_i, which is the representation of article_question_i. Finally, the CLS_i is projected into logit_i which is a scalar to do prediction and compute the loss by Softmax.

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/max.png" width="1000" height="550">

### 3.2 Atten
The main architecture is the same as Max. After getting CLS_j_i, we feed it through a two-layer FNN to learn the attention weight. Then CLS_i is computed by weighted sum of CLS_j_i.

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/atten.png" width="1000" height="550">

### 3.3 Transformer
After getting CLS_j_i, we add a randomly initialized CLS embedding for each question, and feed them to the encoder part of transformer. Then we regard the CLS_i vector of the last Transformer output layer as the representation of article_question_i.

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
