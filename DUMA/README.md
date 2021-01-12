# DUMA (DUal Multi-head Co-Attention)
## 1. 模型架构
模型主要参考了[《DUMA: Reading Comprehension with Transposition Thinking》](https://arxiv.org/pdf/2001.09415.pdf)，架构如下图所示：
<img src="https://github.com/zzshou/RCAM/blob/master/DUMA/model%20architecture.png" width="650" height="650">

首先，对于每个example，我们把5个option填入question中，并且分别与article进行拼接，得到5个序列。接着，我们将这些序列按照max_seq_len进行截断，输入到BERT (论文中用的是albert-xxlarge-v2) 进行encoding，得到每个token经过encoding后的embedding (sequencial output)。此时序列的embedding包含了浏览article/question/option后的信息。

其次，将article与question/option分开，二者输入co-attention层以获取更细微的信息。co-attention层本质上就是transformer中的multi-head attention。**此处co-attention层的构造与原论文中略有不同**。首先我们将question/option视为multi-head attention中的query，article视为key和value，来计算question-option-aware article representation，该过程模拟了人类带着对问题和选项的印象重读文章细节的过程。接着我们将article视为multi-head attention中的query，question-option-aware article representation视为key和value，来计算article-aware question-option representation，该过程模拟了人类在对文章有了更深层次理解后重新思考答案的过程 (p.s. 通过实验发现，这两个步骤的顺序如果颠倒，模型的效果会变差)。这样的co-attention层可以堆叠k次，但论文中指出，k增大并不会产生更好的效果。

最后，co-attention层的输出question-option-aware article representation与article-aware question-option representation仍然是每个token的embedding的形式 (维度是batch_size * num_of_choice, max_seq_len, dim_of_embedding)。我们对其做一维的平均池化 (AvgPool1d)，得到两个维度是 (batch_size * num_of_choice, dim_of_embedding)的向量，包含了汇总的信息 (p.s. 我们通过实验发现AvgPool1d的效果好于MaxPool1d)。然后，我们将两个向量通过concatenation的方式进行融合，得到维度是 (batch_size * num_of_choice, dim_of_embedding * 2)的向量 (p.s. 我们通过实验发现concatenation的效果好于element-wise multiplication, element-wise summation)。最后，我们将该向量输入到一个单层的fully-connected neural network (FNN)，得到article-question-option_i的logit。通过对logits做softmax即可得到每个option的概率，并且计算loss (p.s. 我们通过实验发现，单层的FNN比两层的第一层带激活函数的FNN效果更好)。

## 2. 模型训练和验证
可以直接在命令行运行Train.py文件，并快速调节超参数：
```
$ python Train.py -train_data_path='/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_train.jsonl' \
          -dev_data_path='/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_dev.jsonl' \
          -n_choice=5 \
          -max_seq_len=150 \
          -bert_model='albert-xxlarge-v2' \
          -n_layer=1 \
          -n_head=24 \
          -d_k=64 \
          -d_v=64 \
          -dropout=0.1 \
          -do_train=True \  
          -do_eval=True \
          -evaluate_steps=400 \
          -max_train_steps=-1 \
          -n_epoch=3 \
          -batch_size=2 \
          -gradient_accumulation_steps=1 \
          -max_grad_norm=10.0 \
          -weight_decay=0.01 \
          -lr=5e-6 \
          -save_path='/content/drive/My Drive/SemEval2021-task4/model/log/'
```
