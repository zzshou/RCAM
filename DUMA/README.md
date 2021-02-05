# XRJL at SemEval-2020 Task 4: Dual Multi-Head Co-Attention for Abstract Meaning Understanding
## 1. Model Architecture

<img src="https://github.com/zzshou/RCAM/blob/master/DUMA/model%20architecture.png" width="600" height="800">

## 2. 模型训练和验证
可以直接在命令行运行Run.py文件，并快速调节超参数 (每个参数的具体含义请参照Config.py)：
```
$ python Run.py -train_data_path '/content/drive/My Drive/SemEval2021-task4/data/training_data/Task_1_train.jsonl' \
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
          -evaluate_steps=400 \
          -max_train_steps=-1 \
          -n_epoch=3 \
          -batch_size=2 \
          -gradient_accumulation_steps=1 \
          -max_grad_norm=10.0 \
          -weight_decay=0.01 \
          -lr=5e-6 \
          -save_path='/content/drive/My Drive/SemEval2021-task4/log/task_1_train/'
```
针对Task1, 我在Colab Pro上将改进的DUMA模型和原论文中的DUMA模型使用相同的参数在训练集上都训练了3轮，每训练400步在验证集上评估结果，并保存验证集上准确率最高的模型的参数。每轮训练时间大概在1小时，上述配置占用显存15.5GB。最终训练集的准确率达到92%以上，验证集上结果如下：  

|          training steps          |          改进DUMA          | DUMA |
|:-----------------------:|:-----:|:-----:|
|  400  |  eval_loss = 0.6223<br>eval_accuracy = 76.94% | eval_loss = 0.6338<br>eval_accuracy = 78.02% |
|  800  |  eval_loss = 0.5807<br>eval_accuracy = 80.76% | eval_loss = 0.5894<br>eval_accuracy = 79.57% |
| 1200 |  eval_loss = 0.5501<br>eval_accuracy = 81.12% | eval_loss = 0.5197<br>eval_accuracy = 81.96% |
| 1600 |  eval_loss = 0.5483<br>eval_accuracy = 82.92% | eval_loss = 0.5399<br>eval_accuracy = **83.63%** |
|      2000     |  eval_loss = 0.6838<br>eval_accuracy = 84.11% | eval_loss = 0.6288<br>eval_accuracy = 82.92% |
|      2400     |  eval_loss = 0.6677<br>eval_accuracy = 84.59% | eval_loss = 0.7144<br>eval_accuracy = 82.08% |
|       2800      |  eval_loss = 0.7931<br>eval_accuracy = **85.42%** | eval_loss = 0.7659<br>eval_accuracy = 82.92% |
|       3200      |  eval_loss = 0.7076<br>eval_accuracy = 84.23% | eval_loss = 0.8202<br>eval_accuracy = 82.68% |
|      3600      |  eval_loss = 0.749<br>eval_accuracy = 84.23% | eval_loss = 0.8123<br>eval_accuracy = 83.03% |
|      4000      |   eval_loss = 0.8107<br>eval_accuracy = 84.11%  | eval_loss = 0.8508<br>eval_accuracy = 83.15% |

可以看出改进的模型在验证集上的准确率优于原论文中的模型，且更不易出现过拟合。

(p.s. 该超参数的配置并不代表最优，只是一次实验的结果，可以通过继续调节超参数以获得更好的结果。)


## 3. 模型预测
可以直接在命令行运行Run.py文件:

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

## 4. 其他参考文献
2020年发表的一篇论文[《Multi-task Learning with Multi-head Attention for Multi-choice Reading Comprehension》](https://arxiv.org/pdf/2003.04992.pdf)对阅读理解多选题近期的模型进行了总结，并在DUMA模型的基础上进行了多任务学习 (Multi-task Learning)，即先在DREAM和RACE数据集上进行粗略的学习，再在DREAM数据集上进一步学习，取得了DREAM排行榜上SOTA的效果。

我采用了该方法进行了尝试，即先在TASK_1和TASK_2的训练集进行粗略的学习(training epoch=1)，再在TASK_1的训练集进一步学习，最后在TASK_1的验证集上进行评估，效果不升反降 (基于albert_base_v2，albert_xxlarge_v2还没有尝试)。此外，直接在TASK_1和TASK_2的训练集进行细致的学习 (加大training epoch)，在TASK_1的验证集上进行评估，效果不升反降 (基于albert_base_v2，albert_xxlarge_v2还没有尝试)。

## 5. 一些思考
感觉可以在DUMA模型的思路上进行改进，来更好地适应抽象阅读理解的任务。比如借助CNN来提取句子的summary信息？灵感来自于这篇attention与CNN结合的论文[《ABCNN: Attention-Based Convolutional Neural Networkfor Modeling Sentence Pairs》](https://arxiv.org/pdf/1512.05193.pdf)。 或许可以设计出其他更好的模型，在多个阅读理解多选题任务上得到提升。
