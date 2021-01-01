# window slice
This is a brief description of the model.

## 1. Data Review
Firstly, let's see an example from the training data of task_1:

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/example.png" width="1000" height="240">

As you can see, an example contains 4 parts:
* article : the article that provide the context for the question. (The article is too long in this example, so we omit the middle part which is marked in yellow)
* question : the question with a placeholder remaining to be filled.
* options : five answer options for the question. Model are required to select the true answer from 5 options.
* label : index of the answer in options

## 2. Data Preprocess

<img src="https://github.com/zzshou/RCAM/blob/master/window%20slice/pictures/data_process.png" width="800" height="180">

## 3. Model Building

## 4. Model Training & Evaluating
