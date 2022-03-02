# SemEval2021-Reading-Comprehension-of-Abstract-Meaning

This is the repository for our ACL-IJCNLP 2021 SemEval workshop paper [XRJL-HKUST at SemEval-2021 Task 4: WordNet-Enhanced Dual Multi-head Co-Attention for Reading Comprehension of Abstract Meaning](https://aclanthology.org/2021.semeval-1.105/). It includes code for baseline models and data. For more details, please see the **"Model"** file.

## Data
### The dataset is described in [SemEval 2021 Task 4: Reading Comprehension of Abstract Meaning](https://competitions.codalab.org/competitions/26153).
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
