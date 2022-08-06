# Coding Assignment 2: Hackathon - Binary Sentiment Analysis

## Introduction

The second coding assignment asks you to implement a simple natural language processing model for sentiment analysis on the Amazon Review Dataset [kaggle page of this coding assignment](https://www.kaggle.com/c/gistailab-ca2-hackathon)

You can use some deep learning libraries (e.g., PyTorch, Tensorflow) to accelerate your code with CUDA back-end.

**Note**: we will use `Python 3.x` for the project.

## Deadline
December 19, 2020 9:00AM KST (*No delay is permitted.*)

### Submission checklist
* Push your code to [github classroom page's CA2 section](https://classroom.github.com/a/urINAJis)
* Submit your report to [Gradescope 'CA2 (Hackathon) Report' section](https://www.gradescope.com/courses/301439)
* Submit your entry to [Kaggle](https://www.kaggle.com/c/gistailab-ca2-hackathon)


---
## What to submit
**Push to your github classroom** 

- All of the python files listed above (under "Files you'll edit"). 
  - **Caution:** DO NOT UPLOAD THE DATASET

**Upload to your gradescope page** 
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).

---
## Construct the dataset (10%)

Construct the training set for the amazon review dataset as instructed and report the following statistics.

`REPORT1`: Please fill the below table in the report

| Statistics                                                        | Value  |
|-------------------------------------------------------------------|---|
|     the total number of unique words   in T                       | Plz, fill this |
|     the total number of training   examples in T                  | Plz, fill this  |
|     the ratio of positive examples to   negative examples in T    | Plz, fill this  |
|     the average length of document in   T                         | Plz, fill this  |
|     the max length of document in T                               | Plz, fill this |

---
## Performance of deep neural network for classification (40%)

Suggested hyperparameters: 

1. Data processing
   1. Word embedding dimension: 100
   1. Word Index: keep the most frequent 10k words

1. CNN
   1. Network: Word embedding lookup layer -> 1D CNN layer -> fully connected layer -> output prediction
   1. Number of filters: 100
   1. Filter length: 3
   1. CNN Activation: Relu
   1. Fully connected layer dimension 100, activation: None (i.e. this layer is linear)

1. RNN
   1. Network: Word embedding lookup layer -> LSTM layer -> fully connected layer(on the hidden state of the last LSTM cell) -> output prediction
   1. Hidden dimension for LSTM cell: 100
   1. Activation for LSTM cell: tanh
   1. Fully connected layer dimension 100, activation: None (i.e. this layer is linear)

`REPORT2`: Please fill the below table in the report

|                                        |     Accuracy    |     Training time (in seconds)    |
|----------------------------------------|-----------------|------------------------------------|
|     RNN w/o pretrained   embedding     | Plz, fill this  | Plz, fill this                     |
|     RNN w/ pretrained embedding        | Plz, fill this  | Plz, fill this                     |
|     CNN w/o pretrained   embedding     | Plz, fill this  | Plz, fill this                     |
|     CNN w/ pretrained   embedding      | Plz, fill this  | Plz, fill this                     |


---
## Training behavior (20%)

Plot the training/testing objective, training/testing accuracy over time for the 4 model combinations (correspond to 4 rows in the above table). In other word, there should be 2*4=8 graphs in total, each of which contains two curves (training and testing).

`REPORT3`: RNN **w/o** pretrained embedding
* training/testing objective over time
* training/testing accuracy over time

`REPORT4`: RNN **w/** pretrained embedding
* training/testing objective over time
* training/testing accuracy over time

`REPORT5`: CNN **w/o** pretrained embedding
* training/testing objective over time
* training/testing accuracy over time

`REPORT6`: CNN **w/** pretrained embedding
* training/testing objective over time
* training/testing accuracy over time

---
## Analysis of results (20%)

`REPORT7`: Discuss the complete set of experimental results, comparing the algorithms to each other. 

`REPORT8`: Discuss your observations about the various algorithms, *i.e.*, differences in how they performed, different parameters, what worked well and didn't, patterns/trends you observed across the set of experiments, *etc.*

`REPORT9`: Try to explain why certain algorithms or approaches behaved the way they did.

---
## The software implementation (10%)

Add detailed descriptions about software implementation & data preprocessing, including:

`REPORT10`: A description of what you did to preprocess the dataset to make your implementations easier or more efficient.

`REPORT11`: A description of major data structures (if any); any programming tools or libraries that you used;

`REPORT12`: Strengths and weaknesses of your design, and any problems that your system encountered;

---
### Caution
1. Use your `GIST ID` for your team name in the kaggle submission, otherwise we can't figure out who you are.
1. Do not over-engineer your method by tuning hyper-parameters heavily.

### Academic dishonesty
We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.