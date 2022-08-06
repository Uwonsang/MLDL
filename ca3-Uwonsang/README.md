# Coding Assignment 3: Image classification with imbalanced CIFAR100 dataset

## Introduction

We will implement neural network models for image classification problem with label noise and class imbalance.

The last assignment let you implement any neural network models for image classification on CIFAR100 dataset with imblanced distribution.
Noisy label here means that some of the given labels are not correct (e.g., it is a dog image but labeled as a cat).

We recommend you to use PyTorch as your deep learning libary but we welcome you to use other deep learning libraries (e.g., Tensorflow).

**Note**: we will use `Python 3.x` for the project. (We have tested all codes with Python 3.7.7)

---
## Deadline
June 10, 2021 11:59PM KST (*One day delay is permitted with linear scale score deduction.*)

### Submission checklist
* Push your code to [our github classroom page's CA3 section](https://classroom.github.com/a/rF48OFiH)
* Submit your report to [Gradescope 'CA3 Report' section](https://www.gradescope.com/courses/251016)
* Submit your entry to [Kaggle leaderboard](https://www.kaggle.com/c/cifar100-image-classification-with-long-tail/leaderboard)

---
## Preparation

### Installing prerequisites

The prerequisite usually refers to the necessary library that your code can run with. They are also known as `dependency`. We have prepared a few libraries for you to start with. To install the prerequisite, simply type in the shell prompt (not in a python interpreter) the following:

```
$ pip install -r requirements.txt
```

### Download the dataset

Go to [dataset page in our kaggle page for this challenge (CIFAR100-NoisyLabel)](https://www.kaggle.com/c/cifar100-image-classification-with-noisy-labels/data) to download the dataset. Copy (or move) the dataset into `./dataset` sub-directory.

---
## Files

**Files you'll edit:**

* `datasets.py`: Data provider. 
  - Implement functions to read the `./dataset/data/cifar100_nl.csv` file. 
  - The recommended interface has been noted in the file as comments.

---
## What to submit
**Push to your github classroom** 

- All of the python files listed above (under "Files you'll edit"). 
  - **Caution:** DO NOT UPLOAD THE DATASET

**Upload your report to Gradescope**
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).
  - If you do not want to use LaTeX, use any other wordprocessor and render it to PDF.



---
### Note
**Academic dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting help:** You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, class time, and Piazza are there for your support; please use them. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.

---
## Prepare the dataset (20%)

Read the csv to load the dataset.

```
>>> import datasets
>>> dataset_nl = datasets.C100Dataset('./dataset/data/cifar100_nl.csv')
>>> [data_nl_tr_x, data_nl_tr_y, data_nl_val_x, data_nl_val_y] = dataset_odn.getDataset()
```

Each line of the dataset's `csv` file follows the below format:
```
filename,classname
```

*Note*: You may ignore the warning of `Fontconfig warning: ignoring UTF-8: not a valid region tag` after `import datasets` command.

**Note**: Do not use PyTorch's CIFAR100 dataset loader. Use your own dataset loader.

---
## Image classification with dataset of long tail distribution (75%)

Perform image classification using the given dataset of long tail version of CIFAR-100 dataset (`cifar100_lt.csv`).

`REPORT7`: Describe model your have used (1. architecture overview and 2. any specialty of this model.)

`REPORT8`: Report both the training and testing accuracy (by submitting your entries in Kaggle) in a plot (x: epoch, y: accuracy). 

`REPORT9`: Discuss any ideas to improve the accuracy (e.g., new architecture, using new layers, using new loss)


---
## Compete with others on the accuracy using Kaggle (5%)

**Caution**
1. Use your github ID for your name, otherwise we can't figure out who you are.
1. Do not over-engineer your method by tuning hyper-parameters heavily.

Unlike CA1, you now do not know the true label of test set. Use your trainig set for validation (by designated held-out set or cross validation). Your public leaderboard ranking will be different from the private leaderboard's ranking.
