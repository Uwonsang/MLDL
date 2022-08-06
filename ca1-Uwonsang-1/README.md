# Coding Assignment 1: Regression models to estimate housing price.

## Introduction

The first coding assignment asks you to implement a regression model or a classification model to predict house price given features, and compete with your colleagues in [kaggle page of this coding assignment](https://www.kaggle.com/t/f27310592a9f4022ac6a9dc9a561be37)

You can use some deep learning libraries (e.g., PyTorch, Tensorflow) to accelerate your code with CUDA back-end.

**Note**: we will use `Python 3.x` for the project.

## Deadline
November 3, 2020 11:59PM KST (*One day delay is permitted with linear scale score deduction.*)

### Submission checklist
* Push your code to [github classroom page's CA1 section](https://classroom.github.com/a/ERn1MWva)
* Submit your report to [Gradescope 'CA1 Report' section](https://www.gradescope.com/courses/301439)
* Submit your entry to [Kaggle](https://www.kaggle.com/t/f27310592a9f4022ac6a9dc9a561be37)


---
## What to submit
**Push to your github classroom** 

- All of the python files listed above (under "Files you'll edit"). 
  - **Caution:** DO NOT UPLOAD THE DATASET
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).

---
## Prepare the dataset (10%)

Read the csv to load the dataset.

```
>>> import datasets
>>> price_dataset = datasets.PriceDataset()
>>> [tr_x, tr_y, val_x, val_y] = price_dataset.getDataset()
```

You may ignore the warning of `Fontconfig warning: ignoring UTF-8: not a valid region tag` after `import datasets` command.

---
## Implement your own deep neural network baseline model for this problem (40%)

`REPORT1`: Show the architecture of your model

`REPORT2`: Discuss why you choose the network architecture

`REPORT3`: Error analysis 1 - show some (about 10) example of wrong predictions

`REPORT4`: Error analysis 2 - show some (about 10) examples of correct predictions

`REPORT5`: Error analysis 3 - discuss pattern of wrong/correct predictions

---
## Improve your networks and Compete with others on the accuracy using Kaggle (50%)

`REPORT6`: Discuss your ideas to improve the accuracy

`REPORT7`: Qualitative comparison 1 - show 10 examples of improvement (same examples that were wrongly predicted with your baseline model but are correctly predicted with the improved model.)

`REPORT8`: Qualitative comparison 2 - discuss why these improvement was possible

`REPORT9`: Quantitative comparison - compare the accuracy of baseline model and the improved model

Upload your improved model's entry to kaggle leaderboard

**Caution**
1. Use your `GIST ID` for your team name, otherwise we can't figure out who you are.
1. Do not over-engineer your method by tuning hyper-parameters heavily.

---
### Note
**Academic dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting help:** You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, class time, and Piazza are there for your support; please use them. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.
