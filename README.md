# matr <img src='img/logo.png' align="right" height="140" />

###  Tools for Matrices and Machine Learning Models

## Introduction

This R package provides a consistent NumPy inspired API for working with.
matrices in R. It contains the most common matrix operations and implementations
for many machine learning models designed to operate over R matrices.
**Disclaimer:** These algorithms are primarily implemented for educational
purposes and should not be used in production code. However, I think the
simplistic nature of this  package makes it easy to distill the mechanics of
these models while retaining the performance of C++.

| Model | Status |
| ----- | ------ |
| Neural Network | ![Implemented](https://img.shields.io/badge/Implemented-green) |
| Decision Tree | ![Implemented](https://img.shields.io/badge/Implemented-green) |
| Boosted Tree | ![Planned](https://img.shields.io/badge/Planned-red)
| KNN | ![Planned](https://img.shields.io/badge/Planned-red)
| K-Means | ![Planned](https://img.shields.io/badge/Planned-red)

## Navigating this Repo

* `DESCRIPTION`: Information about package build
* `NAMESPACE`: Determines exports (autogenerated)
* `R/`
  * `RcppExports.R`: Determines C++ linked functions (autogenerated)
  * `generics.R`: Defines generics for prediction, plotting, etc
  * `model_fit.R`: Defined helpers for fitting models
  * `ssdata.rda`: Contains classification datasets
* `README.md`: Info about this repo
* `matr.Rproj`: Sets working directory
* `img/`
  * `logo.png`: matr logo
  * `nn_plot.jpg`: Network plot for README
* `man/`
  * `fit_network.Rd`: Documentation (autogenerated)
  * `train_test.Rd`: Documentation (autogenerated)
* `report/`
  * `loss_df.csv`: Results from model fit
  * `model.R`: Model fit script
  * `neurons_df.csv`: Model fit results
  * `proposal.qmd`: ISYE 6740 project proposal
  * `report.qmd`: ISYE 6740 project report
* `src/`
  * `RcppExports.cpp`: Determines C++ linked functions (autogenerated)
  * `decision_tree.cpp`: Implementation of decision tree in C++
  * `neural_network.cpp`: Implementation of neural networks in C++

## Getting Started

To get started simply install the package from GitHub:

```r
devtools::install_github("https://github.com/walkerjameschris/matr")
```

To train a model, load the package and data. The `fit_network()` function
accepts the training data and labels (one hot encoded) as matrices. You can tune
the number of hidden layer neurons (`neurons`) in addition to the learning rate
(`learn_rate`), the max number of iterations (`epoch`), and a random seed.

```r
data <- matr:::mnist

network <-
  matr::fit_network(
    X = data$X,
    Y = data$Y,
    neurons = 5,
    epoch = 1000,
    learn_rate = 0.0001
  )

predict(network)
```
