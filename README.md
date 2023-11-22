# matr <img src='img/logo.png' align="right" height="150" />

### A Simple R Package for Estimating Neural Network Clasifiers

## Introduction

This R package provides a simple API for estimating neural network classifiers
in R. The neural network itself is implemented using a C++ backend and is linked
to R via Rcpp. The package was originally designed as my project for ISYE 6740.

## Navigating this Repo

* `DESCRIPTION`: Package build information
* `NAMESPACE`: Defines exported object
* `R/`
  * `RcppExports.R`: Defines C++ exports
  * `generics.R`: Generic functions for printing
  * `model_fit.R`: Functions for fitting a network
  * `sysdata.rda`: Bundled data with the package
* `README.md`: Information about this repo
* `img/`
  * `logo.png`: The package logo
* `man/`
  * `fit_network.Rd`: Documentation for fitting a network
  * `train_test.Rd`: Documentation for train/test split
* `matr.Rproj`: Sets working directory
* `report/`
  * `loss_df.csv`: Data for final models
  * `model.R`: Script which fits models
  * `neurons_df.csv`: Data for neuron experiments
  * `report.qmd`: Report as a Qmd
* `src/`
    * `RcppExports.cpp`: Automatically defines C++ exports
    * `neural_network.cpp`: Neural network in C++

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
