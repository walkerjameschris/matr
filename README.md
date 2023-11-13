# deepspace <img src='img/logo.png' align="right" height="140" />

###  A Simple R Package for Estimating Classification Algorithims

## Introduction

This R package provides a simple API for estimating machine learning classifiers
in R. This repo began as a from scratch implementation of a neural network in
for my class project for ISYE 6740. The algorithms are implemented using a C++
backed and are linked to R via Rcpp. Here is a list of algorithms implemented
in this package in addition to several methods planned for the future:

| Model | Status |
| ----- | ------ |
| Neural Network | ![Implemented](https://img.shields.io/badge/Implemented-green) |
| Decision Tree | ![In Progress](https://img.shields.io/badge/In%Progress-yellow) |
| Boosted Tree | ![Planned](https://img.shields.io/badge/Planned-red)
| KNN | ![Planned](https://img.shields.io/badge/Planned-red)
| K-Means | ![Planned](https://img.shields.io/badge/Planned-red)

**Disclaimer:** These algorithms are primarily implemented for educational
purposes and should not be used in production code. However, I think the
simplistic nature of this  package makes it easy to distill the mechanics of
these models while retaining the performance of C++.

## Navigating this Repo

* `DESCRIPTION`: Information about package build
* `NAMESPACE`: Determines exports (autogenerated)
* `R/`
  * `RcppExports.R`: Determines C++ linked functions (autogenerated)
  * `generics.R`: Defines generics for prediction, plotting, etc
  * `model_fit.R`: Defined helpers for fitting models
  * `ssdata.rda`: Contains classification datasets
* `README.md`: Info about this repo
* `deepspace.Rproj`: Sets working directory
* `img/`
  * `logo.png`: deepspace logo
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
devtools::install_github("https://github.com/walkerjameschris/deepspace")
```

To train a model, load the package and data. The `fit_network()` function
accepts the training data and labels (one hot encoded) as matrices. You can tune
the number of hidden layer neurons (`neurons`) in addition to the learning rate
(`learn_rate`), the max number of iterations (`epoch`), and a random seed.

```r
data <- deepspace:::mnist

network <-
  deepspace::fit_network(
    X = data$X,
    Y = data$Y,
    neurons = 5,
    epoch = 1000,
    learn_rate = 0.0001
  )

predict(network)

plot(network)
```

<img src='img/nn_plot.jpg' height="350"/>
