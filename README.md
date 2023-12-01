# matr <img src="img/logo.png" align="right" height="160"/>

### A Simple R Package for Estimating Neural Network Models using a C++ Backend

## Introduction

This R package provides a simple API for estimating neural network models in R using a C++ backend via Rcpp. The package was originally designed as my project for ISYE 6740.

## Getting Started

To get started simply install the package from GitHub:

``` r
devtools::install_github("https://github.com/walkerjameschris/matr")
```

To train a model, load the package and data. The `fit_network()` function accepts the training data and labels (one hot encoded) as matrices. You can tune the number of hidden layer neurons (`neurons`) in addition to the learning rate (`learn_rate`), the max number of iterations (`epoch`), and a random seed.

``` r
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
