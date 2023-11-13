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
| Neural Network | Implemented |
| Decision Tree | Implemented |
| Boosted Tree | Planned |
| KNN | Planned |
| K-Means | Planned |

## Disclaimer

These algorithms are primarily implemented for educational purposes and should
not be used in production code. However, I think the simplistic nature of this 
package makes it easy to distill the mechanics of these models while retaining
the performance of C++.

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
