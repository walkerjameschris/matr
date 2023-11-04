#### Setup ####

# model.R
# Chris Walker

# Fits a deepspace model using captured images

library(deepspace)
library(furrr)
library(dplyr)
library(withr)

data <- deepspace:::asl

#### Augment Data ####

future::plan(
  strategy = future::multisession,
  workers = 4
)

X <-
  tidyr::expand_grid(
    norm = c(0, 0.05, 0.1),
    span = c(0.85, 0.9, 0.95, 1.00, 1.05, 1.1)
  ) |>
  furrr::future_pmap(function(norm, span) {

    shape <- dim(data$X)
    value <- prod(shape)
    
    norm <-
      matrix(
        data = rnorm(value, sd = norm),
        nrow = shape[1]
      )
    
    pmin(data$X * norm * span, 1)
    
  }) |>
  withr::with_seed(
    seed = 123
  ) |>
  do.call(
    what = rbind,
    args = _
  )

augments <- nrow(X) / nrow(data$Y)

Y <-
  purrr::map(
    seq(augments),
    ~ data$Y
  ) |>
  do.call(
    what = rbind,
    args = _
  )

#### Apply Train and Test ####

train_ind <-
  sample(
    x = seq(nrow(Y)),
    size = 20000
  ) |>
  withr::with_seed(
    seed = 123
  )

X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

Y_train <- Y[train_ind, ]
Y_test <- Y[-train_ind, ]

#### Fit Models ####

models <-
  tidyr::expand_grid(
    neurons = c(25, 50, 100),
    rate = c(0.001, 0.0001, 0.00001)
  ) |>
  purrr::pmap(.progress = TRUE, function(neurons, rate) {
    
    deepspace::fit_network(
      X = X,
      Y = Y,
      neurons = neurons,
      epoch = 100L,
      learn_rate = rate,
      seed = 124
    )
    
  })




