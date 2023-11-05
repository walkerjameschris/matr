#### Setup ####

# model.R
# Chris Walker

# Fits a deepspace model using captured images

library(deepspace)
library(dplyr)
library(withr)

attach(deepspace:::asl)

#### Train Test Split ####

n_obs <- nrow(X)

train_ind <-
  sample(
    x = seq(n_obs),
    size = round(n_obs * 0.6),
  ) |>
  withr::with_seed(
    seed = 123
  )

train <- list(X = X[train_ind, ], Y =  Y[train_ind, ])
test <- list(X = X[-train_ind, ], Y = Y[-train_ind, ])

#### Tune Parameters ####

tune_grid <-
  tidyr::expand_grid(
    neurons = c(25, 50, 100),
    learn_rate = c(0.0001, 0.00001),
    epoch = c(25, 100, 250)
  ) |>
  dplyr::mutate(
    id = as.character(dplyr::row_number())
  ) |>
  dplyr::arrange(
    desc(neurons),
    desc(epoch),
    learn_rate
  )

models <-
  tune_grid |>
  purrr::pmap(function(neurons, learn_rate, epoch, id) {
    
      deepspace::fit_network(
        X = train$X,
        Y = train$Y,
        neurons = neurons,
        epoch = epoch,
        learn_rate = learn_rate
      )
    
  })

truth <- apply(test$Y, 1, which.max)

tune_results <-
  models |>
  purrr::map(
    predict,
    newdata = test$X
  ) |>
  purrr::map(
    ~ dplyr::mutate(.x, truth)
  ) |>
  dplyr::bind_rows(
    .id = "id"
  ) |>
  dplyr::left_join(
    y = tune_grid,
    by = "id"
  ) |>
  dplyr::group_by(
    neurons,
    learn_rate,
    epoch
  ) |>
  dplyr::mutate(
    pred = dplyr::case_when(
      V1 > V2 & V1 > V3 ~ 1,
      V2 > V1 & V2 > V3 ~ 2,
      V3 > V1 & V3 > V2 ~ 3
    )
  ) |>
  dplyr::reframe(
    accuracy = mean(truth == pred)
  ) |>
  dplyr::arrange(
    desc(accuracy)
  )

#### Expand Dataset ####

transform <- function(X, Y) {
  
  X <-
    tidyr::expand_grid(
      noise = c(0.00, 0.05, 0.10),
      light = c(0.85, 1.00, 1.15)
    ) |>
    purrr::pmap(function(noise, light) {
      
      value <- prod(dim(X))
      noise <- matrix(rnorm(value, 1, noise), nrow = nrow(X))
      X * noise * light
      
    }) |>
    do.call(
      what = rbind,
      args = _
    ) |>
    withr::with_seed(
      seed = 123
    )
  
  reps <- nrow(X) / nrow(Y)
  
  Y <-
    purrr::map(seq(reps), ~ Y) |>
    do.call(
      what = rbind,
      args = _
    )
  
  tibble::lst(X, Y)
}

train_new <- with(train, transform(X, Y))
test_new <- with(test, transform(X, Y))

#### Fit Final Model ####

best_tune <-
  tune_results |>
  dplyr::slice_max(
    order_by = accuracy,
    n = 1
  ) |>
  as.list()

network <-
  deepspace::fit_network(
    X = train_new$X,
    Y = train_new$Y,
    neurons = best_tune$neurons,
    epoch = best_tune$epoch,
    learn_rate = 1e-5
  )

#### Headline Performance ####

truth_new <- apply(test_new$Y, 1, which.max)

predict(
  object = network,
  newdata = test_new$X
) |>
  dplyr::mutate(
    real = truth_new,
    pred = dplyr::case_when(
      V1 > V2 & V1 > V3 ~ 1,
      V2 > V1 & V2 > V3 ~ 2,
      V3 > V1 & V3 > V2 ~ 3
    )
  ) |>
  dplyr::reframe(
    accuracy = mean(real == pred)
  )
