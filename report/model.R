#### Setup ####

# model.R
# Chris Walker

# Fits a deepspace model using captured images

library(deepspace)
library(dplyr)
library(withr)

data <- deepspace:::asl

data <-
  deepspace::train_test(
    X = data$X,
    Y = data$Y
  )

#### Tune Neurons ####

models <-
  seq(20, 200, 20) |>
  purrr::map(function(neurons) {
    
      deepspace::fit_network(
        X = data$train$X,
        Y = data$train$Y,
        neurons = neurons
      )
    
  })

real <- apply(data$test$Y, 1, which.max)

neurons_df <-
  models |>
  purrr::map(function(model) {
    
    pred <-
      model |>
      predict(data$test$X) |>
      as.matrix() |>
      apply(
        MARGIN = 1,
        FUN = which.max
      )
    
    tibble::tibble(
      neurons = model$neurons,
      time = model$time,
      accuracy = mean(pred == real)
    )
  }) |>
  dplyr::bind_rows()

neurons_df |>
  ggplot2::ggplot(
    ggplot2::aes(
      x = neurons,
      y = time
    )
  ) +
  ggplot2::geom_line()

