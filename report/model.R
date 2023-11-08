#### Setup ####

# model.R
# Chris Walker

# Fits a deepspace model using captured images

library(deepspace)
library(dplyr)
library(withr)
library(foreach)
library(doParallel)

doParallel::registerDoParallel(cores = 4)

data <- deepspace:::asl

data <-
  deepspace::train_test(
    X = data$X,
    Y = data$Y
  )

#### Tune Neurons ####

models <-
  c(25, 50, 100, 200, 400) |>
  foreach::foreach(x = _) %dopar% {
    
    deepspace::fit_network(
      X = data$train$X,
      Y = data$train$Y,
      neurons = x
    )
    
  }

readr::write_rds(models, "~/Documents/models.Rds")

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
      minutes = model$time,
      accuracy = mean(pred == real),
      converged = model$converged
    )
  }) |>
  dplyr::bind_rows()

neurons_df |>
  ggplot2::ggplot(
    ggplot2::aes(
      x = neurons,
      y = accuracy
    )
  ) +
  ggplot2::geom_line() +
  ggplot2::theme_minimal() +
  ggplot2::labs(
    x = "Neurons",
    y = "Training Accuracy",
    title = "Neurons vs Accuracy"
  ) +
  ggplot2::scale_y_continuous(
    labels = scales::label_percent(1)
  )

models |>
  purrr::map(function(model) {
   
    tibble::tibble(
      loss = model$loss_hist,
      neurons = model$neurons
    ) |>
      dplyr::mutate(
        epoch = dplyr::row_number()
      )
    
  }) |>
  dplyr::bind_rows() |>
  ggplot2::ggplot(
    ggplot2::aes(
      x = epoch,
      y = loss,
      color = factor(neurons)
    )
  ) +
  ggplot2::geom_line() +
  ggplot2::theme_minimal() +
  ggplot2::labs(
    x = "Epoch",
    y = "Loss",
    title = "Loss Curves for Neural Networks",
    color = "Neurons"
  )
