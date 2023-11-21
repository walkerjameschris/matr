#### Setup ####

# model.R
# Chris Walker

# Fits a matr model using captured images

library(matr)
library(dplyr)
library(withr)
library(foreach)
library(doParallel)

doParallel::registerDoParallel(cores = 4)

data <- matr:::asl

data <-
  matr::train_test(
    X = data$X,
    Y = data$Y
  )

#### Tune Neurons ####

models <-
  c(25, 50, 100, 200, 400) |>
  foreach::foreach(x = _) %dopar% {
    
    matr::fit_network(
      X = data$train$X,
      Y = data$train$Y,
      neurons = x
    )
    
  }

readr::write_rds(models, "~/Documents/models.Rds")

real <- apply(data$train$Y, 1, which.max)

neurons_df <-
  models |>
  purrr::map(function(model) {
    
    pred <-
      predict(
        model,
        newdata = data$train$X
      ) |>
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
  readr::write_csv(
    here::here("report/neurons_df.csv") 
  )

loss_df <-
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
  dplyr::bind_rows()

loss_df |>
  readr::write_csv(
    here::here("report/loss_df.csv") 
  )

#### Create Modified Data ####

mod_grid <-
  tidyr::expand_grid(
    normal = c(0.00, 0.02, 0.04, 0.06, 0.08),
    bright = c(0.70, 1.00, 1.30)
  )

mod_X <-
  mod_grid |>
  purrr::pmap(function(normal, bright) {
    
    x_df <- data$test$X
    vals <- prod(dim(x_df))
    rows <- nrow(x_df)

    norm_val <-
      rnorm(vals, 0, normal) |>
      matrix(nrow = rows)

    temp <- x_df * bright + norm_val
    matr:::matrix_min_max(temp)
  }) |>
  do.call(
    what = rbind,
    args = _
  ) |>
  withr::with_seed(
    seed = 123
  )

mod_real <- rep(real, nrow(mod_grid))

#### Test Randomized Samples ####

model_200 <- purrr::pluck(models, 4)

preds <-
  predict(
    model_200,
    newdata = mod_X
  ) |>
  as.matrix() |>
  apply(
    MARGIN = 1,
    FUN = which.max
  )

mean(preds == mod_real)
