#' Train Test Split
#' 
#' Splits X and Y data matrices into train and test
#'
#' @param X Matrix of training data
#' @param Y Matrix of training labels
#' @param train_prop Proportion of data for training
#' @param seed Random seed value
#' 
#' @import purrr withr
#' @return A list of matrices
#' @examples
#' data <- deepspace:::mnist
#' 
#' data_split <-
#'   deepspace::train_test(
#'     X = data$X,
#'     Y = data$Y
#'   )
#' 
#' @export
train_test <- function(X, Y, train_prop = 0.6, seed = 123) {
  
  n_obs <- nrow(X)
  
  indices <-
    sample(
      x = seq(n_obs),
      size = round(n_obs * 0.6),
    ) |>
    withr::with_seed(
      seed = seed
    )
  
  list(
    train = list(X = X[indices, ], Y = Y[indices, ]),
    test = list(X = X[-indices, ], Y = Y[-indices, ])
  )
}

#' Fit a Neural Network Classifier
#' 
#' Fits a neural network using C++ backend.
#'
#' @param X Matrix of training data
#' @param Y Matrix of training labels
#' @param neurons Number of hidden layer neurons
#' @param epoch Number of learning epochs
#' @param learn_rate Learning rate
#' @param seed Random seed value
#' 
#' @import cli withr
#' @return A list as class deepspace neural network
#' @examples
#' data <- deepspace:::mnist
#' 
#' network <-
#'   deepspace::fit_network(
#'     X = data$X,
#'     Y = data$Y,
#'     neurons = 5,
#'     epoch = 1000,
#'     learn_rate = 0.0001
#'   )
#' 
#' predict(network)
#' 
#' @export
fit_network <- function(X, Y,
                        neurons = 3L,
                        epoch = 250L,
                        learn_rate = 0.00001,
                        seed = 123) {
  
  start <- Sys.time()
  
  network <-
    withr::with_seed( 
      code = deepspace:::initialize(X, Y, neurons),
      seed = seed
    )
  
  cli::cli_progress_bar(
    name = "Training network",
    total = epoch,
    clear = FALSE
  )
  
  for (i in seq(epoch)) {
    network <- propagate_back(network, Y, learn_rate)
    cli::cli_progress_update(status = network$loss)
  }
  
  cli::cli_progress_done()
  
  network$time <-
    difftime(
      time1 = Sys.time(),
      time2 = start,
      units = "mins"
    ) |>
    as.numeric() |>
    round(2)
  
  network$neurons <- neurons
  network$epoch <- epoch
  network$learn_rate <- learn_rate
  network$seed <- seed
  
  attr(network, "class") <- "deepspace_network"
  network
}
