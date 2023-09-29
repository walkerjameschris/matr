#### Setup ####

# model_tools.R
# Chris Walker

# Functions for estimating neural networks

#' Fit a Neural Network
#'
#' @param X Training data
#' @param Y Training labels one hot encoded
#' @param neurons Hidden layer neurons
#' @param epoch Number of epochs
#' @param alpha Learning rate
#'
#' @return Neural network
#' @export
fit_network <- function(X, Y,
                        neurons = 5L,
                        epoch = 10000L,
                        alpha = 0.001) {
  
  start <- Sys.time()
  network <- initialize(X, Y, neurons)
  loss <- numeric(epoch)
  
  cli::cli_alert_success("Network initialized")
  
  cli::cli_progress_bar(
    name = "Training network",
    total = epoch,
    clear = FALSE
  )
  
  for (i in seq(epoch)) {
    network <- propagate_back(network, Y, alpha)
    loss[i] <- network$loss
    cli::cli_progress_update()
  }
  
  cli::cli_process_done()
  
  network$loss <- loss
  network$time <- as.numeric(difftime(Sys.time(), x, units = "mins"))
  attr(network, "class") <- "visionary_network"
  network
}

#' @method predict visionary_network
#' @export
predict.visionary_network <- function(x, newdata, ...) {
  
  if (!missing(newdata)) {
    x$before <- add_ones(newdata)
  }
  
  feed_forward(x) |>
    purrr::pluck("a2") |>
    as.data.frame() |>
    tibble::as_tibble()
}

#' @method print visionary_network
#' @export
print.visionary_network <- function(x, ...) {
  
  cli::cli_h1("A Visionary Neural Network")
  cli::cli_h2("Final Loss: {round(tail(x$loss, 1), 2)}")
  cli::cli_h2("Elapsed Minutes: {round(x$time, 2)}")
  
  cli::cli_inform(c(
    "*" = glue::glue("{nrow(x$before)} Input Neurons"),
    "*" = glue::glue("{nrow(x$hidden)} Hidden Neurons"),
    "*" = glue::glue("{nrow(x$output)} Output Neurons")
  ))
}
