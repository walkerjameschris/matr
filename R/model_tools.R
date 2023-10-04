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
#' @export
fit_network <- function(X, Y,
                        neurons = 3L,
                        epoch = 100L,
                        learn_rate = 0.0001,
                        seed = 123) {
  
  start <- Sys.time()
  
  network <-
    withr::with_seed( 
      code = initialize(X, Y, neurons),
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
  
  attr(network, "class") <- "deepspace_network"
  network
}

#' @export
#' @import tibble
#' @method predict deepspace_network
predict.deepspace_network <- function(x, newdata, ...) {
  
  if (!missing(newdata)) {
    x$before <- add_ones(newdata)
  }
  
  feed_forward(x)$a3 |>
    as.data.frame() |>
    tibble::as_tibble()
}

#' @export
#' @import purrr cli glue
#' @method print deepspace_network
print.deepspace_network <- function(x, ...) {
  
  dimensions <-
    purrr::map(x[1:3], ncol) |>
    glue::glue_collapse("-")
  
  details <-
    list(
      "*" = "Final Loss: {x$loss}",
      "*" = "Elapsed Time: {x$time} Minutes",
      "*" = "Network Dimensions: {dimensions}"
    ) |>
    purrr::map_chr(~ glue::glue(.x, x = x))
  
  cli::cli_h1("A Deepspace Neural Network")
  cli::cli_h2("Network Information:")
  cli::cli_inform(details)
}
