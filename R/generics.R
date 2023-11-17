#' @export
#' @import purrr cli glue
#' @method print deepspace_network
print.deepspace_network <- function(x, ...) {
  
  i <- c("before", "hide_a", "hide_b", "output")
  
  dimensions <-
    purrr::map(x[i], ncol) |>
    glue::glue_collapse("-")
  
  details <-
    list(
      "*" = "Final Loss: {round(x$loss, 3)}",
      "*" = "Elapsed Time: {x$time} Minutes",
      "*" = "Network Dimensions: {dimensions}",
      "*" = "Learning Rate: {x$learn_rate}",
      "*" = "Max Number of Epochs: {x$epoch}",
      "*" = "Converged: {x$converged}"
    ) |>
    purrr::map_chr(~ glue::glue(.x, x = x))
  
  cli::cli_h1("A Deepspace Neural Network")
  cli::cli_h2("Network Information:")
  cli::cli_inform(details)
}

#' @export
#' @import tibble
#' @method predict deepspace_network
predict.deepspace_network <- function(x, newdata, ...) {
  
  if (!missing(newdata)) {
    x$before <- add_ones(newdata)
  }
  
  feed_forward(x) |>
    purrr::chuck("a3") |>
    as.data.frame() |>
    tibble::as_tibble()
}

#' @export
#' @import tibble cli
#' @method print deepspace_tree
print.deepspace_tree <- function(x, ...) {
  
  cli::cli_h1("A Deepspace Decision Tree")
  cli::cli_h2("Binary Classification")
  cli::cli_inform(c(
    "*" = glue::glue("Min Split: {x$min_split}"),
    "i" = "Call `predict()` to make predictions."
  ))
}

#' @export
#' @import tibble
#' @method predict deepspace_tree
predict.deepspace_tree <- function(x, newdata, ...) {
  
  if (missing(newdata)) {
    newdata <- x$train
  }
  
  recurse_pred_tree_all(
    tree = x,
    X = newdata
  ) |>
    tibble::as_tibble()
}
