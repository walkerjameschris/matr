#' @export
#' @import purrr cli glue
#' @method print matr_network
print.matr_network <- function(x, ...) {
  
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
  
  cli::cli_h1("A matr Neural Network")
  cli::cli_h2("Network Information:")
  cli::cli_inform(details)
}

#' @export
#' @import tibble
#' @method predict matr_network
predict.matr_network <- function(x, newdata, ...) {
  
  if (!missing(newdata)) {
    x$before <- add_ones(newdata)
  }
  
  feed_forward(x) |>
    purrr::chuck("a3") |>
    as.data.frame() |>
    tibble::as_tibble()
}
