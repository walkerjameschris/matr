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
    purrr::map(x[1:4], ncol) |>
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

#' @export
#' @import tibble dplyr tidyr
#' @method plot deepspace_network
plot.deepspace_network <- function(x, ...) {
  
  purrr::set_names(x[2:4], NULL) |>
    purrr::imap(function(layer, id) {
      as.data.frame(layer) |>
        dplyr::mutate(
          curr_id = id + 1,
          prev_id = id,
          prev_neuron = dplyr::row_number()
        ) |>
        tidyr::pivot_longer(
          cols = dplyr::starts_with("V"),
          names_to = "curr_neuron",
          values_to = "weight"
        ) |>
        dplyr::mutate(
          curr_neuron =
            stringr::str_sub(curr_neuron, 2) |>
            as.numeric()
        )
    }) |>
    dplyr::bind_rows() |>
    dplyr::mutate(
      edge_id = dplyr::row_number(),
      weight = abs(weight)
    ) |>
    ggplot2::ggplot(
      ggplot2::aes(
        x = prev_id,
        y = prev_neuron,
        color = weight
      )
    ) +
    ggplot2::geom_segment(
      ggplot2::aes(
        xend = curr_id,
        yend = curr_neuron,
        group = edge_id,
        alpha = weight
      )
    ) +
    ggplot2::geom_point(
      size = 5
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold")
    ) +
    ggplot2::labs(
      title = "A Deepspace Neural Network",
      subtitle = "4-Layer Deep Learning Network",
      y = "Neuron",
      x = "Layer",
      alpha = "Weight",
      color = "Weight"
    ) +
    ggplot2::scale_color_distiller(
      palette = "Blues",
      direction = 1
    )
}
