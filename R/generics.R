#' @export
#' @import purrr cli glue
#' @method print deepspace_network
print.deepspace_network <- function(x, ...) {
  
  i <- c("before", "hide_a", "hide_b", "output")
  
  dimensions <-
    purrr::map(x[x], ncol) |>
    glue::glue_collapse("-")
  
  details <-
    list(
      "*" = "Final Loss: {x$loss}",
      "*" = "Elapsed Time: {x$time} Minutes",
      "*" = "Network Dimensions: {dimensions}",
      "*" = "Learning Rate: {x$learn_rate}",
      "*" = "Number of Epochs: {x$epoch}"
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
  
  i <- c("hide_a", "hide_b", "output")
  
  x[i] |>
    purrr::set_names(NULL) |>
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
