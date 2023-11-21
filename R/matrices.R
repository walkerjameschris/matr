#' Create a Matrix
#' 
#' A NumPy like interface for building matrices
#'
#' @param ... Vectors as rows of the matrix
#' 
#' @import purrr 
#' @return A matrix
#' @examples
#' matr::new_matrix(
#'   c(1, 2, 3),
#'   c(3, 2, 1),
#'   c(1, 1, 2)
#' )
#' @export
new_matrix <- function(...) {
  
  args <- list(...)
  lens <- unique(purrr::map_dbl(args, length))
  type <- purrr::map(args, typeof)
  
  if (length(lens) >= 2) {
    cli::cli_abort("All vectors must be the same length!")
  }
  
  matrix(
    data = unlist(args),
    nrow = length(args),
    byrow = TRUE
  )
}
