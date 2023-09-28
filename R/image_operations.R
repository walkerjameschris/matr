#### Setup ####

# image_operations.R
# Chris Walker

# Functions for capturing, loading, and working with images

#' Capture Images from a Webcam
#'
#' @param path Where the images will be stored
#' @param n The number of images to capture
#' 
#' @import purrr
#' @return TRUE if successful
#' @export
capture_images <- function(path, n = 50, device = NULL) {
  
  if (is.null(device)) {
    device <- "/dev/video0"
  }
  
  seq(n) |>
    purrr::walk(.progress = TRUE, function(i) {
      
      command <-
        glue::glue(
          "fswebcam -d {devide} ",
          "--no-banner --greyscale",
          path, "img_{i}.jpg"
        )
      
      system(
        command = command,
        ignore.stdout = TRUE,
        ignore.stderr = FALSE,
        intern = TRUE
      )
    })
  
  invisible(TRUE)
}

#' Load Images from a Directory as Data
#'
#' @param path A path to an image or directory of images
#' @param label A label (true label) for training
#' @param x_range Subset of x axis pixels for training
#' @param y_range Subset of y axis pixels for training
#' @param pattern A regex pattern for file matching
#'
#' @import purrr fs jpeg tibble dplyr
#' @return A tibble
load_images <- function(path,
                        label = 1,
                        x_range = seq(140, 499, 10),
                        y_range = seq(001, 360, 10),
                        pattern = ".jpg$") {
  
  if (fs::is_dir(path)) {
    path <-
      list.files(
        path = path,
        full.names = TRUE,
        pattern = pattern
      )
  }
  
  path |>
    purrr::map(jpeg::readJPEG) |>
    purrr::map(~ .x[y_range, x_range, 1]) |>
    purrr::map(as.vector) |>
    do.call(
      what = rbind,
      args = _
    ) |>
    as.data.frame() |>
    tibble::as_tibble() |>
    dplyr::mutate(label) |>
    dplyr::relocate(label)
}

#' Initialize a Live Video Stream
#'
#' @param model A neural network model
#'
#' @return Nothing
#' @export
live_stream <- function(model, device = NULL) {
  
  cli::cli_inform(c("i" = "This will loop until the escape key is pressed."))
  Sys.sleep(3)
  
  path <- tempfile(fileext = ".jpg")
  
  while (TRUE) {
    capture_images(path, 1, device)
    image <- load_images(path)
    print(predict(model, newdata = image))
    file.remove(path)
  }
}
