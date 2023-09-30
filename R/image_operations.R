#' Capture Images from a Webcam
#'
#' Uses `fswebcam` to capture images in a loop and save to a directory
#'
#' @param path Where the images will be stored
#' @param n The number of images to capture
#' @param device Path to linux image device
#' 
#' @import purrr glue
#' @return TRUE
#' @export
capture_images <- function(path, n = 50, device = NULL) {
  
  if (is.null(device)) {
    device <- "/dev/video0"
  }
  
  seq(n) |>
    purrr::walk(.progress = TRUE, function(i) {
      
      command <-
        glue::glue(
          "fswebcam -d {device} ",
          "--no-banner --greyscale ",
          path, "img_{i}.jpg"
        )
      
      system(
        command = command,
        ignore.stdout = TRUE,
        ignore.stderr = TRUE,
        intern = TRUE
      )
    })
  
  invisible(TRUE)
}

#' Load Images from a Directory as Data
#' 
#' Load images from a directory or specific path
#'
#' @param path A path to an image or directory of images
#' @param label A label (true label) for training
#' @param x_range Subset of x axis pixels to load
#' @param y_range Subset of y axis pixels to load
#' @param pattern A regex pattern for file matching
#'
#' @import purrr fs jpeg tibble dplyr
#' @return A tibble
#' @export
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
#' @param device Path to linux image device
#'
#' @export
live_stream <- function(model, device = NULL) {
  
  cli::cli_inform(c(
    "i" = "This will loop until the escape key is pressed.",
    "v" = "Streaming in 3 seconds..."
  ))
  
  Sys.sleep(3)
  
  dir <- glue::glue(tempdir(), "/")
  img <- glue::glue("{dir}img_1.jpg")
  
  while (TRUE) {
    capture_images(dir, 1, device)
    image <- load_images(img)
    print(predict(model, newdata = image))
    file.remove(img)
  }
}
