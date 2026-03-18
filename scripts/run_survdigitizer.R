args <- commandArgs(trailingOnly = TRUE)

read_flag <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) {
    stop(sprintf("Missing required flag: %s", flag), call. = FALSE)
  }
  args[index + 1]
}

handle_step_error <- function(error, step) {
  stop(sprintf("SurvdigitizeR failed in %s: %s", step, conditionMessage(error)), call. = FALSE)
}

collapse_break_matches <- function(break_positions, label_table, start_col, end_col) {
  if (nrow(label_table) == 0) {
    return(numeric(0))
  }

  centers <- (label_table[[start_col]] + label_table[[end_col]]) / 2
  pxl_loc <- rep(NA_real_, nrow(label_table))

  for (break_pos in break_positions) {
    matches <- which(break_pos >= label_table[[start_col]] & break_pos <= label_table[[end_col]])
    if (length(matches) == 0) {
      next
    }

    best_match <- matches[which.min(abs(centers[matches] - break_pos))]
    if (is.na(pxl_loc[best_match])) {
      pxl_loc[best_match] <- break_pos
      next
    }

    if (abs(centers[best_match] - break_pos) < abs(centers[best_match] - pxl_loc[best_match])) {
      pxl_loc[best_match] <- break_pos
    }
  }

  pxl_loc
}

safe_range_detect <- function(step1_fig, step2_axes, x_start, x_end, x_increment, y_start, y_increment, y_end, y_text_vertical) {
  X_actual <- seq(x_start, x_end, by = x_increment)
  Y_actual <- seq(y_start, y_end, by = y_increment)
  fig_bw <- step1_fig[, , 3]

  fig_x <- fig_bw[-step2_axes$yaxis, step2_axes$xaxis]
  fig_y <- fig_bw[step2_axes$yaxis, -step2_axes$xaxis]

  loc_y_start <- which.max(rowSums(fig_x < 0.9)) - 1
  if (loc_y_start == 0) {
    loc_y_start <- which.max(diff(rowSums(fig_x < 0.9)))
  }

  loc_of_breaks_x <- which(fig_x[(loc_y_start - 1), ] < 0.9)
  loc_of_breaks_x <- loc_of_breaks_x[c(0, diff(loc_of_breaks_x)) != 1]
  x_pixels_increment <- as.numeric(names(which.max(table(diff(loc_of_breaks_x)))))

  if (length(loc_of_breaks_x) == length(X_actual) - 1) {
    loc_of_breaks_x <- c(loc_of_breaks_x[1] - x_pixels_increment, loc_of_breaks_x)
  }

  xaxis_cimg <- imager::as.cimg(fig_x[loc_y_start:1, ])
  xaxis_gray <- as.matrix(xaxis_cimg)
  x1 <- tesseract::ocr(magick::image_read(png::writePNG(image = xaxis_gray)), HOCR = TRUE)
  x1.tbl <- SurvdigitizeR:::fun_ocrtotbl(x1)
  x1.tbl <- x1.tbl[x1.tbl$word %in% as.character(X_actual), ]

  if (sum(x1.tbl$confidence > 90) > 0) {
    x1.tbl <- x1.tbl[x1.tbl$confidence > 90, ]
  }

  if (nrow(x1.tbl) > 0) {
    mean_dist <- mean(abs(x1.tbl$y0 - loc_y_start))
    x1.tbl <- x1.tbl[which(abs(x1.tbl$y0 - loc_y_start) <= (mean_dist + 10)), ]
  }

  if (length(loc_of_breaks_x) == 1 || (length(loc_of_breaks_x) * 2 < length(X_actual))) {
    max_conf_ind <- which(sort(x1.tbl$confidence, index.return = TRUE, decreasing = TRUE)$ix == 1)
    max_conf_ind2 <- which(sort(x1.tbl$confidence, index.return = TRUE, decreasing = TRUE)$ix == 2)
    word_diff <- abs(as.numeric(x1.tbl$word[max_conf_ind2]) - as.numeric(x1.tbl$word[max_conf_ind]))
    pix_diff <- abs(
      as.numeric((x1.tbl$x0[max_conf_ind2] + x1.tbl$x1[max_conf_ind2]) / 2) -
        as.numeric((x1.tbl$x0[max_conf_ind] + x1.tbl$x1[max_conf_ind]) / 2)
    )
    x_pixels_increment <- pix_diff / (word_diff / x_increment)
    x1.tbl <- x1.tbl[max_conf_ind, ]
    pxl_loc <- (x1.tbl$x0 + x1.tbl$x1) / 2
    X_0pixel <- (length(seq(0, x1.tbl$word, by = x_increment)) - 1) * x_pixels_increment - pxl_loc
  } else {
    x1.tbl$pxl_loc <- collapse_break_matches(loc_of_breaks_x, x1.tbl, "x0", "x1")
    break_length <- length(loc_of_breaks_x)
    number_length <- length(X_actual)

    if (break_length < number_length * 1.5) {
      x1.tbl.matched <- x1.tbl[!is.na(x1.tbl$pxl_loc), ]
      if (nrow(x1.tbl.matched) == 0) {
        x1.tbl.matched <- x1.tbl
      }

      x1.tbl.best <- x1.tbl.matched[which.max(x1.tbl.matched$confidence), ]
      X_0pixel <- (length(seq(0, x1.tbl.best$word, by = x_increment)) - 1) * x_pixels_increment - x1.tbl.best$pxl_loc
      if (!is.finite(X_0pixel) || X_0pixel < 0) {
        X_0pixel <- min(loc_of_breaks_x)
      }
    } else {
      x1.tbl <- x1.tbl[!is.na(x1.tbl$pxl_loc), ]
      if (nrow(x1.tbl) >= 2) {
        word_diff <- as.numeric(x1.tbl$word[2]) - as.numeric(x1.tbl$word[1])
        pix_diff <- as.numeric(x1.tbl$pxl_loc[2]) - as.numeric(x1.tbl$pxl_loc[1])
        x_pixels_increment <- pix_diff / (word_diff / x_increment)
        x1.tbl <- x1.tbl[which.max(x1.tbl$confidence), ]
        X_0pixel <- (length(seq(0, x1.tbl$word, by = x_increment)) - 1) * x_pixels_increment - x1.tbl$pxl_loc
      } else if (nrow(x1.tbl) == 1) {
        x1.tbl <- x1.tbl[1, ]
        X_0pixel <- (length(seq(0, x1.tbl$word, by = x_increment)) - 1) * x_pixels_increment - x1.tbl$pxl_loc
      } else {
        X_0pixel <- min(loc_of_breaks_x)
      }
    }
  }

  fig_y <- fig_bw[step2_axes$yaxis, -step2_axes$xaxis]
  loc_x_start <- which.max(colSums(fig_y < 0.9)) - 1
  if (loc_x_start == 0) {
    loc_x_start <- which.max(diff(colSums(fig_y < 0.9)))
  }

  y_breaks_loc <- which(fig_y[, (loc_x_start - 3)] < 0.6)
  y_breaks <- diff(y_breaks_loc)
  gr1 <- cumsum(!c(1, y_breaks) == 1)
  y_breaks_act <- vector(length = length(unique(gr1)))
  for (i in seq_along(unique(gr1))) {
    y_breaks_act[i] <- round(mean(y_breaks_loc[gr1 == unique(gr1)[i]]))
  }

  y_breaks_loc <- y_breaks_act
  y_breaks <- y_breaks[y_breaks != 1]
  break_indicator_y <- as.numeric(names(table(diff(y_breaks_act))))[which.max(table(diff(y_breaks_act)))]
  break_indicator_y <- round(break_indicator_y)

  if (length(y_breaks_act) == length(Y_actual) - 1) {
    y_breaks_act <- c(y_breaks_act[1] - break_indicator_y, y_breaks_act)
  }

  fig_y_cut <- fig_y[, (1:loc_x_start)]
  fig_y_flip <- t(fig_y_cut[, dim(fig_y_cut)[2]:1])

  if (y_text_vertical == TRUE) {
    yaxis_cimg <- imager::as.cimg(fig_y_cut)
  } else {
    yaxis_cimg <- imager::as.cimg(fig_y_flip)
  }

  yaxis_gray <- as.matrix(yaxis_cimg)
  img_input <- yaxis_gray[dim(yaxis_gray)[1]:1, ]
  y1 <- tesseract::ocr(magick::image_read(png::writePNG(image = img_input)), HOCR = TRUE)
  y1.tbl <- SurvdigitizeR:::fun_ocrtotbl(y1)
  wl <- y1.tbl$word
  end_with_ <- unlist(lapply(wl, function(x) endsWith(x, "-") | endsWith(x, "_")))
  y1.tbl$word[end_with_] <- unlist(lapply(wl[end_with_], function(x) stringr::str_sub(x, 1, -2)))
  y1.tbl <- y1.tbl[y1.tbl$word %in% as.character(Y_actual), ]

  if (y_text_vertical == TRUE) {
    length_axis <- max(dim(img_input))
    x0_temp <- y1.tbl$x0
    x1_temp <- y1.tbl$x1
    y1.tbl$x0 <- length_axis - y1.tbl$y1
    y1.tbl$x1 <- length_axis - y1.tbl$y0
    y1.tbl$y0 <- x0_temp
    y1.tbl$y1 <- x1_temp
  }

  if (sum(y1.tbl$confidence > 90) > 0) {
    y1.tbl <- y1.tbl[y1.tbl$confidence > 90, ]
  }

  if (y_text_vertical == TRUE) {
    y1.tbl <- y1.tbl[rank(as.numeric(y1.tbl$word)), ]
  }

  if (length(y_breaks_loc) == 1 || (length(y_breaks_loc) * 2 < length(Y_actual))) {
    max_conf_ind <- which(sort(y1.tbl$confidence, index.return = TRUE, decreasing = TRUE)$ix == 1)
    max_conf_ind2 <- which(sort(y1.tbl$confidence, index.return = TRUE, decreasing = TRUE)$ix == 2)
    word_diff <- abs(as.numeric(y1.tbl$word[max_conf_ind2]) - as.numeric(y1.tbl$word[max_conf_ind]))
    pix_diff <- abs(
      as.numeric((y1.tbl$x0[max_conf_ind2] + y1.tbl$x1[max_conf_ind2]) / 2) -
        as.numeric((y1.tbl$x0[max_conf_ind] + y1.tbl$x1[max_conf_ind]) / 2)
    )
    y_pixels_increment <- pix_diff / (word_diff / y_increment)
    y1.tbl <- y1.tbl[max_conf_ind, ]
    pxl_loc <- (y1.tbl$x0 + y1.tbl$x1) / 2
    Y_0pixel <- (length(seq(0, y1.tbl$word, by = y_increment)) - 1) * y_pixels_increment - pxl_loc
    break_indicator_y <- y_pixels_increment
  } else {
    if (dim(y1.tbl)[1] != 0) {
      y1.tbl$pxl_loc <- collapse_break_matches(y_breaks_loc, y1.tbl, "x0", "x1")
      if (nrow(y1.tbl) >= 2) {
        y1.tbl <- y1.tbl[!is.na(y1.tbl$pxl_loc), ]
        word_diff <- abs(as.numeric(y1.tbl$word[2]) - as.numeric(y1.tbl$word[1]))
        pix_diff <- as.numeric(y1.tbl$pxl_loc[2]) - as.numeric(y1.tbl$pxl_loc[1])
        y_pixels_increment <- pix_diff / (word_diff / y_increment)
        y1.tbl.multiple <- y1.tbl
        y1.tbl <- y1.tbl[which.max(y1.tbl$confidence), ]
        Y_0pixel <- (length(seq(0, y1.tbl$word, by = y_increment)) - 1) * y_pixels_increment - y1.tbl$pxl_loc
        if (length(y_breaks_loc) >= (length(Y_actual) * 2 - 2)) {
          match_increment <- y1.tbl.multiple[(y1.tbl.multiple$x0 < y_pixels_increment & y1.tbl.multiple$x1 > y_pixels_increment), ]
          if (nrow(match_increment) == 1) {
            break_indicator_y <- match_increment$pxl_loc
          }
        }
      } else {
        y1.tbl <- y1.tbl[which.max(y1.tbl$confidence), ]
        if (y1.tbl$word != 0) {
          Y_0pixel <- (length(seq(0, y1.tbl$word, by = y_increment)) - 1) * break_indicator_y - y1.tbl$pxl_loc
        } else {
          Y_0pixel <- y1.tbl$pxl_loc
        }
        if (Y_0pixel > min(y_breaks_loc) || Y_0pixel < 0) {
          if (Y_actual[1] > 0) {
            Y_0pixel <- min(y_breaks_loc) - Y_actual[1] / (y_increment / break_indicator_y)
          } else {
            Y_0pixel <- min(y_breaks_loc)
          }
        }
      }
    } else {
      if (Y_actual[1] > 0) {
        Y_0pixel <- min(y_breaks_loc) - Y_actual[1] / (y_increment / break_indicator_y)
      } else {
        Y_0pixel <- min(y_breaks_loc)
      }
    }
  }

  list(
    Y_0pixel = Y_0pixel,
    y_increment = y_increment / break_indicator_y,
    X_0pixel = X_0pixel,
    x_increment = x_increment / x_pixels_increment
  )
}

image_path <- read_flag("--image")
manifest_path <- read_flag("--manifest")
output_csv_path <- read_flag("--output-csv")
output_meta_path <- read_flag("--output-meta")
output_overlay_json_path <- read_flag("--output-overlay-json")

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The jsonlite package is required. Run Rscript scripts/bootstrap_r.R first.", call. = FALSE)
}

if (!requireNamespace("SurvdigitizeR", quietly = TRUE)) {
  stop("The SurvdigitizeR package is required. Run Rscript scripts/bootstrap_r.R first.", call. = FALSE)
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  stop("The dplyr package is required. Run Rscript scripts/bootstrap_r.R first.", call. = FALSE)
}

manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = TRUE)

step1 <- tryCatch(
  SurvdigitizeR:::img_read(path = image_path),
  error = function(error) handle_step_error(error, "Step 1: Loading image")
)

step2 <- tryCatch(
  SurvdigitizeR:::axes_identify(fig.hsl = step1, bg_lightness = 0.3),
  error = function(error) handle_step_error(error, "Step 2: Identifying axes")
)

step3 <- tryCatch(
  SurvdigitizeR:::fig_clean(
    fig.hsl = step2$fig.hsl,
    bg_lightness = 0.3,
    attempt_OCR = FALSE,
    word_sensitivity = 30
  ),
  error = function(error) handle_step_error(error, "Step 3: Cleaning figure")
)

step4 <- tryCatch(
  SurvdigitizeR:::color_cluster(
    fig.df = step3,
    num_curves = manifest$num_curves,
    censoring = FALSE,
    enhance = FALSE
  ),
  error = function(error) handle_step_error(error, "Step 4: Clustering colors")
)

step5 <- tryCatch(
  SurvdigitizeR:::overlap_detect(fig.grp = step4, nr_neighbors = 20),
  error = function(error) handle_step_error(error, "Step 5: Detecting overlaps")
)

step6 <- tryCatch(
  SurvdigitizeR:::lines_isolate(fig.curves = step5),
  error = function(error) handle_step_error(error, "Step 6: Isolating lines")
)

step7 <- tryCatch(
  safe_range_detect(
    step1_fig = step1,
    step2_axes = step2$axes,
    x_start = manifest$x_start,
    x_end = manifest$x_end,
    x_increment = manifest$x_increment,
    y_start = manifest$y_start,
    y_end = manifest$y_end,
    y_increment = manifest$y_increment,
    y_text_vertical = manifest$y_text_vertical
  ),
  error = function(error) handle_step_error(error, "Step 7: Detecting ranges")
)

result <- tryCatch(
  SurvdigitizeR:::fig_summarize(
    lines_vector = step6,
    range_list = step7,
    x_start = manifest$x_start,
    y_start = manifest$y_start,
    y_end = manifest$y_end
  ),
  error = function(error) handle_step_error(error, "Step 8: Summarizing data")
)

if (!is.data.frame(result) || nrow(result) == 0) {
  stop("SurvdigitizeR did not return a non-empty data frame.", call. = FALSE)
}

overlay_curves <- lapply(step6, function(curve_df) {
  data.frame(
    curve = as.integer(curve_df$curve),
    x = as.integer(step2$axes$xaxis[curve_df$x]),
    y = as.integer(step2$axes$yaxis[curve_df$y])
  )
})

overlay_points <- dplyr::bind_rows(overlay_curves)
curve_point_counts <- overlay_points |>
  dplyr::group_by(curve) |>
  dplyr::summarise(point_count = dplyr::n(), .groups = "drop")

overlay_json <- list(
  image = basename(image_path),
  width = dim(step1)[2],
  height = dim(step1)[1],
  curves = lapply(seq_len(nrow(curve_point_counts)), function(index) {
    curve_id <- curve_point_counts$curve[[index]]
    curve_points <- overlay_points[overlay_points$curve == curve_id, c("x", "y")]
    list(
      curve = curve_id,
      point_count = curve_point_counts$point_count[[index]],
      points = lapply(seq_len(nrow(curve_points)), function(point_index) {
        list(
          x = curve_points$x[[point_index]],
          y = curve_points$y[[point_index]]
        )
      })
    )
  })
)

utils::write.csv(result, output_csv_path, row.names = FALSE)

meta <- list(
  image = basename(image_path),
  rows = nrow(result),
  columns = names(result),
  curves = if ("curve" %in% names(result)) sort(unique(result$curve)) else NULL,
  overlay = list(
    width = overlay_json$width,
    height = overlay_json$height,
    total_points = nrow(overlay_points),
    curves = lapply(seq_len(nrow(curve_point_counts)), function(index) {
      list(
        curve = curve_point_counts$curve[[index]],
        point_count = curve_point_counts$point_count[[index]]
      )
    })
  )
)

jsonlite::write_json(meta, output_meta_path, auto_unbox = TRUE, pretty = TRUE)
jsonlite::write_json(overlay_json, output_overlay_json_path, auto_unbox = TRUE, pretty = TRUE)
