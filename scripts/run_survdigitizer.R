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
  SurvdigitizeR:::range_detect(
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
