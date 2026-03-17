args <- commandArgs(trailingOnly = TRUE)

read_flag <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) {
    stop(sprintf("Missing required flag: %s", flag), call. = FALSE)
  }
  args[index + 1]
}

image_path <- read_flag("--image")
manifest_path <- read_flag("--manifest")
output_csv_path <- read_flag("--output-csv")
output_meta_path <- read_flag("--output-meta")

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The jsonlite package is required. Run Rscript scripts/bootstrap_r.R first.", call. = FALSE)
}

if (!requireNamespace("SurvdigitizeR", quietly = TRUE)) {
  stop("The SurvdigitizeR package is required. Run Rscript scripts/bootstrap_r.R first.", call. = FALSE)
}

manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = TRUE)

result <- SurvdigitizeR::survival_digitize(
  img_path = image_path,
  num_curves = manifest$num_curves,
  x_start = manifest$x_start,
  x_end = manifest$x_end,
  x_increment = manifest$x_increment,
  y_start = manifest$y_start,
  y_end = manifest$y_end,
  y_increment = manifest$y_increment,
  y_text_vertical = manifest$y_text_vertical,
  attempt_OCR = FALSE,
  censoring = FALSE
)

utils::write.csv(result, output_csv_path, row.names = FALSE)

summary <- list(
  image = basename(image_path),
  rows = nrow(result),
  columns = names(result),
  curves = if ("curve" %in% names(result)) sort(unique(result$curve)) else NULL
)

jsonlite::write_json(summary, output_meta_path, auto_unbox = TRUE, pretty = TRUE)

