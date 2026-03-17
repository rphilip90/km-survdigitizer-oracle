packages <- c("jsonlite", "remotes")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

if (!requireNamespace("SurvdigitizeR", quietly = TRUE)) {
  remotes::install_github("Pechli-Lab/SurvdigitizeR")
}

message("R bootstrap completed.")

