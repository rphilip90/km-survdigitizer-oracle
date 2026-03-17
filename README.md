# KM SurvdigitizeR Oracle

A small hosted batch app for Kaplan-Meier image digitization using OpenAI Vision for parameter inference and `SurvdigitizeR` for extraction.

## What It Does

1. Upload a folder or zip of KM images.
2. Generate a structured manifest per image with OpenAI Vision.
3. Auto-run `SurvdigitizeR` on high-confidence images.
4. Queue ambiguous images for review.
5. Export combined outputs, manifests, source images, and run logs.

## Why This Repo Exists

The JavaScript GitHub Pages prototype is now parked as an experiment. This repo is the production path for a small hosted workflow on Oracle Always Free.

## Tech Stack

- Python + FastAPI for the web app and batch orchestration
- SQLite for local job state
- OpenAI API for image-to-manifest inference
- `Rscript` + `SurvdigitizeR` for curve extraction
- Caddy or Nginx on the same Oracle VM for reverse proxying
- Local disk for uploads, manifests, outputs, and logs

## Manifest Fields

Each image gets a manifest containing:

- `image_id`
- `filename`
- `num_curves`
- `x_start`, `x_end`, `x_increment`
- `y_start`, `y_end`, `y_increment`
- `y_text_vertical`
- optional `rotation`, `crop_hint`, `notes`
- `llm_confidence`
- `review_required`

## Local Setup

### 1. Create a Python virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Create your environment file

```powershell
Copy-Item env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

### 3. Install the R dependencies

```powershell
Rscript scripts/bootstrap_r.R
```

### 4. Start the app

```powershell
.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit `http://127.0.0.1:8000`.

## Oracle Always Free Deployment

This repo is designed for a single Ubuntu VM on Oracle Cloud Always Free.

### Recommended shape

- Ubuntu VM on Ampere A1 Flex
- 2 OCPU / 12 GB RAM
- one public IP
- Caddy reverse proxy on port 80
- FastAPI app on `127.0.0.1:8000`

### Deploy outline

1. Provision the Oracle VM.
2. Install Python 3.12+, R 4.5+, and system packages required by `SurvdigitizeR`.
3. Clone this repo to `/opt/km-survdigitizer-oracle`.
4. Create a virtual environment and install Python requirements.
5. Run `Rscript scripts/bootstrap_r.R`.
6. Copy `.env` with your OpenAI API key.
7. Install the provided systemd unit and Caddy config from `deploy/`.
8. Start the service and browse to the VM public IP.
9. Keep the instance inside Oracle Always Free limits; no managed database or load balancer is required.

## Notes

- v1 has no authentication.
- `rotation` is applied automatically before the R call.
- `crop_hint` is saved for review, but cropping is not automatically applied in v1.
- Reviewed images can be rerun with the saved manifest even if the original LLM confidence remains below the auto-approve threshold.
- The only paid component is the OpenAI API.
