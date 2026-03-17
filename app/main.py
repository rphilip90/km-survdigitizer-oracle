from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import io
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .manifest_service import ManifestService
from .runner import DigitizerRunner
from .schemas import ImageManifest
from .store import (
    append_image_log,
    create_batch,
    create_image,
    get_batch,
    get_image,
    init_db,
    list_batches,
    refresh_batch_status,
    serialize_manifest,
    update_image,
)


app = FastAPI(title=settings.app_title)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

executor = ThreadPoolExecutor(max_workers=settings.max_workers)
manifest_service = ManifestService(settings)
digitizer_runner = DigitizerRunner(settings)

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@app.on_event("startup")
def startup() -> None:
    init_db(settings)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "batches": list_batches(settings),
            "app_title": settings.app_title,
        },
    )


@app.post("/batches")
async def create_batch_route(
    request: Request,
    batch_label: str = Form(default=""),
    zip_file: UploadFile | None = File(default=None),
    images: list[UploadFile] | None = File(default=None),
):
    extracted_files: list[tuple[str, bytes]] = []
    label = batch_label.strip()

    if zip_file and zip_file.filename:
        zip_bytes = await zip_file.read()
        label = label or Path(zip_file.filename).stem
        extracted_files.extend(read_images_from_zip(zip_bytes))

    if images:
        for image in images:
            if not image.filename:
                continue
            suffix = Path(image.filename).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                continue
            extracted_files.append((Path(image.filename).name, await image.read()))

    if not extracted_files:
        raise HTTPException(status_code=400, detail="Upload a zip file or one or more image files.")

    label = label or f"Batch {uuid.uuid4().hex[:8]}"
    batch_id = create_batch(settings, label=label, image_count=len(extracted_files))
    batch_dir = settings.upload_dir / batch_id / "original"
    batch_dir.mkdir(parents=True, exist_ok=True)
    seen_names: dict[str, int] = {}

    for filename, content in extracted_files:
        unique_name = uniquify_filename(filename, seen_names)
        destination = batch_dir / unique_name
        destination.write_bytes(content)
        image_id = create_image(settings, batch_id=batch_id, filename=unique_name, original_path=str(destination))
        append_image_log(settings, image_id, "queued", "Image uploaded and queued for processing.")

    executor.submit(process_batch, batch_id)

    return RedirectResponse(url=f"/batches/{batch_id}", status_code=303)


@app.get("/batches/{batch_id}")
def batch_detail(request: Request, batch_id: str):
    batch = get_batch(settings, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return templates.TemplateResponse(
        "batch.html",
        {
            "request": request,
            "batch": batch,
            "threshold": settings.auto_approve_threshold,
        },
    )


@app.get("/images/{image_id}")
def image_detail(request: Request, image_id: str):
    image = get_image(settings, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    batch = get_batch(settings, image["batch_id"])
    return templates.TemplateResponse(
        "image.html",
        {
            "request": request,
            "image": image,
            "batch": batch,
        },
    )


@app.get("/images/{image_id}/raw")
def image_raw(image_id: str):
    image = get_image(settings, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image["original_path"])


@app.get("/images/{image_id}/artifact/{kind}")
def image_artifact(image_id: str, kind: str):
    image = get_image(settings, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    artifact_map = {
        "prepared": image.get("prepared_path"),
        "csv": image.get("output_csv_path"),
        "meta": image.get("output_meta_path"),
        "log": image.get("output_log_path"),
    }
    artifact_path = artifact_map.get(kind)
    if not artifact_path or not Path(artifact_path).exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(artifact_path)


@app.post("/images/{image_id}/review")
async def save_review(
    image_id: str,
    num_curves: int = Form(...),
    x_start: float = Form(...),
    x_end: float = Form(...),
    x_increment: float = Form(...),
    y_start: float = Form(...),
    y_end: float = Form(...),
    y_increment: float = Form(...),
    y_text_vertical: str = Form(default="false"),
    rotation: int = Form(default=0),
    crop_hint: str = Form(default=""),
    notes: str = Form(default=""),
    llm_confidence: float = Form(default=1.0),
    review_required: str | None = Form(default=None),
    rerun_after_save: str | None = Form(default=None),
):
    image = get_image(settings, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    rerun_requested = rerun_after_save == "true"
    keep_in_review = review_required == "on" and not rerun_requested

    manifest = ImageManifest(
        image_id=image["id"],
        filename=image["filename"],
        num_curves=num_curves,
        x_start=x_start,
        x_end=x_end,
        x_increment=x_increment,
        y_start=y_start,
        y_end=y_end,
        y_increment=y_increment,
        y_text_vertical=y_text_vertical.lower() == "true",
        rotation=rotation,
        crop_hint=crop_hint or None,
        notes=notes or None,
        llm_confidence=llm_confidence,
        review_required=keep_in_review,
    )

    update_image(
        settings,
        image_id,
        manifest_json=serialize_manifest(manifest.model_dump()),
        llm_confidence=manifest.llm_confidence,
        review_required=1 if manifest.review_required else 0,
        status="queued" if rerun_requested and not manifest.review_required else "needs_review",
        error_message=(
            "Clear 'Keep this image in review' and save before rerun."
            if manifest.review_required
            else None
        ),
    )
    append_image_log(
        settings,
        image_id,
        "review_saved",
        (
            "Review saved and image queued for digitization."
            if rerun_requested and not manifest.review_required
            else "Review saved; image remains paused for manual review."
        ),
        level="info" if not manifest.review_required else "warning",
    )

    if rerun_requested and not manifest.review_required:
        executor.submit(process_single_image, image_id, False)

    return RedirectResponse(url=f"/images/{image_id}", status_code=303)


@app.post("/images/{image_id}/rerun")
def rerun_image(image_id: str):
    image = get_image(settings, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    manifest = image.get("manifest")
    if manifest and manifest.get("review_required"):
        update_image(
            settings,
            image_id,
            status="needs_review",
            error_message="Clear 'Keep this image in review' and save before rerun.",
        )
        append_image_log(
            settings,
            image_id,
            "rerun_blocked",
            "Rerun was blocked because the image is still marked for review.",
            level="warning",
        )
        return RedirectResponse(url=f"/images/{image_id}", status_code=303)

    append_image_log(settings, image_id, "rerun_requested", "Manual rerun requested.")
    executor.submit(process_single_image, image_id, False)
    return RedirectResponse(url=f"/images/{image_id}", status_code=303)


@app.get("/batches/{batch_id}/export")
def export_batch(batch_id: str):
    batch = get_batch(settings, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    export_path = build_export_archive(batch)
    return FileResponse(
        export_path,
        media_type="application/zip",
        filename=f"{batch['label'].replace(' ', '_')}_export.zip",
    )


def process_batch(batch_id: str) -> None:
    batch = get_batch(settings, batch_id)
    if not batch:
        return
    for image in batch["images"]:
        process_single_image(image["id"], True)


def process_single_image(image_id: str, regenerate_manifest: bool) -> None:
    image = get_image(settings, image_id)
    if not image:
        return

    try:
        if regenerate_manifest or not image.get("manifest"):
            append_image_log(settings, image_id, "processing_manifest", "Generating manifest from uploaded image.")
            update_image(settings, image_id, status="processing_manifest", error_message=None)
            manifest = manifest_service.generate_manifest(
                image_id=image["id"],
                filename=image["filename"],
                image_path=Path(image["original_path"]),
            )
            update_image(
                settings,
                image_id,
                manifest_json=serialize_manifest(manifest.model_dump()),
                llm_confidence=manifest.llm_confidence,
                review_required=1 if manifest.review_required else 0,
            )
            append_image_log(
                settings,
                image_id,
                "manifest_generated",
                f"Manifest generated at confidence {manifest.llm_confidence:.2f}.",
            )
        else:
            manifest = ImageManifest.model_validate(image["manifest"])
            append_image_log(settings, image_id, "manifest_loaded", "Using the saved reviewed manifest.")

        requires_review = (
            manifest.should_review(settings.auto_approve_threshold)
            if regenerate_manifest
            else manifest.review_required
        )

        if requires_review:
            update_image(
                settings,
                image_id,
                status="needs_review",
                error_message=(
                    "Manifest needs review before digitization."
                    if manifest.review_required
                    else f"Confidence {manifest.llm_confidence:.2f} is below auto-approve threshold."
                ),
            )
            append_image_log(
                settings,
                image_id,
                "needs_review",
                "Processing paused because the manifest needs manual review.",
                level="warning",
            )
            return

        append_image_log(settings, image_id, "processing_digitizer", "Running SurvdigitizeR extraction.")
        update_image(settings, image_id, status="processing_digitizer", error_message=None)
        prepared_path, output_csv_path, output_meta_path, output_log_path = digitizer_runner.run(
            batch_id=image["batch_id"],
            image_id=image["id"],
            image_path=Path(image["original_path"]),
            manifest=manifest,
        )
        update_image(
            settings,
            image_id,
            prepared_path=str(prepared_path),
            output_csv_path=str(output_csv_path),
            output_meta_path=str(output_meta_path),
            output_log_path=str(output_log_path),
            status="completed",
            error_message=None,
            review_required=0,
        )
        append_image_log(
            settings,
            image_id,
            "completed",
            "Digitization completed and artifacts were written successfully.",
        )
    except Exception as error:  # noqa: BLE001
        update_image(settings, image_id, status="failed", error_message=str(error))
        append_image_log(
            settings,
            image_id,
            "failed",
            str(error),
            level="error",
        )


def build_export_archive(batch: dict) -> Path:
    export_dir = settings.export_dir / batch["id"]
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "batch_export.zip"

    summary_buffer = io.StringIO()
    writer = csv.writer(summary_buffer)
    writer.writerow(["image_id", "filename", "status", "confidence", "review_required", "error"])
    for image in batch["images"]:
        writer.writerow([
            image["id"],
            image["filename"],
            image["status"],
            image.get("llm_confidence") or "",
            image.get("review_required"),
            image.get("error_message") or "",
        ])

    with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("batch-summary.csv", summary_buffer.getvalue())

        for image in batch["images"]:
            if image.get("manifest_json"):
                archive.writestr(f"manifests/{image['id']}.json", image["manifest_json"])
            if image.get("original_path") and Path(image["original_path"]).exists():
                archive.write(image["original_path"], arcname=f"images/{image['filename']}")
            if image.get("output_csv_path") and Path(image["output_csv_path"]).exists():
                archive.write(image["output_csv_path"], arcname=f"results/{image['id']}.csv")
            if image.get("output_meta_path") and Path(image["output_meta_path"]).exists():
                archive.write(image["output_meta_path"], arcname=f"results/{image['id']}.meta.json")
            if image.get("output_log_path") and Path(image["output_log_path"]).exists():
                archive.write(image["output_log_path"], arcname=f"logs/{image['id']}.log")

    return export_path


def read_images_from_zip(payload: bytes) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            suffix = Path(member.filename).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                continue
            files.append((Path(member.filename).name, archive.read(member.filename)))
    return files


def uniquify_filename(filename: str, seen_names: dict[str, int]) -> str:
    if filename not in seen_names:
        seen_names[filename] = 1
        return filename

    path = Path(filename)
    index = seen_names[filename] + 1
    candidate = f"{path.stem}-{index}{path.suffix}"
    while candidate in seen_names:
        index += 1
        candidate = f"{path.stem}-{index}{path.suffix}"

    seen_names[filename] = index
    seen_names[candidate] = 1
    return candidate
