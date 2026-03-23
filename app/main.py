from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import io
import json
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .manifest_service import ManifestService
from .preflight import build_preflight_report
from .runner import DigitizerRunner, PreflightArtifacts
from .schemas import ImageManifest, PreflightReport
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
    serialize_preflight,
    update_batch,
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
            "default_threshold": settings.auto_approve_threshold,
        },
    )


@app.post("/batches")
async def create_batch_route(
    request: Request,
    batch_label: str = Form(default=""),
    auto_approve_threshold: float = Form(default=settings.auto_approve_threshold),
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
    batch_id = create_batch(
        settings,
        label=label,
        image_count=len(extracted_files),
        auto_approve_threshold=auto_approve_threshold,
    )
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
            "batch_runtime_progress": build_batch_runtime_progress(batch),
        },
    )


@app.post("/batches/{batch_id}/threshold")
async def update_batch_threshold(
    batch_id: str,
    auto_approve_threshold: float = Form(...),
):
    batch = get_batch(settings, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    update_batch(settings, batch_id, auto_approve_threshold=auto_approve_threshold)
    return RedirectResponse(url=f"/batches/{batch_id}", status_code=303)


@app.get("/images/{image_id}")
def image_detail(request: Request, image_id: str):
    image = get_image(settings, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    batch = get_batch(settings, image["batch_id"])
    review_workflow = build_review_workflow(image)
    return templates.TemplateResponse(
        "image.html",
        {
            "request": request,
            "image": image,
            "batch": batch,
            "review_workflow": review_workflow,
            "runtime_progress": build_image_runtime_progress(image),
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
        "review": image.get("review_overlay_path"),
        "annotated": image.get("annotated_path"),
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
    crop_left: str = Form(default=""),
    crop_top: str = Form(default=""),
    crop_right: str = Form(default=""),
    crop_bottom: str = Form(default=""),
    exclusion_regions_json: str = Form(default="[]"),
    exclusion_polygons_json: str = Form(default="[]"),
    crop_hint: str = Form(default=""),
    notes: str = Form(default=""),
    llm_confidence: float = Form(default=1.0),
    review_required: str | None = Form(default=None),
    rerun_after_save: str | None = Form(default=None),
    approve_crop: str | None = Form(default=None),
):
    image = get_image(settings, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    batch = get_batch(settings, image["batch_id"])
    threshold = batch.get("auto_approve_threshold", settings.auto_approve_threshold) if batch else settings.auto_approve_threshold

    crop_approved = approve_crop == "true"
    rerun_requested = rerun_after_save == "true" or crop_approved
    keep_in_review = False if crop_approved else (review_required == "on" and not rerun_requested)

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
        crop_left=parse_optional_float(crop_left),
        crop_top=parse_optional_float(crop_top),
        crop_right=parse_optional_float(crop_right),
        crop_bottom=parse_optional_float(crop_bottom),
        exclusion_regions=parse_exclusion_regions(exclusion_regions_json),
        exclusion_polygons=parse_exclusion_polygons(exclusion_polygons_json),
        crop_hint=crop_hint or None,
        notes=notes or None,
        llm_confidence=llm_confidence,
        review_required=keep_in_review,
    )

    try:
        preview_result = ensure_preflight_preview(image, manifest)
        preview_fields, report, preview_payload, effective_manifest = unpack_preflight_result(preview_result, manifest)
        saved_manifest = effective_manifest.model_copy(
            update={"review_required": effective_manifest.review_required or report.blocking}
        )
        next_status = "queued" if rerun_requested and not saved_manifest.review_required else "needs_review"
        error_message = build_review_message(saved_manifest, threshold, report, False)
        queued_runner = process_single_image
        if crop_approved and rerun_requested and not saved_manifest.review_required:
            approved_prepared_path = digitizer_runner.commit_approved_crop(
                batch_id=image["batch_id"],
                image_id=image_id,
                prepared_path=Path(preview_fields["prepared_path"]),
                preview_payload=preview_payload,
            )
            preview_fields["prepared_path"] = str(approved_prepared_path)
            queued_runner = process_prepared_image
        if not rerun_requested and not saved_manifest.review_required and not report.blocking:
            error_message = (
                report.warnings[0]
                if report.warnings
                else "Review saved. Click 'Rerun Image' when you are ready to continue."
            )
        if next_status == "queued":
            error_message = None

        update_image(
            settings,
            image_id,
            manifest_json=serialize_manifest(saved_manifest.model_dump()),
            llm_confidence=saved_manifest.llm_confidence,
            review_required=1 if saved_manifest.review_required else 0,
            status=next_status,
            error_message=error_message,
            **preview_fields,
        )
        append_image_log(
            settings,
            image_id,
            "crop_approved" if crop_approved else "review_saved",
            (
                "Suggested crop and masks were approved and the image was queued for digitization."
                if crop_approved and rerun_requested and not saved_manifest.review_required
                else "Suggested crop approval was saved, but pre-flight still requires manual review."
                if crop_approved
                else "Review saved and image queued for digitization."
                if rerun_requested and not saved_manifest.review_required
                else "Review saved; image remains paused for manual review."
            ),
            level="info" if not saved_manifest.review_required else "warning",
        )

        if rerun_requested and not saved_manifest.review_required:
            if queued_runner is process_prepared_image:
                executor.submit(process_prepared_image, image_id)
            else:
                executor.submit(process_single_image, image_id, False)
    except Exception as error:  # noqa: BLE001
        update_image(settings, image_id, status="failed", error_message=str(error))
        append_image_log(
            settings,
            image_id,
            "preflight_failed",
            str(error),
            level="error",
        )

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

    preflight_report = image.get("preflight_report")
    if preflight_report and preflight_report.get("blocking"):
        update_image(
            settings,
            image_id,
            status="needs_review",
            error_message=first_blocking_message(preflight_report),
        )
        append_image_log(
            settings,
            image_id,
            "rerun_blocked",
            "Rerun was blocked because pre-flight checks still have blocking failures.",
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
    batch = get_batch(settings, image["batch_id"])
    threshold = batch.get("auto_approve_threshold", settings.auto_approve_threshold) if batch else settings.auto_approve_threshold

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

        preview_result = ensure_preflight_preview(image, manifest)
        preview_fields, report, _, effective_manifest = unpack_preflight_result(preview_result, manifest)
        manifest_for_run = effective_manifest.model_copy(
            update={"review_required": effective_manifest.review_required or report.blocking}
        )
        requires_review = manifest_for_run.should_review(threshold) if regenerate_manifest else manifest_for_run.review_required

        if requires_review:
            review_message = build_review_message(manifest_for_run, threshold, report, regenerate_manifest)
            update_image(
                settings,
                image_id,
                manifest_json=serialize_manifest(manifest_for_run.model_dump()),
                llm_confidence=manifest_for_run.llm_confidence,
                review_required=1 if manifest_for_run.review_required else 0,
                status="needs_review",
                error_message=review_message,
                **preview_fields,
            )
            append_image_log(
                settings,
                image_id,
                "needs_review",
                "Processing paused because pre-flight checks or the manifest still need manual review.",
                level="warning",
            )
            return

        append_image_log(settings, image_id, "processing_digitizer", "Running the extraction workflow.")
        update_image(settings, image_id, status="processing_digitizer", error_message=None)
        prepared_path, output_csv_path, output_meta_path, output_log_path, annotated_path = digitizer_runner.run(
            batch_id=image["batch_id"],
            image_id=image["id"],
            image_path=Path(image["original_path"]),
            manifest=manifest_for_run,
            prepared_path=Path(preview_fields["prepared_path"]) if preview_fields.get("prepared_path") else None,
        )
        update_image(
            settings,
            image_id,
            prepared_path=str(prepared_path),
            annotated_path=str(annotated_path),
            output_csv_path=str(output_csv_path),
            output_meta_path=str(output_meta_path),
            output_log_path=str(output_log_path),
            manifest_json=serialize_manifest(manifest_for_run.model_dump()),
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


def process_prepared_image(image_id: str) -> None:
    image = get_image(settings, image_id)
    if not image:
        return
    try:
        manifest = ImageManifest.model_validate(image["manifest"])
        prepared_path_value = image.get("prepared_path")
        if not prepared_path_value:
            raise RuntimeError("No approved prepared image was available for rerun.")
        prepared_path = Path(prepared_path_value)
        if not prepared_path.exists():
            raise RuntimeError("The approved prepared image no longer exists on disk.")

        append_image_log(settings, image_id, "processing_digitizer", "Running the extraction workflow on the approved cropped image.")
        update_image(settings, image_id, status="processing_digitizer", error_message=None)
        prepared_path, output_csv_path, output_meta_path, output_log_path, annotated_path = digitizer_runner.run(
            batch_id=image["batch_id"],
            image_id=image["id"],
            image_path=Path(image["original_path"]),
            manifest=manifest,
            prepared_path=prepared_path,
        )
        update_image(
            settings,
            image_id,
            prepared_path=str(prepared_path),
            annotated_path=str(annotated_path),
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
        append_image_log(settings, image_id, "failed", str(error), level="error")


def ensure_preflight_preview(
    image: dict,
    manifest: ImageManifest,
) -> tuple[dict[str, str | None], PreflightReport, dict, ImageManifest]:
    append_image_log(
        settings,
        image["id"],
        "processing_preflight",
        "Running staged pre-flight checks before digitization.",
    )
    artifacts = digitizer_runner.run_preflight(
        batch_id=image["batch_id"],
        image_id=image["id"],
        image_path=Path(image["original_path"]),
        manifest=manifest,
    )
    report = build_preflight_report(
        manifest=manifest,
        prepared_path=artifacts.prepared_path,
        preview_payload=artifacts.preview_payload,
    )

    effective_manifest = manifest
    fallback = maybe_apply_panel_crop_fallback(image, manifest, report)
    if fallback is not None:
        effective_manifest, artifacts, report = fallback
        append_image_log(
            settings,
            image["id"],
            "crop_suggested",
            "A suggested panel crop was prepared after full-image axis detection failed. Review the crop and rerun.",
            level="warning",
        )

    append_image_log(
        settings,
        image["id"],
        "preflight_ready",
        (
            "Pre-flight checks found blocking issues that require review."
            if report.blocking
            else "Pre-flight checks completed successfully."
        ),
        level="warning" if report.blocking else "info",
    )
    return {
        "prepared_path": str(artifacts.prepared_path),
        "review_overlay_path": str(artifacts.review_overlay_path),
        "output_log_path": str(artifacts.output_log_path),
        "preflight_json": serialize_preflight(report.model_dump()),
    }, report, artifacts.preview_payload, effective_manifest


def unpack_preflight_result(
    result: tuple,
    fallback_manifest: ImageManifest,
) -> tuple[dict[str, str | None], PreflightReport, dict, ImageManifest]:
    if len(result) == 4:
        preview_fields, report, preview_payload, effective_manifest = result
        return preview_fields, report, preview_payload, effective_manifest
    preview_fields, report, preview_payload = result
    return preview_fields, report, preview_payload, fallback_manifest


def maybe_apply_panel_crop_fallback(
    image: dict,
    manifest: ImageManifest,
    report: PreflightReport,
) -> tuple[ImageManifest, PreflightArtifacts, PreflightReport] | None:
    if not should_try_panel_crop_fallback(manifest, report):
        return None

    for candidate in build_panel_crop_candidates(manifest):
        try:
            artifacts = digitizer_runner.run_preflight(
                batch_id=image["batch_id"],
                image_id=f"{image['id']}-fallback",
                image_path=Path(image["original_path"]),
                manifest=candidate,
            )
            candidate_report = build_preflight_report(
                manifest=candidate,
                prepared_path=artifacts.prepared_path,
                preview_payload=artifacts.preview_payload,
            )
        except Exception:
            continue

        if not candidate_report.blocking:
            return candidate, artifacts, candidate_report

    return None


def should_try_panel_crop_fallback(manifest: ImageManifest, report: PreflightReport) -> bool:
    has_crop = all(
        value is not None
        for value in (manifest.crop_left, manifest.crop_top, manifest.crop_right, manifest.crop_bottom)
    )
    has_masks = bool(manifest.exclusion_regions or manifest.exclusion_polygons)
    if has_crop or has_masks:
        return False

    for check in report.checks:
        if check.id == "axis-detection" and check.status == "fail":
            return True
    return False


def build_panel_crop_candidates(manifest: ImageManifest) -> list[ImageManifest]:
    hint_prefix = "Automatic panel crop suggestion: exclude the lower risk table/caption and keep the full KM panel."
    notes_prefix = (
        "Automatic fallback crop was generated because full-image axis detection failed. "
        "Review the crop before rerun."
    )

    def merged_text(prefix: str, existing: str | None) -> str:
        return f"{prefix} {existing}".strip() if existing else prefix

    common_updates = {
        "review_required": True,
        "crop_hint": merged_text(hint_prefix, manifest.crop_hint),
        "notes": merged_text(notes_prefix, manifest.notes),
    }

    def candidate(**updates: object) -> ImageManifest:
        payload = manifest.model_dump()
        payload.update(common_updates)
        payload.update(updates)
        return ImageManifest.model_validate(payload)

    return [
        candidate(
            crop_left=0.12,
            crop_top=0.02,
            crop_right=0.94,
            crop_bottom=0.64,
            exclusion_regions=[
                {
                    "left": 0.58,
                    "top": 0.02,
                    "right": 0.98,
                    "bottom": 0.24,
                    "label": "summary block",
                }
            ],
            exclusion_polygons=[],
        ),
        candidate(
            crop_left=0.14,
            crop_top=0.02,
            crop_right=0.92,
            crop_bottom=0.60,
            exclusion_regions=[],
            exclusion_polygons=[],
        ),
        candidate(
            crop_left=0.12,
            crop_top=0.02,
            crop_right=0.94,
            crop_bottom=0.60,
            exclusion_regions=[],
            exclusion_polygons=[],
        ),
    ]


def parse_optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def parse_exclusion_regions(value: str) -> list[dict]:
    stripped = value.strip()
    if not stripped:
        return []
    parsed = json.loads(stripped)
    if not isinstance(parsed, list):
        raise ValueError("exclusion_regions_json must be a JSON array")
    return parsed


def parse_exclusion_polygons(value: str) -> list[dict]:
    stripped = value.strip()
    if not stripped:
        return []
    parsed = json.loads(stripped)
    if not isinstance(parsed, list):
        raise ValueError("exclusion_polygons_json must be a JSON array")
    return parsed


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
            if image.get("preflight_json"):
                archive.writestr(f"preflight/{image['id']}.json", image["preflight_json"])
            if image.get("original_path") and Path(image["original_path"]).exists():
                archive.write(image["original_path"], arcname=f"images/{image['filename']}")
            if image.get("prepared_path") and Path(image["prepared_path"]).exists():
                archive.write(image["prepared_path"], arcname=f"prepared/{image['id']}{Path(image['prepared_path']).suffix}")
            if image.get("review_overlay_path") and Path(image["review_overlay_path"]).exists():
                archive.write(image["review_overlay_path"], arcname=f"review/{image['id']}.png")
            if image.get("annotated_path") and Path(image["annotated_path"]).exists():
                archive.write(image["annotated_path"], arcname=f"annotated/{image['id']}.png")
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


def first_blocking_message(preflight_report: dict | None) -> str:
    if not preflight_report:
        return "Pre-flight checks require manual review before rerun."
    for check in preflight_report.get("checks", []):
        if check.get("status") == "fail":
            return check.get("message") or "Pre-flight checks require manual review before rerun."
    return "Pre-flight checks require manual review before rerun."


def first_blocking_check(preflight_report: dict | None) -> dict | None:
    if not preflight_report:
        return None
    for check in preflight_report.get("checks", []):
        if check.get("status") == "fail":
            return check
    return None


def build_review_workflow(image: dict) -> dict:
    manifest = image.get("manifest") or {}
    preflight_report = image.get("preflight_report") or {}
    blocking_check = first_blocking_check(preflight_report)
    has_crop = all(
        manifest.get(field_name) is not None
        for field_name in ("crop_left", "crop_top", "crop_right", "crop_bottom")
    )
    has_masks = bool(manifest.get("exclusion_regions") or manifest.get("exclusion_polygons"))

    blocker_hint = build_blocker_hint(blocking_check)
    default_message = (
        "Start with one crop box around the KM panel. Keep the full x-axis, y-axis, tick marks, and axis labels inside it."
        if not has_crop
        else blocker_hint
    )

    if has_masks and not has_crop:
        badge = "crop required"
        badge_class = "needs_review"
        primary_action_label = "Step 1: Draw Crop First"
        current_message = (
            "Masks are already saved, but the page still needs a crop box around the plot panel. "
            "Until that crop exists, rerun will only pause in review again."
        )
        primary_action_disabled = True
    elif preflight_report.get("blocking"):
        badge = "recheck needed"
        badge_class = "needs_review"
        primary_action_label = "Save Changes and Recheck"
        current_message = blocker_hint
        primary_action_disabled = False
    else:
        badge = "ready"
        badge_class = "completed"
        primary_action_label = "Use Crop and Rerun"
        current_message = (
            "The crop and mask setup is ready. Use the main button to re-run pre-flight and continue into extraction."
        )
        primary_action_disabled = False

    steps = [
        {
            "id": "crop",
            "title": "Draw Crop",
            "body": "Draw one crop box around the plot panel. Keep the full axes, tick marks, and axis labels inside the box.",
            "state": "done" if has_crop else "active",
        },
        {
            "id": "mask",
            "title": "Mask Non-Plot Areas",
            "body": "Use Mask only for risk tables, legends, summary blocks, or captions outside the plot panel.",
            "state": "done" if has_masks else ("ready" if has_crop else "pending"),
        },
        {
            "id": "preview",
            "title": "Check The Preview",
            "body": "The preview below should show only the plot panel you want extracted. If text or tables remain, tighten the crop or add masks.",
            "state": "ready" if has_crop else "pending",
        },
        {
            "id": "rerun",
            "title": "Recheck And Run",
            "body": "When the crop looks right, use the main button. If pre-flight still blocks, the blocker note above tells you what to correct next.",
            "state": "ready" if has_crop else "pending",
        },
    ]

    return {
        "badge": badge,
        "badge_class": badge_class,
        "current_message": current_message,
        "default_message": default_message,
        "blocker_hint": blocker_hint,
        "blocking": bool(preflight_report.get("blocking")),
        "has_crop": has_crop,
        "has_masks": has_masks,
        "primary_action_label": primary_action_label,
        "primary_action_disabled": primary_action_disabled,
        "steps": steps,
    }


def build_blocker_hint(blocking_check: dict | None) -> str:
    if not blocking_check:
        return "Save changes to refresh the preview and pre-flight checks."

    blocker_id = blocking_check.get("id")
    blocker_message = blocking_check.get("message") or "Pre-flight checks require review."
    blocker_hints = {
        "manifest-crop-required": (
            "Step 1: draw a crop box around the KM panel first. Keep the full axes inside the crop, then use masks only for material outside the plot."
        ),
        "manifest-exclusions": (
            "Move or shrink the masks so they no longer touch the x-axis or y-axis guard bands. The axes and tick labels must stay visible."
        ),
        "prepared-axis-contact": (
            "Widen the crop slightly. The detected plot is touching the image edge, which usually means the crop cut into the axis envelope."
        ),
        "axis-plot-size": (
            "Crop tighter around the actual KM panel and remove tables or summary blocks with masks. The detected plot bounds are still collapsing."
        ),
        "axis-spans": (
            "The plot detector is not finding a full x-axis or y-axis. Re-crop so the complete panel, including both axes, is inside the box."
        ),
        "range-break-match": (
            "The crop now looks usable, but the axis settings still do not match the visible tick marks. Confirm the x/y limits and every visible tick increment."
        ),
        "range-detection": (
            "The image reached range detection but the axis labels or breaks are still ambiguous. Tighten the crop around the plot and confirm the axis fields."
        ),
    }
    return blocker_hints.get(blocker_id, blocker_message)


def build_image_runtime_progress(image: dict) -> dict | None:
    status = image.get("status")
    if status not in {"queued", "processing_manifest", "processing_digitizer"}:
        return None

    latest_log = image.get("latest_log") or {}
    latest_stage = (latest_log.get("stage") or "").lower()
    current_stage = "queued"
    if status == "processing_digitizer":
        current_stage = "digitizer"
    elif latest_stage == "processing_preflight":
        current_stage = "preflight"
    elif status == "processing_manifest":
        current_stage = "manifest"

    order = ["queued", "manifest", "preflight", "digitizer", "completed"]
    labels = {
        "queued": "Queued",
        "manifest": "Read Figure",
        "preflight": "Check Crop + Axes",
        "digitizer": "Extract Curves",
        "completed": "Results Ready",
    }
    progress_map = {
        "queued": 8,
        "manifest": 32,
        "preflight": 56,
        "digitizer": 82,
        "completed": 100,
    }
    headline_map = {
        "queued": "Queued for the next worker slot",
        "manifest": "Reading figure structure and axis settings",
        "preflight": "Running pre-flight checks before extraction",
        "digitizer": "Extracting curves and writing artifacts",
        "completed": "Results are ready",
    }
    detail = latest_log.get("message") or (
        "This page refreshes automatically while the run is active."
    )
    current_index = order.index(current_stage)
    steps = [
        {
            "id": step_id,
            "label": labels[step_id],
            "state": "done" if index < current_index else "current" if index == current_index else "pending",
        }
        for index, step_id in enumerate(order)
    ]
    return {
        "headline": headline_map[current_stage],
        "detail": detail,
        "progress_percent": progress_map[current_stage],
        "steps": steps,
        "refresh_seconds": 5,
    }


def build_batch_runtime_progress(batch: dict) -> dict | None:
    if batch.get("status") not in {"queued", "processing"}:
        return None

    images = batch.get("images") or []
    queued_count = sum(1 for image in images if image.get("status") == "queued")
    manifest_count = sum(1 for image in images if image.get("status") == "processing_manifest")
    digitizer_count = sum(1 for image in images if image.get("status") == "processing_digitizer")
    review_count = sum(1 for image in images if image.get("status") == "needs_review")
    completed_count = sum(1 for image in images if image.get("status") == "completed")

    if manifest_count > 0:
        headline = "Reading uploaded figures"
    elif digitizer_count > 0:
        headline = "Running extraction across the batch"
    elif queued_count > 0:
        headline = "Queued images are waiting to start"
    else:
        headline = "Refreshing batch state"

    detail_parts = []
    if manifest_count:
        detail_parts.append(f"{manifest_count} reading figure structure")
    if digitizer_count:
        detail_parts.append(f"{digitizer_count} extracting curves")
    if queued_count:
        detail_parts.append(f"{queued_count} queued")
    if review_count:
        detail_parts.append(f"{review_count} in review")
    if completed_count:
        detail_parts.append(f"{completed_count} completed")

    return {
        "headline": headline,
        "detail": " · ".join(detail_parts) if detail_parts else "Batch state is being refreshed.",
        "refresh_seconds": 5,
    }


def build_review_message(
    manifest: ImageManifest,
    threshold: float,
    report: PreflightReport,
    apply_threshold: bool,
) -> str:
    reasons: list[str] = []
    if report.blocking:
        reasons.append(first_blocking_message(report.model_dump()))
    if manifest.review_required and not report.blocking:
        if manifest.crop_hint and manifest.crop_hint.startswith("Automatic panel crop suggestion:"):
            reasons.append(f"{manifest.crop_hint} Review the crop before digitization.")
        else:
            reasons.append("Manifest needs review before digitization.")
    elif apply_threshold and manifest.llm_confidence < threshold and not manifest.review_required:
        reasons.append(f"Confidence {manifest.llm_confidence:.2f} is below the batch threshold of {threshold:.2f}.")
    elif not report.blocking and report.warnings:
        reasons.append(report.warnings[0])
    return " ".join(reasons) if reasons else "Manual review is required before digitization."
