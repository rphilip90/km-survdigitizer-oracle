from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .preflight import PREFLIGHT_VERSION


IMAGE_STATUSES = {
    "queued",
    "processing_manifest",
    "needs_review",
    "processing_digitizer",
    "completed",
    "failed",
}
REVIEW_REASON_LABELS = {
    "needs_crop": "Needs Crop",
    "needs_axis_review": "Needs Axis Review",
    "ready_to_run": "Ready To Run",
    "engine_failed": "Engine Failed",
}
DIAGNOSTIC_LABELS = {
    "layout_axis_failure": "Layout / axis isolation",
    "collapsed_plot_bounds": "Collapsed plot bounds",
    "mask_overlaps_axis": "Mask overlaps axis",
    "range_calibration_sparse": "Sparse range calibration",
    "engine_runtime_failure": "Engine runtime failure",
    "suspicious_output": "Suspicious extracted output",
    "ready_to_run": "Ready to run",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(settings: Settings) -> None:
    with connect(settings.db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS batches (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                status TEXT NOT NULL,
                auto_approve_threshold REAL,
                image_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                prepared_path TEXT,
                review_overlay_path TEXT,
                annotated_path TEXT,
                manifest_json TEXT,
                preflight_json TEXT,
                llm_confidence REAL,
                review_required INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                output_csv_path TEXT,
                output_meta_path TEXT,
                output_log_path TEXT,
                processing_log_json TEXT NOT NULL DEFAULT '[]',
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_images_batch_id ON images(batch_id);
            CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
            """
        )
        ensure_column(connection, "batches", "auto_approve_threshold", "REAL")
        ensure_column(connection, "images", "annotated_path", "TEXT")
        ensure_column(connection, "images", "review_overlay_path", "TEXT")
        ensure_column(connection, "images", "output_log_path", "TEXT")
        ensure_column(connection, "images", "processing_log_json", "TEXT")
        ensure_column(connection, "images", "preflight_json", "TEXT")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def create_batch(
    settings: Settings,
    label: str,
    image_count: int,
    auto_approve_threshold: float | None = None,
) -> str:
    batch_id = uuid.uuid4().hex
    now = utc_now()
    threshold = auto_approve_threshold if auto_approve_threshold is not None else settings.auto_approve_threshold
    with connect(settings.db_path) as connection:
        connection.execute(
            """
            INSERT INTO batches (id, label, status, auto_approve_threshold, image_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (batch_id, label, "queued", threshold, image_count, now, now),
        )
        connection.commit()
    return batch_id


def update_batch(settings: Settings, batch_id: str, **fields: Any) -> None:
    if not fields:
        return

    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in fields.keys())
    values = list(fields.values()) + [batch_id]

    with connect(settings.db_path) as connection:
        connection.execute(
            f"UPDATE batches SET {assignments} WHERE id = ?",
            values,
        )
        connection.commit()


def create_image(settings: Settings, batch_id: str, filename: str, original_path: str) -> str:
    image_id = uuid.uuid4().hex
    now = utc_now()
    with connect(settings.db_path) as connection:
        connection.execute(
            """
            INSERT INTO images (
                id, batch_id, filename, original_path, status, processing_log_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (image_id, batch_id, filename, original_path, "queued", "[]", now, now),
        )
        connection.commit()
    refresh_batch_status(settings, batch_id)
    return image_id


def list_batches(settings: Settings, limit: int = 20) -> list[dict[str, Any]]:
    with connect(settings.db_path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM batches
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    batches = [row_to_dict(row) for row in rows]
    for batch in batches:
        if batch.get("auto_approve_threshold") is None:
            batch["auto_approve_threshold"] = settings.auto_approve_threshold
    return batches


def get_batch(settings: Settings, batch_id: str) -> dict[str, Any] | None:
    with connect(settings.db_path) as connection:
        batch_row = connection.execute(
            "SELECT * FROM batches WHERE id = ?",
            (batch_id,),
        ).fetchone()
        if not batch_row:
            return None
        image_rows = connection.execute(
            """
            SELECT * FROM images
            WHERE batch_id = ?
            ORDER BY created_at ASC
            """,
            (batch_id,),
        ).fetchall()
    batch = row_to_dict(batch_row)
    if batch.get("auto_approve_threshold") is None:
        batch["auto_approve_threshold"] = settings.auto_approve_threshold
    batch["images"] = [deserialize_image_row(row_to_dict(row)) for row in image_rows]
    return batch


def get_image(settings: Settings, image_id: str) -> dict[str, Any] | None:
    with connect(settings.db_path) as connection:
        row = connection.execute(
            "SELECT * FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
    return deserialize_image_row(row_to_dict(row))


def update_image(settings: Settings, image_id: str, **fields: Any) -> None:
    if not fields:
        return

    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in fields.keys())
    values = list(fields.values()) + [image_id]

    with connect(settings.db_path) as connection:
        connection.execute(
            f"UPDATE images SET {assignments} WHERE id = ?",
            values,
        )
        batch_id_row = connection.execute(
            "SELECT batch_id FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
        connection.commit()

    if batch_id_row:
        refresh_batch_status(settings, batch_id_row["batch_id"])


def refresh_batch_status(settings: Settings, batch_id: str) -> None:
    with connect(settings.db_path) as connection:
        rows = connection.execute(
            "SELECT status FROM images WHERE batch_id = ?",
            (batch_id,),
        ).fetchall()
        statuses = [row["status"] for row in rows]
        if not statuses:
            next_status = "queued"
        elif any(status in {"processing_manifest", "processing_digitizer"} for status in statuses):
            next_status = "processing"
        elif any(status == "failed" for status in statuses):
            next_status = "failed"
        elif any(status == "needs_review" for status in statuses):
            next_status = "needs_review"
        elif all(status == "completed" for status in statuses):
            next_status = "completed"
        else:
            next_status = "queued"

        connection.execute(
            "UPDATE batches SET status = ?, updated_at = ? WHERE id = ?",
            (next_status, utc_now(), batch_id),
        )
        connection.commit()


def serialize_manifest(manifest: dict[str, Any] | None) -> str | None:
    if manifest is None:
        return None
    return json.dumps(manifest, indent=2)


def serialize_preflight(report: dict[str, Any] | None) -> str | None:
    if report is None:
        return None
    return json.dumps(report, indent=2)


def append_image_log(
    settings: Settings,
    image_id: str,
    stage: str,
    message: str,
    level: str = "info",
) -> dict[str, Any] | None:
    timestamp = utc_now()
    entry = {
        "timestamp": timestamp,
        "stage": stage,
        "level": level,
        "message": message,
    }

    with connect(settings.db_path) as connection:
        row = connection.execute(
            "SELECT batch_id, processing_log_json FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
        if not row:
            return None

        existing_logs = []
        if row["processing_log_json"]:
            existing_logs = json.loads(row["processing_log_json"])
        existing_logs.append(entry)
        existing_logs = existing_logs[-50:]

        connection.execute(
            """
            UPDATE images
            SET processing_log_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(existing_logs, indent=2), timestamp, image_id),
        )
        connection.execute(
            "UPDATE batches SET updated_at = ? WHERE id = ?",
            (timestamp, row["batch_id"]),
        )
        connection.commit()

    return entry


def deserialize_image_row(image: dict[str, Any] | None) -> dict[str, Any] | None:
    if image is None:
        return None
    if image.get("manifest_json"):
        image["manifest"] = json.loads(image["manifest_json"])
    else:
        image["manifest"] = None
    if image.get("preflight_json"):
        image["preflight_report"] = json.loads(image["preflight_json"])
    else:
        image["preflight_report"] = None
    if image.get("processing_log_json"):
        image["processing_log"] = json.loads(image["processing_log_json"])
    else:
        image["processing_log"] = []
    image["latest_log"] = image["processing_log"][-1] if image["processing_log"] else None
    image["review_required"] = bool(image.get("review_required"))
    preflight_report = image.get("preflight_report") or {}
    preflight_stale = bool(preflight_report) and preflight_report.get("preflight_version") != PREFLIGHT_VERSION
    image["preflight_stale"] = preflight_stale
    if image.get("preflight_report") is not None:
        image["preflight_report"]["stale"] = preflight_stale
    image["effective_preflight_blocking"] = bool(preflight_report.get("blocking")) and not preflight_stale

    review_state = derive_review_state(image)
    image.update(review_state)
    return image


def derive_review_state(image: dict[str, Any]) -> dict[str, Any]:
    manifest = image.get("manifest") or {}
    preflight_report = image.get("preflight_report") or {}
    status = image.get("status")
    has_crop = all(
        manifest.get(field_name) is not None
        for field_name in ("crop_left", "crop_top", "crop_right", "crop_bottom")
    )
    has_suggested_preparation = bool(preflight_report.get("suggested_preparation"))
    preflight_stale = bool(image.get("preflight_stale"))
    effective_preflight_blocking = bool(image.get("effective_preflight_blocking"))

    diagnostic_category = preflight_report.get("diagnostic_category")
    review_reason = preflight_report.get("review_reason")

    if status == "failed":
        review_reason = "engine_failed"
        diagnostic_category = diagnostic_category or "engine_runtime_failure"
    elif status == "completed":
        review_reason = "ready_to_run"
        diagnostic_category = diagnostic_category or "ready_to_run"
    elif status == "needs_review":
        if diagnostic_category == "suspicious_output":
            review_reason = "needs_axis_review"
        elif preflight_stale:
            review_reason = "needs_crop" if has_suggested_preparation or not has_crop else "needs_axis_review"
            diagnostic_category = diagnostic_category or ("layout_axis_failure" if review_reason == "needs_crop" else "range_calibration_sparse")
        elif effective_preflight_blocking:
            review_reason = review_reason or "needs_axis_review"
            diagnostic_category = diagnostic_category or "layout_axis_failure"
        elif manifest.get("review_required"):
            if has_suggested_preparation or not has_crop:
                review_reason = "needs_crop"
                diagnostic_category = diagnostic_category or "layout_axis_failure"
            else:
                review_reason = review_reason or "needs_axis_review"
                diagnostic_category = diagnostic_category or "range_calibration_sparse"
        else:
            review_reason = "ready_to_run"
            diagnostic_category = diagnostic_category or "ready_to_run"
    else:
        review_reason = review_reason or ("engine_failed" if status == "failed" else "ready_to_run")
        diagnostic_category = diagnostic_category or ("engine_runtime_failure" if status == "failed" else "ready_to_run")

    user_message = derive_user_message(image, review_reason, diagnostic_category)
    status_badge_label = derive_status_badge_label(status, review_reason)

    return {
        "review_reason": review_reason,
        "review_reason_label": REVIEW_REASON_LABELS.get(review_reason, review_reason.replace("_", " ").title()),
        "diagnostic_category": diagnostic_category,
        "diagnostic_label": DIAGNOSTIC_LABELS.get(diagnostic_category or "", (diagnostic_category or "Uncategorized").replace("_", " ").title()),
        "user_message": user_message,
        "status_badge_label": status_badge_label,
        "status_description": derive_status_description(status, review_reason, preflight_stale),
        "preflight_version": preflight_report.get("preflight_version"),
    }


def derive_status_badge_label(status: str | None, review_reason: str | None) -> str:
    if status == "needs_review":
        return REVIEW_REASON_LABELS.get(review_reason or "needs_axis_review", "Needs Review")
    if not status:
        return "Unknown"
    return status.replace("_", " ").title()


def derive_status_description(status: str | None, review_reason: str | None, preflight_stale: bool) -> str:
    if status == "processing_manifest":
        return "Reading axes, tick spacing, and curve count."
    if status == "processing_digitizer":
        return "Running the extraction workflow."
    if status == "completed":
        return "Extraction finished successfully."
    if status == "failed":
        return "Processing stopped with an unrecoverable engine or infrastructure error."
    if status == "needs_review":
        if preflight_stale:
            return "Saved checks were generated by an older workflow version and should be refreshed."
        if review_reason == "needs_crop":
            return "Paused until the plot panel crop and masks are confirmed."
        if review_reason == "ready_to_run":
            return "Review is saved and this image is ready to rerun."
        return "Paused until axis settings or tick spacing are confirmed."
    return "Waiting to start."


def derive_user_message(image: dict[str, Any], review_reason: str | None, diagnostic_category: str | None) -> str:
    if image.get("status") == "failed":
        return image.get("error_message") or "The extraction workflow failed."

    if image.get("preflight_stale"):
        return "Saved pre-flight checks were generated by an older workflow version. Save review or rerun this image to refresh them."

    if image.get("status") == "needs_review":
        if image.get("effective_preflight_blocking"):
            blocking_message = first_blocking_message(image.get("preflight_report"))
            if blocking_message:
                return blocking_message
        if review_reason == "needs_crop":
            crop_hint = (image.get("manifest") or {}).get("crop_hint")
            return crop_hint or "Crop the full KM panel first, then mask only non-plot blocks such as risk tables or summary text."
        if review_reason == "ready_to_run":
            return "The crop and axis settings look usable. Save review and rerun to continue into extraction."
        if diagnostic_category == "range_calibration_sparse":
            return "Confirm the axis limits and every visible tick increment before rerun."
        return image.get("error_message") or "Confirm the manifest before rerun."

    return image.get("error_message") or (image.get("latest_log") or {}).get("message") or ""


def first_blocking_message(preflight_report: dict | None) -> str | None:
    if not preflight_report:
        return None
    for check in preflight_report.get("checks", []):
        if check.get("status") == "fail":
            return check.get("message")
    return None


def ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing_columns = {row["name"] for row in rows}
    if column_name in existing_columns:
        return

    connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    connection.commit()
