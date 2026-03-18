from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from .config import Settings


IMAGE_STATUSES = {
    "queued",
    "processing_manifest",
    "needs_review",
    "processing_digitizer",
    "completed",
    "failed",
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
                annotated_path TEXT,
                manifest_json TEXT,
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
        ensure_column(connection, "images", "output_log_path", "TEXT")
        ensure_column(connection, "images", "processing_log_json", "TEXT")


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
    if image.get("processing_log_json"):
        image["processing_log"] = json.loads(image["processing_log_json"])
    else:
        image["processing_log"] = []
    image["latest_log"] = image["processing_log"][-1] if image["processing_log"] else None
    image["review_required"] = bool(image.get("review_required"))
    return image


def ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing_columns = {row["name"] for row in rows}
    if column_name in existing_columns:
        return

    connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    connection.commit()
