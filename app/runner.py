from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import uuid

from PIL import Image

from .config import Settings
from .schemas import ImageManifest


class DigitizerRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare_image(self, batch_id: str, image_path: Path, manifest: ImageManifest) -> Path:
        prepared_dir = self.settings.upload_dir / batch_id / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = prepared_dir / f"{image_path.stem}-{uuid.uuid4().hex}{image_path.suffix}"

        rotation = manifest.rotation or 0
        if rotation == 0:
            shutil.copy2(image_path, prepared_path)
            return prepared_path

        with Image.open(image_path) as image:
            rotated = image.rotate(-rotation, expand=True)
            rotated.save(prepared_path)
        return prepared_path

    def run(self, batch_id: str, image_id: str, image_path: Path, manifest: ImageManifest) -> tuple[Path, Path, Path, Path]:
        batch_manifest_dir = self.settings.manifest_dir / batch_id
        batch_result_dir = self.settings.result_dir / batch_id
        batch_log_dir = self.settings.log_dir / batch_id
        batch_manifest_dir.mkdir(parents=True, exist_ok=True)
        batch_result_dir.mkdir(parents=True, exist_ok=True)
        batch_log_dir.mkdir(parents=True, exist_ok=True)

        prepared_path = self.prepare_image(batch_id=batch_id, image_path=image_path, manifest=manifest)
        manifest_path = batch_manifest_dir / f"{image_id}.json"
        output_csv_path = batch_result_dir / f"{image_id}.csv"
        output_meta_path = batch_result_dir / f"{image_id}.meta.json"
        output_log_path = batch_log_dir / f"{image_id}.log"

        manifest_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                self.settings.rscript_bin,
                str(self.settings.runner_script),
                "--image",
                str(prepared_path),
                "--manifest",
                str(manifest_path),
                "--output-csv",
                str(output_csv_path),
                "--output-meta",
                str(output_meta_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        output_log_path.write_text(
            "\n".join(
                [
                    f"Command: {self.settings.rscript_bin} {self.settings.runner_script}",
                    f"Prepared image: {prepared_path}",
                    "",
                    "STDOUT:",
                    result.stdout or "",
                    "",
                    "STDERR:",
                    result.stderr or "",
                ]
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        if result.returncode != 0:
            message = (result.stderr or result.stdout or "SurvdigitizeR run failed").strip()
            raise RuntimeError(message)

        return prepared_path, output_csv_path, output_meta_path, output_log_path
