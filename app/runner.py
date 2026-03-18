from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import uuid

from PIL import Image, ImageDraw

from .config import Settings
from .schemas import ImageManifest


OVERLAY_COLORS = [
    (0, 153, 255, 210),
    (255, 82, 82, 210),
    (46, 204, 113, 210),
    (255, 193, 7, 210),
    (156, 39, 176, 210),
    (255, 111, 0, 210),
]


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

    def run(self, batch_id: str, image_id: str, image_path: Path, manifest: ImageManifest) -> tuple[Path, Path, Path, Path, Path]:
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
        output_overlay_points_path = batch_result_dir / f"{image_id}.overlay.json"
        output_annotated_path = batch_result_dir / f"{image_id}.annotated.png"
        output_log_path = batch_log_dir / f"{image_id}.log"

        manifest_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                self.settings.rscript_bin,
                "--vanilla",
                str(self.settings.runner_script),
                "--image",
                str(prepared_path),
                "--manifest",
                str(manifest_path),
                "--output-csv",
                str(output_csv_path),
                "--output-meta",
                str(output_meta_path),
                "--output-overlay-json",
                str(output_overlay_points_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        output_log_path.write_text(
            "\n".join(
                [
                    f"Command: {self.settings.rscript_bin} --vanilla {self.settings.runner_script}",
                    f"Prepared image: {prepared_path}",
                    f"Overlay points JSON: {output_overlay_points_path}",
                    f"Annotated overlay image: {output_annotated_path}",
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

        if not output_csv_path.exists():
            raise RuntimeError("SurvdigitizeR finished without writing the CSV output.")

        if not output_meta_path.exists():
            raise RuntimeError("SurvdigitizeR finished without writing the metadata output.")

        if not output_overlay_points_path.exists():
            raise RuntimeError("SurvdigitizeR finished without writing the overlay point output.")

        self.render_digitized_overlay(
            prepared_path=prepared_path,
            overlay_points_path=output_overlay_points_path,
            output_annotated_path=output_annotated_path,
        )

        if not output_annotated_path.exists():
            raise RuntimeError("Digitizer finished without writing the annotated overlay image.")

        return prepared_path, output_csv_path, output_meta_path, output_log_path, output_annotated_path

    def render_digitized_overlay(
        self,
        prepared_path: Path,
        overlay_points_path: Path,
        output_annotated_path: Path,
    ) -> Path:
        overlay_payload = json.loads(overlay_points_path.read_text(encoding="utf-8"))

        with Image.open(prepared_path) as source_image:
            canvas = source_image.convert("RGBA")

        draw = ImageDraw.Draw(canvas, "RGBA")
        width, height = canvas.size

        for curve_index, curve in enumerate(overlay_payload.get("curves", [])):
            color = OVERLAY_COLORS[curve_index % len(OVERLAY_COLORS)]
            points = curve.get("points") or []
            if not points:
                continue

            line_points = [
                (
                    max(0, min(int(point["x"]), width - 1)),
                    max(0, min(int(point["y"]), height - 1)),
                )
                for point in points
            ]

            if len(line_points) >= 2:
                draw.line(line_points, fill=color, width=2)

            sample_step = max(1, len(line_points) // 40)
            marker_indices = sorted({0, len(line_points) - 1, *range(0, len(line_points), sample_step)})
            for marker_index in marker_indices:
                x_coord, y_coord = line_points[marker_index]
                draw.ellipse(
                    (x_coord - 3, y_coord - 3, x_coord + 3, y_coord + 3),
                    fill=(255, 255, 255, 170),
                    outline=color,
                    width=2,
                )

        output_annotated_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_annotated_path)
        return output_annotated_path
