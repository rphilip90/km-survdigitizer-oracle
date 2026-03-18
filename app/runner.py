from __future__ import annotations

from pathlib import Path
import json
import math
import shutil
import subprocess
import uuid

from PIL import Image, ImageDraw, ImageFont

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

REVIEW_GRID_COLOR = (102, 204, 255, 120)
REVIEW_AXIS_COLOR = (255, 165, 0, 210)
REVIEW_TICK_COLOR = (255, 0, 102, 220)
REVIEW_LABEL_BG = (255, 255, 255, 200)
REVIEW_LABEL_TEXT = (15, 23, 42, 255)


class DigitizerRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare_image(self, batch_id: str, image_path: Path, manifest: ImageManifest) -> Path:
        prepared_dir = self.settings.upload_dir / batch_id / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = prepared_dir / f"{image_path.stem}-{uuid.uuid4().hex}{image_path.suffix}"

        rotation = manifest.rotation or 0
        has_crop = all(
            value is not None
            for value in [manifest.crop_left, manifest.crop_top, manifest.crop_right, manifest.crop_bottom]
        )
        if rotation == 0 and not has_crop:
            shutil.copy2(image_path, prepared_path)
            return prepared_path

        with Image.open(image_path) as image:
            working_image = image.copy()

            if has_crop:
                width, height = working_image.size
                left = int(width * manifest.crop_left)
                top = int(height * manifest.crop_top)
                right = int(width * manifest.crop_right)
                bottom = int(height * manifest.crop_bottom)

                left = max(0, min(left, width - 1))
                top = max(0, min(top, height - 1))
                right = max(left + 1, min(right, width))
                bottom = max(top + 1, min(bottom, height))

                if right - left < 20 or bottom - top < 20:
                    raise RuntimeError("The requested crop bounds are too small to isolate a usable plot panel.")

                working_image = working_image.crop((left, top, right, bottom))

            if rotation != 0:
                working_image = working_image.rotate(-rotation, expand=True)

            working_image.save(prepared_path)
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

    def build_review_preview(
        self,
        batch_id: str,
        image_id: str,
        image_path: Path,
        manifest: ImageManifest,
    ) -> tuple[Path, Path]:
        batch_manifest_dir = self.settings.manifest_dir / batch_id
        batch_result_dir = self.settings.result_dir / batch_id
        batch_manifest_dir.mkdir(parents=True, exist_ok=True)
        batch_result_dir.mkdir(parents=True, exist_ok=True)

        prepared_path = self.prepare_image(batch_id=batch_id, image_path=image_path, manifest=manifest)
        manifest_path = batch_manifest_dir / f"{image_id}.json"
        review_preview_path = batch_result_dir / f"{image_id}.review.json"
        review_overlay_path = batch_result_dir / f"{image_id}.review.png"

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
                "--output-review-json",
                str(review_preview_path),
                "--preview-only",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            message = (result.stderr or result.stdout or "Review preview generation failed").strip()
            raise RuntimeError(message)

        if not review_preview_path.exists():
            raise RuntimeError("Review preview generation finished without writing preview metadata.")

        self.render_review_overlay(
            prepared_path=prepared_path,
            review_preview_path=review_preview_path,
            output_review_overlay_path=review_overlay_path,
            manifest=manifest,
        )

        if not review_overlay_path.exists():
            raise RuntimeError("Review preview generation finished without writing the review grid image.")

        return prepared_path, review_overlay_path

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

    def render_review_overlay(
        self,
        prepared_path: Path,
        review_preview_path: Path,
        output_review_overlay_path: Path,
        manifest: ImageManifest,
    ) -> Path:
        preview_payload = json.loads(review_preview_path.read_text(encoding="utf-8"))

        with Image.open(prepared_path) as source_image:
            canvas = source_image.convert("RGBA")

        draw = ImageDraw.Draw(canvas, "RGBA")
        font = ImageFont.load_default()
        width, height = canvas.size

        plot = preview_payload.get("plot_bounds") or {}
        left = self._clamp(float(plot.get("left", 0)), 0, width - 1)
        right = self._clamp(float(plot.get("right", width - 1)), 0, width - 1)
        top = self._clamp(float(plot.get("top", 0)), 0, height - 1)
        bottom = self._clamp(float(plot.get("bottom", height - 1)), 0, height - 1)

        if right <= left or bottom <= top:
            raise RuntimeError("Review preview metadata did not contain usable plot bounds.")

        tick_values_x = self._generate_tick_values(manifest.x_start, manifest.x_end, manifest.x_increment)
        tick_values_y = self._generate_tick_values(manifest.y_start, manifest.y_end, manifest.y_increment)

        for tick_value in tick_values_x:
            position = self._axis_position(tick_value, manifest.x_start, manifest.x_end, left, right)
            draw.line((position, top, position, bottom), fill=REVIEW_GRID_COLOR, width=1)
            draw.line((position, bottom, position, min(height - 1, bottom + 8)), fill=REVIEW_TICK_COLOR, width=2)

        for tick_value in tick_values_y:
            position = self._axis_position(tick_value, manifest.y_start, manifest.y_end, bottom, top)
            draw.line((left, position, right, position), fill=REVIEW_GRID_COLOR, width=1)
            draw.line((max(0, left - 8), position, left, position), fill=REVIEW_TICK_COLOR, width=2)

        draw.rectangle((left, top, right, bottom), outline=REVIEW_AXIS_COLOR, width=2)
        self._draw_preview_badges(draw, font, left, top, manifest)

        output_review_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_review_overlay_path)
        return output_review_overlay_path

    def _draw_preview_badges(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        left: float,
        top: float,
        manifest: ImageManifest,
    ) -> None:
        labels = [
            f"X: {self._format_value(manifest.x_start)} to {self._format_value(manifest.x_end)} by {self._format_value(manifest.x_increment)}",
            f"Y: {self._format_value(manifest.y_start)} to {self._format_value(manifest.y_end)} by {self._format_value(manifest.y_increment)}",
        ]
        anchor_x = int(max(8, left))
        anchor_y = int(max(8, top - 40))

        for index, label in enumerate(labels):
            box = draw.textbbox((0, 0), label, font=font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
            top_offset = anchor_y + index * (text_height + 8)
            padding_x = 6
            padding_y = 4
            draw.rounded_rectangle(
                (
                    anchor_x,
                    top_offset,
                    anchor_x + text_width + padding_x * 2,
                    top_offset + text_height + padding_y * 2,
                ),
                radius=8,
                fill=REVIEW_LABEL_BG,
                outline=REVIEW_AXIS_COLOR,
                width=1,
            )
            draw.text(
                (anchor_x + padding_x, top_offset + padding_y),
                label,
                font=font,
                fill=REVIEW_LABEL_TEXT,
            )

    @staticmethod
    def _generate_tick_values(start: float, end: float, increment: float) -> list[float]:
        if increment <= 0 or end <= start:
            return [start, end]

        tick_values: list[float] = []
        epsilon = max(abs(increment) * 1e-6, 1e-9)
        max_steps = max(2, int(math.ceil((end - start) / increment)) + 2)

        for step in range(max_steps):
            value = start + step * increment
            if value > end + epsilon:
                break
            if abs(value - end) <= epsilon:
                value = end
            tick_values.append(value)

        if not tick_values or abs(tick_values[-1] - end) > epsilon:
            tick_values.append(end)

        return tick_values

    @staticmethod
    def _axis_position(value: float, start: float, end: float, pixel_start: float, pixel_end: float) -> float:
        if end == start:
            return pixel_start
        ratio = (value - start) / (end - start)
        ratio = max(0.0, min(1.0, ratio))
        return pixel_start + (pixel_end - pixel_start) * ratio

    @staticmethod
    def _format_value(value: float) -> str:
        return f"{value:g}"

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))
