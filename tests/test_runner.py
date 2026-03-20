from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest import mock

from PIL import Image, ImageDraw

from app.runner import DigitizerRunner
from app.schemas import ImageManifest
from tests.test_store import make_settings


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        self.settings = make_settings(self.root)
        self.settings.ensure_directories()
        self.image_path = self.root / "source.png"
        Image.new("RGB", (20, 20), "white").save(self.image_path)
        self.manifest = ImageManifest(
            image_id="image-1",
            filename="source.png",
            num_curves=1,
            x_start=0,
            x_end=10,
            x_increment=1,
            y_start=0,
            y_end=1,
            y_increment=0.1,
            y_text_vertical=False,
            rotation=0,
            crop_hint=None,
            notes=None,
            llm_confidence=1.0,
            review_required=False,
        )

    def test_runner_requires_csv_output(self) -> None:
        runner = DigitizerRunner(self.settings)
        fake_result = SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch("app.runner.subprocess.run", return_value=fake_result):
            with self.assertRaisesRegex(RuntimeError, "CSV output"):
                runner.run("batch-1", "image-1", self.image_path, self.manifest)

    def test_prepare_image_applies_crop_bounds(self) -> None:
        runner = DigitizerRunner(self.settings)
        large_image_path = self.root / "large-source.png"
        Image.new("RGB", (120, 80), "white").save(large_image_path)
        cropped_manifest = self.manifest.model_copy(
            update={
                "crop_left": 0.25,
                "crop_top": 0.10,
                "crop_right": 0.75,
                "crop_bottom": 0.90,
            }
        )

        prepared_path = runner.prepare_image("batch-1", large_image_path, cropped_manifest)

        with Image.open(prepared_path) as prepared_image:
            self.assertEqual(prepared_image.size, (60, 64))

    def test_prepare_image_masks_exclusion_regions(self) -> None:
        runner = DigitizerRunner(self.settings)
        masked_image_path = self.root / "masked-source.png"
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 0, 99, 20), fill="black")
        image.save(masked_image_path)
        masked_manifest = ImageManifest.model_validate(
            {
                **self.manifest.model_dump(),
                "exclusion_regions": [
                    {
                        "left": 0.6,
                        "top": 0.0,
                        "right": 1.0,
                        "bottom": 0.2,
                        "label": "top summary",
                    }
                ]
            }
        )

        prepared_path = runner.prepare_image("batch-1", masked_image_path, masked_manifest)

        with Image.open(prepared_path) as prepared_image:
            self.assertNotEqual(prepared_image.getpixel((80, 10)), (0, 0, 0))

    def test_runner_requires_meta_output(self) -> None:
        runner = DigitizerRunner(self.settings)

        def fake_prepare_image(batch_id: str, image_path: Path, manifest: ImageManifest) -> Path:  # noqa: ARG001
            prepared_dir = self.settings.upload_dir / batch_id / "prepared"
            prepared_dir.mkdir(parents=True, exist_ok=True)
            prepared_path = prepared_dir / image_path.name
            prepared_path.write_bytes(image_path.read_bytes())
            return prepared_path

        def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
            result_dir = self.settings.result_dir / "batch-1"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "image-1.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with (
            mock.patch.object(runner, "prepare_image", side_effect=fake_prepare_image),
            mock.patch("app.runner.subprocess.run", side_effect=fake_run),
        ):
            with self.assertRaisesRegex(RuntimeError, "metadata output"):
                runner.run("batch-1", "image-1", self.image_path, self.manifest)

    def test_render_digitized_overlay_writes_png(self) -> None:
        runner = DigitizerRunner(self.settings)
        prepared_path = self.root / "prepared.png"
        output_annotated_path = self.root / "annotated.png"
        overlay_points_path = self.root / "overlay.json"
        Image.new("RGB", (40, 30), "white").save(prepared_path)
        overlay_points_path.write_text(
            json.dumps(
                {
                    "curves": [
                        {
                            "curve": 1,
                            "points": [
                                {"x": 5, "y": 5},
                                {"x": 10, "y": 10},
                                {"x": 20, "y": 12},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        runner.render_digitized_overlay(prepared_path, overlay_points_path, output_annotated_path)

        self.assertTrue(output_annotated_path.exists())

    def test_render_review_overlay_writes_png(self) -> None:
        runner = DigitizerRunner(self.settings)
        prepared_path = self.root / "review-prepared.png"
        output_review_path = self.root / "review-overlay.png"
        review_preview_path = self.root / "review.json"
        Image.new("RGB", (120, 80), "white").save(prepared_path)
        review_preview_path.write_text(
            json.dumps(
                {
                    "width": 120,
                    "height": 80,
                    "plot_bounds": {
                        "left": 20,
                        "right": 100,
                        "top": 10,
                        "bottom": 60,
                    },
                }
            ),
            encoding="utf-8",
        )

        runner.render_review_overlay(prepared_path, review_preview_path, output_review_path, self.manifest)

        self.assertTrue(output_review_path.exists())

    def test_run_preflight_writes_preview_artifacts(self) -> None:
        runner = DigitizerRunner(self.settings)

        def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
            result_dir = self.settings.result_dir / "batch-1"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "image-1.preflight.json").write_text(
                json.dumps(
                    {
                        "width": 20,
                        "height": 20,
                        "source_width": 20,
                        "source_height": 20,
                        "plot_bounds": {
                            "left": 3,
                            "right": 18,
                            "top": 2,
                            "bottom": 17,
                        },
                        "metrics": {
                            "prepared_width": 20,
                            "prepared_height": 20,
                            "plot_width": 15,
                            "plot_height": 15,
                        },
                        "stage_errors": [],
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch("app.runner.subprocess.run", side_effect=fake_run):
            artifacts = runner.run_preflight("batch-1", "image-1", self.image_path, self.manifest)

        self.assertTrue(artifacts.review_overlay_path.exists())
        self.assertEqual(artifacts.preview_payload["metrics"]["plot_width"], 15)

    def test_commit_approved_crop_uses_plot_bounds(self) -> None:
        runner = DigitizerRunner(self.settings)
        prepared_path = self.root / "commit-source.png"
        Image.new("RGB", (200, 160), "white").save(prepared_path)
        preview_payload = {
            "plot_bounds": {
                "left": 40,
                "right": 160,
                "top": 20,
                "bottom": 120,
            }
        }

        approved_path = runner.commit_approved_crop("batch-1", "image-1", prepared_path, preview_payload)

        self.assertTrue(approved_path.exists())
        with Image.open(approved_path) as approved_image:
            self.assertLess(approved_image.size[0], 200)
            self.assertLess(approved_image.size[1], 160)
            self.assertGreaterEqual(approved_image.size[0], 120)
            self.assertGreaterEqual(approved_image.size[1], 100)

    def test_runner_requires_overlay_output(self) -> None:
        runner = DigitizerRunner(self.settings)

        def fake_prepare_image(batch_id: str, image_path: Path, manifest: ImageManifest) -> Path:  # noqa: ARG001
            prepared_dir = self.settings.upload_dir / batch_id / "prepared"
            prepared_dir.mkdir(parents=True, exist_ok=True)
            prepared_path = prepared_dir / image_path.name
            prepared_path.write_bytes(image_path.read_bytes())
            return prepared_path

        def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
            result_dir = self.settings.result_dir / "batch-1"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "image-1.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            (result_dir / "image-1.meta.json").write_text("{\"rows\": 1}", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with (
            mock.patch.object(runner, "prepare_image", side_effect=fake_prepare_image),
            mock.patch("app.runner.subprocess.run", side_effect=fake_run),
        ):
            with self.assertRaisesRegex(RuntimeError, "overlay point output"):
                runner.run("batch-1", "image-1", self.image_path, self.manifest)


if __name__ == "__main__":
    unittest.main()
