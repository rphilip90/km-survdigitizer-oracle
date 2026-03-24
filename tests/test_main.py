from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app import main
from app.runner import PreflightArtifacts
from app.schemas import ImageManifest, PreflightCheck, PreflightReport
from app.store import create_batch, create_image, get_image, init_db, serialize_manifest, update_batch, update_image
from tests.test_store import make_settings


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def submit(self, fn, *args, **kwargs):  # noqa: ANN001
        self.calls.append((fn, args, kwargs))
        return None


class MainFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        self.settings = make_settings(self.root)
        self.settings.ensure_directories()
        init_db(self.settings)

        self.image_path = self.settings.upload_dir / "test.png"
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_path.write_bytes(b"fake-image")

        self.batch_id = create_batch(self.settings, "main-test", 1)
        self.image_id = create_image(self.settings, self.batch_id, "test.png", str(self.image_path))

        self.fake_executor = FakeExecutor()
        self.patches = [
            mock.patch.object(main, "settings", self.settings),
            mock.patch.object(main, "executor", self.fake_executor),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_preflight_result(
        self,
        blocking: bool = False,
        warning: str | None = None,
    ) -> tuple[dict[str, str | None], PreflightReport, dict]:
        prepared_path = self.root / "preflight-prepared.png"
        review_overlay_path = self.root / "preflight-review.png"
        preflight_log_path = self.root / "preflight.log"
        prepared_path.write_text("prepared", encoding="utf-8")
        review_overlay_path.write_text("overlay", encoding="utf-8")
        preflight_log_path.write_text("preflight", encoding="utf-8")

        checks = [
            PreflightCheck(
                id="axis-plot-size",
                stage="axis_preflight",
                severity="blocking" if blocking else "info",
                status="fail" if blocking else "pass",
                message="Detected plot bounds are collapsed." if blocking else "Detected plot bounds are large enough to continue.",
                evidence={"plot_width": 2} if blocking else {"plot_width": 180},
            )
        ]
        warnings = [warning] if warning else []
        report = PreflightReport(
            blocking=blocking,
            checks=checks,
            warnings=warnings,
            metrics={"plot_width": 2 if blocking else 180},
        )
        preview_payload = {
            "plot_bounds": {
                "left": 20,
                "right": 160,
                "top": 15,
                "bottom": 110,
            }
        }
        return {
            "prepared_path": str(prepared_path),
            "review_overlay_path": str(review_overlay_path),
            "output_log_path": str(preflight_log_path),
            "preflight_json": main.serialize_preflight(report.model_dump()),
        }, report, preview_payload

    def test_home_page_uses_product_branding(self) -> None:
        with TestClient(main.app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Survival Curve Studio", response.text)
        self.assertIn("Review-first batch extraction for Kaplan-Meier figures", response.text)
        self.assertIn("Start New Batch", response.text)

    def test_save_review_and_rerun_clears_review_flag(self) -> None:
        preflight_result = self.make_preflight_result()
        with (
            TestClient(main.app) as client,
            mock.patch.object(main, "ensure_preflight_preview", return_value=preflight_result),
        ):
            response = client.post(
                f"/images/{self.image_id}/review",
                data={
                    "num_curves": 2,
                    "llm_confidence": 0.4,
                    "x_start": 0,
                    "x_end": 60,
                    "x_increment": 10,
                    "y_start": 0,
                    "y_end": 100,
                    "y_increment": 25,
                    "y_text_vertical": "true",
                    "rotation": 0,
                    "crop_hint": "",
                    "notes": "route-test",
                    "review_required": "on",
                    "rerun_after_save": "true",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        image = get_image(self.settings, self.image_id)
        self.assertFalse(image["review_required"])
        self.assertEqual(image["status"], "queued")
        self.assertEqual(len(self.fake_executor.calls), 1)
        self.assertEqual(self.fake_executor.calls[0][0], main.process_single_image)

    def test_approve_crop_button_clears_review_and_reruns(self) -> None:
        preflight_result = self.make_preflight_result()
        approved_path = self.root / "approved-prepared.png"
        approved_path.write_text("approved", encoding="utf-8")
        with (
            TestClient(main.app) as client,
            mock.patch.object(main, "ensure_preflight_preview", return_value=preflight_result),
            mock.patch.object(main.digitizer_runner, "commit_approved_crop", return_value=approved_path),
        ):
            response = client.post(
                f"/images/{self.image_id}/review",
                data={
                    "num_curves": 2,
                    "llm_confidence": 0.4,
                    "x_start": 0,
                    "x_end": 60,
                    "x_increment": 10,
                    "y_start": 0,
                    "y_end": 100,
                    "y_increment": 25,
                    "y_text_vertical": "true",
                    "rotation": 0,
                    "crop_left": "0.12",
                    "crop_top": "0.10",
                    "crop_right": "0.94",
                    "crop_bottom": "0.88",
                    "exclusion_regions_json": json.dumps(
                        [
                            {
                                "left": 0.62,
                                "top": 0.03,
                                "right": 0.98,
                                "bottom": 0.20,
                                "label": "top summary",
                            }
                        ]
                    ),
                    "crop_hint": "exclude risk table and top summary",
                    "notes": "approve-crop",
                    "review_required": "on",
                    "approve_crop": "true",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        image = get_image(self.settings, self.image_id)
        self.assertFalse(image["review_required"])
        self.assertEqual(image["status"], "queued")
        self.assertAlmostEqual(image["manifest"]["crop_left"], 0.12)
        self.assertEqual(Path(image["prepared_path"]), approved_path)
        self.assertEqual(len(self.fake_executor.calls), 1)
        self.assertEqual(self.fake_executor.calls[0][0], main.process_prepared_image)
        stages = [entry["stage"] for entry in image["processing_log"]]
        self.assertIn("crop_approved", stages)

    def test_save_review_generates_preview_when_image_stays_in_review(self) -> None:
        preflight_result = self.make_preflight_result()

        with (
            TestClient(main.app) as client,
            mock.patch.object(
                main,
                "ensure_preflight_preview",
                return_value=preflight_result,
            ),
        ):
            response = client.post(
                f"/images/{self.image_id}/review",
                data={
                    "num_curves": 2,
                    "llm_confidence": 0.4,
                    "x_start": 0,
                    "x_end": 60,
                    "x_increment": 5,
                    "y_start": 0,
                    "y_end": 100,
                    "y_increment": 25,
                    "y_text_vertical": "true",
                    "rotation": 0,
                    "crop_left": "0.15",
                    "crop_top": "0.20",
                    "crop_right": "0.90",
                    "crop_bottom": "0.82",
                    "exclusion_regions_json": json.dumps(
                        [
                            {
                                "left": 0.62,
                                "top": 0.03,
                                "right": 0.98,
                                "bottom": 0.20,
                                "label": "top summary",
                            }
                        ]
                    ),
                    "crop_hint": "",
                    "notes": "preview-test",
                    "exclusion_polygons_json": json.dumps(
                        [
                            {
                                "label": "manual polygon 1",
                                "points": [
                                    {"x": 0.70, "y": 0.15},
                                    {"x": 0.88, "y": 0.12},
                                    {"x": 0.90, "y": 0.24},
                                ],
                            }
                        ]
                    ),
                    "review_required": "on",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        image = get_image(self.settings, self.image_id)
        self.assertTrue(image["review_required"])
        self.assertEqual(image["status"], "needs_review")
        self.assertEqual(Path(image["review_overlay_path"]), Path(preflight_result[0]["review_overlay_path"]))
        self.assertAlmostEqual(image["manifest"]["crop_left"], 0.15)
        self.assertAlmostEqual(image["manifest"]["crop_bottom"], 0.82)
        self.assertEqual(len(image["manifest"]["exclusion_regions"]), 1)
        self.assertEqual(len(image["manifest"]["exclusion_polygons"]), 1)

    def test_image_review_page_shows_zoomable_manual_crop_controls(self) -> None:
        manifest = ImageManifest(
            image_id=self.image_id,
            filename="test.png",
            num_curves=2,
            x_start=0,
            x_end=60,
            x_increment=5,
            y_start=0,
            y_end=100,
            y_increment=25,
            y_text_vertical=True,
            rotation=0,
            crop_hint="manual crop recommended: exclude risk table below x-axis",
            notes="ui-smoke",
            llm_confidence=0.52,
            review_required=True,
        )
        update_image(
            self.settings,
            self.image_id,
            manifest_json=serialize_manifest(manifest.model_dump()),
            review_required=1,
            status="needs_review",
        )

        with TestClient(main.app) as client:
            response = client.get(f"/images/{self.image_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Manual Crop Review", response.text)
        self.assertIn("Step 1: Draw Crop First", response.text)
        self.assertIn("Needs Crop", response.text)
        self.assertIn("Zoom", response.text)
        self.assertIn("manual crop recommended", response.text)
        self.assertIn("zoom-slider", response.text)

    def test_image_review_page_explains_crop_first_when_masks_exist(self) -> None:
        manifest = ImageManifest(
            image_id=self.image_id,
            filename="test.png",
            num_curves=2,
            x_start=0,
            x_end=60,
            x_increment=5,
            y_start=0,
            y_end=100,
            y_increment=25,
            y_text_vertical=True,
            rotation=0,
            exclusion_regions=[
                {
                    "left": 0.65,
                    "top": 0.04,
                    "right": 0.92,
                    "bottom": 0.18,
                    "label": "summary block",
                }
            ],
            crop_hint="manual crop recommended: exclude summary block",
            notes="workflow-smoke",
            llm_confidence=0.52,
            review_required=True,
        )
        update_image(
            self.settings,
            self.image_id,
            manifest_json=serialize_manifest(manifest.model_dump()),
            review_required=1,
            status="needs_review",
        )

        with TestClient(main.app) as client:
            response = client.get(f"/images/{self.image_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("What To Do Next", response.text)
        self.assertIn("Step 1: Draw Crop First", response.text)
        self.assertIn("Draw the crop first", response.text)
        self.assertIn("Needs Crop", response.text)
        self.assertIn("exclusion-regions-input", response.text)
        self.assertIn("Mask unlocks after the crop is set", response.text)
        self.assertIn("Draw the crop box first to unlock Mask.", response.text)

    def test_batch_page_shows_progress_panel(self) -> None:
        second_image_path = self.settings.upload_dir / "second.png"
        second_image_path.parent.mkdir(parents=True, exist_ok=True)
        second_image_path.write_bytes(b"fake-image-2")
        second_image_id = create_image(self.settings, self.batch_id, "second.png", str(second_image_path))

        update_image(self.settings, self.image_id, status="completed")
        update_image(self.settings, second_image_id, status="needs_review", review_required=1)

        with TestClient(main.app) as client:
            response = client.get(f"/batches/{self.batch_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Workflow progress", response.text)
        self.assertIn("progress-bar", response.text)
        self.assertIn("Ready results", response.text)

    def test_batch_page_shows_live_refresh_progress_when_processing(self) -> None:
        update_image(self.settings, self.image_id, status="processing_manifest")

        with TestClient(main.app) as client:
            response = client.get(f"/batches/{self.batch_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("data-live-progress", response.text)
        self.assertIn("Refreshing in", response.text)
        self.assertIn("live-refresh-fill", response.text)

    def test_image_page_shows_live_run_progress_when_processing(self) -> None:
        update_image(self.settings, self.image_id, status="processing_digitizer")

        with TestClient(main.app) as client:
            response = client.get(f"/images/{self.image_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("data-live-progress", response.text)
        self.assertIn("Extracting curves and writing artifacts", response.text)
        self.assertIn("live-stage-current", response.text)

    def test_process_single_image_logs_review_pause(self) -> None:
        manifest = ImageManifest(
            image_id=self.image_id,
            filename="test.png",
            num_curves=2,
            x_start=0,
            x_end=60,
            x_increment=10,
            y_start=0,
            y_end=100,
            y_increment=25,
            y_text_vertical=True,
            rotation=0,
            crop_hint=None,
            notes="review-pause",
            llm_confidence=0.45,
            review_required=True,
        )
        preflight_result = self.make_preflight_result()

        with (
            mock.patch.object(main.manifest_service, "generate_manifest", return_value=manifest),
            mock.patch.object(main, "ensure_preflight_preview", return_value=preflight_result),
        ):
            main.process_single_image(self.image_id, True)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "needs_review")
        self.assertEqual(Path(image["review_overlay_path"]), Path(preflight_result[0]["review_overlay_path"]))
        stages = [entry["stage"] for entry in image["processing_log"]]
        self.assertEqual(stages, ["processing_manifest", "manifest_generated", "needs_review"])

    def test_process_single_image_uses_batch_threshold(self) -> None:
        update_batch(self.settings, self.batch_id, auto_approve_threshold=0.90)
        manifest = ImageManifest(
            image_id=self.image_id,
            filename="test.png",
            num_curves=2,
            x_start=0,
            x_end=60,
            x_increment=10,
            y_start=0,
            y_end=100,
            y_increment=25,
            y_text_vertical=True,
            rotation=0,
            crop_hint=None,
            notes="threshold-check",
            llm_confidence=0.85,
            review_required=False,
        )
        preflight_result = self.make_preflight_result()

        with (
            mock.patch.object(main.manifest_service, "generate_manifest", return_value=manifest),
            mock.patch.object(main, "ensure_preflight_preview", return_value=preflight_result),
        ):
            main.process_single_image(self.image_id, True)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "needs_review")
        self.assertIn("below the batch threshold of 0.90", image["error_message"])
        self.assertEqual(Path(image["review_overlay_path"]), Path(preflight_result[0]["review_overlay_path"]))

    def test_process_single_image_pauses_on_blocking_preflight(self) -> None:
        manifest = ImageManifest(
            image_id=self.image_id,
            filename="test.png",
            num_curves=2,
            x_start=0,
            x_end=60,
            x_increment=10,
            y_start=0,
            y_end=100,
            y_increment=25,
            y_text_vertical=True,
            rotation=0,
            crop_hint=None,
            notes="blocked",
            llm_confidence=0.95,
            review_required=False,
        )
        preflight_result = self.make_preflight_result(blocking=True)

        with (
            mock.patch.object(main.manifest_service, "generate_manifest", return_value=manifest),
            mock.patch.object(main, "ensure_preflight_preview", return_value=preflight_result),
        ):
            main.process_single_image(self.image_id, True)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "needs_review")
        self.assertTrue(image["review_required"])
        self.assertIn("Detected plot bounds are collapsed", image["error_message"])

    def test_process_single_image_logs_runner_failure(self) -> None:
        manifest = ImageManifest(
            image_id=self.image_id,
            filename="test.png",
            num_curves=2,
            x_start=0,
            x_end=60,
            x_increment=10,
            y_start=0,
            y_end=100,
            y_increment=25,
            y_text_vertical=True,
            rotation=0,
            crop_hint=None,
            notes="runner-failure",
            llm_confidence=0.9,
            review_required=False,
        )
        update_image(
            self.settings,
            self.image_id,
            manifest_json=serialize_manifest(manifest.model_dump()),
            review_required=0,
            status="queued",
        )
        preflight_result = self.make_preflight_result()

        with (
            mock.patch.object(main, "ensure_preflight_preview", return_value=preflight_result),
            mock.patch.object(main.digitizer_runner, "run", side_effect=RuntimeError("digitizer blew up")),
        ):
            main.process_single_image(self.image_id, False)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "failed")
        self.assertEqual(image["error_message"], "digitizer blew up")
        stages = [entry["stage"] for entry in image["processing_log"]]
        self.assertEqual(stages, ["manifest_loaded", "processing_digitizer", "failed"])

    def test_process_single_image_logs_manifest_failure(self) -> None:
        with mock.patch.object(
            main.manifest_service,
            "generate_manifest",
            side_effect=RuntimeError("vision failed"),
        ):
            main.process_single_image(self.image_id, True)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "failed")
        self.assertEqual(image["error_message"], "vision failed")
        stages = [entry["stage"] for entry in image["processing_log"]]
        self.assertEqual(stages, ["processing_manifest", "failed"])

    def test_process_single_image_logs_completion(self) -> None:
        manifest = ImageManifest(
            image_id=self.image_id,
            filename="test.png",
            num_curves=2,
            x_start=0,
            x_end=60,
            x_increment=10,
            y_start=0,
            y_end=100,
            y_increment=25,
            y_text_vertical=True,
            rotation=0,
            crop_hint=None,
            notes="success",
            llm_confidence=0.9,
            review_required=False,
        )
        update_image(
            self.settings,
            self.image_id,
            manifest_json=serialize_manifest(manifest.model_dump()),
            review_required=0,
            status="queued",
        )
        preflight_result = self.make_preflight_result()

        prepared_path = self.root / "prepared.png"
        annotated_path = self.root / "annotated.png"
        output_csv_path = self.root / "result.csv"
        output_meta_path = self.root / "result.meta.json"
        output_log_path = self.root / "result.log"
        for path in [prepared_path, annotated_path, output_csv_path, output_log_path]:
            path.write_text("ok", encoding="utf-8")
        output_meta_path.write_text(
            json.dumps(
                {
                    "overlay": {
                        "total_points": 20,
                        "curves": [
                            {"curve": 1, "point_count": 10},
                            {"curve": 2, "point_count": 10},
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(main, "ensure_preflight_preview", return_value=preflight_result),
            mock.patch.object(
                main.digitizer_runner,
                "run",
                return_value=(prepared_path, output_csv_path, output_meta_path, output_log_path, annotated_path),
            ),
        ):
            main.process_single_image(self.image_id, False)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "completed")
        self.assertIsNone(image["error_message"])
        self.assertEqual(Path(image["annotated_path"]), annotated_path)
        self.assertEqual(Path(image["output_csv_path"]), output_csv_path)
        stages = [entry["stage"] for entry in image["processing_log"]]
        self.assertEqual(stages, ["manifest_loaded", "processing_digitizer", "completed"])

    def test_process_single_image_pauses_on_suspicious_completed_output(self) -> None:
        manifest = ImageManifest(
            image_id=self.image_id,
            filename="test.png",
            num_curves=2,
            x_start=0,
            x_end=60,
            x_increment=10,
            y_start=0,
            y_end=100,
            y_increment=25,
            y_text_vertical=True,
            rotation=0,
            crop_hint=None,
            notes="integrity-check",
            llm_confidence=0.9,
            review_required=False,
        )
        update_image(
            self.settings,
            self.image_id,
            manifest_json=serialize_manifest(manifest.model_dump()),
            review_required=0,
            status="queued",
        )
        preflight_result = self.make_preflight_result()

        prepared_path = self.root / "prepared.png"
        annotated_path = self.root / "annotated.png"
        output_csv_path = self.root / "result.csv"
        output_meta_path = self.root / "result.meta.json"
        output_log_path = self.root / "result.log"
        for path in [prepared_path, annotated_path, output_csv_path, output_log_path]:
            path.write_text("ok", encoding="utf-8")
        output_meta_path.write_text(
            json.dumps(
                {
                    "overlay": {
                        "total_points": 4,
                        "curves": [{"curve": 1, "point_count": 4}],
                    }
                }
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(main, "ensure_preflight_preview", return_value=preflight_result),
            mock.patch.object(
                main.digitizer_runner,
                "run",
                return_value=(prepared_path, output_csv_path, output_meta_path, output_log_path, annotated_path),
            ),
        ):
            main.process_single_image(self.image_id, False)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "needs_review")
        self.assertEqual(image["review_reason"], "needs_axis_review")
        self.assertEqual(image["diagnostic_category"], "suspicious_output")

    def test_export_batch_includes_annotated_overlay(self) -> None:
        annotated_path = self.root / "annotated.png"
        prepared_path = self.root / "prepared.png"
        output_csv_path = self.root / "result.csv"
        output_meta_path = self.root / "result.meta.json"
        output_log_path = self.root / "result.log"
        for path in [annotated_path, prepared_path, output_csv_path, output_meta_path, output_log_path]:
            path.write_text("artifact", encoding="utf-8")

        update_image(
            self.settings,
            self.image_id,
            annotated_path=str(annotated_path),
            prepared_path=str(prepared_path),
            output_csv_path=str(output_csv_path),
            output_meta_path=str(output_meta_path),
            output_log_path=str(output_log_path),
            status="completed",
        )

        batch = main.get_batch(self.settings, self.batch_id)
        export_path = main.build_export_archive(batch)

        import zipfile

        with zipfile.ZipFile(export_path) as archive:
            names = set(archive.namelist())

        self.assertIn(f"annotated/{self.image_id}.png", names)
        self.assertIn(f"prepared/{self.image_id}.png", names)

    def test_ensure_preflight_preview_applies_panel_crop_fallback_after_axis_failure(self) -> None:
        manifest = ImageManifest(
            image_id=self.image_id,
            filename="test.png",
            num_curves=2,
            x_start=0,
            x_end=72,
            x_increment=6,
            y_start=0,
            y_end=100,
            y_increment=20,
            y_text_vertical=True,
            rotation=0,
            crop_hint=None,
            notes=None,
            llm_confidence=0.6,
            review_required=False,
        )

        first_prepared = self.root / "first-prepared.png"
        first_review = self.root / "first-review.png"
        first_log = self.root / "first.log"
        second_prepared = self.root / "second-prepared.png"
        second_review = self.root / "second-review.png"
        second_log = self.root / "second.log"
        for path in [first_prepared, first_review, first_log, second_prepared, second_review, second_log]:
            path.write_text("artifact", encoding="utf-8")

        first_payload = {
            "plot_bounds": None,
            "metrics": {
                "prepared_width": 958,
                "prepared_height": 615,
            },
            "stage_errors": [
                {
                    "step": "Step 2: Identifying axes",
                    "message": "'x' must be an array of at least two dimensions",
                }
            ],
        }
        second_payload = {
            "plot_bounds": {"left": 78, "right": 543, "top": 24, "bottom": 381},
            "metrics": {
                "prepared_width": 580,
                "prepared_height": 426,
                "plot_width": 466,
                "plot_height": 358,
                "plot_aspect_ratio": 1.30,
                "x_axis_pixel_span": 466,
                "y_axis_pixel_span": 358,
                "cleaned_object_count": 3668,
                "adaptive_sampsize": 500,
                "detected_x_breaks": 12,
                "detected_y_breaks": 5,
                "x_pixels_increment": 55.0,
                "y_pixels_increment": 53.0,
            },
            "stage_errors": [],
        }

        side_effects = [
            PreflightArtifacts(
                prepared_path=first_prepared,
                review_overlay_path=first_review,
                preview_payload=first_payload,
                output_log_path=first_log,
            ),
            PreflightArtifacts(
                prepared_path=second_prepared,
                review_overlay_path=second_review,
                preview_payload=second_payload,
                output_log_path=second_log,
            ),
        ]

        with mock.patch.object(main.digitizer_runner, "run_preflight", side_effect=side_effects) as run_preflight:
            preview_fields, report, _payload, effective_manifest = main.ensure_preflight_preview(
                {"id": self.image_id, "batch_id": self.batch_id, "original_path": str(self.image_path)},
                manifest,
            )

        self.assertFalse(report.blocking)
        self.assertEqual(run_preflight.call_count, 2)
        self.assertAlmostEqual(effective_manifest.crop_left or 0, 0.12)
        self.assertAlmostEqual(effective_manifest.crop_bottom or 0, 0.64)
        self.assertEqual(len(effective_manifest.exclusion_regions), 1)
        self.assertEqual(Path(preview_fields["prepared_path"]), second_prepared)


if __name__ == "__main__":
    unittest.main()
