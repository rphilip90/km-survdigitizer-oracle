from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app import main
from app.schemas import ImageManifest
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

    def test_save_review_and_rerun_clears_review_flag(self) -> None:
        with TestClient(main.app) as client:
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

        with mock.patch.object(main.manifest_service, "generate_manifest", return_value=manifest):
            main.process_single_image(self.image_id, True)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "needs_review")
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

        with mock.patch.object(main.manifest_service, "generate_manifest", return_value=manifest):
            main.process_single_image(self.image_id, True)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "needs_review")
        self.assertIn("below the batch threshold of 0.90", image["error_message"])

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

        with mock.patch.object(main.digitizer_runner, "run", side_effect=RuntimeError("digitizer blew up")):
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

        prepared_path = self.root / "prepared.png"
        annotated_path = self.root / "annotated.png"
        output_csv_path = self.root / "result.csv"
        output_meta_path = self.root / "result.meta.json"
        output_log_path = self.root / "result.log"
        for path in [prepared_path, annotated_path, output_csv_path, output_meta_path, output_log_path]:
            path.write_text("ok", encoding="utf-8")

        with mock.patch.object(
            main.digitizer_runner,
            "run",
            return_value=(prepared_path, output_csv_path, output_meta_path, output_log_path, annotated_path),
        ):
            main.process_single_image(self.image_id, False)

        image = get_image(self.settings, self.image_id)
        self.assertEqual(image["status"], "completed")
        self.assertIsNone(image["error_message"])
        self.assertEqual(Path(image["annotated_path"]), annotated_path)
        self.assertEqual(Path(image["output_csv_path"]), output_csv_path)
        stages = [entry["stage"] for entry in image["processing_log"]]
        self.assertEqual(stages, ["manifest_loaded", "processing_digitizer", "completed"])

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


if __name__ == "__main__":
    unittest.main()
