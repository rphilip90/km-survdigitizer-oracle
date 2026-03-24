from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.config import Settings
from app.store import append_image_log, create_batch, create_image, get_batch, get_image, init_db, serialize_preflight, update_image


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_settings(root: Path) -> Settings:
    data_dir = root / "data"
    return Settings(
        app_title="Test App",
        app_host="127.0.0.1",
        app_port=8000,
        data_dir=data_dir,
        db_path=data_dir / "test.sqlite3",
        upload_dir=data_dir / "uploads",
        manifest_dir=data_dir / "manifests",
        result_dir=data_dir / "results",
        export_dir=data_dir / "exports",
        log_dir=data_dir / "logs",
        openai_api_key=None,
        openai_model="gpt-5-mini",
        auto_approve_threshold=0.82,
        rscript_bin="Rscript",
        runner_script=REPO_ROOT / "scripts" / "run_survdigitizer.R",
        max_workers=1,
    )


class StoreLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        self.settings = make_settings(self.root)
        self.settings.ensure_directories()
        init_db(self.settings)

    def test_append_image_log_persists_timeline(self) -> None:
        batch_id = create_batch(self.settings, "store-test", 1)
        image_path = self.settings.upload_dir / "image.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"test")
        image_id = create_image(self.settings, batch_id, "image.png", str(image_path))

        append_image_log(self.settings, image_id, "queued", "Image queued.")
        append_image_log(self.settings, image_id, "processing_manifest", "Generating manifest.")

        image = get_image(self.settings, image_id)
        self.assertEqual(len(image["processing_log"]), 2)
        self.assertEqual(image["latest_log"]["stage"], "processing_manifest")
        self.assertEqual(image["latest_log"]["message"], "Generating manifest.")

    def test_batch_persists_auto_approve_threshold(self) -> None:
        batch_id = create_batch(self.settings, "threshold-test", 1, auto_approve_threshold=0.91)

        batch = get_batch(self.settings, batch_id)

        self.assertEqual(batch["auto_approve_threshold"], 0.91)

    def test_image_persists_preflight_report(self) -> None:
        batch_id = create_batch(self.settings, "preflight-test", 1)
        image_path = self.settings.upload_dir / "image.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"test")
        image_id = create_image(self.settings, batch_id, "image.png", str(image_path))

        update_image(
            self.settings,
            image_id,
            preflight_json=serialize_preflight(
                {
                    "blocking": True,
                    "checks": [{"id": "axis-plot-size", "status": "fail"}],
                    "warnings": ["collapsed axis"],
                    "metrics": {"plot_width": 2},
                }
            ),
        )

        image = get_image(self.settings, image_id)

        self.assertTrue(image["preflight_report"]["blocking"])
        self.assertEqual(image["preflight_report"]["metrics"]["plot_width"], 2)

    def test_stale_preflight_does_not_block_effective_review_state(self) -> None:
        batch_id = create_batch(self.settings, "stale-preflight", 1)
        image_path = self.settings.upload_dir / "image.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"test")
        image_id = create_image(self.settings, batch_id, "image.png", str(image_path))

        update_image(
            self.settings,
            image_id,
            status="needs_review",
            review_required=1,
            preflight_json=serialize_preflight(
                {
                    "blocking": True,
                    "checks": [{"id": "axis-plot-size", "status": "fail"}],
                    "warnings": ["collapsed axis"],
                    "metrics": {"plot_width": 2},
                    "preflight_version": "legacy",
                    "review_reason": "needs_crop",
                    "diagnostic_category": "collapsed_plot_bounds",
                }
            ),
        )

        image = get_image(self.settings, image_id)

        self.assertTrue(image["preflight_stale"])
        self.assertFalse(image["effective_preflight_blocking"])


if __name__ == "__main__":
    unittest.main()
