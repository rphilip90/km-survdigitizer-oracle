from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

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


if __name__ == "__main__":
    unittest.main()
