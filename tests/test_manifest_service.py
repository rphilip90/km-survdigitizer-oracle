from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.manifest_service import ManifestService
from app.schemas import ImageManifest
from tests.test_store import make_settings


class ManifestParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        self.settings = make_settings(self.root)
        self.service = ManifestService(self.settings)

    def test_parse_json_payload_strips_code_fences(self) -> None:
        raw = """```json
{"image_id":"1","filename":"plot.png","num_curves":"2","x_start":"0","x_end":"60","x_increment":"10","y_start":"0","y_end":"100","y_increment":"25","y_text_vertical":"true","rotation":"0","crop_hint":null,"notes":null,"llm_confidence":"0.8","review_required":"false"}
```"""

        parsed = self.service.parse_json_payload(raw)
        manifest = ImageManifest.model_validate(parsed)

        self.assertEqual(manifest.num_curves, 2)
        self.assertEqual(manifest.x_increment, 10)
        self.assertFalse(manifest.review_required)

    def test_manifest_normalizes_increment_note_to_minor_tick(self) -> None:
        manifest = ImageManifest.model_validate(
            {
                "image_id": "1",
                "filename": "plot.png",
                "num_curves": 2,
                "x_start": 0,
                "x_end": 60,
                "x_increment": "10 (minor ticks at 5 months visible)",
                "y_start": 0,
                "y_end": 100,
                "y_increment": 25,
                "y_text_vertical": True,
                "rotation": 0,
                "crop_hint": None,
                "notes": "OpenAI guess",
                "llm_confidence": 0.81,
                "review_required": False,
            }
        )

        self.assertEqual(manifest.x_increment, 5)
        self.assertTrue(manifest.review_required)
        self.assertIn("x_increment normalized", manifest.notes)

    def test_manifest_normalizes_percentage_confidence(self) -> None:
        manifest = ImageManifest.model_validate(
            {
                "image_id": "1",
                "filename": "plot.png",
                "num_curves": 2,
                "x_start": 0,
                "x_end": 60,
                "x_increment": 10,
                "y_start": 0,
                "y_end": 100,
                "y_increment": 25,
                "y_text_vertical": "true",
                "rotation": "0",
                "crop_hint": None,
                "notes": None,
                "llm_confidence": "87%",
                "review_required": "false",
            }
        )

        self.assertAlmostEqual(manifest.llm_confidence, 0.87)
        self.assertTrue(manifest.y_text_vertical)

    def test_manifest_accepts_normalized_crop_bounds(self) -> None:
        manifest = ImageManifest.model_validate(
            {
                "image_id": "1",
                "filename": "plot.png",
                "num_curves": 2,
                "x_start": 0,
                "x_end": 30,
                "x_increment": 5,
                "y_start": 0,
                "y_end": 100,
                "y_increment": 20,
                "y_text_vertical": "true",
                "rotation": "0",
                "crop_left": "0.10",
                "crop_top": "0.20",
                "crop_right": "0.90",
                "crop_bottom": "0.80",
                "exclusion_regions": [
                    {
                        "left": 0.65,
                        "top": 0.0,
                        "right": 1.0,
                        "bottom": 0.2,
                        "label": "top summary",
                    }
                ],
                "crop_hint": "exclude risk table",
                "notes": None,
                "llm_confidence": 0.7,
                "review_required": "true",
            }
        )

        self.assertAlmostEqual(manifest.crop_left, 0.10)
        self.assertAlmostEqual(manifest.crop_bottom, 0.80)
        self.assertEqual(manifest.crop_hint, "exclude risk table")
        self.assertEqual(len(manifest.exclusion_regions), 1)
        self.assertEqual(manifest.exclusion_regions[0].label, "top summary")

    def test_prompt_disables_ai_crop_coordinate_guessing(self) -> None:
        prompt = self.service.build_prompt("image-1", "plot.png")

        self.assertIn("Do not guess crop coordinates or exclusion-mask coordinates", prompt)
        self.assertIn("Always return crop_left, crop_top, crop_right, and crop_bottom as null", prompt)
        self.assertIn("Always return exclusion_polygons as []", prompt)


if __name__ == "__main__":
    unittest.main()
