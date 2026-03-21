from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from app.preflight import build_preflight_report
from app.schemas import ImageManifest


class PreflightReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        self.prepared_path = self.root / "prepared.png"
        Image.new("RGB", (400, 300), "white").save(self.prepared_path)
        self.manifest = ImageManifest(
            image_id="image-1",
            filename="prepared.png",
            num_curves=2,
            x_start=0,
            x_end=60,
            x_increment=5,
            y_start=0,
            y_end=100,
            y_increment=10,
            y_text_vertical=True,
            rotation=0,
            crop_left=0.1,
            crop_top=0.1,
            crop_right=0.9,
            crop_bottom=0.9,
            exclusion_regions=[],
            crop_hint=None,
            notes=None,
            llm_confidence=0.8,
            review_required=False,
        )
        self.base_payload = {
            "plot_bounds": {"left": 40, "right": 360, "top": 30, "bottom": 250},
            "metrics": {
                "prepared_width": 400,
                "prepared_height": 300,
                "plot_width": 320,
                "plot_height": 220,
                "plot_aspect_ratio": 320 / 220,
                "x_axis_pixel_span": 320,
                "y_axis_pixel_span": 220,
                "cleaned_object_count": 900,
                "adaptive_sampsize": 500,
                "detected_x_breaks": 13,
                "detected_y_breaks": 11,
            },
            "stage_errors": [],
        }

    def test_blocks_when_exclusion_regions_cover_axis_zone(self) -> None:
        manifest = ImageManifest.model_validate(
            {
                **self.manifest.model_dump(),
                "exclusion_regions": [
                    {
                        "left": 0.1,
                        "top": 0.78,
                        "right": 0.85,
                        "bottom": 0.9,
                        "label": "risk table too high",
                    }
                ],
            }
        )

        report = build_preflight_report(manifest, self.prepared_path, self.base_payload)

        self.assertTrue(report.blocking)
        self.assertTrue(any(check.id == "manifest-exclusions" and check.status == "fail" for check in report.checks))

    def test_blocks_when_exclusion_polygon_covers_axis_zone(self) -> None:
        manifest = ImageManifest.model_validate(
            {
                **self.manifest.model_dump(),
                "exclusion_polygons": [
                    {
                        "label": "risk table polygon",
                        "points": [
                            {"x": 0.12, "y": 0.76},
                            {"x": 0.84, "y": 0.78},
                            {"x": 0.84, "y": 0.89},
                            {"x": 0.12, "y": 0.89},
                        ],
                    }
                ],
            }
        )

        report = build_preflight_report(manifest, self.prepared_path, self.base_payload)

        self.assertTrue(report.blocking)
        self.assertTrue(any(check.id == "manifest-exclusions" and check.status == "fail" for check in report.checks))

    def test_requires_crop_when_large_masks_exist_without_crop(self) -> None:
        manifest = ImageManifest.model_validate(
            {
                **self.manifest.model_dump(),
                "crop_left": None,
                "crop_top": None,
                "crop_right": None,
                "crop_bottom": None,
                "exclusion_polygons": [
                    {
                        "label": "bottom risk table",
                        "points": [
                            {"x": 0.05, "y": 0.6},
                            {"x": 0.95, "y": 0.6},
                            {"x": 0.9, "y": 0.98},
                            {"x": 0.05, "y": 0.98},
                        ],
                    }
                ],
            }
        )

        report = build_preflight_report(manifest, self.prepared_path, self.base_payload)

        self.assertTrue(report.blocking)
        self.assertTrue(any(check.id == "manifest-crop-required" and check.status == "fail" for check in report.checks))

    def test_blocks_on_collapsed_axis_geometry(self) -> None:
        payload = {
            **self.base_payload,
            "metrics": {
                **self.base_payload["metrics"],
                "plot_width": 2,
                "plot_height": 220,
                "plot_aspect_ratio": 2 / 220,
                "x_axis_pixel_span": 2,
            },
        }

        report = build_preflight_report(self.manifest, self.prepared_path, payload)

        self.assertTrue(report.blocking)
        self.assertTrue(any(check.stage == "axis_preflight" and check.status == "fail" for check in report.checks))

    def test_warns_when_cleaned_object_count_is_sparse(self) -> None:
        payload = {
            **self.base_payload,
            "metrics": {
                **self.base_payload["metrics"],
                "cleaned_object_count": 448,
                "adaptive_sampsize": 448,
            },
        }

        report = build_preflight_report(self.manifest, self.prepared_path, payload)

        self.assertFalse(report.blocking)
        self.assertTrue(any(check.stage == "cluster_preflight" and check.status == "warn" for check in report.checks))

    def test_blocks_when_range_detection_mismatches_manifest(self) -> None:
        payload = {
            **self.base_payload,
            "metrics": {
                **self.base_payload["metrics"],
                "detected_x_breaks": 3,
                "detected_y_breaks": 2,
            },
        }

        report = build_preflight_report(self.manifest, self.prepared_path, payload)

        self.assertTrue(report.blocking)
        self.assertTrue(any(check.stage == "range_preflight" and check.status == "fail" for check in report.checks))


if __name__ == "__main__":
    unittest.main()
