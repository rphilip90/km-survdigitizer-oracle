from __future__ import annotations

from pathlib import Path
import math

from PIL import Image

from .schemas import ImageManifest, PreflightCheck, PreflightReport


MIN_PREPARED_WIDTH = 160
MIN_PREPARED_HEIGHT = 120
MIN_PLOT_WIDTH = 40
MIN_PLOT_HEIGHT = 40
MIN_AXIS_SPAN = 40
MAX_REASONABLE_TICKS = 200
WARN_MASK_COVERAGE = 0.25
FAIL_MASK_COVERAGE = 0.45
WARN_OBJECT_COUNT = 500
FAIL_OBJECT_COUNT = 60


def build_preflight_report(
    manifest: ImageManifest,
    prepared_path: Path,
    preview_payload: dict,
) -> PreflightReport:
    metrics = dict(preview_payload.get("metrics") or {})
    stage_errors = list(preview_payload.get("stage_errors") or [])

    if not metrics.get("prepared_width") or not metrics.get("prepared_height"):
        with Image.open(prepared_path) as image:
            metrics.setdefault("prepared_width", image.width)
            metrics.setdefault("prepared_height", image.height)

    metrics["expected_x_ticks"] = _expected_tick_count(manifest.x_start, manifest.x_end, manifest.x_increment)
    metrics["expected_y_ticks"] = _expected_tick_count(manifest.y_start, manifest.y_end, manifest.y_increment)
    metrics["mask_coverage_ratio"] = round(_estimate_mask_coverage_ratio(manifest), 4)
    metrics["exclusion_region_count"] = len(manifest.exclusion_regions)
    metrics["crop_area_ratio"] = round(_estimate_crop_area_ratio(manifest), 4)

    checks: list[PreflightCheck] = []
    checks.extend(_build_manifest_checks(manifest, metrics))
    checks.extend(_build_prepared_image_checks(manifest, metrics, preview_payload))
    checks.extend(_build_axis_checks(metrics, stage_errors))
    checks.extend(_build_cluster_checks(metrics, stage_errors))
    checks.extend(_build_range_checks(metrics, stage_errors))

    warnings = [check.message for check in checks if check.status != "pass"]
    blocking = any(check.status == "fail" and check.severity == "blocking" for check in checks)
    return PreflightReport(
        blocking=blocking,
        checks=checks,
        warnings=warnings,
        metrics=metrics,
    )


def _build_manifest_checks(manifest: ImageManifest, metrics: dict) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []

    checks.append(
        _check(
            "manifest-ranges",
            "manifest_preflight",
            "pass",
            (
                f"Axis ranges look valid: X {manifest.x_start:g} to {manifest.x_end:g} by {manifest.x_increment:g}; "
                f"Y {manifest.y_start:g} to {manifest.y_end:g} by {manifest.y_increment:g}."
            ),
            evidence={
                "expected_x_ticks": metrics["expected_x_ticks"],
                "expected_y_ticks": metrics["expected_y_ticks"],
                "num_curves": manifest.num_curves,
            },
        )
    )

    if metrics["expected_x_ticks"] > MAX_REASONABLE_TICKS or metrics["expected_y_ticks"] > MAX_REASONABLE_TICKS:
        checks.append(
            _check(
                "manifest-dense-ticks",
                "manifest_preflight",
                "warn",
                "The manifest implies a very dense tick grid. Confirm that the increment is the visible minor-tick spacing.",
                evidence={
                    "expected_x_ticks": metrics["expected_x_ticks"],
                    "expected_y_ticks": metrics["expected_y_ticks"],
                },
            )
        )
    else:
        checks.append(
            _check(
                "manifest-ticks",
                "manifest_preflight",
                "pass",
                "Tick counts derived from the manifest are within a reasonable range.",
                evidence={
                    "expected_x_ticks": metrics["expected_x_ticks"],
                    "expected_y_ticks": metrics["expected_y_ticks"],
                },
            )
        )

    crop_values = [manifest.crop_left, manifest.crop_top, manifest.crop_right, manifest.crop_bottom]
    if all(value is None for value in crop_values):
        checks.append(
            _check(
                "manifest-crop",
                "manifest_preflight",
                "pass",
                "No crop is applied; the full image will be passed into preflight.",
            )
        )
    else:
        checks.append(
            _check(
                "manifest-crop",
                "manifest_preflight",
                "pass",
                "A normalized plot crop is defined.",
                evidence={
                    "crop_left": manifest.crop_left,
                    "crop_top": manifest.crop_top,
                    "crop_right": manifest.crop_right,
                    "crop_bottom": manifest.crop_bottom,
                    "crop_area_ratio": metrics["crop_area_ratio"],
                },
            )
        )

    overlap = _axis_guard_overlap(manifest)
    if overlap["blocking"]:
        checks.append(
            _check(
                "manifest-exclusions",
                "manifest_preflight",
                "fail",
                "One or more exclusion regions overlap the likely x-axis or y-axis guard bands. Adjust the masks before rerun.",
                evidence=overlap,
            )
        )
    elif overlap["warning"]:
        checks.append(
            _check(
                "manifest-exclusions",
                "manifest_preflight",
                "warn",
                "Exclusion regions come close to the likely axis guard bands. Confirm that the full axes and labels are still visible.",
                evidence=overlap,
            )
        )
    else:
        checks.append(
            _check(
                "manifest-exclusions",
                "manifest_preflight",
                "pass",
                "Exclusion regions stay clear of the likely axis guard bands.",
                evidence={"mask_coverage_ratio": metrics["mask_coverage_ratio"]},
            )
        )

    return checks


def _build_prepared_image_checks(manifest: ImageManifest, metrics: dict, preview_payload: dict) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    width = int(metrics.get("prepared_width") or 0)
    height = int(metrics.get("prepared_height") or 0)

    if width < MIN_PREPARED_WIDTH or height < MIN_PREPARED_HEIGHT:
        checks.append(
            _check(
                "prepared-size",
                "prepared_image_preflight",
                "fail",
                "The prepared image is too small for reliable axis and curve detection.",
                evidence={"prepared_width": width, "prepared_height": height},
            )
        )
    else:
        checks.append(
            _check(
                "prepared-size",
                "prepared_image_preflight",
                "pass",
                "Prepared image dimensions are large enough for digitization.",
                evidence={"prepared_width": width, "prepared_height": height},
            )
        )

    mask_coverage = float(metrics.get("mask_coverage_ratio") or 0.0)
    if mask_coverage >= FAIL_MASK_COVERAGE:
        checks.append(
            _check(
                "prepared-mask-coverage",
                "prepared_image_preflight",
                "fail",
                "The exclusion masks blank out too much of the prepared image. Reduce the masked area or widen the crop.",
                evidence={"mask_coverage_ratio": mask_coverage},
            )
        )
    elif mask_coverage >= WARN_MASK_COVERAGE:
        checks.append(
            _check(
                "prepared-mask-coverage",
                "prepared_image_preflight",
                "warn",
                "The exclusion masks cover a large share of the prepared image. Confirm that the plot panel still has enough visible structure.",
                evidence={"mask_coverage_ratio": mask_coverage},
            )
        )
    else:
        checks.append(
            _check(
                "prepared-mask-coverage",
                "prepared_image_preflight",
                "pass",
                "Mask coverage is within a reasonable range.",
                evidence={"mask_coverage_ratio": mask_coverage},
            )
        )

    plot = preview_payload.get("plot_bounds") or {}
    if plot:
        left = float(plot.get("left", 0))
        right = float(plot.get("right", width - 1))
        top = float(plot.get("top", 0))
        bottom = float(plot.get("bottom", height - 1))
        edge_contacts = {
            "left_margin": round(left, 2),
            "top_margin": round(top, 2),
            "right_margin": round(max(0.0, width - right - 1), 2),
            "bottom_margin": round(max(0.0, height - bottom - 1), 2),
        }
        touching_edges = sum(1 for value in edge_contacts.values() if value <= 1.5)
        if touching_edges >= 2:
            checks.append(
                _check(
                    "prepared-axis-contact",
                    "prepared_image_preflight",
                    "fail",
                    "Detected plot bounds are pressed against multiple image edges. The crop or masks likely cut into the axis envelope.",
                    evidence=edge_contacts,
                )
            )
        elif touching_edges == 1 and (
            manifest.crop_left is not None or manifest.exclusion_regions
        ):
            checks.append(
                _check(
                    "prepared-axis-contact",
                    "prepared_image_preflight",
                    "warn",
                    "Detected plot bounds touch the prepared image edge. Double-check that the crop still preserves the full axes and tick labels.",
                    evidence=edge_contacts,
                )
            )
        else:
            checks.append(
                _check(
                    "prepared-axis-contact",
                    "prepared_image_preflight",
                    "pass",
                    "Detected plot bounds sit safely inside the prepared image.",
                    evidence=edge_contacts,
                )
            )

    return checks


def _build_axis_checks(metrics: dict, stage_errors: list[dict]) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    axis_error = _find_stage_error(stage_errors, "Step 2: Identifying axes")
    if axis_error:
        return [
            _check(
                "axis-detection",
                "axis_preflight",
                "fail",
                f"Axis identification failed before digitization: {axis_error['message']}",
                evidence=axis_error,
            )
        ]

    plot_width = int(metrics.get("plot_width") or 0)
    plot_height = int(metrics.get("plot_height") or 0)
    x_span = int(metrics.get("x_axis_pixel_span") or 0)
    y_span = int(metrics.get("y_axis_pixel_span") or 0)
    aspect_ratio = float(metrics.get("plot_aspect_ratio") or 0.0)

    if plot_width < MIN_PLOT_WIDTH or plot_height < MIN_PLOT_HEIGHT:
        checks.append(
            _check(
                "axis-plot-size",
                "axis_preflight",
                "fail",
                "Detected plot bounds are collapsed. Review the crop and exclusion regions before rerun.",
                evidence={
                    "plot_width": plot_width,
                    "plot_height": plot_height,
                },
            )
        )
    else:
        checks.append(
            _check(
                "axis-plot-size",
                "axis_preflight",
                "pass",
                "Detected plot bounds are large enough to continue.",
                evidence={"plot_width": plot_width, "plot_height": plot_height},
            )
        )

    if aspect_ratio <= 0 or aspect_ratio < 0.2 or aspect_ratio > 8:
        checks.append(
            _check(
                "axis-aspect-ratio",
                "axis_preflight",
                "fail",
                "The detected plot aspect ratio is implausible for a usable KM panel.",
                evidence={"plot_aspect_ratio": round(aspect_ratio, 4)},
            )
        )
    else:
        checks.append(
            _check(
                "axis-aspect-ratio",
                "axis_preflight",
                "pass",
                "Detected plot aspect ratio is plausible.",
                evidence={"plot_aspect_ratio": round(aspect_ratio, 4)},
            )
        )

    if x_span < MIN_AXIS_SPAN or y_span < MIN_AXIS_SPAN:
        checks.append(
            _check(
                "axis-spans",
                "axis_preflight",
                "fail",
                "The detected x-axis or y-axis span is too short. Axis detection likely locked onto the wrong region.",
                evidence={"x_axis_pixel_span": x_span, "y_axis_pixel_span": y_span},
            )
        )
    else:
        checks.append(
            _check(
                "axis-spans",
                "axis_preflight",
                "pass",
                "Detected axis spans look plausible.",
                evidence={"x_axis_pixel_span": x_span, "y_axis_pixel_span": y_span},
            )
        )

    return checks


def _build_cluster_checks(metrics: dict, stage_errors: list[dict]) -> list[PreflightCheck]:
    fig_clean_error = _find_stage_error(stage_errors, "Step 3: Cleaning figure")
    if fig_clean_error:
        if str(fig_clean_error.get("message", "")).startswith("Skipped because axis identification failed"):
            return [
                _check(
                    "cluster-clean-figure",
                    "cluster_preflight",
                    "warn",
                    "Cluster pre-flight was skipped because axis detection already failed earlier in the pipeline.",
                    evidence=fig_clean_error,
                )
            ]
        return [
            _check(
                "cluster-clean-figure",
                "cluster_preflight",
                "fail",
                f"Figure cleaning failed before clustering: {fig_clean_error['message']}",
                evidence=fig_clean_error,
            )
        ]

    object_count = int(metrics.get("cleaned_object_count") or 0)
    sampsize = int(metrics.get("adaptive_sampsize") or 0)
    if object_count <= 0:
        return [
            _check(
                "cluster-object-count",
                "cluster_preflight",
                "fail",
                "No cleaned plot objects were available for color clustering.",
                evidence={"cleaned_object_count": object_count},
            )
        ]

    if object_count < FAIL_OBJECT_COUNT:
        return [
            _check(
                "cluster-object-count",
                "cluster_preflight",
                "fail",
                "Too few cleaned plot objects remain after preprocessing. Review the crop and exclusion masks before rerun.",
                evidence={
                    "cleaned_object_count": object_count,
                    "adaptive_sampsize": sampsize,
                },
            )
        ]

    if object_count < WARN_OBJECT_COUNT:
        return [
            _check(
                "cluster-object-count",
                "cluster_preflight",
                "warn",
                "The cleaned plot is sparse. The wrapper will clamp CLARA sampsize automatically, but the panel still needs manual verification.",
                evidence={
                    "cleaned_object_count": object_count,
                    "adaptive_sampsize": sampsize,
                },
            )
        ]

    return [
        _check(
            "cluster-object-count",
            "cluster_preflight",
            "pass",
            "Cleaned object count is sufficient for color clustering.",
            evidence={
                "cleaned_object_count": object_count,
                "adaptive_sampsize": sampsize,
            },
        )
    ]


def _build_range_checks(metrics: dict, stage_errors: list[dict]) -> list[PreflightCheck]:
    range_error = _find_stage_error(stage_errors, "Step 7: Detecting ranges")
    if range_error:
        if str(range_error.get("message", "")).startswith("Skipped because axis identification failed"):
            return [
                _check(
                    "range-detection",
                    "range_preflight",
                    "warn",
                    "Range pre-flight was skipped because axis detection already failed earlier in the pipeline.",
                    evidence=range_error,
                )
            ]
        return [
            _check(
                "range-detection",
                "range_preflight",
                "fail",
                f"Range detection failed before digitization: {range_error['message']}",
                evidence=range_error,
            )
        ]

    expected_x = int(metrics.get("expected_x_ticks") or 0)
    expected_y = int(metrics.get("expected_y_ticks") or 0)
    detected_x = int(metrics.get("detected_x_breaks") or 0)
    detected_y = int(metrics.get("detected_y_breaks") or 0)

    x_ratio = _safe_ratio(detected_x, expected_x)
    y_ratio = _safe_ratio(detected_y, expected_y)

    evidence = {
        "expected_x_ticks": expected_x,
        "detected_x_breaks": detected_x,
        "expected_y_ticks": expected_y,
        "detected_y_breaks": detected_y,
        "x_break_ratio": round(x_ratio, 4) if x_ratio is not None else None,
        "y_break_ratio": round(y_ratio, 4) if y_ratio is not None else None,
    }

    if detected_x == 0 or detected_y == 0:
        return [
            _check(
                "range-break-match",
                "range_preflight",
                "fail",
                "No usable tick breaks were detected on one or both axes. Review the crop, masks, and manifest tick spacing.",
                evidence=evidence,
            )
        ]

    if (x_ratio is not None and x_ratio < 0.5) or (y_ratio is not None and y_ratio < 0.5):
        return [
            _check(
                "range-break-match",
                "range_preflight",
                "fail",
                "Detected tick geometry is too far from the manifest expectation. Confirm the axis limits and every visible tick increment.",
                evidence=evidence,
            )
        ]

    if (x_ratio is not None and x_ratio < 0.8) or (y_ratio is not None and y_ratio < 0.8):
        return [
            _check(
                "range-break-match",
                "range_preflight",
                "warn",
                "Detected tick geometry is incomplete relative to the manifest expectation. The image can proceed only after manual review.",
                evidence=evidence,
            )
        ]

    return [
        _check(
            "range-break-match",
            "range_preflight",
            "pass",
            "Detected tick geometry agrees with the current manifest.",
            evidence=evidence,
        )
    ]


def _check(
    check_id: str,
    stage: str,
    status: str,
    message: str,
    evidence: dict | None = None,
) -> PreflightCheck:
    severity = {
        "pass": "info",
        "warn": "warning",
        "fail": "blocking",
    }[status]
    return PreflightCheck(
        id=check_id,
        stage=stage,
        severity=severity,
        status=status,
        message=message,
        evidence=evidence or None,
    )


def _expected_tick_count(start: float, end: float, increment: float) -> int:
    if increment <= 0 or end <= start:
        return 0
    return max(2, int(math.floor(((end - start) / increment) + 1e-6)) + 1)


def _estimate_crop_area_ratio(manifest: ImageManifest) -> float:
    if manifest.crop_left is None:
        return 1.0
    return max(0.0, (manifest.crop_right - manifest.crop_left) * (manifest.crop_bottom - manifest.crop_top))


def _estimate_mask_coverage_ratio(manifest: ImageManifest) -> float:
    crop_left = manifest.crop_left if manifest.crop_left is not None else 0.0
    crop_top = manifest.crop_top if manifest.crop_top is not None else 0.0
    crop_right = manifest.crop_right if manifest.crop_right is not None else 1.0
    crop_bottom = manifest.crop_bottom if manifest.crop_bottom is not None else 1.0
    crop_area = max(1e-9, (crop_right - crop_left) * (crop_bottom - crop_top))

    masked_area = 0.0
    for region in manifest.exclusion_regions:
        left = max(crop_left, region.left)
        top = max(crop_top, region.top)
        right = min(crop_right, region.right)
        bottom = min(crop_bottom, region.bottom)
        if right <= left or bottom <= top:
            continue
        masked_area += (right - left) * (bottom - top)
    return min(1.0, masked_area / crop_area)


def _axis_guard_overlap(manifest: ImageManifest) -> dict[str, float | bool]:
    if not manifest.exclusion_regions:
        return {"blocking": False, "warning": False, "left_overlap": 0.0, "bottom_overlap": 0.0}

    if manifest.rotation not in (None, 0):
        return {"blocking": False, "warning": True, "left_overlap": 0.0, "bottom_overlap": 0.0}

    crop_left = manifest.crop_left if manifest.crop_left is not None else 0.0
    crop_top = manifest.crop_top if manifest.crop_top is not None else 0.0
    crop_right = manifest.crop_right if manifest.crop_right is not None else 1.0
    crop_bottom = manifest.crop_bottom if manifest.crop_bottom is not None else 1.0
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    if crop_width <= 0 or crop_height <= 0:
        return {"blocking": True, "warning": False, "left_overlap": 1.0, "bottom_overlap": 1.0}

    left_overlap = 0.0
    bottom_overlap = 0.0
    left_guard = 0.14
    bottom_guard = 0.18

    for region in manifest.exclusion_regions:
        left = max(crop_left, region.left)
        top = max(crop_top, region.top)
        right = min(crop_right, region.right)
        bottom = min(crop_bottom, region.bottom)
        if right <= left or bottom <= top:
            continue

        rel_left = (left - crop_left) / crop_width
        rel_right = (right - crop_left) / crop_width
        rel_top = (top - crop_top) / crop_height
        rel_bottom = (bottom - crop_top) / crop_height

        left_overlap = max(left_overlap, max(0.0, min(rel_right, left_guard) - rel_left))
        bottom_overlap = max(bottom_overlap, max(0.0, rel_bottom - max(rel_top, 1 - bottom_guard)))

    blocking = left_overlap >= 0.05 or bottom_overlap >= 0.05
    warning = not blocking and (left_overlap > 0 or bottom_overlap > 0)
    return {
        "blocking": blocking,
        "warning": warning,
        "left_overlap": round(left_overlap, 4),
        "bottom_overlap": round(bottom_overlap, 4),
    }


def _find_stage_error(stage_errors: list[dict], step_name: str) -> dict | None:
    for error in stage_errors:
        if error.get("step") == step_name:
            return error
    return None


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
