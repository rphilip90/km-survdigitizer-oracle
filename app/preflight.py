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
    metrics["exclusion_polygon_count"] = len(manifest.exclusion_polygons)
    metrics["crop_area_ratio"] = round(_estimate_crop_area_ratio(manifest), 4)

    checks: list[PreflightCheck] = []
    checks.extend(_build_manifest_checks(manifest, metrics, preview_payload))
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


def _build_manifest_checks(manifest: ImageManifest, metrics: dict, preview_payload: dict) -> list[PreflightCheck]:
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
    has_crop = all(value is not None for value in crop_values)
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

    if not has_crop and (manifest.exclusion_regions or manifest.exclusion_polygons):
        checks.append(
            _check(
                "manifest-crop-required",
                "manifest_preflight",
                "fail",
                "Masks are present but no crop rectangle is set. Draw a crop around the plot panel before rerun.",
                evidence={
                    "mask_coverage_ratio": metrics["mask_coverage_ratio"],
                    "exclusion_polygons": metrics["exclusion_polygon_count"],
                    "exclusion_regions": metrics["exclusion_region_count"],
                },
            )
        )

    overlap = _axis_guard_overlap(manifest, preview_payload, metrics)
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
                evidence={
                    "mask_coverage_ratio": metrics["mask_coverage_ratio"],
                    "exclusion_polygons": metrics["exclusion_polygon_count"],
                },
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
    x_pixels_increment = _safe_float(metrics.get("x_pixels_increment"))
    y_pixels_increment = _safe_float(metrics.get("y_pixels_increment"))
    geometric_fallback_ready = (
        x_pixels_increment is not None and x_pixels_increment > 0
        and y_pixels_increment is not None and y_pixels_increment > 0
    )

    x_ratio = _safe_ratio(detected_x, expected_x)
    y_ratio = _safe_ratio(detected_y, expected_y)

    evidence = {
        "expected_x_ticks": expected_x,
        "detected_x_breaks": detected_x,
        "expected_y_ticks": expected_y,
        "detected_y_breaks": detected_y,
        "x_break_ratio": round(x_ratio, 4) if x_ratio is not None else None,
        "y_break_ratio": round(y_ratio, 4) if y_ratio is not None else None,
        "x_pixels_increment": round(x_pixels_increment, 4) if x_pixels_increment is not None else None,
        "y_pixels_increment": round(y_pixels_increment, 4) if y_pixels_increment is not None else None,
    }

    if detected_x == 0 or detected_y == 0:
        if geometric_fallback_ready:
            return [
                _check(
                    "range-break-match",
                    "range_preflight",
                    "warn",
                    "Tick-label detection is sparse, but geometric axis spacing was inferred successfully. Manual review is still recommended before final extraction.",
                    evidence=evidence,
                )
            ]
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
        if geometric_fallback_ready:
            return [
                _check(
                    "range-break-match",
                    "range_preflight",
                    "warn",
                    "Tick-label detection is incomplete, but geometric axis spacing was inferred successfully. Confirm the axis settings, then extraction can continue.",
                    evidence=evidence,
                )
            ]
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
    for polygon in manifest.exclusion_polygons:
        polygon_area = _polygon_area(polygon.points)
        masked_area += polygon_area
    return min(1.0, masked_area / crop_area)


def _axis_guard_overlap(manifest: ImageManifest, preview_payload: dict, metrics: dict) -> dict[str, float | bool]:
    if not manifest.exclusion_regions and not manifest.exclusion_polygons:
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

    prepared_width = float(metrics.get("prepared_width") or 0.0)
    prepared_height = float(metrics.get("prepared_height") or 0.0)
    plot_bounds = preview_payload.get("plot_bounds") or {}

    if prepared_width > 0 and prepared_height > 0 and plot_bounds:
        left_guard_right = min(
            prepared_width,
            float(plot_bounds.get("left", 0.0)) + max(24.0, float(metrics.get("plot_width") or 0.0) * 0.06),
        )
        bottom_guard_top = max(
            0.0,
            float(plot_bounds.get("bottom", prepared_height)) - max(12.0, float(metrics.get("plot_height") or 0.0) * 0.04),
        )
        bottom_guard_bottom = min(
            prepared_height,
            float(plot_bounds.get("bottom", prepared_height)) + max(32.0, float(metrics.get("plot_height") or 0.0) * 0.1),
        )
        left_guard_rect = (
            crop_left,
            crop_top,
            clamp_ratio(left_guard_right / prepared_width),
            crop_bottom,
        )
        bottom_guard_rect = (
            crop_left,
            clamp_ratio(bottom_guard_top / prepared_height),
            crop_right,
            clamp_ratio(bottom_guard_bottom / prepared_height),
        )
    else:
        left_guard_rect = (crop_left, crop_top, min(crop_right, crop_left + (crop_width * 0.14)), crop_bottom)
        bottom_guard_rect = (crop_left, max(crop_top, crop_bottom - (crop_height * 0.18)), crop_right, crop_bottom)

    left_guard_area = _rect_area(*left_guard_rect)
    bottom_guard_area = _rect_area(*bottom_guard_rect)

    left_overlap = 0.0
    bottom_overlap = 0.0

    for region in manifest.exclusion_regions:
        left = max(crop_left, region.left)
        top = max(crop_top, region.top)
        right = min(crop_right, region.right)
        bottom = min(crop_bottom, region.bottom)
        if right <= left or bottom <= top:
            continue

        if left_guard_area > 0:
            left_overlap = max(left_overlap, _rect_intersection_area((left, top, right, bottom), left_guard_rect) / left_guard_area)
        if bottom_guard_area > 0:
            bottom_overlap = max(bottom_overlap, _rect_intersection_area((left, top, right, bottom), bottom_guard_rect) / bottom_guard_area)

    for polygon in manifest.exclusion_polygons:
        normalized_points = [(point.x, point.y) for point in polygon.points]
        clipped_points = [
            (
                clamp_ratio(max(crop_left, min(crop_right, x_value))),
                clamp_ratio(max(crop_top, min(crop_bottom, y_value))),
            )
            for x_value, y_value in normalized_points
        ]
        if len(clipped_points) < 3:
            continue

        if left_guard_area > 0:
            left_overlap = max(left_overlap, _polygon_rectangle_overlap_ratio(clipped_points, left_guard_rect, left_guard_area))
        if bottom_guard_area > 0:
            bottom_overlap = max(bottom_overlap, _polygon_rectangle_overlap_ratio(clipped_points, bottom_guard_rect, bottom_guard_area))

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


def _safe_float(value: object) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rect_area(left: float, top: float, right: float, bottom: float) -> float:
    return max(0.0, right - left) * max(0.0, bottom - top)


def _rect_intersection_area(rect_a: tuple[float, float, float, float], rect_b: tuple[float, float, float, float]) -> float:
    left = max(rect_a[0], rect_b[0])
    top = max(rect_a[1], rect_b[1])
    right = min(rect_a[2], rect_b[2])
    bottom = min(rect_a[3], rect_b[3])
    return _rect_area(left, top, right, bottom)


def _polygon_rectangle_overlap_ratio(
    points: list[tuple[float, float]],
    rect: tuple[float, float, float, float],
    rect_area: float,
) -> float:
    if rect_area <= 0 or len(points) < 3:
        return 0.0

    clipped = _clip_polygon_to_rect(points, rect)
    if len(clipped) < 3:
        return 0.0
    return min(1.0, _polygon_area_xy(clipped) / rect_area)


def _clip_polygon_to_rect(
    points: list[tuple[float, float]],
    rect: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    left, top, right, bottom = rect

    def clip(points_to_clip: list[tuple[float, float]], inside, intersect):
        if not points_to_clip:
            return []
        output: list[tuple[float, float]] = []
        previous = points_to_clip[-1]
        previous_inside = inside(previous)
        for current in points_to_clip:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersect(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous = current
            previous_inside = current_inside
        return output

    def intersect_vertical(a: tuple[float, float], b: tuple[float, float], x_value: float) -> tuple[float, float]:
        if b[0] == a[0]:
            return (x_value, a[1])
        ratio = (x_value - a[0]) / (b[0] - a[0])
        return (x_value, a[1] + ratio * (b[1] - a[1]))

    def intersect_horizontal(a: tuple[float, float], b: tuple[float, float], y_value: float) -> tuple[float, float]:
        if b[1] == a[1]:
            return (a[0], y_value)
        ratio = (y_value - a[1]) / (b[1] - a[1])
        return (a[0] + ratio * (b[0] - a[0]), y_value)

    clipped = clip(points, lambda point: point[0] >= left, lambda a, b: intersect_vertical(a, b, left))
    clipped = clip(clipped, lambda point: point[0] <= right, lambda a, b: intersect_vertical(a, b, right))
    clipped = clip(clipped, lambda point: point[1] >= top, lambda a, b: intersect_horizontal(a, b, top))
    clipped = clip(clipped, lambda point: point[1] <= bottom, lambda a, b: intersect_horizontal(a, b, bottom))
    return clipped


def _polygon_area_xy(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += (point[0] * next_point[1]) - (next_point[0] * point[1])
    return abs(area) / 2.0


def _polygon_area(points) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += (point.x * next_point.y) - (next_point.x * point.y)
    return abs(area) / 2.0
