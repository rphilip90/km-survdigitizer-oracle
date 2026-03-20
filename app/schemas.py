from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+")
PLAIN_NUMBER_PATTERN = re.compile(r"^\s*[-+]?\d*\.?\d+\s*%?\s*$")
BOOLEAN_TRUE_VALUES = {"true", "1", "yes", "y", "on"}
BOOLEAN_FALSE_VALUES = {"false", "0", "no", "n", "off"}
FLOAT_FIELDS = {
    "x_start",
    "x_end",
    "x_increment",
    "y_start",
    "y_end",
    "y_increment",
    "llm_confidence",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
}
INTEGER_FIELDS = {"num_curves", "rotation"}
BOOLEAN_FIELDS = {"y_text_vertical", "review_required"}
INCREMENT_FIELDS = {"x_increment", "y_increment"}
OPTIONAL_FLOAT_FIELDS = {"crop_left", "crop_top", "crop_right", "crop_bottom"}


def extract_numbers(value: str) -> list[float]:
    return [float(match.group(0)) for match in NUMERIC_PATTERN.finditer(value)]


def parse_boolean(value: Any) -> bool | Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in BOOLEAN_TRUE_VALUES:
            return True
        if lowered in BOOLEAN_FALSE_VALUES:
            return False
    return value


def normalize_numeric_value(field_name: str, value: Any) -> tuple[Any, str | None, bool]:
    if isinstance(value, bool) or value is None:
        return value, None, False
    if isinstance(value, (int, float)):
        return value, None, False
    if not isinstance(value, str):
        return value, None, False

    stripped = value.strip()
    numbers = extract_numbers(stripped)
    if not numbers:
        return value, None, False

    normalized = numbers[0]
    ambiguous = False

    if field_name in INCREMENT_FIELDS and "minor" in stripped.lower():
        positive_numbers = [number for number in numbers if number > 0]
        if len(positive_numbers) >= 2:
            normalized = min(positive_numbers)
            ambiguous = True

    if field_name == "llm_confidence" and "%" in stripped:
        normalized = normalized / 100 if normalized > 1 else normalized

    if field_name in INTEGER_FIELDS:
        normalized = int(round(normalized))

    note = None
    if PLAIN_NUMBER_PATTERN.fullmatch(stripped) is None:
        note = f"{field_name} normalized from '{stripped}' to {normalized}"

    return normalized, note, ambiguous


class ImageManifest(BaseModel):
    image_id: str
    filename: str
    num_curves: int = Field(ge=1, le=12)
    x_start: float
    x_end: float
    x_increment: float = Field(gt=0)
    y_start: float
    y_end: float
    y_increment: float = Field(gt=0)
    y_text_vertical: bool
    rotation: int | None = 0
    crop_left: float | None = None
    crop_top: float | None = None
    crop_right: float | None = None
    crop_bottom: float | None = None
    exclusion_regions: list["ExclusionRegion"] = Field(default_factory=list)
    crop_hint: str | None = None
    notes: str | None = None
    llm_confidence: float = Field(ge=0, le=1)
    review_required: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        payload = dict(value)
        notes = []
        if payload.get("notes"):
            notes.append(str(payload["notes"]).strip())

        force_review = False

        for field_name in FLOAT_FIELDS | INTEGER_FIELDS:
            if field_name not in payload:
                continue
            if field_name in OPTIONAL_FLOAT_FIELDS and isinstance(payload[field_name], str) and not payload[field_name].strip():
                payload[field_name] = None
                continue
            normalized, note, ambiguous = normalize_numeric_value(field_name, payload[field_name])
            payload[field_name] = normalized
            if note:
                notes.append(note)
            if ambiguous:
                force_review = True

        for field_name in BOOLEAN_FIELDS:
            if field_name in payload:
                payload[field_name] = parse_boolean(payload[field_name])

        if "llm_confidence" in payload and isinstance(payload["llm_confidence"], (int, float)) and payload["llm_confidence"] > 1:
            payload["llm_confidence"] = payload["llm_confidence"] / 100

        if force_review:
            payload["review_required"] = True

        payload["notes"] = "\n".join(note for note in notes if note) or None
        return payload

    @model_validator(mode="after")
    def validate_ranges(self) -> "ImageManifest":
        if self.x_end <= self.x_start:
            raise ValueError("x_end must be greater than x_start")
        if self.y_end <= self.y_start:
            raise ValueError("y_end must be greater than y_start")
        if self.rotation not in (None, 0, 90, 180, 270):
            raise ValueError("rotation must be one of 0, 90, 180, 270")
        crop_values = [self.crop_left, self.crop_top, self.crop_right, self.crop_bottom]
        defined_crops = [value for value in crop_values if value is not None]
        if defined_crops and len(defined_crops) != 4:
            raise ValueError("crop_left, crop_top, crop_right, and crop_bottom must all be provided together")
        for value in defined_crops:
            if not 0 <= value <= 1:
                raise ValueError("crop bounds must be between 0 and 1")
        if self.crop_right is not None and self.crop_left is not None and self.crop_right <= self.crop_left:
            raise ValueError("crop_right must be greater than crop_left")
        if self.crop_bottom is not None and self.crop_top is not None and self.crop_bottom <= self.crop_top:
            raise ValueError("crop_bottom must be greater than crop_top")
        return self

    def should_review(self, threshold: float) -> bool:
        return self.review_required or self.llm_confidence < threshold


class ExclusionRegion(BaseModel):
    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)
    label: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "ExclusionRegion":
        if self.right <= self.left:
            raise ValueError("exclusion region right must be greater than left")
        if self.bottom <= self.top:
            raise ValueError("exclusion region bottom must be greater than top")
        return self


class PreflightCheck(BaseModel):
    id: str
    stage: Literal[
        "manifest_preflight",
        "prepared_image_preflight",
        "axis_preflight",
        "cluster_preflight",
        "range_preflight",
    ]
    severity: Literal["info", "warning", "blocking"]
    status: Literal["pass", "warn", "fail"]
    message: str
    evidence: dict[str, Any] | None = None


class PreflightReport(BaseModel):
    blocking: bool = False
    checks: list[PreflightCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


ImageManifest.model_rebuild()
