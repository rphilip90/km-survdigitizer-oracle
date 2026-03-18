from __future__ import annotations

import re
from typing import Any

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
}
INTEGER_FIELDS = {"num_curves", "rotation"}
BOOLEAN_FIELDS = {"y_text_vertical", "review_required"}
INCREMENT_FIELDS = {"x_increment", "y_increment"}


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
        return self

    def should_review(self, threshold: float) -> bool:
        return self.review_required or self.llm_confidence < threshold
