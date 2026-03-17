from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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

