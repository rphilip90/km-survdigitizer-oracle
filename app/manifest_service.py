from __future__ import annotations

from pathlib import Path
import base64
import json
import mimetypes
import re

from openai import OpenAI

from .config import Settings
from .schemas import ImageManifest


class ManifestService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def generate_manifest(self, image_id: str, filename: str, image_path: Path) -> ImageManifest:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        prompt = self.build_prompt(image_id=image_id, filename=filename)
        data_url = self.image_to_data_url(image_path)

        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce strict JSON only. "
                        "Return SurvdigitizeR-ready axis parameters for Kaplan-Meier style images. "
                        "If uncertain, set review_required to true and lower llm_confidence."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content or "{}"
        parsed = self.parse_json_payload(raw)
        parsed.setdefault("image_id", image_id)
        parsed.setdefault("filename", filename)
        return ImageManifest.model_validate(parsed)

    def build_prompt(self, image_id: str, filename: str) -> str:
        return f"""
Analyze this Kaplan-Meier or survival-style plot image and extract parameters for the R package SurvdigitizeR.

Return JSON with exactly these keys:
- image_id
- filename
- num_curves
- x_start
- x_end
- x_increment
- y_start
- y_end
- y_increment
- y_text_vertical
- rotation
- crop_left
- crop_top
- crop_right
- crop_bottom
- exclusion_regions
- exclusion_polygons
- crop_hint
- notes
- llm_confidence
- review_required

Important rules:
- Do not guess crop coordinates or exclusion-mask coordinates. Manual crop and mask drawing is handled in the review UI.
- Always return crop_left, crop_top, crop_right, and crop_bottom as null.
- Always return exclusion_regions as [].
- Always return exclusion_polygons as [].
- If the figure contains risk tables, summary tables, captions, panel labels, or other non-plot blocks that likely require manual crop or masking, explain that in notes or crop_hint and set review_required to true.
- If the plot layout is complex or likely needs manual crop before digitization, set review_required to true.
- x_increment and y_increment must be plain JSON numbers only.
- If minor ticks are visibly present, x_increment and y_increment must be the minor tick spacing itself.
- Put any explanation about tick marks, uncertainty, units, or assumptions in notes, not in numeric fields.
- y_text_vertical is true only if the y-axis labels are rotated vertically.
- rotation must be one of 0, 90, 180, 270 and means the clockwise correction needed before digitization.
- crop_hint should briefly explain whether manual crop or masking is recommended, for example "manual crop recommended: exclude risk table below x-axis"; otherwise null.
- review_required must be true if any axis limit, increment, or curve count is uncertain.
- review_required must be true if the image likely needs manual crop or masking before digitization.
- llm_confidence must be a number between 0 and 1.
- Do not include markdown, prose, or extra keys.

Image id: {image_id}
Filename: {filename}
""".strip()

    def image_to_data_url(self, image_path: Path) -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def parse_json_payload(self, raw: str) -> dict:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)
