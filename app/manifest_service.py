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
- crop_hint
- notes
- llm_confidence
- review_required

Important rules:
- Identify the actual plotting panel only. Exclude number-at-risk tables, top summary tables, captions, keywords, and any decorative page content outside the axes.
- crop_left, crop_top, crop_right, and crop_bottom must keep the full x-axis, full y-axis, all visible tick marks, and all axis labels needed for digitization. Do not crop into the axes.
- If the figure contains non-plot blocks such as a risk table, hazard-ratio table, top summary block, or caption that intrude into the crop rectangle, use exclusion_regions to mask them while preserving the full axes.
- exclusion_regions must be a JSON array of objects. Each object must have left, top, right, bottom as normalized numbers between 0 and 1 plus an optional label. Return [] when no masking is needed.
- Prefer a generous crop plus one or more exclusion_regions over an aggressive crop that cuts off axes or tick labels.
- If the full image is already just the plotting panel, set crop_left, crop_top, crop_right, and crop_bottom to null.
- x_increment and y_increment must be plain JSON numbers only.
- If minor ticks are visibly present, x_increment and y_increment must be the minor tick spacing itself.
- Put any explanation about tick marks, uncertainty, units, or assumptions in notes, not in numeric fields.
- y_text_vertical is true only if the y-axis labels are rotated vertically.
- rotation must be one of 0, 90, 180, 270 and means the clockwise correction needed before digitization.
- crop_hint should briefly explain what was excluded, for example "exclude risk table and top summary"; otherwise null.
- review_required must be true if any axis limit, increment, or curve count is uncertain.
- review_required must be true if the plot panel bounds or exclusion_regions are uncertain.
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
