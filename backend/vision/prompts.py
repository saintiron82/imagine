"""
v3 P0: Prompt templates for 2-Stage Vision Pipeline.

STAGE1_PROMPT: Quick classification (1-2s).
STAGE2_PROMPTS: Per-type detailed analysis prompts.
"""

import json
from .schemas import get_schema

STAGE1_PROMPT = """Classify this image into exactly ONE type. Respond with JSON only, no explanation.

{
  "image_type": "ONE OF: character, background, ui_element, item, icon, texture, effect, logo, photo, illustration, other",
  "confidence": "ONE OF: high, medium, low"
}"""

STAGE2_PROMPTS = {
    "character": "Analyze this character image. Output JSON only.\n{schema}",
    "background": "Analyze this background/environment image. Output JSON only.\n{schema}",
    "ui_element": "Analyze this UI element image. Output JSON only.\n{schema}",
    "item": "Analyze this item/object image. Output JSON only.\n{schema}",
    "icon": "Analyze this icon image. Output JSON only.\n{schema}",
    "texture": "Analyze this texture/pattern image. Output JSON only.\n{schema}",
    "effect": "Analyze this visual effect image. Output JSON only.\n{schema}",
    "logo": "Analyze this logo/title image. Output JSON only.\n{schema}",
    "photo": "Analyze this photograph. Output JSON only.\n{schema}",
    "illustration": "Analyze this illustration. Output JSON only.\n{schema}",
    "other": "Analyze this image. Output JSON only.\n{schema}",
}


def get_stage2_prompt(image_type: str) -> str:
    """Build the full Stage 2 prompt with embedded schema."""
    template = STAGE2_PROMPTS.get(image_type, STAGE2_PROMPTS["other"])
    schema = get_schema(image_type)
    return template.replace("{schema}", json.dumps(schema, indent=2))
