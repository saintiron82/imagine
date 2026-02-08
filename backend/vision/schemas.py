"""
v3 P0: JSON Schemas for 2-Stage Vision Pipeline.

STAGE1_SCHEMA: image classification output schema.
STAGE2_SCHEMAS: per-type structured analysis schemas.
"""

STAGE1_SCHEMA = {
    "image_type": "string",
    "confidence": "string",
}

STAGE2_SCHEMAS = {
    "character": {
        "character_type": "string",
        "gender_presentation": "string",
        "age_range": "string",
        "pose": "string",
        "equipment": "array",
        "emotion": "string",
        "art_style": "string",
        "color_palette": "string",
        "caption": "string",
        "tags": "array",
    },
    "background": {
        "scene_type": "string",
        "time_of_day": "string",
        "weather": "string",
        "season": "string",
        "mood": "string",
        "architecture_style": "string",
        "lighting": "string",
        "depth": "string",
        "art_style": "string",
        "color_palette": "string",
        "caption": "string",
        "tags": "array",
    },
    "ui_element": {
        "ui_type": "string",
        "ui_style": "string",
        "color_scheme": "string",
        "has_text": "string",
        "text_content": "string",
        "interactive_elements": "array",
        "platform": "string",
        "art_style": "string",
        "caption": "string",
        "tags": "array",
    },
    "item": {
        "item_type": "string",
        "item_subtype": "string",
        "rarity_feel": "string",
        "material": "string",
        "has_transparency": "string",
        "view_angle": "string",
        "art_style": "string",
        "color_palette": "string",
        "caption": "string",
        "tags": "array",
    },
    "icon": {
        "icon_category": "string",
        "icon_shape": "string",
        "has_border": "string",
        "has_text": "string",
        "text_content": "string",
        "color_palette": "string",
        "art_style": "string",
        "caption": "string",
        "tags": "array",
    },
    "texture": {
        "texture_type": "string",
        "tileable": "string",
        "surface": "string",
        "color_palette": "string",
        "caption": "string",
        "tags": "array",
    },
    "effect": {
        "effect_type": "string",
        "has_transparency": "string",
        "animation_feel": "string",
        "intensity": "string",
        "color_palette": "string",
        "caption": "string",
        "tags": "array",
    },
    # Shared fallback for: logo, photo, illustration, other
    "generic": {
        "subject": "string",
        "art_style": "string",
        "color_palette": "string",
        "mood": "string",
        "has_text": "string",
        "text_content": "string",
        "caption": "string",
        "tags": "array",
    },
}

# Types that use the generic schema
_GENERIC_TYPES = {"logo", "photo", "illustration", "other"}


def get_schema(image_type: str) -> dict:
    """Return the Stage 2 schema for the given image_type."""
    if image_type in _GENERIC_TYPES:
        return STAGE2_SCHEMAS["generic"]
    return STAGE2_SCHEMAS.get(image_type, STAGE2_SCHEMAS["generic"])
