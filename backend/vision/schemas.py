"""
v3 P0: JSON Schemas for 2-Stage Vision Pipeline.

STAGE1_SCHEMA: image classification output schema.
STAGE2_SCHEMAS: per-type structured analysis schemas.
"""

STAGE1_SCHEMA = {
    "image_type": "string",
    "confidence": "string",
}

SPATIAL_OBJECTS_FIELD = (
    "array of visible tag-level objects. Each item must be an object with: "
    "name (English object/tag name), ko_name (Korean object name if known), "
    "locations (array using one or more of: top-left, top, top-right, left, center, right, "
    "bottom-left, bottom, bottom-right), primary_location (one value from locations), "
    "extent (small, medium, large, wide, full), confidence (high, medium, low), "
    "salience (primary, secondary, background). "
    "Choose exactly one primary_location as the representative grid cell. "
    "Use extra locations sparingly for secondary cells only when the object clearly spans them; "
    "locations should contain at most 3 cells total. Include searchable structural objects "
    "such as walls, floors, ceilings, doors, windows, stairs, fences, gates, buildings, shelves, "
    "cabinets, sky, water, and trees when visible."
)

SPATIAL_STRUCTURAL_OBJECTS_FIELD = (
    "array of visible scene-structure objects only. Keep ordinary objects and structural objects separate. "
    "Use this for architecture, room parts, and large environmental anchors such as wall, floor, ceiling, "
    "door, window, stairs, railing, fence, gate, castle, building, shelf, cabinet, road, sky, water, tree, "
    "and mountain. Each item must use the same object-location shape as objects: name, ko_name, locations, "
    "primary_location, extent, confidence, salience. Choose exactly one primary_location first and keep "
    "locations to at most 3 cells total."
)

SPATIAL_DEPTH_LAYERS_FIELD = (
    "array of visible depth-layer evidence. Each item must be an object with: "
    "name (English object/tag name), ko_name (Korean object name if known), "
    "layer (foreground, midground, background), confidence (high, medium, low). "
    "Use only for visible objects whose depth layer is reasonably clear."
)

SPATIAL_RELATIONS_FIELD = (
    "array of maximum 3 high-confidence visible object-to-object spatial relations. Each item must be an object with: "
    "subject (English object/tag name), relation (one of: left_of, right_of, above, below, "
    "in_front_of, behind, on, under, inside, attached_to), "
    "object (English object/tag name), subject_location (optional 3x3 grid cell), "
    "object_location (optional 3x3 grid cell), confidence (high, medium, low). "
    "Only include relations that are directly visible and whose subject/object names match listed objects."
)

SPATIAL_SCHEMA_VERSION_FIELD = "integer. Current value must be 2."

EXTRACTION_QUALITY_FIELD = (
    "object with objects_status, relations_status, depth_status "
    "(each one of: ok, empty, failed, partial), confidence "
    "(high, medium, low), and notes (short string)."
)

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
        schema = dict(STAGE2_SCHEMAS["generic"])
    else:
        schema = dict(STAGE2_SCHEMAS.get(image_type, STAGE2_SCHEMAS["generic"]))
    schema.setdefault("objects", SPATIAL_OBJECTS_FIELD)
    schema.setdefault("structural_objects", SPATIAL_STRUCTURAL_OBJECTS_FIELD)
    schema.setdefault("depth_layers", SPATIAL_DEPTH_LAYERS_FIELD)
    schema.setdefault("relations", SPATIAL_RELATIONS_FIELD)
    schema.setdefault("spatial_schema_version", SPATIAL_SCHEMA_VERSION_FIELD)
    schema.setdefault("extraction_quality", EXTRACTION_QUALITY_FIELD)
    return schema


def inject_hints_to_schema(
    schema: dict, hints: dict[str, list[str]]
) -> dict:
    """
    Inject domain-specific suggested values into a schema.

    Transforms:
        {"scene_type": "string"}
    Into:
        {"scene_type": "string (suggested: dungeon, castle, forest, ...)"}

    Only injects hints for fields that already exist in the schema.
    Fields with "array" type are not modified (tags, equipment, etc.).

    Args:
        schema: Base schema dict (will NOT be mutated)
        hints: {field_name: [suggested_values, ...]}

    Returns:
        New schema dict with hints injected
    """
    if not hints:
        return schema

    result = dict(schema)
    for field_name, values in hints.items():
        if field_name not in result:
            continue
        # Only inject into string fields, not arrays
        if result[field_name] == "string" and values:
            suggested = ", ".join(values)
            result[field_name] = f"string (suggested: {suggested})"

    return result
