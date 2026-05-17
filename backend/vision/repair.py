"""
v3 P0: 3-tier defensive JSON parsing for Vision LLM output.

Tier 1: Direct json.loads
Tier 2: Common error repair (markdown fences, trailing commas, unclosed braces)
Tier 3: Regex field extraction (last resort)

Always returns a dict with at least MINIMUM_GUARANTEED_FIELDS.
"""

import json
import re

MINIMUM_GUARANTEED_FIELDS = {
    "image_type": "other",
    "caption": "",
    "tags": [],
    "objects": [],
    "depth_layers": [],
    "relations": [],
    "art_style": "other",
    "color_palette": "neutral",
}

_SPATIAL_LOCATIONS = {
    "top-left",
    "top",
    "top-right",
    "left",
    "center",
    "right",
    "bottom-left",
    "bottom",
    "bottom-right",
}

_OBJECT_FIELDS = {
    "name",
    "ko_name",
    "locations",
    "primary_location",
    "extent",
    "confidence",
}

_DEPTH_LAYERS = {"foreground", "midground", "background"}
_RELATIONS = {
    "on",
    "under",
    "left_of",
    "right_of",
    "above",
    "below",
    "in_front_of",
    "behind",
    "inside",
    "around",
    "attached_to",
    "near",
    "overlapping",
}


def parse_structured_output(
    raw: str,
    schema: dict,
    image_type: str = "other",
    include_diagnostics: bool = False,
) -> dict:
    """3-tier defensive JSON parsing. Always returns a usable dict."""

    # Tier 1: direct parse
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, dict) and _validate_fields(parsed):
            return _with_diagnostics(
                _sanitize_result(parsed), "direct", False, include_diagnostics
            )
    except json.JSONDecodeError:
        pass

    # Tier 2: common error repair
    try:
        repaired = _repair_common_errors(raw)
        parsed = json.loads(repaired)
        if isinstance(parsed, dict) and _validate_fields(parsed):
            return _with_diagnostics(
                _sanitize_result(parsed), "repaired", True, include_diagnostics
            )
    except (json.JSONDecodeError, ValueError):
        pass

    # Tier 3: regex field extraction (last resort)
    extracted = _extract_fields_fallback(raw, schema)
    extracted.setdefault("image_type", image_type)

    # If caption is empty, salvage raw text
    if not extracted.get("caption"):
        clean_text = re.sub(r'[{}\[\]":]', ' ', raw)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        extracted["caption"] = clean_text[:500]

    return _with_diagnostics(
        _sanitize_result({**MINIMUM_GUARANTEED_FIELDS, **extracted}),
        "fallback",
        True,
        include_diagnostics,
    )


def _with_diagnostics(result: dict, status: str, repaired: bool, include: bool) -> dict:
    if include:
        result = dict(result)
        result["_parse_diagnostics"] = {
            "status": status,
            "repaired": repaired,
        }
    return result


def _repair_common_errors(raw: str) -> str:
    """Fix common LLM JSON mistakes."""
    text = raw.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    # Trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Unclosed braces/brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    text += '}' * max(0, open_braces)
    text += ']' * max(0, open_brackets)

    # Single quotes to double quotes (key patterns only)
    text = re.sub(r"'(\w+)'\s*:", r'"\1":', text)

    return text


def _extract_fields_fallback(raw: str, schema: dict) -> dict:
    """Regex per-field extraction as last resort."""
    result = {}
    for field_name in schema.keys():
        extracted_value = _extract_json_field_value(raw, field_name)
        if extracted_value is not None:
            try:
                parsed_value = json.loads(extracted_value)
                if field_name == "objects":
                    result[field_name] = _coerce_spatial_objects(parsed_value)
                else:
                    result[field_name] = parsed_value
                continue
            except json.JSONDecodeError:
                pass

        if field_name == "objects":
            result.setdefault("objects", [])
            continue

        # String value pattern
        pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
        match = re.search(pattern, raw)
        if match:
            result[field_name] = match.group(1)

        # Array value pattern
        array_pattern = rf'"{field_name}"\s*:\s*\[([^\]]*)\]'
        array_match = re.search(array_pattern, raw)
        if array_match and field_name not in result:
            items = re.findall(r'"([^"]*)"', array_match.group(1))
            result[field_name] = items

    return result


def _validate_fields(parsed: dict) -> bool:
    """Check minimum required fields exist."""
    return "caption" in parsed or "image_type" in parsed


def _sanitize_result(parsed: dict) -> dict:
    result = dict(parsed)
    result["objects"] = _coerce_spatial_objects(result.get("objects", []))
    result["depth_layers"] = _coerce_depth_layers(result.get("depth_layers", []))
    result["relations"] = _coerce_relations(result.get("relations", []))
    return result


def _extract_json_field_value(raw: str, field_name: str) -> str | None:
    match = re.search(rf'"{re.escape(field_name)}"\s*:', raw)
    if not match:
        return None

    i = match.end()
    while i < len(raw) and raw[i].isspace():
        i += 1
    if i >= len(raw):
        return None

    start = i
    opener = raw[i]
    if opener == '"':
        i += 1
        escaped = False
        while i < len(raw):
            ch = raw[i]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                return raw[start : i + 1]
            i += 1
        return None

    if opener not in "[{":
        while i < len(raw) and raw[i] not in ",}\n":
            i += 1
        value = raw[start:i].strip()
        return value or None

    closer = "]" if opener == "[" else "}"
    stack = [closer]
    i += 1
    in_string = False
    escaped = False
    while i < len(raw):
        ch = raw[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in "[{":
                stack.append("]" if ch == "[" else "}")
            elif ch in "]}":
                if not stack or ch != stack[-1]:
                    return None
                stack.pop()
                if not stack:
                    return raw[start : i + 1]
        i += 1

    return None


def _coerce_spatial_objects(value) -> list[dict]:
    if not isinstance(value, list):
        return []

    if value and all(not isinstance(item, dict) for item in value):
        value = _coerce_flat_object_tokens(value)

    objects = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        obj = _sanitize_spatial_object(raw)
        if obj:
            objects.append(obj)
    return objects


def _coerce_flat_object_tokens(tokens: list) -> list[dict]:
    text_tokens = [str(token).strip() for token in tokens if str(token).strip()]
    objects = []
    current = {}
    i = 0
    while i < len(text_tokens):
        token = text_tokens[i]
        if token == "name":
            if current:
                objects.append(current)
            current = {}
            if i + 1 < len(text_tokens) and text_tokens[i + 1] not in _OBJECT_FIELDS:
                current["name"] = text_tokens[i + 1]
                i += 2
                continue
        elif token == "ko_name":
            if i + 1 < len(text_tokens) and text_tokens[i + 1] not in _OBJECT_FIELDS:
                current["ko_name"] = text_tokens[i + 1]
                i += 2
                continue
        elif token == "locations":
            locations = []
            i += 1
            while i < len(text_tokens) and text_tokens[i] not in _OBJECT_FIELDS:
                locations.append(text_tokens[i])
                i += 1
            current["locations"] = locations
            continue
        elif token in {"primary_location", "extent", "confidence"}:
            if i + 1 < len(text_tokens) and text_tokens[i + 1] not in _OBJECT_FIELDS:
                current[token] = text_tokens[i + 1]
                i += 2
                continue
        i += 1

    if current:
        objects.append(current)
    return objects


def _sanitize_spatial_object(raw: dict) -> dict | None:
    name = str(raw.get("name") or "").strip()
    ko_name = str(raw.get("ko_name") or "").strip()
    if not name and not ko_name:
        return None

    raw_locations = raw.get("locations")
    if isinstance(raw_locations, str):
        raw_locations = [raw_locations]
    if not isinstance(raw_locations, list):
        raw_locations = []

    locations = []
    seen = set()
    for loc in raw_locations:
        normalized = _normalize_location(loc)
        if normalized and normalized not in seen:
            seen.add(normalized)
            locations.append(normalized)

    primary_location = _normalize_location(raw.get("primary_location"))
    if primary_location:
        if primary_location in seen:
            locations = [loc for loc in locations if loc != primary_location]
        else:
            seen.add(primary_location)
        locations.insert(0, primary_location)
    elif locations:
        primary_location = locations[0]

    if not primary_location or not locations:
        return None

    extent = str(raw.get("extent") or "").strip().lower()
    if extent not in {"small", "medium", "large", "wide", "full"}:
        extent = ""

    confidence = str(raw.get("confidence") or "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "name": name,
        "ko_name": ko_name,
        "locations": locations,
        "primary_location": primary_location,
        "extent": extent,
        "confidence": confidence,
    }


def _normalize_location(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    return text if text in _SPATIAL_LOCATIONS else None


def _coerce_depth_layers(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    layers = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or raw.get("object") or "").strip()
        ko_name = str(raw.get("ko_name") or "").strip()
        layer = str(raw.get("layer") or "").strip().lower().replace("_", "-")
        if layer not in _DEPTH_LAYERS or not (name or ko_name):
            continue
        confidence = str(raw.get("confidence") or "").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        layers.append({
            "name": name,
            "ko_name": ko_name,
            "layer": layer,
            "confidence": confidence,
        })
    return layers


def _coerce_relations(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    relations = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        subject = str(raw.get("subject") or "").strip()
        obj = str(raw.get("object") or "").strip()
        relation = str(raw.get("relation") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not subject or not obj or relation not in _RELATIONS:
            continue
        confidence = str(raw.get("confidence") or "").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        relations.append({
            "subject": subject,
            "relation": relation,
            "object": obj,
            "subject_location": _normalize_location(raw.get("subject_location")) or "",
            "object_location": _normalize_location(raw.get("object_location")) or "",
            "confidence": confidence,
        })
        if len(relations) >= 5:
            break
    return relations
