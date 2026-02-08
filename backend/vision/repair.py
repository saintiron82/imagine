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
    "art_style": "other",
    "color_palette": "neutral",
}


def parse_structured_output(raw: str, schema: dict, image_type: str = "other") -> dict:
    """3-tier defensive JSON parsing. Always returns a usable dict."""

    # Tier 1: direct parse
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, dict) and _validate_fields(parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    # Tier 2: common error repair
    try:
        repaired = _repair_common_errors(raw)
        parsed = json.loads(repaired)
        if isinstance(parsed, dict) and _validate_fields(parsed):
            return parsed
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

    return {**MINIMUM_GUARANTEED_FIELDS, **extracted}


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
