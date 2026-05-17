"""
v3 P0: Prompt templates for 2-Stage Vision Pipeline.

STAGE1_SYSTEM / STAGE1_USER: Quick classification (1-2s).
STAGE2_SYSTEM / STAGE2_PROMPTS: Per-type detailed analysis prompts.

v3.6: system role separated from user prompt for better JSON compliance.
v3.7: Domain-aware dynamic prompt generation.
      build_stage1_prompt(domain) generates domain-specific Stage 1 prompts.
      get_stage2_prompt() now accepts domain parameter for hint injection.
"""

import json
from typing import TYPE_CHECKING

from .schemas import get_schema, inject_hints_to_schema

if TYPE_CHECKING:
    from .domain_loader import DomainProfile

# ── Stage 1: Classification ─────────────────────────────────────────────
STAGE1_SYSTEM = "You are a strict JSON generator. Output valid JSON only. No explanation, no markdown fences."

STAGE1_USER = """Classify this image into exactly ONE type.

{
  "image_type": "ONE OF: character, background, ui_element, item, icon, texture, effect, logo, photo, illustration, other",
  "confidence": "ONE OF: high, medium, low"
}"""

# Keep legacy alias for backward compatibility with Ollama adapter
STAGE1_PROMPT = STAGE1_USER

# ── Stage 2: Structured Analysis ─────────────────────────────────────────
STAGE2_SYSTEM = "You are a strict JSON generator. Output valid JSON only matching the provided schema. No explanation, no markdown fences."

SPATIAL_OBJECT_INSTRUCTIONS = """Object-location extraction:
- Populate "objects" for the concrete visible elements that justify the caption/tags.
- Use a 3x3 image grid only: top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right.
- Use multiple locations when an element spans more than one grid cell.
- Set primary_location to the most representative cell from locations.
- Use extent small/medium/large/wide/full and confidence high/medium/low.
- Prefer concrete visible elements: moon, character, tree, window, sword, text overlay, sky, wall, water.
- Populate "depth_layers" for clear foreground, midground, background evidence.
- Populate "relations" with maximum 5 visible object-to-object spatial relations.
- Use relation values only from: on, under, left_of, right_of, above, below, in_front_of, behind, inside, around, attached_to, near, overlapping.
- Do not invent invisible objects. If no reliable object can be localized, return "objects": [].
"""

STAGE2_PROMPTS = {
    "character": """Analyze this character image with provided file context.

INSTRUCTIONS:
- Use layer names to identify visual elements (e.g., "head", "body", "weapon")
- Reference folder path for character category/role
- Consider used fonts for design style
- Incorporate text content for character name/attributes

{schema}""",

    "background": """Analyze this background/environment image with provided file context.

INSTRUCTIONS:
- Use layer names to identify scene components (e.g., "sky", "buildings", "foreground")
- Reference folder path for scene category (e.g., "Dungeons", "Cities")
- Consider composition structure from layer organization
- Incorporate text content for location/mood hints

{schema}""",

    "ui_element": """Analyze this UI element with provided file context.

INSTRUCTIONS:
- Use layer names to identify UI components (e.g., "button", "icon", "text")
- Reference text content for labels/function
- Consider used fonts for UI style category
- Incorporate folder path for UI section context

{schema}""",

    "item": """Analyze this item/object image with provided file context.

INSTRUCTIONS:
- Use layer names to identify item parts (e.g., "blade", "handle", "glow_effect")
- Reference folder path for item category (e.g., "Weapons", "Potions")
- Consider text content for item name/properties
- Incorporate design patterns from layer structure

{schema}""",

    "icon": """Analyze this icon image with provided file context.

INSTRUCTIONS:
- Use layer names to identify icon elements (e.g., "symbol", "background", "badge")
- Reference folder path for icon set/category
- Consider design style from layer composition
- Incorporate text content for icon meaning

{schema}""",

    "texture": """Analyze this texture/pattern image with provided file context.

INSTRUCTIONS:
- Use layer names to identify texture layers (e.g., "base", "detail", "overlay")
- Reference folder path for texture category (e.g., "Wood", "Metal", "Fabric")
- Consider pattern type from layer organization
- Incorporate filename hints for material type

{schema}""",

    "effect": """Analyze this visual effect image with provided file context.

INSTRUCTIONS:
- Use layer names to identify effect components (e.g., "glow", "particles", "distortion")
- Reference folder path for effect category (e.g., "Magic", "Fire", "Lightning")
- Consider effect type from layer structure
- Incorporate filename hints for effect purpose

{schema}""",

    "logo": """Analyze this logo/title image with provided file context.

INSTRUCTIONS:
- Use layer names to identify logo elements (e.g., "text", "icon", "background")
- Reference text content for brand/title name
- Consider used fonts for style classification
- Incorporate folder path for logo category

{schema}""",

    "photo": """Analyze this photograph with provided file context.

INSTRUCTIONS:
- Use filename hints for photo subject/location
- Reference folder path for photo category (e.g., "Portraits", "Landscapes")
- Consider EXIF data if available (in context)
- Incorporate text content for captions/notes

{schema}""",

    "illustration": """Analyze this illustration with provided file context.

INSTRUCTIONS:
- Use layer names to identify illustration elements (e.g., "lineart", "colors", "shadows")
- Reference folder path for illustration category
- Consider art style from layer organization
- Incorporate text content for title/description

{schema}""",

    "other": """Analyze this image with provided file context.

INSTRUCTIONS:
- Use any available layer names to identify visual elements
- Reference folder path for image category hints
- Consider filename for content clues
- Incorporate text content if present

{schema}""",
}

# ── Stage 2: Concise variants — type-specific focus + compact output ────────

_CONCISE_OUTPUT_FMT = f"""\nReturn ONLY this JSON (no markdown fences):
{{
  "caption": "one sentence, max 20 words, describe what is visible",
  "tags": ["5-8 concrete tags of objects/attributes visible in the image"],
  "objects": [
    {{
      "name": "visible object/tag name",
      "ko_name": "Korean object name if known",
      "locations": ["one or more of: top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right"],
      "primary_location": "one value from locations",
      "extent": "small|medium|large|wide|full",
      "confidence": "high|medium|low"
    }}
  ],
  "depth_layers": [
    {{
      "name": "visible object/tag name",
      "ko_name": "Korean object name if known",
      "layer": "foreground|midground|background",
      "confidence": "high|medium|low"
    }}
  ],
  "relations": [
    {{
      "subject": "visible object/tag name",
      "relation": "on|under|left_of|right_of|above|below|in_front_of|behind|inside|around|attached_to|near|overlapping",
      "object": "visible object/tag name",
      "subject_location": "optional 3x3 grid cell",
      "object_location": "optional 3x3 grid cell",
      "confidence": "high|medium|low"
    }}
  ],
  "spatial_schema_version": 2,
  "extraction_quality": {{
    "objects_status": "ok|empty|failed|partial",
    "relations_status": "ok|empty|failed|partial",
    "depth_status": "ok|empty|failed|partial",
    "confidence": "high|medium|low",
    "notes": "short reason when empty or partial"
  }},
  "art_style": "one word",
  "color_palette": "max 3 dominant colors"
}}

{SPATIAL_OBJECT_INSTRUCTIONS}"""

STAGE2_PROMPTS_CONCISE = {
    "character": f"""Analyze this character image. Focus on: pose, outfit, weapon/equipment, expression, body type.{_CONCISE_OUTPUT_FMT}""",

    "background": f"""Analyze this background/environment image. Focus on: scene type, visible objects (furniture, architecture, nature), lighting, time of day.{_CONCISE_OUTPUT_FMT}""",

    "ui_element": f"""Analyze this UI element. Focus on: UI component type, layout, text labels, color scheme.{_CONCISE_OUTPUT_FMT}""",

    "item": f"""Analyze this item/object. Focus on: item type, material, shape, visual details.{_CONCISE_OUTPUT_FMT}""",

    "icon": f"""Analyze this icon. Focus on: symbol meaning, shape, color, border style.{_CONCISE_OUTPUT_FMT}""",

    "texture": f"""Analyze this texture/pattern. Focus on: surface material, tileability, color, pattern type.{_CONCISE_OUTPUT_FMT}""",

    "effect": f"""Analyze this visual effect. Focus on: effect type (fire, magic, glow, etc.), color, intensity.{_CONCISE_OUTPUT_FMT}""",

    "logo": f"""Analyze this logo/title. Focus on: text content, font style, brand identity.{_CONCISE_OUTPUT_FMT}""",

    "photo": f"""Analyze this photograph. Focus on: subject, setting, composition, lighting.{_CONCISE_OUTPUT_FMT}""",

    "illustration": f"""Analyze this illustration. Focus on: scene content, visible objects, art style, composition.{_CONCISE_OUTPUT_FMT}""",

    "other": f"""Analyze this image. Focus on: visible objects, scene type, composition.{_CONCISE_OUTPUT_FMT}""",
}

# Legacy alias
STAGE2_USER_CONCISE = STAGE2_PROMPTS_CONCISE["other"]


def _normalize_analysis_profile(profile: dict = None) -> dict:
    if not isinstance(profile, dict):
        return {}
    expected = profile.get("expected_types") or []
    if isinstance(expected, str):
        expected = [expected]
    expected = [str(t).strip() for t in expected if str(t).strip()]
    primary = str(profile.get("primary_type") or "").strip()
    if primary and primary not in expected:
        expected = [primary] + expected
    return {
        "domain_id": str(profile.get("domain_id") or "").strip(),
        "expected_types": expected,
        "primary_type": primary,
        "source": str(profile.get("source") or "").strip(),
    }


def _build_analysis_profile_text(profile: dict = None) -> str:
    normalized = _normalize_analysis_profile(profile)
    expected = normalized.get("expected_types") or []
    primary = normalized.get("primary_type") or (expected[0] if expected else "")
    if not expected and not primary:
        return ""

    lines = [
        "Analysis job profile:",
        f"- Expected types: {', '.join(expected)}" if expected else "",
        f"- primary expected type: {primary}" if primary else "",
        "- Use this as a soft prior for ambiguous cases. Do not override clear visual evidence.",
    ]
    if primary == "background":
        lines.append("- For background-like images, prioritize environment, lighting, composition, and spatial layout.")
    elif primary == "effect":
        lines.append("- For effect-like images, prioritize visible particles, glow, motion, color, and intensity.")
    return "\n".join(line for line in lines if line)


def build_stage1_prompt(
    domain: "DomainProfile" = None,
    analysis_profile: dict = None,
) -> str:
    """
    Build Stage 1 classification prompt, optionally scoped to a domain.

    When a domain is provided, only the domain's image_types are listed
    as choices, producing more focused classification.

    Args:
        domain: Optional DomainProfile. If None, uses the full legacy type list.

    Returns:
        Stage 1 user prompt string
    """
    if domain and domain.image_types:
        types_str = ", ".join(domain.image_types)
    else:
        # Legacy: all 11 types
        types_str = (
            "character, background, ui_element, item, icon, "
            "texture, effect, logo, photo, illustration, other"
        )

    prompt = (
        "Classify this image into exactly ONE type.\n\n"
        "{\n"
        f'  "image_type": "ONE OF: {types_str}",\n'
        '  "confidence": "ONE OF: high, medium, low"\n'
        "}"
    )
    profile_text = _build_analysis_profile_text(analysis_profile)
    if profile_text:
        prompt = f"{prompt}\n\n{profile_text}"
    return prompt


def get_stage2_prompt(
    image_type: str,
    context: dict = None,
    domain: "DomainProfile" = None,
    concise: bool = False,
    analysis_profile: dict = None,
) -> str:
    """
    Build the full Stage 2 prompt with embedded schema.

    Args:
        image_type: Image classification type
        context: Optional file metadata context (v3.1: MC.raw)
                Format: {"file_name": str, "folder_path": str, "layer_names": list, ...}
        domain: Optional DomainProfile for domain-specific hint injection (v3.7)
        concise: Use concise prompt for Qwen3.5 (fewer tokens, faster generation)

    Returns:
        Stage 2 prompt string with schema, domain hints, and context
    """
    if analysis_profile is None and isinstance(context, dict):
        analysis_profile = context.get("analysis_profile")
    profile_text = _build_analysis_profile_text(analysis_profile)

    # v4.2: Concise prompt — type-specific focus + compact output format
    if concise:
        prompt = STAGE2_PROMPTS_CONCISE.get(image_type, STAGE2_PROMPTS_CONCISE["other"])
        if profile_text:
            prompt = f"{prompt}\n\n{profile_text}"
        if context:
            context_text = _build_context_text(context)
            prompt = f"{prompt}\n\n{context_text}"
        return prompt

    template = STAGE2_PROMPTS.get(image_type, STAGE2_PROMPTS["other"])
    schema = get_schema(image_type)

    # v3.7: Inject domain-specific category hints into schema
    if domain:
        hints = domain.get_type_hints(image_type)
        schema = inject_hints_to_schema(schema, hints)

        # Inject extra domain-specific instructions
        extra_instruction = domain.get_type_instruction(image_type)
        if extra_instruction:
            template = f"{template}\n\nDOMAIN-SPECIFIC: {extra_instruction}"

    prompt = template.replace("{schema}", json.dumps(schema, indent=2))
    prompt = f"{prompt}\n\n{SPATIAL_OBJECT_INSTRUCTIONS}"
    if profile_text:
        prompt = f"{prompt}\n\n{profile_text}"

    # v3.1: Inject file metadata context
    if context:
        context_text = _build_context_text(context)
        prompt = f"{prompt}\n\n{context_text}"

    return prompt


def _build_context_text(context: dict) -> str:
    """
    Build context injection text from metadata.

    v3.1: MC.raw - File metadata facts to inject into AI prompt.
    """
    parts = ["File context:"]

    if context.get("file_name"):
        parts.append(f"- File: {context['file_name']}")

    if context.get("folder_path"):
        parts.append(f"- Folder: {context['folder_path']}")

    if context.get("layer_names"):
        layers = context["layer_names"]
        if isinstance(layers, list):
            layers = ", ".join(str(l) for l in layers[:10])  # First 10 layers
        parts.append(f"- Layers: {layers}")

    if context.get("used_fonts"):
        fonts = context["used_fonts"]
        if isinstance(fonts, list):
            fonts = ", ".join(fonts[:5])  # First 5 fonts
        elif fonts:
            fonts = str(fonts)
        if fonts:
            parts.append(f"- Fonts: {fonts}")

    if context.get("ocr_text"):
        ocr = context["ocr_text"][:100]  # First 100 chars
        if ocr:
            parts.append(f"- Text content: {ocr}")

    if context.get("text_content"):
        text = context["text_content"]
        if isinstance(text, list):
            text = " ".join(str(t) for t in text[:3])  # First 3 text layers
        if text:
            parts.append(f"- Layer text: {text[:100]}")

    return "\n".join(parts)
