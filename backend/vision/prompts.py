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
    "character": """Analyze this character image with provided file context.

INSTRUCTIONS:
- Use layer names to identify visual elements (e.g., "head", "body", "weapon")
- Reference folder path for character category/role
- Consider used fonts for design style
- Incorporate text content for character name/attributes

Output JSON only.
{schema}""",

    "background": """Analyze this background/environment image with provided file context.

INSTRUCTIONS:
- Use layer names to identify scene components (e.g., "sky", "buildings", "foreground")
- Reference folder path for scene category (e.g., "Dungeons", "Cities")
- Consider composition structure from layer organization
- Incorporate text content for location/mood hints

Output JSON only.
{schema}""",

    "ui_element": """Analyze this UI element with provided file context.

INSTRUCTIONS:
- Use layer names to identify UI components (e.g., "button", "icon", "text")
- Reference text content for labels/function
- Consider used fonts for UI style category
- Incorporate folder path for UI section context

Output JSON only.
{schema}""",

    "item": """Analyze this item/object image with provided file context.

INSTRUCTIONS:
- Use layer names to identify item parts (e.g., "blade", "handle", "glow_effect")
- Reference folder path for item category (e.g., "Weapons", "Potions")
- Consider text content for item name/properties
- Incorporate design patterns from layer structure

Output JSON only.
{schema}""",

    "icon": """Analyze this icon image with provided file context.

INSTRUCTIONS:
- Use layer names to identify icon elements (e.g., "symbol", "background", "badge")
- Reference folder path for icon set/category
- Consider design style from layer composition
- Incorporate text content for icon meaning

Output JSON only.
{schema}""",

    "texture": """Analyze this texture/pattern image with provided file context.

INSTRUCTIONS:
- Use layer names to identify texture layers (e.g., "base", "detail", "overlay")
- Reference folder path for texture category (e.g., "Wood", "Metal", "Fabric")
- Consider pattern type from layer organization
- Incorporate filename hints for material type

Output JSON only.
{schema}""",

    "effect": """Analyze this visual effect image with provided file context.

INSTRUCTIONS:
- Use layer names to identify effect components (e.g., "glow", "particles", "distortion")
- Reference folder path for effect category (e.g., "Magic", "Fire", "Lightning")
- Consider effect type from layer structure
- Incorporate filename hints for effect purpose

Output JSON only.
{schema}""",

    "logo": """Analyze this logo/title image with provided file context.

INSTRUCTIONS:
- Use layer names to identify logo elements (e.g., "text", "icon", "background")
- Reference text content for brand/title name
- Consider used fonts for style classification
- Incorporate folder path for logo category

Output JSON only.
{schema}""",

    "photo": """Analyze this photograph with provided file context.

INSTRUCTIONS:
- Use filename hints for photo subject/location
- Reference folder path for photo category (e.g., "Portraits", "Landscapes")
- Consider EXIF data if available (in context)
- Incorporate text content for captions/notes

Output JSON only.
{schema}""",

    "illustration": """Analyze this illustration with provided file context.

INSTRUCTIONS:
- Use layer names to identify illustration elements (e.g., "lineart", "colors", "shadows")
- Reference folder path for illustration category
- Consider art style from layer organization
- Incorporate text content for title/description

Output JSON only.
{schema}""",

    "other": """Analyze this image with provided file context.

INSTRUCTIONS:
- Use any available layer names to identify visual elements
- Reference folder path for image category hints
- Consider filename for content clues
- Incorporate text content if present

Output JSON only.
{schema}""",
}


def get_stage2_prompt(image_type: str, context: dict = None) -> str:
    """
    Build the full Stage 2 prompt with embedded schema.

    Args:
        image_type: Image classification type
        context: Optional file metadata context (v3.1: MC.raw)
                Format: {"file_name": str, "folder_path": str, "layer_names": list, ...}

    Returns:
        Stage 2 prompt string with schema and context
    """
    template = STAGE2_PROMPTS.get(image_type, STAGE2_PROMPTS["other"])
    schema = get_schema(image_type)
    prompt = template.replace("{schema}", json.dumps(schema, indent=2))

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
