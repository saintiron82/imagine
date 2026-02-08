"""
Data Cleaner Module - Normalizes and enriches layer names for semantic processing.

Key functions:
- clean_layer_name(): Removes noise like "copy", numbers, special chars
- infer_content_type(): Heuristic tagging based on layer properties
"""

import re
from typing import Optional, Tuple, Literal


# Patterns to remove from layer names
NOISE_PATTERNS = [
    r'\bcopy\b',           # "copy"
    r'\b복사\b',           # Korean "copy"
    r'\bのコピー\s*\d*\b',  # Japanese "copy" (のコピー, のコピー 2)
    r'\d+$',               # Trailing numbers
    r'^\d+\s*',            # Leading numbers
    r'[_\-]+$',            # Trailing underscores/dashes
    r'^[_\-]+',            # Leading underscores/dashes
]

# Meaningless default layer names (checked after cleaning + lowering)
MEANINGLESS_NAMES = {
    # English defaults
    'layer', 'group', 'shape', 'untitled',
    # Korean defaults
    '레이어', '그룹', '모양', '무제',
    # Japanese defaults
    'レイヤー', 'グループ', 'シェイプ',
    # Photoshop adjustment layers (EN)
    'levels', 'curves', 'brightness/contrast', 'hue/saturation',
    'color balance', 'black & white', 'photo filter', 'exposure',
    # Photoshop adjustment layers (JP)
    '明るさ・コントラスト', 'カラーバランス', '色相・彩度', 'トーンカーブ',
    # Single characters / symbols that carry no meaning
    '☆', '★', 'ｔ',
}

# Regex for "Name + number" patterns (Layer 42, グループ 8, etc.)
_NUMBERED_DEFAULT_RE = re.compile(
    r'^(?:layer|group|shape|レイヤー|グループ|シェイプ|'
    r'明るさ・コントラスト|カラーバランス|色相・彩度|'
    r'levels|curves)\s*\d*$',
    re.IGNORECASE,
)


def clean_layer_name(name: str) -> str:
    """
    Clean and normalize a layer name for semantic processing.
    
    Examples:
        "Layer 1 copy 2" -> "Layer"
        "Character_Body_01" -> "Character Body"
        "레이어 1 복사" -> ""  (completely meaningless)
    
    Args:
        name: Raw layer name from PSD
        
    Returns:
        Cleaned layer name, or empty string if meaningless
    """
    if not name:
        return ""
    
    cleaned = name.strip()
    
    # Apply noise patterns
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Replace underscores with spaces
    cleaned = cleaned.replace('_', ' ')
    
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Check if result is meaningless
    if cleaned == "":
        return ""
    if cleaned.lower() in MEANINGLESS_NAMES or cleaned in MEANINGLESS_NAMES:
        return ""
    if _NUMBERED_DEFAULT_RE.match(cleaned):
        return ""

    # Single character (except CJK ideographs which can be meaningful)
    if len(cleaned) == 1 and not ('\u4e00' <= cleaned <= '\u9fff'):
        return ""

    return cleaned


def is_meaningful_name(name: str) -> bool:
    """Check if a layer name carries semantic meaning."""
    cleaned = clean_layer_name(name)
    return len(cleaned) > 0


ContentType = Literal['background', 'character', 'object', 'effect', 'ui', 'text', 'unknown']


def infer_content_type(
    layer_name: str,
    canvas_size: Tuple[int, int],
    layer_size: Tuple[int, int],
    layer_position: Tuple[int, int],
    alpha_ratio: float = 1.0
) -> ContentType:
    """
    Infer the content type of a layer based on heuristics.
    
    This function attempts to classify layers even when names are meaningless,
    using geometric and visual properties.
    
    Args:
        layer_name: The layer name (may be cleaned or raw)
        canvas_size: (width, height) of the canvas
        layer_size: (width, height) of the layer
        layer_position: (left, top) position of the layer
        alpha_ratio: Ratio of non-transparent pixels (0.0 to 1.0)
        
    Returns:
        Inferred content type
    """
    canvas_w, canvas_h = canvas_size
    layer_w, layer_h = layer_size
    left, top = layer_position
    
    canvas_area = canvas_w * canvas_h
    layer_area = layer_w * layer_h
    
    if canvas_area == 0:
        return 'unknown'
    
    area_ratio = layer_area / canvas_area
    
    # Check name first (if meaningful)
    name_lower = layer_name.lower()
    
    # Name-based detection
    if any(kw in name_lower for kw in ['bg', 'background', '배경']):
        return 'background'
    if any(kw in name_lower for kw in ['char', 'character', '캐릭터', 'body', 'face']):
        return 'character'
    if any(kw in name_lower for kw in ['fx', 'effect', '효과', 'glow', 'shadow']):
        return 'effect'
    if any(kw in name_lower for kw in ['ui', 'button', '버튼', 'icon']):
        return 'ui'
    if any(kw in name_lower for kw in ['text', '텍스트', 'label']):
        return 'text'
    
    # Geometric heuristics (when name is meaningless)
    
    # Large layer covering most of canvas -> Background
    if area_ratio > 0.9:
        return 'background'
    
    # Very small layer with low alpha -> Effect
    if area_ratio < 0.05 and alpha_ratio < 0.3:
        return 'effect'
    
    # Medium sized layer near center -> Character/Object
    center_x = canvas_w / 2
    center_y = canvas_h / 2
    layer_center_x = left + layer_w / 2
    layer_center_y = top + layer_h / 2
    
    distance_from_center = (
        ((layer_center_x - center_x) ** 2 + (layer_center_y - center_y) ** 2) ** 0.5
    )
    max_distance = ((canvas_w ** 2 + canvas_h ** 2) ** 0.5) / 2
    
    if max_distance > 0 and distance_from_center / max_distance < 0.3:
        if area_ratio > 0.1:
            return 'character'
        else:
            return 'object'
    
    # Small layer at corners -> UI
    in_corner = (
        (left < canvas_w * 0.1 or left + layer_w > canvas_w * 0.9) and
        (top < canvas_h * 0.1 or top + layer_h > canvas_h * 0.9)
    )
    if in_corner and area_ratio < 0.1:
        return 'ui'
    
    return 'unknown'


def build_semantic_tags(layer_paths: list[str]) -> str:
    """
    Build a semantic tag string from layer paths for embedding.
    
    Args:
        layer_paths: List of layer paths like ["Group1/Layer1", "Group2/Layer2"]
        
    Returns:
        Space-separated cleaned names for embedding
    """
    cleaned_names = []
    
    for path in layer_paths:
        # Split path and clean each component
        parts = path.split('/')
        for part in parts:
            cleaned = clean_layer_name(part)
            if cleaned and cleaned not in cleaned_names:
                cleaned_names.append(cleaned)
    
    return ' '.join(cleaned_names)
