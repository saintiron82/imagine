"""
AssetMeta Schema - Unified data model for image asset metadata.

This schema defines the structure for all parsed image data (PSD, PNG, JPG)
and serves as the contract between parsers and the vector database.
"""

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime


class LayerInfo(BaseModel):
    """Individual layer information within a PSD file."""
    name: str
    path: str  # Full path like "Group1/SubGroup/LayerName"
    kind: Literal['group', 'pixel', 'type', 'shape', 'smartobject', 'adjustment', 'unknown']
    visible: bool = True
    opacity: float = 1.0  # 0.0 to 1.0
    position: Tuple[int, int] = (0, 0)  # (left, top)
    size: Tuple[int, int] = (0, 0)  # (width, height)
    has_clipping: bool = False
    text_content: Optional[str] = None  # For type layers only


class AssetMeta(BaseModel):
    """
    Main metadata model for image assets.

    This model is designed to be directly compatible with vector DB payload.
    All parsed images (PSD, PNG, JPG) produce this unified format.
    """

    # v3 P05: validate on assignment so VLM list/dict outputs get coerced
    # by _coerce_text_field even when set via meta.dominant_color = [...]
    model_config = ConfigDict(validate_assignment=True)

    # === File Information ===
    file_path: str = Field(..., description="Absolute path to the source file")
    file_name: str = Field(..., description="File name with extension")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Image format (lowercase canonical: psd/png/jpg)")
    resolution: Tuple[int, int] = Field(..., description="(width, height) in pixels")

    @field_validator("format", mode="before")
    @classmethod
    def _normalize_format(cls, v):
        """v3 P08: accept any case + leading dot, normalize to lowercase canonical."""
        if v is None:
            return v
        s = str(v).lower().lstrip(".")
        return "jpg" if s == "jpeg" else s

    @field_validator(
        "dominant_color", "ai_style", "color_palette",
        mode="before",
    )
    @classmethod
    def _coerce_text_field(cls, v):
        """v3 P05: VLM may return list/tuple for color/style fields. Join to string."""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v if x)
        if isinstance(v, dict):
            import json as _j
            return _j.dumps(v, ensure_ascii=False)
        return v
    
    # === Vector Source Fields ===
    # These fields are used to generate vectors in Phase 2
    visual_source_path: Optional[str] = Field(
        None, 
        description="Path to the thumbnail image for VV generation"
    )
    semantic_tags: str = Field(
        "",
        description="Cleaned layer names joined for MV generation"
    )
    text_content: List[str] = Field(
        default_factory=list,
        description="Text extracted from text layers or OCR"
    )

    # === AI-Generated Content (Phase 4 → v3.1: mc_caption) ===
    mc_caption: Optional[str] = Field(
        None,
        description="Meta-Context Caption: AI-generated caption with file metadata context"
    )
    ai_tags: List[str] = Field(
        default_factory=list,
        description="AI-extracted tags and keywords from image analysis"
    )
    ocr_text: Optional[str] = Field(
        None,
        description="Text extracted from image using OCR (separate from text layers)"
    )
    dominant_color: Optional[str] = Field(
        None,
        description="Dominant color in hex format (#RRGGBB)"
    )
    ai_style: Optional[str] = Field(
        None,
        description="AI-detected style description (e.g., 'minimalist, modern')"
    )

    # === v3 P0: 2-Stage Vision Classification ===
    image_type: Optional[str] = Field(
        None,
        description="AI-classified image type: character, background, ui_element, item, icon, texture, effect, logo, photo, illustration, other"
    )
    art_style: Optional[str] = Field(
        None,
        description="Art style: realistic, anime, pixel, painterly, cartoon, 3d_render, chibi, sketch, flat_design"
    )
    color_palette: Optional[str] = Field(
        None,
        description="Color palette: warm, cool, monochrome, vibrant, pastel, dark, neutral"
    )
    scene_type: Optional[str] = Field(
        None,
        description="Scene type for backgrounds: alley, forest, dungeon, castle, etc."
    )
    time_of_day: Optional[str] = Field(
        None,
        description="Time of day for backgrounds: dawn, morning, noon, sunset, night, etc."
    )
    weather: Optional[str] = Field(
        None,
        description="Weather for backgrounds: clear, rain, snow, fog, storm, etc."
    )
    character_type: Optional[str] = Field(
        None,
        description="Character subtype: human, monster, animal, robot, fantasy_creature, etc."
    )
    item_type: Optional[str] = Field(
        None,
        description="Item subtype: weapon, armor, potion, tool, accessory, etc."
    )
    ui_type: Optional[str] = Field(
        None,
        description="UI subtype: button, panel, hud, dialog_box, inventory, etc."
    )
    structured_meta: Optional[str] = Field(
        None,
        description="Full Stage 2 JSON output (preserved for flexible querying)"
    )

    # === Content Hash (path-independent file identification) ===
    content_hash: Optional[str] = Field(
        None,
        description="SHA256 content hash for path-independent file identification"
    )

    # === v3 P0: Path Abstraction ===
    storage_root: Optional[str] = Field(
        None,
        description="Storage root path (POSIX normalized)"
    )
    relative_path: Optional[str] = Field(
        None,
        description="Relative path from storage root (POSIX normalized)"
    )

    # === v3.1: Perceptual hash for de-duplication ===
    perceptual_hash: Optional[int] = Field(
        None,
        description="Perceptual hash (dHash) for near-duplicate detection"
    )
    dup_group_id: Optional[int] = Field(
        None,
        description="Duplicate group ID (files with similar perceptual hashes)"
    )

    # === v3 P0: Embedding Version Tracking ===
    embedding_model: Optional[str] = Field(
        None,
        description="Embedding model used for this file (set from active tier config)"
    )
    embedding_version: Optional[int] = Field(
        1,
        description="Embedding version number"
    )

    # === Structural Information (Payload) ===
    layer_tree: Dict = Field(
        default_factory=dict,
        description="Full layer hierarchy as nested dict"
    )
    layer_count: int = Field(0, description="Total number of layers")
    used_fonts: List[str] = Field(
        default_factory=list, 
        description="List of fonts used in text layers"
    )
    
    # === Management Information ===
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional metadata (author, project_tag, etc.)"
    )
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    parsed_at: datetime = Field(default_factory=datetime.now)
    
    # === Reference ===
    thumbnail_url: Optional[str] = Field(
        None,
        description="URL or path to the generated thumbnail"
    )

    # === Folder Discovery Information ===
    folder_path: Optional[str] = Field(
        None,
        description="Relative folder path from discovery root (e.g., 'Characters/Hero')"
    )
    folder_depth: int = Field(
        0,
        description="Depth of file relative to discovery root (0 = root level)"
    )
    folder_tags: List[str] = Field(
        default_factory=list,
        description="Folder names as searchable tags (e.g., ['Characters', 'Hero'])"
    )

    # === v3.1: 3-Tier AI Mode Metadata ===
    mode_tier: str = Field(
        "pro",
        description="AI mode tier: standard | pro | ultra"
    )
    caption_model: str = Field(
        "",
        description="VLM model used for captioning (e.g., Qwen/Qwen3.5-9B)"
    )
    text_embed_model: str = Field(
        "",
        description="MV model (e.g., qwen3-embedding:0.6b)"
    )
    runtime_version: str = Field(
        "",
        description="Ollama/runtime version (e.g., ollama-0.15.2)"
    )
    preprocess_params: Dict = Field(
        default_factory=dict,
        description="Preprocessing parameters: max_edge, aspect_ratio_mode, padding_color"
    )


class ParseResult(BaseModel):
    """Result wrapper for parser output."""
    success: bool
    asset_meta: Optional[AssetMeta] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
