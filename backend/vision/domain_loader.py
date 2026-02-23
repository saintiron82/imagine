"""
Domain Profile Loader for Domain-Aware Classification System.

Loads domain profiles from YAML files, merges with base defaults and
user customizations from config.yaml.

Usage:
    from backend.vision.domain_loader import get_active_domain

    domain = get_active_domain()
    print(domain.image_types)       # ["character", "background", ...]
    print(domain.get_type_hints("background"))  # {"scene_type": [...], ...}
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from backend.utils.config import get_config

logger = logging.getLogger(__name__)

_DOMAINS_DIR = Path(__file__).parent / "domains"

# Default image types when no domain is configured (legacy behavior)
_DEFAULT_IMAGE_TYPES = [
    "character", "background", "ui_element", "item", "icon",
    "texture", "effect", "logo", "photo", "illustration", "other",
]


@dataclass
class DomainProfile:
    """Represents a loaded and merged domain profile."""

    id: str
    name: str
    name_ko: str
    description: str
    image_types: List[str]
    type_hints: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    type_instructions: Dict[str, str] = field(default_factory=dict)
    common_hints: Dict[str, List[str]] = field(default_factory=dict)

    def get_type_hints(self, image_type: str) -> Dict[str, List[str]]:
        """Get merged hints for a specific image_type (type-specific + common)."""
        hints = dict(self.type_hints.get(image_type, {}))
        # Merge common hints for fields not already defined
        for key, values in self.common_hints.items():
            if key not in hints:
                hints[key] = values
        return hints

    def get_type_instruction(self, image_type: str) -> Optional[str]:
        """Get extra prompt instruction for a specific image_type."""
        return self.type_instructions.get(image_type)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file safely."""
    if not path.exists():
        logger.warning(f"Domain file not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load domain file {path}: {e}")
        return {}


def _load_base() -> dict:
    """Load _base.yaml common defaults."""
    return _load_yaml(_DOMAINS_DIR / "_base.yaml")


def _apply_custom_overrides(
    image_types: List[str],
    type_hints: Dict[str, Dict[str, List[str]]],
    overrides: dict,
) -> tuple:
    """
    Apply user customizations from config.yaml classification.custom_overrides.

    Supports:
      - extra_image_types: add types to the domain's list
      - remove_image_types: remove types from the domain's list
      - type_hints: override specific hint lists per type
    """
    if not overrides:
        return image_types, type_hints

    # Add extra image types
    extra = overrides.get("extra_image_types", [])
    if extra:
        image_types = list(image_types) + [t for t in extra if t not in image_types]

    # Remove image types
    remove = overrides.get("remove_image_types", [])
    if remove:
        image_types = [t for t in image_types if t not in remove]

    # Override type hints
    hint_overrides = overrides.get("type_hints", {})
    if hint_overrides:
        type_hints = copy.deepcopy(type_hints)
        for img_type, field_hints in hint_overrides.items():
            if img_type not in type_hints:
                type_hints[img_type] = {}
            for field_name, values in field_hints.items():
                type_hints[img_type][field_name] = list(values)

    return image_types, type_hints


def load_domain(domain_id: str) -> DomainProfile:
    """
    Load a domain profile by ID.

    1. Load _base.yaml (common hints)
    2. Load {domain_id}.yaml
    3. Apply config.yaml custom_overrides
    4. Return merged DomainProfile

    Args:
        domain_id: Domain identifier (e.g., "game_asset", "illustration")

    Returns:
        DomainProfile instance
    """
    # Load base
    base_data = _load_base()
    common_hints = base_data.get("common_hints", {})

    # Load domain YAML
    domain_path = _DOMAINS_DIR / f"{domain_id}.yaml"
    data = _load_yaml(domain_path)

    if not data:
        logger.warning(f"Domain '{domain_id}' not found, using defaults")
        return DomainProfile(
            id=domain_id,
            name=domain_id,
            name_ko=domain_id,
            description="",
            image_types=list(_DEFAULT_IMAGE_TYPES),
            common_hints=common_hints,
        )

    domain_meta = data.get("domain", {})
    image_types = data.get("image_types", list(_DEFAULT_IMAGE_TYPES))
    type_hints = data.get("type_hints", {})
    type_instructions = data.get("type_instructions", {})

    # Apply user custom overrides from config.yaml
    cfg = get_config()
    overrides = cfg.get("classification.custom_overrides")
    if overrides:
        image_types, type_hints = _apply_custom_overrides(
            image_types, type_hints, overrides
        )

    return DomainProfile(
        id=domain_meta.get("id", domain_id),
        name=domain_meta.get("name", domain_id),
        name_ko=domain_meta.get("name_ko", domain_id),
        description=domain_meta.get("description", ""),
        image_types=image_types,
        type_hints=type_hints,
        type_instructions=type_instructions,
        common_hints=common_hints,
    )


def get_active_domain() -> Optional[DomainProfile]:
    """
    Load the currently active domain from config.yaml.

    Reads config.yaml > classification.active_domain.
    Returns None if no domain is configured (legacy mode).

    Returns:
        DomainProfile or None
    """
    cfg = get_config()
    domain_id = cfg.get("classification.active_domain")

    if not domain_id:
        logger.debug("No active domain configured, using legacy mode")
        return None

    logger.info(f"Loading active domain: {domain_id}")
    return load_domain(domain_id)


def list_available_domains() -> List[Dict[str, Any]]:
    """
    List all available domain presets.

    Scans backend/vision/domains/ for YAML files (excluding _base.yaml).

    Returns:
        List of dicts with id, name, name_ko, description, image_types_count
    """
    domains = []
    if not _DOMAINS_DIR.exists():
        return domains

    for path in sorted(_DOMAINS_DIR.glob("*.yaml")):
        if path.stem.startswith("_"):
            continue

        data = _load_yaml(path)
        if not data:
            continue

        meta = data.get("domain", {})
        domains.append({
            "id": meta.get("id", path.stem),
            "name": meta.get("name", path.stem),
            "name_ko": meta.get("name_ko", path.stem),
            "description": meta.get("description", ""),
            "image_types_count": len(data.get("image_types", [])),
            "image_types": data.get("image_types", []),
        })

    return domains
