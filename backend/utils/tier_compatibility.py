"""
Tier Compatibility Matrix

Defines compatibility rules between different AI tiers and
determines required actions for tier transitions.

v3.1.1 - Strict tier compatibility enforcement
"""

from typing import Dict, Any, Optional, Tuple
from enum import Enum


class TierAction(Enum):
    """Actions required for tier transition."""
    NONE = "none"                           # No action needed (same tier)
    REPROCESS_OPTIONAL = "reprocess_optional"  # Optional reprocess for quality
    REPROCESS_REQUIRED = "reprocess_required"  # Mandatory reprocess
    BLOCK = "block"                         # Block operation, user decision needed


class CompatibilityReason(Enum):
    """Reasons for compatibility status."""
    SAME_TIER = "same_tier"
    DIMENSION_UPGRADE = "dimension_upgrade"      # 768 → 1152 → 1664
    DIMENSION_DOWNGRADE = "dimension_downgrade"  # 1664 → 1152 → 768
    MODEL_CHANGE = "model_change"               # Different model family
    QUALITY_UPGRADE = "quality_upgrade"         # Better quality available
    QUALITY_DOWNGRADE = "quality_downgrade"     # Losing quality


# Tier specifications (from config.yaml)
TIER_SPECS = {
    'standard': {
        'visual_model': 'google/siglip2-base-patch16-224',
        'visual_dim': 768,
        'text_model': 'google/gemma-2b-instruct',
        'text_dim': 256,
        'vlm_model': 'Qwen/Qwen3-VL-2B-Instruct',
        'vram_req': '~6GB',
        'quality_tier': 1  # 1=basic, 2=medium, 3=high
    },
    'pro': {
        'visual_model': 'google/siglip2-so400m-patch14-384',
        'visual_dim': 1152,
        'text_model': 'qwen3-embedding:0.6b',
        'text_dim': 1024,
        'vlm_model': 'Qwen/Qwen3-VL-4B-Instruct',
        'vram_req': '8-16GB',
        'quality_tier': 2
    },
    'ultra': {
        'visual_model': 'google/siglip2-giant-opt-patch16-256',
        'visual_dim': 1664,
        'text_model': 'qwen3-embedding:8b',
        'text_dim': 4096,
        'vlm_model': 'qwen3-vl:8b',
        'vram_req': '20GB+',
        'quality_tier': 3
    }
}


# Compatibility Matrix: (from_tier, to_tier) → compatibility rules
TIER_COMPATIBILITY_MATRIX = {
    # ===== SAME TIER (No change) =====
    ('standard', 'standard'): {
        'compatible': True,
        'action': TierAction.NONE,
        'reason': CompatibilityReason.SAME_TIER,
        'message': 'No action required (same tier)',
        'user_prompt': None,
        'auto_allow': True
    },
    ('pro', 'pro'): {
        'compatible': True,
        'action': TierAction.NONE,
        'reason': CompatibilityReason.SAME_TIER,
        'message': 'No action required (same tier)',
        'user_prompt': None,
        'auto_allow': True
    },
    ('ultra', 'ultra'): {
        'compatible': True,
        'action': TierAction.NONE,
        'reason': CompatibilityReason.SAME_TIER,
        'message': 'No action required (same tier)',
        'user_prompt': None,
        'auto_allow': True
    },

    # ===== UPGRADES (Standard → Pro → Ultra) =====
    ('standard', 'pro'): {
        'compatible': False,
        'action': TierAction.REPROCESS_OPTIONAL,
        'reason': CompatibilityReason.QUALITY_UPGRADE,
        'message': 'Upgrading from Standard to Pro tier',
        'user_prompt': (
            'Upgrade to Pro tier detected:\n'
            '  • Better quality models (Qwen3-VL-4B, SigLIP-so400m)\n'
            '  • Dimension: 768 → 1152\n'
            '  • VRAM: ~6GB → 8-16GB\n\n'
            'Reprocess all files for improved quality?'
        ),
        'auto_allow': False  # Ask user
    },
    ('standard', 'ultra'): {
        'compatible': False,
        'action': TierAction.REPROCESS_OPTIONAL,
        'reason': CompatibilityReason.QUALITY_UPGRADE,
        'message': 'Upgrading from Standard to Ultra tier',
        'user_prompt': (
            'Upgrade to Ultra tier detected:\n'
            '  • Highest quality models (Qwen3-VL-8B, SigLIP-giant)\n'
            '  • Dimension: 768 → 1664\n'
            '  • VRAM: ~6GB → 20GB+\n\n'
            'Reprocess all files for maximum quality?'
        ),
        'auto_allow': False
    },
    ('pro', 'ultra'): {
        'compatible': False,
        'action': TierAction.REPROCESS_OPTIONAL,
        'reason': CompatibilityReason.QUALITY_UPGRADE,
        'message': 'Upgrading from Pro to Ultra tier',
        'user_prompt': (
            'Upgrade to Ultra tier detected:\n'
            '  • Highest quality models (Qwen3-VL-8B, SigLIP-giant)\n'
            '  • Dimension: 1152 → 1664\n'
            '  • VRAM: 8-16GB → 20GB+\n\n'
            'Reprocess all files for maximum quality?'
        ),
        'auto_allow': False
    },

    # ===== DOWNGRADES (Ultra → Pro → Standard) =====
    ('ultra', 'standard'): {
        'compatible': False,
        'action': TierAction.REPROCESS_REQUIRED,
        'reason': CompatibilityReason.DIMENSION_DOWNGRADE,
        'message': 'Downgrading from Ultra to Standard tier (MANDATORY REPROCESS)',
        'user_prompt': (
            '⚠️  Downgrade to Standard tier detected:\n'
            '  • Quality loss (Qwen3-VL-8B → Qwen3-VL-2B)\n'
            '  • Dimension: 1664 → 768 (cannot search existing data)\n'
            '  • VRAM: 20GB+ → ~6GB\n\n'
            'All files MUST be reprocessed.\n'
            'Existing embeddings will be deleted.\n\n'
            'Continue with downgrade?'
        ),
        'auto_allow': False  # Block and require confirmation
    },
    ('ultra', 'pro'): {
        'compatible': False,
        'action': TierAction.REPROCESS_REQUIRED,
        'reason': CompatibilityReason.DIMENSION_DOWNGRADE,
        'message': 'Downgrading from Ultra to Pro tier (MANDATORY REPROCESS)',
        'user_prompt': (
            '⚠️  Downgrade to Pro tier detected:\n'
            '  • Quality loss (Qwen3-VL-8B → Qwen3-VL-4B)\n'
            '  • Dimension: 1664 → 1152 (cannot search existing data)\n'
            '  • VRAM: 20GB+ → 8-16GB\n\n'
            'All files MUST be reprocessed.\n\n'
            'Continue with downgrade?'
        ),
        'auto_allow': False
    },
    ('pro', 'standard'): {
        'compatible': False,
        'action': TierAction.REPROCESS_REQUIRED,
        'reason': CompatibilityReason.DIMENSION_DOWNGRADE,
        'message': 'Downgrading from Pro to Standard tier (MANDATORY REPROCESS)',
        'user_prompt': (
            '⚠️  Downgrade to Standard tier detected:\n'
            '  • Quality loss (Qwen3-VL-4B → Qwen3-VL-2B)\n'
            '  • Dimension: 1152 → 768 (cannot search existing data)\n'
            '  • VRAM: 8-16GB → ~6GB\n\n'
            'All files MUST be reprocessed.\n\n'
            'Continue with downgrade?'
        ),
        'auto_allow': False
    },
}


def get_tier_spec(tier: str) -> Optional[Dict[str, Any]]:
    """Get specifications for a tier."""
    return TIER_SPECS.get(tier)


def get_compatibility(from_tier: str, to_tier: str) -> Dict[str, Any]:
    """
    Get compatibility rules for tier transition.

    Args:
        from_tier: Current/DB tier ('standard', 'pro', 'ultra')
        to_tier: Target tier

    Returns:
        Dict with compatibility rules:
            - compatible: bool
            - action: TierAction
            - reason: CompatibilityReason
            - message: str (short description)
            - user_prompt: str or None (message to show user)
            - auto_allow: bool (can proceed without user confirmation)
    """
    key = (from_tier, to_tier)

    if key not in TIER_COMPATIBILITY_MATRIX:
        # Unknown transition - be conservative
        return {
            'compatible': False,
            'action': TierAction.BLOCK,
            'reason': CompatibilityReason.MODEL_CHANGE,
            'message': f'Unknown tier transition: {from_tier} → {to_tier}',
            'user_prompt': f'Tier change detected ({from_tier} → {to_tier}). Reprocess required?',
            'auto_allow': False
        }

    return TIER_COMPATIBILITY_MATRIX[key]


def check_tier_transition(
    db_tier: Optional[str],
    current_tier: str,
    db_dimension: Optional[int],
    current_dimension: int
) -> Dict[str, Any]:
    """
    Check tier transition and determine required action.

    Args:
        db_tier: Tier from database (None if empty DB)
        current_tier: Current tier from config
        db_dimension: Embedding dimension from DB
        current_dimension: Expected dimension for current tier

    Returns:
        Dict with:
            - compatible: bool
            - action: str (TierAction value)
            - reason: str (CompatibilityReason value)
            - message: str
            - user_prompt: str or None
            - auto_allow: bool
            - db_tier: str or None
            - current_tier: str
            - db_dimension: int or None
            - current_dimension: int
    """
    # Empty DB - always compatible
    if db_tier is None:
        return {
            'compatible': True,
            'action': TierAction.NONE.value,
            'reason': CompatibilityReason.SAME_TIER.value,
            'message': 'Empty database - no migration needed',
            'user_prompt': None,
            'auto_allow': True,
            'db_tier': None,
            'current_tier': current_tier,
            'db_dimension': db_dimension,
            'current_dimension': current_dimension
        }

    # Get compatibility rules
    compat = get_compatibility(db_tier, current_tier)

    # Convert enums to strings
    result = {
        'compatible': compat['compatible'],
        'action': compat['action'].value,
        'reason': compat['reason'].value,
        'message': compat['message'],
        'user_prompt': compat['user_prompt'],
        'auto_allow': compat['auto_allow'],
        'db_tier': db_tier,
        'current_tier': current_tier,
        'db_dimension': db_dimension,
        'current_dimension': current_dimension
    }

    return result


def get_migration_steps(from_tier: str, to_tier: str) -> list:
    """
    Get step-by-step migration instructions.

    Args:
        from_tier: Source tier
        to_tier: Target tier

    Returns:
        List of migration steps (strings)
    """
    compat = get_compatibility(from_tier, to_tier)

    if compat['action'] == TierAction.NONE:
        return ["No migration needed"]

    steps = [
        f"1. Backup database:",
        f"   python backend/db/migrate_tier.py --tier {to_tier}",
        f"",
        f"2. Update config.yaml:",
        f"   ai_mode.override: {to_tier}",
        f"",
        f"3. Reprocess all files:",
        f"   python backend/pipeline/ingest_engine.py --discover \"path/to/images\""
    ]

    if compat['action'] == TierAction.REPROCESS_REQUIRED:
        steps.insert(0, "⚠️  MANDATORY REPROCESS - Existing embeddings are incompatible")
    elif compat['action'] == TierAction.REPROCESS_OPTIONAL:
        steps.insert(0, "💡 OPTIONAL REPROCESS - Recommended for quality improvement")

    return steps
