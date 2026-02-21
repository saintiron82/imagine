"""
Tier Compatibility Matrix

Defines compatibility rules between different AI tiers and
determines required actions for tier transitions.

v3.1.1 - Strict tier compatibility enforcement
v6.7   - Unified VV model (so400m-naflex) + MV unification (standard/pro)
         standard↔pro: fully compatible (same VV + MV models)
         ultra: MV-only reprocess (different text embedding model)
"""

from typing import Dict, Any, Optional, Tuple
from enum import Enum


class TierAction(Enum):
    """Actions required for tier transition."""
    NONE = "none"                           # No action needed (same tier)
    REPROCESS_OPTIONAL = "reprocess_optional"  # Optional reprocess for quality
    REPROCESS_REQUIRED = "reprocess_required"  # Mandatory reprocess
    MV_ONLY = "mv_only"                     # Only MV reprocess (VV compatible)
    BLOCK = "block"                         # Block operation, user decision needed


class CompatibilityReason(Enum):
    """Reasons for compatibility status."""
    SAME_TIER = "same_tier"
    SAME_MODELS = "same_models"                 # VV+MV models identical across tiers
    DIMENSION_UPGRADE = "dimension_upgrade"      # 768 → 1152 → 1664
    DIMENSION_DOWNGRADE = "dimension_downgrade"  # 1664 → 1152 → 768
    MODEL_CHANGE = "model_change"               # Different model family
    MV_MODEL_CHANGE = "mv_model_change"         # Only MV model differs (VV same)
    QUALITY_UPGRADE = "quality_upgrade"         # Better quality available
    QUALITY_DOWNGRADE = "quality_downgrade"     # Losing quality


# Tier specifications (from config.yaml)
# v6.7: VV unified to so400m-naflex (1152d) across all tiers
# v6.7: MV unified to 0.6b/1024d for standard+pro, ultra uses 8b/4096d
TIER_SPECS = {
    'standard': {
        'visual_model': 'google/siglip2-so400m-patch16-naflex',
        'visual_dim': 1152,
        'text_model': 'qwen3-embedding:0.6b',
        'text_dim': 1024,
        'vlm_model': 'Qwen/Qwen3-VL-2B-Instruct',
        'vram_req': '~6GB',
        'quality_tier': 1  # 1=basic, 2=medium, 3=high
    },
    'pro': {
        'visual_model': 'google/siglip2-so400m-patch16-naflex',
        'visual_dim': 1152,
        'text_model': 'qwen3-embedding:0.6b',
        'text_dim': 1024,
        'vlm_model': 'Qwen/Qwen3-VL-4B-Instruct',
        'vram_req': '8-16GB',
        'quality_tier': 2
    },
    'ultra': {
        'visual_model': 'google/siglip2-so400m-patch16-naflex',
        'visual_dim': 1152,
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

    # ===== STANDARD ↔ PRO (Fully compatible — same VV + MV models) =====
    ('standard', 'pro'): {
        'compatible': True,
        'action': TierAction.NONE,
        'reason': CompatibilityReason.SAME_MODELS,
        'message': 'Standard ↔ Pro: fully compatible (same VV/MV models)',
        'user_prompt': None,
        'auto_allow': True
    },
    ('pro', 'standard'): {
        'compatible': True,
        'action': TierAction.NONE,
        'reason': CompatibilityReason.SAME_MODELS,
        'message': 'Pro ↔ Standard: fully compatible (same VV/MV models)',
        'user_prompt': None,
        'auto_allow': True
    },

    # ===== STANDARD/PRO ↔ ULTRA (MV-only reprocess — VV compatible) =====
    ('standard', 'ultra'): {
        'compatible': False,
        'action': TierAction.MV_ONLY,
        'reason': CompatibilityReason.MV_MODEL_CHANGE,
        'message': 'Upgrading to Ultra tier (MV reprocess only)',
        'user_prompt': (
            'Upgrade to Ultra tier detected:\n'
            '  • VV (visual): compatible (same SigLIP2 model)\n'
            '  • MV (text): reprocess needed (0.6B → 8B model)\n'
            '  • VLM: Qwen3-VL-2B → Qwen3-VL-8B\n\n'
            'Only MV embeddings need reprocessing (fast, ~0.5s/file).\n'
            'VV embeddings and MC captions are preserved.\n\n'
            'Reprocess MV for Ultra tier?'
        ),
        'auto_allow': False
    },
    ('pro', 'ultra'): {
        'compatible': False,
        'action': TierAction.MV_ONLY,
        'reason': CompatibilityReason.MV_MODEL_CHANGE,
        'message': 'Upgrading to Ultra tier (MV reprocess only)',
        'user_prompt': (
            'Upgrade to Ultra tier detected:\n'
            '  • VV (visual): compatible (same SigLIP2 model)\n'
            '  • MV (text): reprocess needed (0.6B → 8B model)\n'
            '  • VLM: Qwen3-VL-4B → Qwen3-VL-8B\n\n'
            'Only MV embeddings need reprocessing (fast, ~0.5s/file).\n'
            'VV embeddings and MC captions are preserved.\n\n'
            'Reprocess MV for Ultra tier?'
        ),
        'auto_allow': False
    },
    ('ultra', 'pro'): {
        'compatible': False,
        'action': TierAction.MV_ONLY,
        'reason': CompatibilityReason.MV_MODEL_CHANGE,
        'message': 'Downgrading from Ultra to Pro (MV reprocess only)',
        'user_prompt': (
            'Downgrade to Pro tier detected:\n'
            '  • VV (visual): compatible (same SigLIP2 model)\n'
            '  • MV (text): reprocess needed (8B → 0.6B model)\n'
            '  • VLM: Qwen3-VL-8B → Qwen3-VL-4B\n\n'
            'Only MV embeddings need reprocessing (fast, ~0.5s/file).\n'
            'VV embeddings and MC captions are preserved.\n\n'
            'Reprocess MV for Pro tier?'
        ),
        'auto_allow': False
    },
    ('ultra', 'standard'): {
        'compatible': False,
        'action': TierAction.MV_ONLY,
        'reason': CompatibilityReason.MV_MODEL_CHANGE,
        'message': 'Downgrading from Ultra to Standard (MV reprocess only)',
        'user_prompt': (
            'Downgrade to Standard tier detected:\n'
            '  • VV (visual): compatible (same SigLIP2 model)\n'
            '  • MV (text): reprocess needed (8B → 0.6B model)\n'
            '  • VLM: Qwen3-VL-8B → Qwen3-VL-2B\n\n'
            'Only MV embeddings need reprocessing (fast, ~0.5s/file).\n'
            'VV embeddings and MC captions are preserved.\n\n'
            'Reprocess MV for Standard tier?'
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
    elif compat['action'] == TierAction.MV_ONLY:
        steps.insert(0, "📝 MV-ONLY REPROCESS - VV embeddings and MC captions are preserved")
    elif compat['action'] == TierAction.REPROCESS_OPTIONAL:
        steps.insert(0, "💡 OPTIONAL REPROCESS - Recommended for quality improvement")

    return steps
