"""Tier Configuration Loader.

Manages AI model tier selection (standard/pro/ultra) with automatic VRAM
detection and manual override support.
"""

import logging
from typing import Dict, Any, Tuple

from backend.utils.config import get_config
from backend.utils.gpu_detect import get_gpu_vram_mb, select_tier

logger = logging.getLogger(__name__)


def get_active_tier() -> Tuple[str, Dict[str, Any]]:
    """
    현재 활성 티어 및 설정 반환.

    Returns:
        tuple[str, dict]: (tier_name, tier_config)
            - tier_name: "standard" | "pro" | "ultra"
            - tier_config: 해당 티어의 전체 설정 dict

    Selection Priority:
        1. Manual override (config: ai_mode.override)
        2. Auto-detection (config: ai_mode.auto_detect = true)
        3. Default fallback ("pro")

    Examples:
        >>> tier_name, tier_config = get_active_tier()
        >>> print(f"Active tier: {tier_name}")
        >>> print(f"Visual model: {tier_config['visual']['model']}")
        Active tier: pro
        Visual model: google/siglip2-so400m-patch14-384
    """
    cfg = get_config()

    # Priority 1: 수동 오버라이드 확인
    override = cfg.get("ai_mode.override")
    if override:
        logger.info(f"Manual tier override: {override}")
        tier_name = override
    else:
        # Priority 2: 자동 감지
        auto_detect = cfg.get("ai_mode.auto_detect", True)
        if auto_detect:
            vram_mb = get_gpu_vram_mb()
            tier_name = select_tier(vram_mb, cfg)
            logger.info(f"Auto-detected tier: {tier_name} (VRAM: {vram_mb} MB)")
        else:
            # Priority 3: 기본값
            tier_name = "pro"
            logger.info(f"Auto-detect disabled, using default tier: {tier_name}")

    # 티어 설정 로드
    tiers = cfg.get("ai_mode.tiers", {})
    tier_config = tiers.get(tier_name)

    if not tier_config:
        logger.warning(f"Tier '{tier_name}' not found in config, falling back to 'pro'")
        tier_name = "pro"
        tier_config = tiers.get("pro", {})

    if not tier_config:
        logger.error("No valid tier configuration found in config.yaml")
        raise ValueError("Invalid ai_mode.tiers configuration")

    logger.debug(f"Loaded tier config: {tier_name} → {list(tier_config.keys())}")
    return tier_name, tier_config


def get_tier_info(tier_name: str = None) -> Dict[str, Any]:
    """
    특정 티어의 상세 정보 반환 (디버깅용).

    Args:
        tier_name: "standard" | "pro" | "ultra" (None = 현재 활성 티어)

    Returns:
        dict: 티어 정보 요약

    Examples:
        >>> info = get_tier_info("pro")
        >>> print(info)
        {
            'tier': 'pro',
            'visual_model': 'google/siglip2-so400m-patch14-384',
            'vlm_model': 'Qwen/Qwen3-VL-4B-Instruct',
            'text_embed_model': 'qwen3-embedding:0.6b',
            'max_edge': 768,
            'vram_range': '8192-16384 MB'
        }
    """
    if tier_name is None:
        tier_name, tier_config = get_active_tier()
    else:
        cfg = get_config()
        tiers = cfg.get("ai_mode.tiers", {})
        tier_config = tiers.get(tier_name, {})

    if not tier_config:
        return {"error": f"Tier '{tier_name}' not found"}

    # VRAM 범위 계산
    vram_min = tier_config.get("vram_min", 0)
    vram_max = tier_config.get("vram_max", float('inf'))
    if vram_max == float('inf'):
        vram_range = f"≥{vram_min} MB"
    elif vram_min == 0:
        vram_range = f"≤{vram_max} MB"
    else:
        vram_range = f"{vram_min}-{vram_max} MB"

    return {
        "tier": tier_name,
        "visual_model": tier_config.get("visual", {}).get("model"),
        "visual_dimensions": tier_config.get("visual", {}).get("dimensions"),
        "vlm_model": tier_config.get("vlm", {}).get("model"),
        "vlm_backend": tier_config.get("vlm", {}).get("backend"),
        "text_embed_model": tier_config.get("text_embed", {}).get("model"),
        "text_embed_dimensions": tier_config.get("text_embed", {}).get("dimensions"),
        "max_edge": tier_config.get("preprocess", {}).get("max_edge"),
        "aspect_ratio_mode": tier_config.get("preprocess", {}).get("aspect_ratio_mode"),
        "vram_range": vram_range,
    }


def list_all_tiers() -> Dict[str, Dict[str, Any]]:
    """
    모든 티어의 정보 반환 (UI용).

    Returns:
        dict: {tier_name: tier_info, ...}

    Examples:
        >>> tiers = list_all_tiers()
        >>> for name, info in tiers.items():
        ...     print(f"{name}: {info['vram_range']}")
        standard: ≤6144 MB
        pro: 8192-16384 MB
        ultra: ≥20480 MB
    """
    cfg = get_config()
    tiers = cfg.get("ai_mode.tiers", {})

    return {
        tier_name: get_tier_info(tier_name)
        for tier_name in tiers.keys()
    }
