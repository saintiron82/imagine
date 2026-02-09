"""GPU VRAM Detection and Tier Selection Module.

Detects GPU VRAM capacity using PyTorch CUDA APIs and selects the optimal
AI model tier (standard/pro/ultra) based on available memory.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_gpu_vram_mb() -> int:
    """
    GPU VRAM 용량을 MB 단위로 반환.

    Returns:
        int: VRAM (MB), GPU 없으면 0

    Examples:
        >>> vram = get_gpu_vram_mb()
        >>> print(f"VRAM: {vram} MB")
        VRAM: 8192 MB
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, cannot detect GPU")
        return 0

    if not torch.cuda.is_available():
        logger.info("No CUDA GPU detected (CPU-only mode)")
        return 0

    try:
        device = torch.cuda.current_device()
        vram_bytes = torch.cuda.get_device_properties(device).total_memory
        vram_mb = vram_bytes // (1024 * 1024)

        gpu_name = torch.cuda.get_device_name(device)
        logger.info(f"GPU detected: {gpu_name}, VRAM: {vram_mb} MB")

        return vram_mb

    except Exception as e:
        logger.error(f"Failed to detect VRAM: {e}")
        return 0


def select_tier(vram_mb: int, config: Dict[str, Any]) -> str:
    """
    VRAM 용량에 따라 최적 티어 선택.

    Args:
        vram_mb: VRAM (MB)
        config: config.yaml의 전체 설정 dict

    Returns:
        str: "standard" | "pro" | "ultra"

    Selection Logic:
        - 0 MB (CPU-only): standard
        - ≤ 6144 MB (≤6GB): standard
        - 6145-16384 MB (6-16GB): pro
        - ≥ 16385 MB (>16GB): ultra

    Examples:
        >>> tier = select_tier(8192, config)
        >>> print(tier)
        'pro'
    """
    tiers = config.get("ai_mode", {}).get("tiers", {})

    # VRAM 기준 자동 선택
    if vram_mb == 0:
        logger.info("CPU-only mode detected → selecting 'standard' tier")
        return "standard"

    # Standard tier threshold
    standard_max = tiers.get("standard", {}).get("vram_max", 6144)
    if vram_mb <= standard_max:
        logger.info(f"VRAM {vram_mb} MB ≤ {standard_max} MB → 'standard' tier")
        return "standard"

    # Pro tier threshold
    pro_max = tiers.get("pro", {}).get("vram_max", 16384)
    if vram_mb <= pro_max:
        logger.info(f"VRAM {vram_mb} MB ≤ {pro_max} MB → 'pro' tier")
        return "pro"

    # Ultra tier (>16GB)
    logger.info(f"VRAM {vram_mb} MB > {pro_max} MB → 'ultra' tier")
    return "ultra"


def get_gpu_info() -> Dict[str, Any]:
    """
    GPU 상세 정보 반환 (디버깅용).

    Returns:
        dict: GPU 정보 (name, vram_mb, compute_capability, cuda_available)
    """
    try:
        import torch
    except ImportError:
        return {
            "cuda_available": False,
            "error": "PyTorch not installed"
        }

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_count": 0
        }

    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        return {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": device,
            "name": props.name,
            "vram_mb": props.total_memory // (1024 * 1024),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }

    except Exception as e:
        return {
            "cuda_available": True,
            "error": str(e)
        }
