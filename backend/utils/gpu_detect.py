"""GPU VRAM Detection and Tier Selection Module.

Detects GPU VRAM capacity using PyTorch CUDA APIs and selects the optimal
AI model tier (standard/pro/ultra) based on available memory.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def determine_worker_mode(resources: Dict[str, Any], server_tier: str) -> str:
    """서버 tier 기준으로 워커의 processing_mode를 결정.

    워커 자체의 GPU 등급을 정하는 것이 아니라, 서버가 지정한 tier의 VLM을
    해당 워커가 실행할 수 있는지 판단하여 역할을 부여한다.

    Args:
        resources: 워커의 resources_json 딕셔너리
                   (gpu_type: "cuda"/"mps"/None, gpu_memory_total_gb: float 등)
        server_tier: 서버 활성 tier ("standard" / "pro" / "ultra")

    Returns:
        "full"       - 서버 tier VLM 실행 가능 → 전체 Phase P→V→VV→MV 처리
        "embed_only" - VLM 실행 불가 (VRAM 부족 또는 CPU-only) → Phase VV+MV만 처리

    Decision logic:
        1. gpu_type이 None (CPU-only) → embed_only
        2. 워커 VRAM ≥ 서버 tier vram_min → full
        3. 워커 VRAM < 서버 tier vram_min → embed_only

    Examples:
        # 서버 tier = pro (vram_min = 8192 MB)
        determine_worker_mode({"gpu_type": "cuda", "gpu_memory_total_gb": 12.0}, "pro")
        # → "full"   (12GB ≥ 8GB)

        determine_worker_mode({"gpu_type": "cuda", "gpu_memory_total_gb": 6.0}, "pro")
        # → "embed_only"  (6GB < 8GB)

        determine_worker_mode({"gpu_type": None}, "pro")
        # → "embed_only"  (CPU-only)

        determine_worker_mode({"gpu_type": "mps", "gpu_memory_total_gb": 16.0}, "pro")
        # → "full"   (M2 16GB ≥ 8GB)
    """
    from backend.utils.config import get_config

    gpu_type = resources.get("gpu_type")  # "cuda" / "mps" / None

    # GPU 없음 → VLM 실행 불가
    if not gpu_type:
        logger.info("Worker has no GPU (CPU-only) → embed_only mode")
        return "embed_only"

    worker_vram_mb = int(resources.get("gpu_memory_total_gb", 0) * 1024)

    # 서버 tier의 최소 VRAM 요구사항 조회
    cfg = get_config()
    tiers = cfg.get("ai_mode.tiers", {})
    tier_cfg = tiers.get(server_tier, {})
    vram_min_mb = tier_cfg.get("vram_min", 0)  # standard=없음(0), pro=8192, ultra=20480

    if worker_vram_mb >= vram_min_mb:
        logger.info(
            f"Worker VRAM {worker_vram_mb} MB ≥ {server_tier} tier min {vram_min_mb} MB"
            f" → full mode"
        )
        return "full"

    logger.info(
        f"Worker VRAM {worker_vram_mb} MB < {server_tier} tier min {vram_min_mb} MB"
        f" → embed_only mode"
    )
    return "embed_only"


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
