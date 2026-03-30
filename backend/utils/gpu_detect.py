"""GPU VRAM Detection and Tier Selection Module.

Detects GPU VRAM capacity using PyTorch CUDA APIs and selects the optimal
AI model tier (standard/pro/ultra) based on available memory.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def determine_worker_mode(resources: Dict[str, Any], server_tier: str) -> str:
    """서버 tier 기준으로 워커의 단일 역할(processing_mode)을 결정.

    워커는 하나의 역할만 반복 수행하여 모델 교체 비용을 제거한다.
    서버 임베디드 워커만 전체 파이프라인(full)을 수행할 수 있다.

    Args:
        resources: 워커의 resources_json 딕셔너리
                   (gpu_type: "cuda"/"mps"/None, gpu_memory_total_gb: float 등)
        server_tier: 서버 활성 tier ("standard" / "pro" / "ultra")

    Returns:
        "mc"  - VLM 구동 가능 → MC(캡션/태그) 생성만 반복 (VLM 상주)
        "vv"  - GPU 있지만 VLM 불가 → VV(SigLIP2 시각 벡터)만 반복
        "mv"  - CPU만 → MV(Qwen3-Embedding 의미 벡터)만 반복

    Note:
        "full" 모드는 서버 임베디드 워커 전용. 외부 워커에는 배정하지 않는다.
        임베디드 워커는 모든 외부 워커가 안 하는 나머지 phase를 보완 처리한다.

    Decision logic:
        1. gpu_type이 None (CPU-only) → "mv" (텍스트 임베딩은 CPU로도 가능)
        2. 워커 VRAM ≥ 서버 tier vram_min → "mc" (가장 가치 높은 GPU 작업)
        3. 워커 VRAM < 서버 tier vram_min → "vv" (SigLIP2는 2GB면 충분)
    """
    from backend.utils.config import get_config

    gpu_type = resources.get("gpu_type")  # "cuda" / "mps" / None

    # GPU 없음 → MV만 (텍스트 임베딩, CPU 가능)
    if not gpu_type:
        logger.info("Worker has no GPU (CPU-only) → mv mode")
        return "mv"

    worker_vram_mb = int((resources.get("gpu_memory_total_gb") or 0) * 1024)

    # 서버 tier의 최소 VRAM 요구사항 조회
    cfg = get_config()
    tiers = cfg.get("ai_mode.tiers", {})
    tier_cfg = tiers.get(server_tier, {})
    vram_min_mb = tier_cfg.get("vram_min", 0)  # standard=없음(0), pro=8192, ultra=20480
    # 10% margin: allow workers slightly below threshold to still do MC
    vram_threshold = int(vram_min_mb * 0.90)

    if worker_vram_mb >= vram_threshold:
        logger.info(
            f"Worker VRAM {worker_vram_mb} MB ≥ {vram_threshold} MB "
            f"(tier {server_tier} min {vram_min_mb}, 10% margin) → mc mode"
        )
        return "mc"

    logger.info(
        f"Worker VRAM {worker_vram_mb} MB < {vram_threshold} MB "
        f"(tier {server_tier} min {vram_min_mb}) → vv mode"
    )
    return "vv"


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
    except Exception as e:
        logger.warning(f"PyTorch unavailable ({type(e).__name__}), cannot detect GPU")
        return 0

    # CUDA GPU
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            vram_bytes = torch.cuda.get_device_properties(device).total_memory
            vram_mb = vram_bytes // (1024 * 1024)

            gpu_name = torch.cuda.get_device_name(device)
            logger.info(f"GPU detected: {gpu_name}, VRAM: {vram_mb} MB")

            return vram_mb

        except Exception as e:
            logger.error(f"Failed to detect CUDA VRAM: {e}")
            return 0

    # Apple Silicon MPS — unified memory (system RAM = GPU memory)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        import os
        try:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            vram_mb = total_bytes // (1024 * 1024)
            logger.info(f"MPS GPU detected (Apple Silicon), unified memory: {vram_mb} MB")
            return vram_mb
        except (ValueError, OSError):
            try:
                import psutil
                vram_mb = psutil.virtual_memory().total // (1024 * 1024)
                logger.info(f"MPS GPU detected (Apple Silicon), system memory: {vram_mb} MB")
                return vram_mb
            except Exception:
                pass
        logger.warning("MPS available but failed to detect memory")
        return 0

    logger.info("No CUDA/MPS GPU detected (CPU-only mode)")
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

    # CUDA
    if torch.cuda.is_available():
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

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        vram_mb = get_gpu_vram_mb()
        return {
            "cuda_available": False,
            "mps_available": True,
            "name": "Apple Silicon (MPS)",
            "vram_mb": vram_mb,
        }

    return {
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0
    }
