"""
Platform Detection and Optimal Backend Selection

Automatically determines the best vision backend based on:
- Operating system (Windows/Mac/Linux)
- Available software (vLLM, Ollama)
- Hardware capabilities (VRAM, GPU)

v3.1: Cross-platform optimization for ImageParser
- Windows: Ollama (sequential processing)
- Mac/Linux: vLLM or Ollama (batch processing capable)
"""

import platform
import shutil
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def get_platform_info() -> Dict[str, Any]:
    """
    Get detailed platform information.

    Returns:
        Dict with os, arch, python_version
    """
    return {
        "os": platform.system(),  # 'Windows', 'Darwin', 'Linux'
        "os_version": platform.version(),
        "arch": platform.machine(),  # 'AMD64', 'x86_64', 'arm64'
        "python_version": platform.python_version()
    }


def is_vllm_available() -> bool:
    """
    Check if vLLM is available on this system.

    vLLM is Linux/Mac only (requires Unix-like system).

    Returns:
        True if vLLM can be imported and used
    """
    system = platform.system()

    # vLLM is not supported on Windows
    if system == 'Windows':
        logger.debug("vLLM not available: Windows platform")
        return False

    # Check if vLLM is installed
    try:
        import vllm
        logger.debug(f"vLLM available: version {vllm.__version__}")
        return True
    except ImportError:
        logger.debug("vLLM not available: package not installed")
        return False


def is_ollama_available() -> bool:
    """
    Check if Ollama is available on this system.

    Returns:
        True if Ollama server is reachable
    """
    try:
        import requests
        from ..config import OLLAMA_HOST

        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        available = response.status_code == 200

        if available:
            logger.debug(f"Ollama available at {OLLAMA_HOST}")
        else:
            logger.debug(f"Ollama not responding: status {response.status_code}")

        return available

    except Exception as e:
        logger.debug(f"Ollama not available: {e}")
        return False


def get_optimal_backend(tier: str = 'ultra') -> str:
    """
    Determine optimal vision backend for current platform.

    Decision logic:
        Mac/Linux:
            1. vLLM (if available) - best batch performance
            2. Ollama (fallback) - stable but slower

        Windows:
            1. Ollama (only option) - sequential processing
            2. Transformers (fallback) - CPU-based

    Args:
        tier: AI tier name ('standard', 'pro', 'ultra')

    Returns:
        'vllm', 'ollama', or 'transformers'
    """
    system = platform.system()

    logger.info(f"Detecting optimal backend for {system} (tier: {tier})")

    # Mac/Linux: vLLM preferred for batch processing
    if system in ['Darwin', 'Linux']:
        if is_vllm_available():
            logger.info("✓ vLLM available - optimal batch processing")
            return 'vllm'
        elif is_ollama_available():
            logger.info("✓ Ollama available (vLLM not installed)")
            return 'ollama'
        else:
            logger.warning("⚠ Neither vLLM nor Ollama available, falling back to Transformers")
            return 'transformers'

    # Windows: Ollama only option for Qwen3-VL
    elif system == 'Windows':
        if is_ollama_available():
            logger.info("✓ Ollama available (vLLM not supported on Windows)")
            return 'ollama'
        else:
            logger.warning("⚠ Ollama not available, falling back to Transformers")
            return 'transformers'

    else:
        logger.warning(f"Unknown platform: {system}, using Transformers")
        return 'transformers'


def get_optimal_batch_size(backend: str, tier: str = 'ultra') -> int:
    """
    Determine optimal batch size for backend and tier.

    Batch size rules:
        vLLM:
            - Standard: 8
            - Pro: 12
            - Ultra: 16

        Ollama:
            - All tiers: 1 (Vision API doesn't benefit from batching)

        Transformers:
            - Standard: 20
            - Pro: 10
            - Ultra: 1 (if using heavy model)

    Args:
        backend: 'vllm', 'ollama', or 'transformers'
        tier: 'standard', 'pro', or 'ultra'

    Returns:
        Recommended batch size
    """
    if backend == 'vllm':
        # vLLM supports excellent batch processing
        batch_sizes = {
            'standard': 8,
            'pro': 12,
            'ultra': 16
        }
        return batch_sizes.get(tier, 16)

    elif backend == 'ollama':
        # Ollama Vision API performs worse with batching
        # Always use sequential processing
        return 1

    elif backend == 'transformers':
        # Transformers supports batch processing well
        batch_sizes = {
            'standard': 20,
            'pro': 10,
            'ultra': 1  # Heavy models need sequential
        }
        return batch_sizes.get(tier, 10)

    else:
        logger.warning(f"Unknown backend: {backend}, using batch_size=1")
        return 1


def get_platform_recommendations() -> Dict[str, Any]:
    """
    Get comprehensive platform recommendations.

    Returns:
        Dict with:
            - platform: OS name
            - optimal_backend: Recommended backend
            - optimal_batch_size: Recommended batch size
            - available_backends: List of available backends
            - warnings: List of warnings/recommendations
    """
    system = platform.system()
    optimal_backend = get_optimal_backend()
    optimal_batch_size = get_optimal_batch_size(optimal_backend)

    available = []
    if is_vllm_available():
        available.append('vllm')
    if is_ollama_available():
        available.append('ollama')
    available.append('transformers')  # Always available

    warnings = []

    # Platform-specific warnings
    if system == 'Windows' and optimal_backend == 'ollama':
        warnings.append(
            "Windows: Using Ollama (sequential processing). "
            "For faster batch processing, consider Mac/Linux with vLLM."
        )

    if optimal_backend == 'transformers':
        warnings.append(
            "Falling back to Transformers. "
            "For better performance, install Ollama or vLLM (Mac/Linux only)."
        )

    if system == 'Darwin' and 'vllm' not in available:
        warnings.append(
            "Mac detected: vLLM not installed. "
            "Install vLLM for 3-5x faster batch processing: pip install vllm"
        )

    return {
        "platform": system,
        "optimal_backend": optimal_backend,
        "optimal_batch_size": optimal_batch_size,
        "available_backends": available,
        "warnings": warnings
    }


def print_platform_info():
    """Print platform information and recommendations."""
    info = get_platform_info()
    recs = get_platform_recommendations()

    print("\n" + "="*70)
    print("PLATFORM INFORMATION")
    print("="*70)
    print(f"OS: {info['os']} {info['os_version']}")
    print(f"Architecture: {info['arch']}")
    print(f"Python: {info['python_version']}")

    print("\n" + "="*70)
    print("BACKEND RECOMMENDATIONS")
    print("="*70)
    print(f"Optimal Backend: {recs['optimal_backend']}")
    print(f"Optimal Batch Size: {recs['optimal_batch_size']}")
    print(f"Available Backends: {', '.join(recs['available_backends'])}")

    if recs['warnings']:
        print("\n" + "="*70)
        print("WARNINGS")
        print("="*70)
        for warning in recs['warnings']:
            print(f"⚠ {warning}")

    print("="*70 + "\n")


if __name__ == '__main__':
    # Test platform detection
    logging.basicConfig(level=logging.INFO)
    print_platform_info()
