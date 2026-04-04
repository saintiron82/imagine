"""
Worker capability spec — hardware profile collected once at startup.

Sent to server on connect so scheduler can classify worker GPU class
and estimate initial batch sizes before actual speed measurements.

Separate from resource_monitor.py (real-time metrics for throttle).
This is a one-time snapshot of what the worker CAN do.
"""

import logging
import os
import platform
import socket
from typing import Dict, Any

logger = logging.getLogger(__name__)


def collect_capability() -> Dict[str, Any]:
    """Collect worker hardware capability spec.

    Returns:
        {
            "hostname": "my-pc",
            "os": "Darwin" | "Windows" | "Linux",
            "cpu_count": 10,
            "memory_total_gb": 32.0,
            "gpu_name": "Apple M5" | "NVIDIA RTX 4090" | "",
            "gpu_type": "mps" | "cuda" | "cpu",
            "vram_gb": 32.0,
            "is_metal": True | False,
            "compute_capability": "8.9" | None,
        }
    """
    spec = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "cpu_count": os.cpu_count() or 1,
        "memory_total_gb": _get_system_memory_gb(),
        "gpu_name": "",
        "gpu_type": "cpu",
        "vram_gb": 0.0,
        "is_metal": False,
        "compute_capability": None,
    }

    # Try GPU detection
    try:
        import torch
    except ImportError:
        return spec

    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            spec["gpu_name"] = props.name
            spec["gpu_type"] = "cuda"
            spec["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
            spec["compute_capability"] = f"{props.major}.{props.minor}"
        except Exception as e:
            logger.warning(f"CUDA detection failed: {e}")
        return spec

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        spec["gpu_type"] = "mps"
        spec["is_metal"] = True
        spec["gpu_name"] = _get_apple_chip_name()
        spec["vram_gb"] = _get_system_memory_gb()  # unified memory
        return spec

    return spec


def _get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    # Fallback: read from /proc or sysctl
    try:
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return round(int(result.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def _get_apple_chip_name() -> str:
    """Get Apple chip name (M1/M2/M3/M4/M5)."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        brand = result.stdout.strip()
        if brand:
            return f"Apple Silicon ({brand})"
    except Exception:
        pass
    return "Apple Silicon (MPS)"
