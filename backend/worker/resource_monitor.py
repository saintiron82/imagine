"""
Resource Monitor — lightweight system metrics collector for worker heartbeat.

Collects CPU, system memory, and GPU metrics (CUDA/MPS) with full error
tolerance. Any individual metric that fails simply returns None for that field.

Also provides throttle_level judgement based on resource pressure:
    normal   → all clear
    warning  → approaching limits, should reduce batch size
    danger   → high pressure, should pause briefly
    critical → imminent OOM/thermal shutdown, must stop and unload

Called every heartbeat interval (~30s) so it must be fast and non-blocking.

Platform dispatch:
    - CUDA: torch.cuda API for memory, nvidia-smi subprocess for temperature
    - MPS (macOS): torch.mps API for allocated memory, IOKit not attempted
      (unreliable without root); temperature field returns None
    - CPU-only: GPU fields all None
"""

import logging
import platform
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Type alias for the metrics dictionary
MetricsDict = Dict[str, Any]

# Expected keys in the returned dict (for validation / documentation)
_METRIC_KEYS = (
    "cpu_percent",
    "memory_percent",
    "memory_used_gb",
    "memory_total_gb",
    "gpu_memory_used_gb",
    "gpu_memory_total_gb",
    "gpu_memory_percent",
    "gpu_temperature_c",
    "gpu_type",
)


def _empty_metrics() -> MetricsDict:
    """Return a metrics dict with all values set to None."""
    return {k: None for k in _METRIC_KEYS}


# ---------------------------------------------------------------------------
# CPU / System Memory (psutil)
# ---------------------------------------------------------------------------

def _collect_cpu_metrics(metrics: MetricsDict) -> None:
    """Populate cpu_percent, memory_percent, memory_used_gb, memory_total_gb."""
    try:
        import psutil
    except ImportError:
        logger.debug("psutil not installed — CPU/memory metrics unavailable")
        return

    try:
        # cpu_percent with interval=None returns since last call (non-blocking)
        metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
    except Exception as exc:
        logger.debug(f"Failed to read CPU percent: {exc}")

    try:
        vm = psutil.virtual_memory()
        metrics["memory_percent"] = vm.percent
        metrics["memory_used_gb"] = round(vm.used / (1024 ** 3), 2)
        metrics["memory_total_gb"] = round(vm.total / (1024 ** 3), 2)
    except Exception as exc:
        logger.debug(f"Failed to read system memory: {exc}")


# ---------------------------------------------------------------------------
# CUDA GPU
# ---------------------------------------------------------------------------

def _collect_cuda_metrics(metrics: MetricsDict) -> None:
    """Populate GPU fields via torch.cuda + nvidia-smi for temperature."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
    except ImportError:
        return

    metrics["gpu_type"] = "cuda"

    # Memory via torch.cuda
    try:
        mem_used = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        metrics["gpu_memory_used_gb"] = round(mem_used, 2)
        metrics["gpu_memory_total_gb"] = round(mem_total, 2)
        if mem_total > 0:
            metrics["gpu_memory_percent"] = round(mem_used / mem_total * 100, 1)
    except Exception as exc:
        logger.debug(f"Failed to read CUDA memory: {exc}")

    # Temperature via nvidia-smi (subprocess, ~50ms)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            temp = int(result.stdout.strip().split("\n")[0])
            metrics["gpu_temperature_c"] = temp
    except FileNotFoundError:
        logger.debug("nvidia-smi not found — GPU temperature unavailable")
    except Exception as exc:
        logger.debug(f"Failed to read GPU temperature via nvidia-smi: {exc}")


# ---------------------------------------------------------------------------
# MPS GPU (Apple Silicon)
# ---------------------------------------------------------------------------

def _collect_mps_metrics(metrics: MetricsDict) -> None:
    """Populate GPU fields via torch.mps on macOS Apple Silicon."""
    try:
        import torch
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return
    except ImportError:
        return

    metrics["gpu_type"] = "mps"

    # MPS memory — torch.mps provides allocated/driver allocated
    try:
        mem_allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
        metrics["gpu_memory_used_gb"] = round(mem_allocated, 2)

        # MPS has driver_allocated_memory (total allocated by Metal driver)
        try:
            mem_driver = torch.mps.driver_allocated_memory() / (1024 ** 3)
            metrics["gpu_memory_total_gb"] = round(mem_driver, 2)
            if mem_driver > 0:
                metrics["gpu_memory_percent"] = round(mem_allocated / mem_driver * 100, 1)
        except (AttributeError, RuntimeError):
            # driver_allocated_memory may not exist on older PyTorch
            pass
    except Exception as exc:
        logger.debug(f"Failed to read MPS memory: {exc}")

    # Temperature: IOKit requires root or entitlements on modern macOS;
    # not attempted to keep the module lightweight and permission-free.
    # gpu_temperature_c stays None for MPS.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_metrics() -> MetricsDict:
    """
    Collect system resource metrics for worker heartbeat reporting.

    Returns a dict with the following keys (any value may be None if
    the corresponding metric could not be collected):

        cpu_percent        : float  — CPU usage (0-100)
        memory_percent     : float  — System RAM usage (0-100)
        memory_used_gb     : float  — System RAM used (GB)
        memory_total_gb    : float  — System RAM total (GB)
        gpu_memory_used_gb : float  — GPU VRAM used (GB)
        gpu_memory_total_gb: float  — GPU VRAM total (GB)
        gpu_memory_percent : float  — GPU VRAM usage (0-100)
        gpu_temperature_c  : int    — GPU temperature (Celsius)
        gpu_type           : str    — "cuda" | "mps" | None

    This function never raises; all errors are caught and logged at DEBUG.
    """
    metrics = _empty_metrics()

    # 1. CPU / system memory
    _collect_cpu_metrics(metrics)

    # 2. GPU (try CUDA first, then MPS)
    system = platform.system()
    if system != "Darwin":
        # Non-macOS: try CUDA
        _collect_cuda_metrics(metrics)
    else:
        # macOS: try MPS first, then CUDA (for eGPU setups)
        _collect_mps_metrics(metrics)
        if metrics["gpu_type"] is None:
            _collect_cuda_metrics(metrics)

    return metrics


# ---------------------------------------------------------------------------
# Throttle Level
# ---------------------------------------------------------------------------

# Default thresholds — overridden by config.yaml > worker.throttle.thresholds
_DEFAULT_THRESHOLDS = {
    "gpu_memory": {"warning": 70, "danger": 85, "critical": 95},
    "cpu":        {"warning": 80, "danger": 90, "critical": 95},
    "memory":     {"warning": 75, "danger": 85, "critical": 95},
    "gpu_temp":   {"warning": 75, "danger": 85, "critical": 90},
}

# Ordered levels from best to worst (used for max-takes-all logic)
_LEVEL_ORDER = ("normal", "warning", "danger", "critical")
_LEVEL_RANK = {level: idx for idx, level in enumerate(_LEVEL_ORDER)}


def _load_thresholds() -> dict:
    """Load throttle thresholds from config.yaml, falling back to defaults."""
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        user_thresholds = cfg.get("worker", {}).get("throttle", {}).get("thresholds", {})
        if user_thresholds:
            merged = {}
            for key in _DEFAULT_THRESHOLDS:
                merged[key] = {**_DEFAULT_THRESHOLDS[key], **(user_thresholds.get(key, {}))}
            return merged
    except Exception:
        pass
    return _DEFAULT_THRESHOLDS


def _classify_metric(value: Optional[float], thresholds: dict) -> str:
    """Classify a single metric value against its thresholds.

    Args:
        value: Metric value (0-100 for percent, Celsius for temp). None = skip.
        thresholds: Dict with 'warning', 'danger', 'critical' keys.

    Returns:
        One of 'normal', 'warning', 'danger', 'critical'.
    """
    if value is None:
        return "normal"  # Missing metrics don't contribute to throttle

    if value >= thresholds["critical"]:
        return "critical"
    elif value >= thresholds["danger"]:
        return "danger"
    elif value >= thresholds["warning"]:
        return "warning"
    return "normal"


def get_throttle_level(metrics: MetricsDict) -> str:
    """Determine the throttle level based on collected resource metrics.

    Evaluates GPU memory, CPU, system memory, and GPU temperature against
    configurable thresholds. The worst (highest) level across all metrics
    becomes the overall throttle level.

    If GPU metrics are None (no GPU or unsupported), only CPU and system
    memory are considered.

    Args:
        metrics: Dict from collect_metrics().

    Returns:
        One of 'normal', 'warning', 'danger', 'critical'.
    """
    thresholds = _load_thresholds()

    levels = []

    # GPU memory (percent)
    levels.append(_classify_metric(
        metrics.get("gpu_memory_percent"),
        thresholds["gpu_memory"],
    ))

    # CPU percent
    levels.append(_classify_metric(
        metrics.get("cpu_percent"),
        thresholds["cpu"],
    ))

    # System memory percent
    levels.append(_classify_metric(
        metrics.get("memory_percent"),
        thresholds["memory"],
    ))

    # GPU temperature (Celsius)
    levels.append(_classify_metric(
        metrics.get("gpu_temperature_c"),
        thresholds["gpu_temp"],
    ))

    # Max-takes-all: worst level wins
    worst = "normal"
    worst_rank = 0
    for level in levels:
        rank = _LEVEL_RANK[level]
        if rank > worst_rank:
            worst_rank = rank
            worst = level

    return worst


# ---------------------------------------------------------------------------
# CLI quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)
    result = collect_metrics()
    level = get_throttle_level(result)
    result["_throttle_level"] = level
    print(json.dumps(result, indent=2))
