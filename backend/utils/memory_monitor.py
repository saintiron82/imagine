"""
Runtime memory monitoring for adaptive batch pipeline.

Tracks process RSS and GPU memory to determine memory pressure level.
Designed for macOS Apple Silicon (Unified Memory) but supports CUDA too.

v3.4: Used by AdaptiveBatchController to dynamically adjust batch sizes.

Usage:
    monitor = MemoryMonitor.get_instance()
    monitor.set_baseline()
    snap = monitor.snapshot()
    print(f"Pressure: {snap.pressure_ratio:.0%}")
"""

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Point-in-time memory state."""
    timestamp: float
    rss_bytes: int
    system_available_bytes: int
    system_total_bytes: int
    gpu_allocated_bytes: int = 0
    gpu_driver_bytes: int = 0

    @property
    def rss_gb(self) -> float:
        return self.rss_bytes / (1024 ** 3)

    @property
    def gpu_allocated_gb(self) -> float:
        return self.gpu_allocated_bytes / (1024 ** 3)

    @property
    def gpu_driver_gb(self) -> float:
        return self.gpu_driver_bytes / (1024 ** 3)

    @property
    def system_available_gb(self) -> float:
        return self.system_available_bytes / (1024 ** 3)

    @property
    def system_total_gb(self) -> float:
        return self.system_total_bytes / (1024 ** 3)

    @property
    def pressure_ratio(self) -> float:
        """
        Memory pressure as 0.0 (free) to 1.0 (full).

        Apple Silicon Unified Memory: RSS includes GPU allocations,
        so use max(rss, gpu_driver) to avoid double-counting.
        """
        if self.system_total_bytes == 0:
            return 0.0
        effective = max(self.rss_bytes, self.gpu_driver_bytes)
        return min(1.0, effective / self.system_total_bytes)


# Pressure level thresholds
PRESSURE_LOW = 0.50
PRESSURE_MEDIUM = 0.65
PRESSURE_HIGH = 0.75
PRESSURE_CRITICAL = 0.85
PRESSURE_DANGER = 0.90


class MemoryMonitor:
    """
    Runtime memory monitor singleton.

    Tracks process RSS + GPU memory and provides pressure-based
    recommendations for batch sizing.
    """

    _instance: Optional["MemoryMonitor"] = None

    def __init__(self):
        self._baseline: Optional[MemorySnapshot] = None
        self._history: List[MemorySnapshot] = []
        self._has_psutil = False
        self._has_mps = False
        self._has_cuda = False
        self._system_total = 0

        # Detect available APIs
        try:
            import psutil
            self._has_psutil = True
            self._system_total = psutil.virtual_memory().total
        except ImportError:
            logger.warning("psutil not available, memory monitoring limited")

        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._has_mps = True
            elif torch.cuda.is_available():
                self._has_cuda = True
        except ImportError:
            pass

    @classmethod
    def get_instance(cls) -> "MemoryMonitor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def snapshot(self) -> MemorySnapshot:
        """Capture current memory state."""
        rss = 0
        sys_avail = 0
        sys_total = self._system_total or (32 * 1024 ** 3)
        gpu_alloc = 0
        gpu_driver = 0

        if self._has_psutil:
            import psutil
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss
            vm = psutil.virtual_memory()
            sys_avail = vm.available
            sys_total = vm.total

        if self._has_mps:
            import torch
            try:
                gpu_alloc = torch.mps.current_allocated_memory()
                gpu_driver = torch.mps.driver_allocated_memory()
            except Exception:
                pass
        elif self._has_cuda:
            import torch
            try:
                gpu_alloc = torch.cuda.memory_allocated()
                gpu_driver = torch.cuda.memory_reserved()
            except Exception:
                pass

        snap = MemorySnapshot(
            timestamp=time.time(),
            rss_bytes=rss,
            system_available_bytes=sys_avail,
            system_total_bytes=sys_total,
            gpu_allocated_bytes=gpu_alloc,
            gpu_driver_bytes=gpu_driver,
        )
        self._history.append(snap)
        # Keep history bounded
        if len(self._history) > 200:
            self._history = self._history[-100:]
        return snap

    def get_pressure(self) -> float:
        """Current memory pressure ratio (0.0 ~ 1.0)."""
        return self.snapshot().pressure_ratio

    def set_baseline(self):
        """Set current state as baseline for delta tracking."""
        self._baseline = self.snapshot()

    def get_phase_delta_gb(self) -> float:
        """RSS growth since baseline (GB)."""
        if not self._baseline:
            return 0.0
        current = self.snapshot()
        return current.rss_gb - self._baseline.rss_gb

    def force_cleanup(self):
        """Aggressive GC + GPU cache clear."""
        gc.collect()
        gc.collect()
        if self._has_mps:
            import torch
            try:
                torch.mps.empty_cache()
                torch.mps.synchronize()
            except Exception:
                pass
        elif self._has_cuda:
            import torch
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def log_status(self, label: str = ""):
        """Log current memory state."""
        snap = self.snapshot()
        logger.info(
            f"[MEM:{label}] RSS={snap.rss_gb:.1f}GB "
            f"GPU={snap.gpu_allocated_gb:.1f}GB "
            f"Driver={snap.gpu_driver_gb:.1f}GB "
            f"Avail={snap.system_available_gb:.1f}GB "
            f"Pressure={snap.pressure_ratio:.0%}"
        )
