"""
Adaptive Batch Controller for memory-aware pipeline processing.

Starts from small batch sizes and grows incrementally based on
runtime memory pressure. Automatically shrinks on high pressure.

v3.4: Central controller used by all pipeline phases (VLM, VV, MV).

Algorithm:
    1. Start at initial_batch (2 for VLM, 4 for VV, 16 for MV)
    2. After each sub-batch: measure memory pressure
    3. pressure < 50%  → batch + 1 (linear growth)
       50~65%          → hold (stable)
       65~75%          → batch - 1 (linear shrink)
       75~85%          → batch = 1 (emergency)
       85~90%          → force_cleanup, re-evaluate
       > 90%           → abort phase

    Linear +1/-1 ensures overshoot is at most 1 file worth of memory.
    Example: 2→3→4→...→32(safe)→33(pressure detected)→32(shrink back)

Usage:
    controller = AdaptiveBatchController()
    while work_remaining:
        bs = controller.get_batch_size('vlm')
        process(items[:bs])
        decision = controller.after_sub_batch('vlm', bs)
        if decision.abort:
            break
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .memory_monitor import (
    MemoryMonitor,
    PRESSURE_LOW,
    PRESSURE_MEDIUM,
    PRESSURE_HIGH,
    PRESSURE_CRITICAL,
    PRESSURE_DANGER,
)

logger = logging.getLogger(__name__)


# Default phase limits
DEFAULT_PHASE_LIMITS = {
    'vlm': {'min': 1, 'max': 8, 'initial': 2},
    'vv': {'min': 1, 'max': 64, 'initial': 4},
    'mv': {'min': 2, 'max': 64, 'initial': 16},
}


@dataclass
class BatchDecision:
    """Result of batch size decision after a sub-batch."""
    batch_size: int
    reason: str       # growth / stable / shrink / emergency / abort
    pressure: float
    phase: str

    @property
    def abort(self) -> bool:
        return self.batch_size == 0 or self.reason == 'abort'


class AdaptiveBatchController:
    """
    Memory-pressure-aware batch size controller.

    Manages batch sizes for all pipeline phases independently.
    Each phase has its own current size, limits, and growth state.
    """

    GROWTH_STEP = 1       # +1 linear growth (safe: overshoot is only 1 file)
    SHRINK_STEP = 1       # -1 linear shrink

    def __init__(
        self,
        phase_limits: Optional[Dict] = None,
        memory_budget_gb: float = 20.0,
    ):
        self._monitor = MemoryMonitor.get_instance()
        self._budget_gb = memory_budget_gb
        self._limits = dict(DEFAULT_PHASE_LIMITS)
        if phase_limits:
            for k, v in phase_limits.items():
                self._limits[k] = {**self._limits.get(k, {}), **v}

        self._current_sizes: Dict[str, int] = {}
        self._decisions: List[BatchDecision] = []
        self._consecutive_shrinks: Dict[str, int] = {}

    def get_batch_size(self, phase: str) -> int:
        """Get current batch size for a phase."""
        if phase not in self._current_sizes:
            limits = self._limits.get(phase, {'initial': 2})
            self._current_sizes[phase] = limits.get('initial', 2)
        return self._current_sizes[phase]

    def set_phase_max(self, phase: str, max_batch: int):
        """Override max batch size for a phase (e.g., from SigLIP2 discovery)."""
        if phase in self._limits:
            self._limits[phase]['max'] = max_batch
        else:
            self._limits[phase] = {'min': 1, 'max': max_batch, 'initial': 2}

    def after_sub_batch(self, phase: str, processed_count: int) -> BatchDecision:
        """
        Called after each sub-batch completes. Decides next batch size.

        Args:
            phase: 'vlm', 'vv', 'mv'
            processed_count: files processed in this sub-batch

        Returns:
            BatchDecision with next batch size and reason
        """
        limits = self._limits.get(phase, {'min': 1, 'max': 64})
        min_bs = limits.get('min', 1)
        max_bs = limits.get('max', 64)
        current = self._current_sizes.get(phase, limits.get('initial', 2))
        pressure = self._monitor.get_pressure()

        # Danger zone: try cleanup first
        if pressure >= PRESSURE_DANGER:
            self._monitor.force_cleanup()
            pressure = self._monitor.get_pressure()
            if pressure >= PRESSURE_DANGER:
                decision = BatchDecision(0, 'abort', pressure, phase)
                self._decisions.append(decision)
                logger.error(
                    f"[ADAPTIVE:{phase}] ABORT: pressure={pressure:.0%} "
                    f"after cleanup"
                )
                return decision

        # Decide new batch size (+1/-1 linear to avoid memory spikes)
        if pressure >= PRESSURE_CRITICAL:
            new_size = min_bs
            reason = 'emergency'
            self._consecutive_shrinks[phase] = \
                self._consecutive_shrinks.get(phase, 0) + 1
        elif pressure >= PRESSURE_HIGH:
            new_size = max(min_bs, current - self.SHRINK_STEP)
            reason = 'shrink'
            self._consecutive_shrinks[phase] = \
                self._consecutive_shrinks.get(phase, 0) + 1
        elif pressure < PRESSURE_LOW:
            new_size = min(max_bs, current + self.GROWTH_STEP)
            reason = 'growth'
            self._consecutive_shrinks[phase] = 0
        else:
            new_size = current
            reason = 'stable'
            self._consecutive_shrinks[phase] = 0

        # Clamp
        new_size = max(min_bs, min(max_bs, new_size))

        self._current_sizes[phase] = new_size
        decision = BatchDecision(new_size, reason, pressure, phase)
        self._decisions.append(decision)

        logger.info(
            f"[ADAPTIVE:{phase}] B:{current}→{new_size} "
            f"({reason}, pressure={pressure:.0%})"
        )
        return decision

    def reset_phase(self, phase: str):
        """Reset a phase's batch size state (call between phases)."""
        self._current_sizes.pop(phase, None)
        self._consecutive_shrinks.pop(phase, None)

    def should_abort_phase(self, phase: str) -> bool:
        """True if phase has had 5+ consecutive shrinks."""
        return self._consecutive_shrinks.get(phase, 0) >= 5

    def get_summary(self) -> str:
        """Human-readable summary of all decisions."""
        if not self._decisions:
            return "No decisions yet"
        lines = []
        for d in self._decisions[-10:]:
            lines.append(
                f"  {d.phase}: B={d.batch_size} ({d.reason}, {d.pressure:.0%})"
            )
        return "\n".join(lines)
