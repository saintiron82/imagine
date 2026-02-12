"""
Adaptive Batch Controller — throughput-optimized pipeline processing.

Triangular-step growth driven by measured throughput (items/sec):

  1. GROWING:  +1, +2, +3, +4... (accelerating triangular step)
             — keep growing while throughput improves
             — enter REFINING when throughput drops
  2. REFINING: reset step to +1 from last_fast, regrow toward first_slower
             (naturally converges as ceiling tightens each round)
  3. LOCKED:   optimal found, hold steady (re-enter if throughput degrades 20%+)

Memory pressure acts as a safety ceiling only (CRITICAL/DANGER → emergency).

Example (optimal ~ 34):
  GROWING:  2(+1)→3(+2)→5(+3)→8(+4)→12(+5)→17(+6)→23(+7)→30(+8)→38(slower!)
  REFINING: fast=30, step=1 → 31(+1)→33(+2)→36(+3,slower!)
            fast=33, step=1 → 34(+1)→36(+2,slower!)
            fast=34, step=1 → 35(+1,slower!) → gap=1 → LOCKED at 34

v3.5: Throughput-driven with memory safety ceiling.

Usage:
    controller = AdaptiveBatchController()
    while work_remaining:
        bs = controller.get_batch_size('vlm')
        t0 = time.perf_counter()
        process(items[:bs])
        elapsed = time.perf_counter() - t0
        decision = controller.after_sub_batch('vlm', bs, elapsed)
        if decision.abort:
            break
"""

import logging
from dataclasses import dataclass, field
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


# Default phase limits (max is a safety ceiling; real limit is found by throughput)
DEFAULT_PHASE_LIMITS = {
    'vlm': {'min': 1, 'max': 128, 'initial': 2},
    'vv': {'min': 1, 'max': 256, 'initial': 4},
    'mv': {'min': 2, 'max': 256, 'initial': 16},
}

# State machine states
STATE_GROWING = 'growing'
STATE_REFINING = 'refining'
STATE_LOCKED = 'locked'

# Throughput comparison thresholds
THROUGHPUT_DROP_THRESHOLD = 0.95    # 5% drop = slower
THROUGHPUT_DEGRADE_THRESHOLD = 0.80  # 20% drop from best = re-enter REFINING


@dataclass
class BatchDecision:
    """Result of batch size decision after a sub-batch."""
    batch_size: int
    reason: str       # growth / refine / locked / emergency / abort
    pressure: float
    phase: str
    state: str = ''   # growing / refining / locked
    throughput: float = 0.0  # items/sec for this sub-batch

    @property
    def abort(self) -> bool:
        return self.batch_size == 0 or self.reason == 'abort'


@dataclass
class PhaseState:
    """Per-phase adaptive state."""
    current: int = 2
    state: str = STATE_GROWING
    step_increment: int = 1                  # Triangular step (grows: 1, 2, 3, 4...)
    last_fast: Optional[int] = None          # Largest batch with good throughput
    first_slower: Optional[int] = None       # Smallest batch where throughput dropped
    consecutive_shrinks: int = 0
    # Throughput tracking
    prev_throughput: Optional[float] = None  # Previous sub-batch items/sec
    best_throughput: Optional[float] = None  # Best observed items/sec
    best_batch_size: Optional[int] = None    # Batch size at best throughput


class AdaptiveBatchController:
    """
    Throughput-driven adaptive batch controller.

    Primary criterion: items/sec — find the batch size that maximizes throughput.
    Safety ceiling: memory pressure — CRITICAL/DANGER triggers emergency shrink/abort.
    """

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

        self._states: Dict[str, PhaseState] = {}
        self._decisions: List[BatchDecision] = []

    def _get_state(self, phase: str) -> PhaseState:
        if phase not in self._states:
            limits = self._limits.get(phase, {'initial': 2})
            self._states[phase] = PhaseState(current=limits.get('initial', 2))
        return self._states[phase]

    def get_batch_size(self, phase: str) -> int:
        """Get current batch size for a phase."""
        return self._get_state(phase).current

    def set_phase_max(self, phase: str, max_batch: int):
        """Override max batch size for a phase (e.g., from SigLIP2 discovery)."""
        if phase in self._limits:
            self._limits[phase]['max'] = max_batch
        else:
            self._limits[phase] = {'min': 1, 'max': max_batch, 'initial': 2}

    def after_sub_batch(self, phase: str, processed_count: int, elapsed_sec: float = 0.0) -> BatchDecision:
        """
        Called after each sub-batch. Decides next batch size based on throughput.

        Args:
            phase: 'vlm', 'vv', 'mv'
            processed_count: items processed in this sub-batch
            elapsed_sec: wall-clock time for this sub-batch (seconds)

        Returns:
            BatchDecision with next batch size and reason
        """
        limits = self._limits.get(phase, {'min': 1, 'max': 64})
        min_bs = limits.get('min', 1)
        max_bs = limits.get('max', 64)
        ps = self._get_state(phase)
        pressure = self._monitor.get_pressure()

        # Compute throughput
        throughput = processed_count / max(elapsed_sec, 0.001) if elapsed_sec > 0 else 0.0

        # ── Memory safety ceiling (any state) ──
        if pressure >= PRESSURE_DANGER:
            self._monitor.force_cleanup()
            pressure = self._monitor.get_pressure()
            if pressure >= PRESSURE_DANGER:
                decision = BatchDecision(0, 'abort', pressure, phase, ps.state, throughput)
                self._decisions.append(decision)
                logger.error(
                    f"[ADAPTIVE:{phase}] ABORT: pressure={pressure:.0%}"
                )
                return decision

        if pressure >= PRESSURE_CRITICAL:
            ps.current = min_bs
            ps.step_increment = 1
            ps.consecutive_shrinks += 1
            decision = BatchDecision(min_bs, 'emergency', pressure, phase, ps.state, throughput)
            self._decisions.append(decision)
            logger.warning(
                f"[ADAPTIVE:{phase}] EMERGENCY → B={min_bs} "
                f"(pressure={pressure:.0%}, {throughput:.1f} items/s)"
            )
            return decision

        # ── Throughput-driven state machine ──
        if ps.state == STATE_GROWING:
            decision = self._handle_growing(ps, throughput, min_bs, max_bs, pressure, phase)
        elif ps.state == STATE_REFINING:
            decision = self._handle_refining(ps, throughput, min_bs, max_bs, pressure, phase)
        else:  # LOCKED
            decision = self._handle_locked(ps, throughput, min_bs, max_bs, pressure, phase)

        # Update throughput history
        ps.prev_throughput = throughput
        if ps.best_throughput is None or throughput > ps.best_throughput:
            ps.best_throughput = throughput
            ps.best_batch_size = ps.current

        self._decisions.append(decision)
        logger.info(
            f"[ADAPTIVE:{phase}] B:{processed_count}→{decision.batch_size} "
            f"({decision.reason}, {decision.state}, step={ps.step_increment}, "
            f"{throughput:.1f} items/s, pressure={pressure:.0%})"
        )
        return decision

    def _is_slower(self, ps: PhaseState, throughput: float) -> bool:
        """Check if throughput dropped compared to previous measurement."""
        if ps.prev_throughput is None or ps.prev_throughput <= 0:
            return False  # First measurement — can't compare
        return throughput < ps.prev_throughput * THROUGHPUT_DROP_THRESHOLD

    def _handle_growing(
        self, ps: PhaseState, throughput: float,
        min_bs: int, max_bs: int, pressure: float, phase: str
    ) -> BatchDecision:
        """GROWING state: triangular step until throughput drops.

        First sub-batch has no comparison → always grow.
        Subsequent: grow if throughput holds, refine if it drops.
        """
        current = ps.current

        if self._is_slower(ps, throughput):
            # Throughput dropped! Record boundary and enter REFINING
            ps.first_slower = current
            ps.step_increment = 1
            ps.current = ps.last_fast or max(min_bs, current - 1)
            ps.state = STATE_REFINING
            ps.consecutive_shrinks = 0

            # Check if already converged
            gap = ps.first_slower - (ps.last_fast or min_bs)
            if gap <= 1:
                ps.current = ps.last_fast or min_bs
                ps.state = STATE_LOCKED
                return BatchDecision(ps.current, 'locked', pressure, phase, STATE_LOCKED, throughput)

            return BatchDecision(ps.current, 'refine', pressure, phase, STATE_REFINING, throughput)

        else:
            # Throughput same or better — record and keep growing
            ps.last_fast = current
            next_size = min(max_bs, current + ps.step_increment)
            ps.step_increment += 1
            ps.current = next_size
            ps.consecutive_shrinks = 0
            return BatchDecision(next_size, 'growth', pressure, phase, STATE_GROWING, throughput)

    def _handle_refining(
        self, ps: PhaseState, throughput: float,
        min_bs: int, max_bs: int, pressure: float, phase: str
    ) -> BatchDecision:
        """REFINING state: triangular step from last_fast, capped at first_slower.

        Throughput drop → tighten ceiling, go back to last_fast.
        Throughput holds → grow toward ceiling.
        """
        current = ps.current

        if self._is_slower(ps, throughput):
            # Tighten upper bound, reset step, go back to last_fast
            ps.first_slower = min(ps.first_slower or current, current)
            ps.step_increment = 1
            if ps.last_fast is not None and ps.last_fast < current:
                ps.current = ps.last_fast
            else:
                ps.current = max(min_bs, current - 1)
                ps.last_fast = None

        else:
            # Throughput ok: raise lower bound and grow toward ceiling
            ps.last_fast = max(ps.last_fast or current, current)
            ceiling = (ps.first_slower or max_bs) - 1
            next_size = min(ceiling, current + ps.step_increment)
            ps.step_increment += 1
            ps.current = max(current, next_size)

        # Check convergence (gap <= 1 means we found the boundary)
        gap = (ps.first_slower or max_bs) - (ps.last_fast or min_bs)
        if gap <= 1:
            ps.current = ps.last_fast or min_bs
            ps.state = STATE_LOCKED
            ps.consecutive_shrinks = 0
            return BatchDecision(ps.current, 'locked', pressure, phase, STATE_LOCKED, throughput)

        return BatchDecision(ps.current, 'refine', pressure, phase, STATE_REFINING, throughput)

    def _handle_locked(
        self, ps: PhaseState, throughput: float,
        min_bs: int, max_bs: int, pressure: float, phase: str
    ) -> BatchDecision:
        """LOCKED state: hold at optimal. Re-enter REFINING if throughput degrades."""
        current = ps.current

        # Check if throughput degraded significantly from best
        if (ps.best_throughput is not None and ps.best_throughput > 0
                and throughput < ps.best_throughput * THROUGHPUT_DEGRADE_THRESHOLD):
            # Conditions changed — re-enter REFINING
            ps.first_slower = None
            ps.last_fast = None
            ps.step_increment = 1
            ps.state = STATE_REFINING
            ps.consecutive_shrinks += 1
            return BatchDecision(current, 'refine', pressure, phase, STATE_REFINING, throughput)

        # Hold steady
        return BatchDecision(current, 'locked', pressure, phase, STATE_LOCKED, throughput)

    def reset_phase(self, phase: str):
        """Reset a phase's state (call between phases)."""
        self._states.pop(phase, None)

    def should_abort_phase(self, phase: str) -> bool:
        """True if phase has had 5+ consecutive shrinks."""
        ps = self._states.get(phase)
        return ps is not None and ps.consecutive_shrinks >= 5

    def get_summary(self) -> str:
        """Human-readable summary of recent decisions."""
        if not self._decisions:
            return "No decisions yet"
        lines = []
        for d in self._decisions[-10:]:
            lines.append(
                f"  {d.phase}: B={d.batch_size} ({d.reason}/{d.state}, "
                f"{d.throughput:.1f} items/s, {d.pressure:.0%})"
            )
        return "\n".join(lines)
