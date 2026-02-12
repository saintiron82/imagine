"""
Adaptive Batch Controller for memory-aware pipeline processing.

Triangular-step growth with auto-refinement:

  1. GROWING:  +1, +2, +3, +4... (accelerating triangular step)
  2. REFINING: on pressure, reset step to +1, regrow from last_safe
              (naturally converges as ceiling tightens each round)
  3. LOCKED:   optimal found, hold steady (re-enter if conditions change)

Example (optimal ~ 34):
  GROWING:  2(+1)→3(+2)→5(+3)→8(+4)→12(+5)→17(+6)→23(+7)→30(+8)→38(pressure!)
  REFINING: safe=30, step=1 → 31(+1)→33(+2)→36(+3,pressure!)
            safe=33, step=1 → 34(+1)→36(+2,pressure!)
            safe=34, step=1 → 35(+1,pressure!) → gap=1 → LOCKED at 34

v3.4: Central controller used by all pipeline phases (VLM, VV, MV).

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


# Default phase limits
DEFAULT_PHASE_LIMITS = {
    'vlm': {'min': 1, 'max': 8, 'initial': 2},
    'vv': {'min': 1, 'max': 64, 'initial': 4},
    'mv': {'min': 2, 'max': 64, 'initial': 16},
}

# State machine states
STATE_GROWING = 'growing'
STATE_REFINING = 'refining'
STATE_LOCKED = 'locked'


@dataclass
class BatchDecision:
    """Result of batch size decision after a sub-batch."""
    batch_size: int
    reason: str       # growth / stable / shrink / refine / locked / emergency / abort
    pressure: float
    phase: str
    state: str = ''   # growing / refining / locked

    @property
    def abort(self) -> bool:
        return self.batch_size == 0 or self.reason == 'abort'


@dataclass
class PhaseState:
    """Per-phase adaptive state."""
    current: int = 2
    state: str = STATE_GROWING
    step_increment: int = 1                  # Triangular step (grows: 1, 2, 3, 4...)
    last_safe: Optional[int] = None          # Largest batch that was safe
    first_unsafe: Optional[int] = None       # Smallest batch that triggered pressure
    consecutive_shrinks: int = 0


class AdaptiveBatchController:
    """
    Triangular-step adaptive batch controller.

    GROWING:  accelerating steps (+1, +2, +3...) until pressure.
    REFINING: reset step to +1 from last_safe, re-grow toward ceiling.
    LOCKED:   hold at optimal, re-enter if conditions change.
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

    def after_sub_batch(self, phase: str, processed_count: int) -> BatchDecision:
        """
        Called after each sub-batch. Decides next batch size via state machine.

        Args:
            phase: 'vlm', 'vv', 'mv'
            processed_count: files processed in this sub-batch

        Returns:
            BatchDecision with next batch size and reason
        """
        limits = self._limits.get(phase, {'min': 1, 'max': 64})
        min_bs = limits.get('min', 1)
        max_bs = limits.get('max', 64)
        ps = self._get_state(phase)
        pressure = self._monitor.get_pressure()

        # ── Emergency / Danger handling (any state) ──
        if pressure >= PRESSURE_DANGER:
            self._monitor.force_cleanup()
            pressure = self._monitor.get_pressure()
            if pressure >= PRESSURE_DANGER:
                decision = BatchDecision(0, 'abort', pressure, phase, ps.state)
                self._decisions.append(decision)
                logger.error(
                    f"[ADAPTIVE:{phase}] ABORT: pressure={pressure:.0%}"
                )
                return decision

        if pressure >= PRESSURE_CRITICAL:
            ps.current = min_bs
            ps.step_increment = 1
            ps.consecutive_shrinks += 1
            decision = BatchDecision(min_bs, 'emergency', pressure, phase, ps.state)
            self._decisions.append(decision)
            logger.warning(
                f"[ADAPTIVE:{phase}] EMERGENCY → B={min_bs} "
                f"(pressure={pressure:.0%})"
            )
            return decision

        # ── Pressure classification ──
        is_pressured = pressure >= PRESSURE_HIGH    # >= 75%
        is_safe = pressure < PRESSURE_LOW           # < 50%

        # ── State machine ──
        if ps.state == STATE_GROWING:
            decision = self._handle_growing(ps, is_pressured, is_safe, min_bs, max_bs, pressure, phase)
        elif ps.state == STATE_REFINING:
            decision = self._handle_refining(ps, is_pressured, is_safe, min_bs, max_bs, pressure, phase)
        else:  # LOCKED
            decision = self._handle_locked(ps, is_pressured, is_safe, min_bs, max_bs, pressure, phase)

        self._decisions.append(decision)
        logger.info(
            f"[ADAPTIVE:{phase}] B:{processed_count}→{decision.batch_size} "
            f"({decision.reason}, {decision.state}, step={ps.step_increment}, "
            f"pressure={pressure:.0%})"
        )
        return decision

    def _handle_growing(
        self, ps: PhaseState, is_pressured: bool, is_safe: bool,
        min_bs: int, max_bs: int, pressure: float, phase: str
    ) -> BatchDecision:
        """GROWING state: triangular step (+1, +2, +3...) until pressure.

        In GROWING, only HIGH pressure (>=75%) triggers REFINING.
        Medium pressure (50-75%) still grows — we want to find the limit fast.
        """
        current = ps.current

        if is_pressured:
            # Pressure hit! Record boundary and enter REFINING
            ps.first_unsafe = current
            ps.step_increment = 1
            ps.current = ps.last_safe or max(min_bs, current - 1)
            ps.state = STATE_REFINING
            ps.consecutive_shrinks = 0

            # Check if already converged
            gap = ps.first_unsafe - (ps.last_safe or min_bs)
            if gap <= 1:
                ps.current = ps.last_safe or min_bs
                ps.state = STATE_LOCKED
                return BatchDecision(ps.current, 'locked', pressure, phase, STATE_LOCKED)

            return BatchDecision(ps.current, 'refine', pressure, phase, STATE_REFINING)

        else:
            # Safe or medium: record and keep growing with triangular step.
            # HIGH (75%) is the only brake in GROWING — be aggressive to find limit.
            ps.last_safe = current
            next_size = min(max_bs, current + ps.step_increment)
            ps.step_increment += 1
            ps.current = next_size
            ps.consecutive_shrinks = 0
            return BatchDecision(next_size, 'growth', pressure, phase, STATE_GROWING)

    def _handle_refining(
        self, ps: PhaseState, is_pressured: bool, is_safe: bool,
        min_bs: int, max_bs: int, pressure: float, phase: str
    ) -> BatchDecision:
        """REFINING state: triangular step from last_safe, capped at first_unsafe.

        Only HIGH pressure (>=75%) is treated as failure.
        Medium/safe both grow — must converge toward the boundary.
        """
        current = ps.current

        if is_pressured:
            # Tighten upper bound, reset step, go back to last_safe
            ps.first_unsafe = min(ps.first_unsafe or current, current)
            ps.step_increment = 1
            if ps.last_safe is not None and ps.last_safe < current:
                ps.current = ps.last_safe
            else:
                ps.current = max(min_bs, current - 1)
                ps.last_safe = None

        else:
            # Safe or medium: raise lower bound and grow toward ceiling
            ps.last_safe = max(ps.last_safe or current, current)
            ceiling = (ps.first_unsafe or max_bs) - 1
            next_size = min(ceiling, current + ps.step_increment)
            ps.step_increment += 1
            ps.current = max(current, next_size)

        # Check convergence (gap <= 1 means we found the boundary)
        gap = (ps.first_unsafe or max_bs) - (ps.last_safe or min_bs)
        if gap <= 1:
            ps.current = ps.last_safe or min_bs
            ps.state = STATE_LOCKED
            ps.consecutive_shrinks = 0
            return BatchDecision(ps.current, 'locked', pressure, phase, STATE_LOCKED)

        return BatchDecision(ps.current, 'refine', pressure, phase, STATE_REFINING)

    def _handle_locked(
        self, ps: PhaseState, is_pressured: bool, is_safe: bool,
        min_bs: int, max_bs: int, pressure: float, phase: str
    ) -> BatchDecision:
        """LOCKED state: hold at optimal. Re-enter REFINING if conditions worsen."""
        current = ps.current

        if is_pressured:
            # Conditions worsened — shrink by 1 and re-refine
            ps.first_unsafe = current
            ps.current = max(min_bs, current - 1)
            ps.last_safe = None
            ps.step_increment = 1
            ps.state = STATE_REFINING
            ps.consecutive_shrinks += 1
            return BatchDecision(ps.current, 'shrink', pressure, phase, STATE_REFINING)

        # Hold steady
        return BatchDecision(current, 'locked', pressure, phase, STATE_LOCKED)

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
                f"  {d.phase}: B={d.batch_size} ({d.reason}/{d.state}, {d.pressure:.0%})"
            )
        return "\n".join(lines)
