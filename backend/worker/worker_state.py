"""
Worker State Machine — manages worker lifecycle states based on schedule,
resource throttling, and idle timeout.

States:
    ACTIVE   — Normal operation: claim jobs, process batches.
    IDLE     — Schedule inactive OR idle timeout expired: no job claims,
               all models unloaded.
    RESTING  — Throttle critical: no job claims, models already unloaded
               by the throttle handler.

Transitions:
    ACTIVE  -> IDLE     : schedule becomes inactive
    ACTIVE  -> IDLE     : idle timeout expired (no jobs for N minutes)
    ACTIVE  -> RESTING  : throttle_level == "critical"
    IDLE    -> ACTIVE   : schedule becomes active + (has_pending_jobs OR idle not expired)
    RESTING -> ACTIVE   : throttle_level drops below "critical"
    RESTING -> IDLE     : schedule becomes inactive while resting
    IDLE    -> RESTING  : (not expected, but handled) throttle critical while idle
"""

import enum
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class WorkerState(enum.Enum):
    """Worker lifecycle states."""
    ACTIVE = "active"
    IDLE = "idle"
    RESTING = "resting"


def _get_idle_unload_minutes() -> float:
    """Read idle_unload_minutes from user-settings.yaml > worker section.

    Returns the configured value (in minutes), or 10.0 as default.
    A value of 0 or negative disables idle timeout.
    """
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        val = cfg.get("worker.idle_unload_minutes")
        if val is not None:
            return float(val)
    except Exception:
        pass
    return 10.0


class WorkerStateMachine:
    """Manages worker state transitions based on schedule, throttle level,
    and idle timeout.

    Args:
        on_enter_idle: Callback invoked when transitioning TO idle state.
            Typically triggers full model unload.
        on_enter_active: Callback invoked when transitioning TO active state.
            Models reload lazily on first use, so this is informational only.
        on_enter_resting: Callback invoked when transitioning TO resting state.
            Models should already be unloaded by throttle handler.
    """

    def __init__(
        self,
        on_enter_idle: Optional[Callable] = None,
        on_enter_active: Optional[Callable] = None,
        on_enter_resting: Optional[Callable] = None,
    ):
        self._state = WorkerState.ACTIVE
        self._on_enter_idle = on_enter_idle
        self._on_enter_active = on_enter_active
        self._on_enter_resting = on_enter_resting

        # Idle timeout tracking
        self._last_job_time: float = time.time()
        self._idle_due_to_timeout: bool = False

    @property
    def state(self) -> WorkerState:
        """Current worker state."""
        return self._state

    @property
    def state_name(self) -> str:
        """Current state as a string value (for heartbeat reporting)."""
        return self._state.value

    @property
    def idle_unload_minutes(self) -> float:
        """Configured idle timeout in minutes (0 = disabled)."""
        return _get_idle_unload_minutes()

    def record_job_activity(self):
        """Record that a job was processed (start or completion).

        Resets the idle timeout timer. Call this whenever a batch
        completes or a new job starts processing.
        """
        self._last_job_time = time.time()
        self._idle_due_to_timeout = False

    def update(
        self,
        is_scheduled_active: bool,
        throttle_level: str = "normal",
        has_pending_jobs: bool = True,
    ) -> bool:
        """Evaluate conditions and transition state if needed.

        Args:
            is_scheduled_active: Whether the current time is within the
                configured active schedule window.
            throttle_level: Current resource throttle level
                ('normal', 'warning', 'danger', 'critical').
            has_pending_jobs: Whether there are pending jobs. When False
                and idle_unload_minutes has elapsed since last job activity,
                the worker transitions to IDLE to unload models. When True
                and currently idle due to timeout, automatically wakes to
                ACTIVE.

        Returns:
            True if the state changed, False otherwise.
        """
        old_state = self._state
        new_state = self._determine_state(
            is_scheduled_active, throttle_level, has_pending_jobs
        )

        if new_state == old_state:
            return False

        self._state = new_state
        reason = ""
        if new_state == WorkerState.IDLE and self._idle_due_to_timeout:
            minutes = self.idle_unload_minutes
            reason = f" (idle timeout: {minutes:.0f}min with no jobs)"
        logger.info(f"Worker state: {old_state.value} -> {new_state.value}{reason}")

        # Fire transition callbacks
        if new_state == WorkerState.IDLE:
            self._fire_callback(self._on_enter_idle, "on_enter_idle")
        elif new_state == WorkerState.ACTIVE:
            self._fire_callback(self._on_enter_active, "on_enter_active")
        elif new_state == WorkerState.RESTING:
            self._fire_callback(self._on_enter_resting, "on_enter_resting")

        return True

    def _is_idle_timeout_expired(self) -> bool:
        """Check if idle timeout has expired (no job activity for N minutes).

        Returns False if idle timeout is disabled (value <= 0).
        """
        minutes = self.idle_unload_minutes
        if minutes <= 0:
            return False
        elapsed = time.time() - self._last_job_time
        return elapsed >= (minutes * 60)

    def _determine_state(
        self, is_scheduled_active: bool, throttle_level: str,
        has_pending_jobs: bool = True,
    ) -> WorkerState:
        """Pure logic: determine what state we should be in.

        Priority:
            1. Schedule inactive -> IDLE (highest priority)
            2. Throttle critical -> RESTING
            3. Idle timeout expired AND no pending jobs -> IDLE
            4. Has pending jobs while idle-due-to-timeout -> ACTIVE (wake up)
            5. Otherwise -> ACTIVE
        """
        if not is_scheduled_active:
            return WorkerState.IDLE

        if throttle_level == "critical":
            return WorkerState.RESTING

        # Idle timeout: transition to IDLE when no jobs for N minutes
        if self._is_idle_timeout_expired() and not has_pending_jobs:
            self._idle_due_to_timeout = True
            return WorkerState.IDLE

        # Wake from idle-timeout when new jobs arrive
        if self._idle_due_to_timeout and has_pending_jobs:
            self._idle_due_to_timeout = False
            return WorkerState.ACTIVE

        # If currently idle due to timeout but timeout no longer expired
        # (e.g., record_job_activity was called), return to active
        if self._idle_due_to_timeout and not self._is_idle_timeout_expired():
            self._idle_due_to_timeout = False
            return WorkerState.ACTIVE

        return WorkerState.ACTIVE

    def _fire_callback(self, callback: Optional[Callable], name: str):
        """Safely invoke a callback."""
        if callback is None:
            return
        try:
            callback()
        except Exception as e:
            logger.warning(f"State callback {name} failed: {e}")
