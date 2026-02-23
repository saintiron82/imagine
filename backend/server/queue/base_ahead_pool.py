"""Base class for server-side ahead-of-demand processing pools.

Provides common thread lifecycle management, config access, and stats interface.
Subclasses implement _loop() for their specific processing logic
and _unload_models() for cleanup on stop.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from backend.db.sqlite_client import SQLiteDB

logger = logging.getLogger(__name__)


class BaseAheadPool(ABC):
    """Base class for background daemon pools (ParseAhead, EmbedAhead).

    Provides common thread management, config access, and lifecycle.
    Subclasses implement _loop() for their specific processing logic
    and _unload_models() for cleanup.

    Demand signal: ParseAheadPool only runs when workers are actively
    claiming jobs. record_claim() is called by JobQueueManager on each
    successful claim, and has_recent_demand() gates the pool.
    """

    # Shared demand signal: updated by JobQueueManager.claim_jobs()
    _last_claim_time: float = 0.0
    _claim_inactivity_timeout: float = 120.0  # 2 min without claim → pause
    # Per-worker demand: session_id -> (last_claim_count, timestamp)
    _worker_demand: Dict[int, Tuple[int, float]] = {}

    @classmethod
    def record_claim(cls, session_id: Optional[int] = None, count: int = 0):
        """Called by JobQueueManager when a worker claims jobs.

        Args:
            session_id: Worker session ID (tracks per-worker demand).
            count: Number of jobs actually claimed.
        """
        cls._last_claim_time = time.time()
        if session_id is not None and count > 0:
            cls._worker_demand[session_id] = (count, time.time())

    @classmethod
    def has_recent_demand(cls) -> bool:
        """True if any worker claimed jobs within the inactivity timeout."""
        return (time.time() - cls._last_claim_time) < cls._claim_inactivity_timeout

    @classmethod
    def get_total_demand(cls) -> int:
        """Sum of recent per-worker claim counts.

        Only includes workers that claimed within the inactivity timeout.
        This predicts the next round of claims based on actual consumption.
        Also prunes stale entries to prevent unbounded dict growth.
        """
        cutoff = time.time() - cls._claim_inactivity_timeout
        # Prune stale entries (prevents infinite growth from offline workers)
        stale_keys = [
            sid for sid, (_, ts) in cls._worker_demand.items() if ts <= cutoff
        ]
        for sid in stale_keys:
            del cls._worker_demand[sid]

        return sum(count for count, _ in cls._worker_demand.values())

    def __init__(self, db: SQLiteDB):
        self.db = db
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def _pool_name(self) -> str:
        """Human-readable pool name derived from class name."""
        return self.__class__.__name__

    def start(self):
        """Start the background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning(f"{self._pool_name} already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name=self._pool_name,
            daemon=True,
        )
        self._thread.start()
        logger.info(f"{self._pool_name} started")

    def stop(self):
        """Gracefully stop the daemon thread and unload models."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                logger.warning(f"{self._pool_name} thread did not stop within timeout")
            else:
                logger.info(f"{self._pool_name} stopped")
        self._thread = None
        self._unload_models()

    @abstractmethod
    def _loop(self):
        """Main processing loop -- runs in daemon thread."""
        ...

    @abstractmethod
    def _unload_models(self):
        """Unload any loaded models on stop."""
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """Return current pool statistics."""
        ...

    def _get_config_value(self, dotted_key: str, default):
        """Read a value from config.yaml by dotted key."""
        try:
            from backend.utils.config import get_config
            return get_config().get(dotted_key, default)
        except Exception:
            return default
