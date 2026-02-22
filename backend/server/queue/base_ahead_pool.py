"""Base class for server-side ahead-of-demand processing pools.

Provides common thread lifecycle management, config access, and stats interface.
Subclasses implement _loop() for their specific processing logic
and _unload_models() for cleanup on stop.
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional

from backend.db.sqlite_client import SQLiteDB

logger = logging.getLogger(__name__)


class BaseAheadPool(ABC):
    """Base class for background daemon pools (ParseAhead, EmbedAhead).

    Provides common thread management, config access, and lifecycle.
    Subclasses implement _loop() for their specific processing logic
    and _unload_models() for cleanup.
    """

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
