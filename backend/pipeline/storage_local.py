"""
storage_local.py — StorageBackend for local/Electron IPC mode.

Wraps DBWriteQueue for async incremental writes to local SQLite DB.
Used by ingest_engine.py.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from backend.pipeline.protocols import PhaseItem, StorageBackend

logger = logging.getLogger("pipeline.storage_local")


class LocalDBStorage:
    """
    Async local DB storage via DBWriteQueue.

    Each save_*() call submits a Future to the write queue.
    flush() blocks until all pending writes complete.
    """

    def __init__(self, db_writer):
        """
        Args:
            db_writer: DBWriteQueue instance from ingest_engine
                       Must have submit_vision_fields() and submit_vectors()
        """
        self._writer = db_writer

    def save_vision(self, item: PhaseItem, vision_result: dict) -> None:
        if not item.file_id or not vision_result:
            return
        try:
            self._writer.submit_vision_fields(item.file_id, vision_result)
        except Exception as e:
            logger.error(f"[FAIL:vision-store] {item.file_name}: {e}")

    def save_vv(self, item: PhaseItem, vv_vec: Any,
                structure_vec: Any = None) -> None:
        if not item.file_id or vv_vec is None:
            return
        try:
            self._writer.submit_vectors(
                item.file_id,
                vv_vec=vv_vec,
                mv_vec=None,
                structure_vec=structure_vec,
            )
        except Exception as e:
            logger.error(f"[FAIL:vv-store] {item.file_name}: {e}")

    def save_mv(self, item: PhaseItem, mv_vec: Any, text: str) -> None:
        if not item.file_id or mv_vec is None:
            return
        try:
            self._writer.submit_vectors(
                item.file_id,
                vv_vec=None,
                mv_vec=mv_vec,
                structure_vec=None,
            )
        except Exception as e:
            logger.error(f"[FAIL:mv-store] {item.file_name}: {e}")

    def flush(self) -> None:
        try:
            self._writer.flush(timeout=60.0)
        except Exception as e:
            logger.error(f"[DBQ] flush failed: {e}")
