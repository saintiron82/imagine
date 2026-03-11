"""
storage_direct.py — StorageBackend for server auto mode (parse_ahead).

Direct synchronous writes to SQLite DB.
Used by parse_ahead.py's _process_auto_batch().
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from backend.pipeline.protocols import PhaseItem, StorageBackend

logger = logging.getLogger("pipeline.storage_direct")


class DirectSQLStorage:
    """
    Synchronous direct DB storage.

    Writes to SQLite DB immediately (no async queue).
    Suitable for server-side processing where a single writer is guaranteed.
    """

    def __init__(self, db):
        """
        Args:
            db: SQLiteDB instance (backend.db.sqlite_client.SQLiteDB)
        """
        self._db = db

    def save_vision(self, item: PhaseItem, vision_result: dict) -> None:
        if not item.file_id or not vision_result:
            return
        try:
            self._db.update_vision_fields(item.file_id, vision_result)
        except Exception as e:
            logger.error(f"[FAIL:vision-store] {item.file_name}: {e}")

    def save_vv(self, item: PhaseItem, vv_vec: Any,
                structure_vec: Any = None) -> None:
        if not item.file_id or vv_vec is None:
            return
        try:
            self._db.store_vectors(
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
            self._db.store_vectors(
                item.file_id,
                vv_vec=None,
                mv_vec=mv_vec,
                structure_vec=None,
            )
        except Exception as e:
            logger.error(f"[FAIL:mv-store] {item.file_name}: {e}")

    def flush(self) -> None:
        # Direct writes are already committed; checkpoint WAL
        try:
            if hasattr(self._db, "checkpoint"):
                self._db.checkpoint()
        except Exception as e:
            logger.debug(f"WAL checkpoint note: {e}")
