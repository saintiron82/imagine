"""
storage_direct.py — StorageBackend for server auto mode (parse_ahead).

Direct synchronous writes to SQLite DB.
Used by parse_ahead.py's _process_auto_batch().
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from backend.pipeline.protocols import PhaseItem, StorageBackend

logger = logging.getLogger("pipeline.storage_direct")

# Fields that update_vision_fields() accepts
_VISION_FIELD_MAP = {
    "caption": "mc_caption",
    "tags": "ai_tags",
}
_VISION_PASSTHROUGH = {
    "image_type", "art_style", "scene_type", "ocr_text",
    "dominant_color", "character_type", "item_type", "ui_type",
}


class DirectSQLStorage:
    """
    Synchronous direct DB storage.

    Writes to SQLite DB immediately (no async queue).
    Suitable for server-side processing where a single writer is guaranteed.

    DB API:
      - update_vision_fields(file_path: str, fields: dict) — keyed by file_path
      - upsert_vectors(file_id: int, vv_vec, mv_vec, structure_vec) — keyed by file_id
    """

    def __init__(self, db):
        """
        Args:
            db: SQLiteDB instance (backend.db.sqlite_client.SQLiteDB)
        """
        self._db = db

    def save_vision(self, item: PhaseItem, vision_result: dict) -> None:
        if not item.file_path or not vision_result:
            return
        try:
            # Transform VLM result dict to DB column names
            fields = {}
            for src, dst in _VISION_FIELD_MAP.items():
                if src in vision_result:
                    fields[dst] = vision_result[src]
            for key in _VISION_PASSTHROUGH:
                if vision_result.get(key) is not None:
                    fields[key] = vision_result[key]

            if fields:
                # update_vision_fields uses file_path (not file_id)
                self._db.update_vision_fields(item.file_path, fields)
        except Exception as e:
            logger.error(f"[FAIL:vision-store] {item.file_name}: {e}")

    def save_vv(self, item: PhaseItem, vv_vec: Any,
                structure_vec: Any = None) -> None:
        if not item.file_id or vv_vec is None:
            return
        try:
            self._db.upsert_vectors(
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
            self._db.upsert_vectors(
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
