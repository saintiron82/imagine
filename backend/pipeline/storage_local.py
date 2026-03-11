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

# Fields that submit_vision() accepts (same mapping as storage_direct.py)
_VISION_FIELD_MAP = {
    "caption": "mc_caption",
    "tags": "ai_tags",
}
_VISION_PASSTHROUGH = {
    "image_type", "art_style", "scene_type", "ocr_text",
    "dominant_color", "character_type", "item_type", "ui_type",
}


class LocalDBStorage:
    """
    Async local DB storage via DBWriteQueue.

    Each save_*() call submits a Future to the write queue.
    flush() blocks until all pending writes complete.

    DB API (DBWriteQueue):
      - submit_vision(file_path: str, fields: dict) — keyed by file_path
      - submit_vectors(file_id: int, vv_vec, mv_vec, structure_vec) — keyed by file_id
    """

    def __init__(self, db_writer):
        """
        Args:
            db_writer: DBWriteQueue instance from ingest_engine
                       Must have submit_vision() and submit_vectors()
        """
        self._writer = db_writer

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
                self._writer.submit_vision(item.file_path, fields)
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
