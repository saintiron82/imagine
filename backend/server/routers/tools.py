"""
Admin tools router — repair, maintenance, diagnostics.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db_safe, require_admin
from backend.utils.meta_helpers import meta_to_dict

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin-tools"])

# ── Repair-parse state (module-level singleton) ─────────────────

_repair_state = {
    "running": False,
    "progress": {
        "done": 0,
        "failed": 0,
        "total": 0,
        "current_file": "",
    },
}
_repair_lock = threading.Lock()


def _run_repair_parse(db_path: str):
    """Background thread: re-parse all files and update metadata JSON column."""
    global _repair_state

    # Each thread needs its own DB connection (sqlite threading.local)
    from backend.db.sqlite_client import SQLiteDB
    db = SQLiteDB(db_path)

    try:
        cursor = db.conn.cursor()
        cursor.execute("SELECT id, file_path, file_name, format FROM files")
        rows = cursor.fetchall()

        total = len(rows)
        with _repair_lock:
            _repair_state["progress"]["total"] = total

        logger.info(f"[repair-parse] Starting repair for {total} files")

        done = 0
        failed = 0

        for file_id, file_path, file_name, fmt in rows:
            with _repair_lock:
                _repair_state["progress"]["current_file"] = file_name or file_path

            # Skip WebDAV files (no local access)
            if file_path and file_path.startswith("webdav://"):
                logger.debug(f"[repair-parse] Skip WebDAV: {file_path}")
                with _repair_lock:
                    failed += 1
                    _repair_state["progress"]["failed"] = failed
                continue

            try:
                p = Path(file_path)
                if not p.exists():
                    logger.debug(f"[repair-parse] File not found, skip: {file_path}")
                    with _repair_lock:
                        failed += 1
                        _repair_state["progress"]["failed"] = failed
                    continue

                # Select parser by extension
                from backend.parser.psd_parser import PSDParser
                from backend.parser.image_parser import ImageParser

                ext = p.suffix.lower()
                if ext == ".psd":
                    parser = PSDParser()
                elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"):
                    parser = ImageParser()
                else:
                    logger.debug(f"[repair-parse] Unsupported format, skip: {file_path}")
                    with _repair_lock:
                        failed += 1
                        _repair_state["progress"]["failed"] = failed
                    continue

                result = parser.parse(p)
                if not result.success or result.asset_meta is None:
                    logger.warning(f"[repair-parse] Parse failed: {file_path} — {result.errors}")
                    with _repair_lock:
                        failed += 1
                        _repair_state["progress"]["failed"] = failed
                    continue

                meta_dict = meta_to_dict(result.asset_meta)

                # Build metadata JSON (same structure as upsert_metadata)
                metadata_json = {
                    "layer_tree": meta_dict.get("layer_tree"),
                    "semantic_tags": meta_dict.get("semantic_tags"),
                    "text_content": meta_dict.get("text_content"),
                    "layer_count": meta_dict.get("layer_count"),
                    "used_fonts": meta_dict.get("used_fonts"),
                }

                cur = db.conn.cursor()
                cur.execute(
                    "UPDATE files SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata_json, ensure_ascii=False), file_id),
                )
                db._refresh_fts_row(cur, file_id)
                db.conn.commit()

                with _repair_lock:
                    done += 1
                    _repair_state["progress"]["done"] = done

                if done % 50 == 0:
                    logger.info(f"[repair-parse] Progress: {done}/{total} done, {failed} failed")

            except Exception as e:
                logger.warning(f"[repair-parse] Error on {file_path}: {e}")
                try:
                    db.conn.rollback()
                except Exception:
                    pass
                with _repair_lock:
                    failed += 1
                    _repair_state["progress"]["failed"] = failed

        logger.info(f"[repair-parse] Complete: {done}/{total} done, {failed} failed")

    except Exception as e:
        logger.error(f"[repair-parse] Fatal error: {e}")
    finally:
        with _repair_lock:
            _repair_state["running"] = False
            _repair_state["progress"]["current_file"] = ""
        try:
            db.close()
        except Exception:
            pass


@router.post("/admin/tools/repair-parse")
def start_repair_parse(
    _user: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Re-parse all files and repair metadata JSON column.

    Runs in background thread. Returns immediately with total count.
    """
    global _repair_state

    with _repair_lock:
        if _repair_state["running"]:
            return {
                "success": False,
                "detail": "Repair already running",
                "progress": dict(_repair_state["progress"]),
            }

    # Count files
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files")
    total = cursor.fetchone()[0]

    if total == 0:
        return {"success": False, "detail": "No files in database"}

    # Get DB path for background thread (needs its own connection)
    from backend.server.config import get_db_path
    db_path = get_db_path()

    with _repair_lock:
        _repair_state["running"] = True
        _repair_state["progress"] = {
            "done": 0,
            "failed": 0,
            "total": total,
            "current_file": "",
        }

    t = threading.Thread(
        target=_run_repair_parse,
        args=(db_path,),
        daemon=True,
        name="repair-parse",
    )
    t.start()

    return {
        "success": True,
        "task_id": "repair-parse",
        "total": total,
    }


@router.get("/admin/tools/repair-parse/status")
def get_repair_parse_status(
    _user: dict = Depends(require_admin),
):
    """Check repair-parse progress."""
    with _repair_lock:
        return {
            "running": _repair_state["running"],
            "progress": dict(_repair_state["progress"]),
        }
