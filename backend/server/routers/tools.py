"""
Admin tools router — repair, maintenance, diagnostics.
"""

import json
import logging
import threading
import time
from pathlib import Path

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
        "skipped": 0,
        "total": 0,
        "current_file": "",
        "phase": "",
    },
}
_repair_lock = threading.Lock()


def _update_progress(**kwargs):
    with _repair_lock:
        _repair_state["progress"].update(kwargs)


def _run_repair_parse(db_path: str):
    """Background thread: re-parse accessible files and update metadata + FTS."""
    logger.info("[repair-parse] Thread started")

    try:
        from backend.db.sqlite_client import SQLiteDB
        db = SQLiteDB(db_path)
    except Exception as e:
        logger.error(f"[repair-parse] DB connection failed: {e}")
        with _repair_lock:
            _repair_state["running"] = False
        return

    try:
        cursor = db.conn.cursor()
        cursor.execute("SELECT id, file_path, file_name, format FROM files ORDER BY id")
        rows = cursor.fetchall()

        total = len(rows)
        _update_progress(total=total, phase="parsing")
        logger.info(f"[repair-parse] {total} files to process")

        done = 0
        failed = 0
        skipped = 0

        # Pre-import parsers
        from backend.parser.psd_parser import PSDParser
        from backend.parser.image_parser import ImageParser
        psd_parser = PSDParser()
        img_parser = ImageParser()

        for file_id, file_path, file_name, fmt in rows:
            display = file_name or Path(file_path).name if file_path else f"id={file_id}"
            _update_progress(current_file=display)

            # Skip WebDAV files
            if file_path and file_path.startswith("webdav://"):
                skipped += 1
                _update_progress(skipped=skipped)
                continue

            if not file_path:
                skipped += 1
                _update_progress(skipped=skipped)
                continue

            try:
                p = Path(file_path)
                if not p.exists():
                    skipped += 1
                    _update_progress(skipped=skipped)
                    continue

                ext = p.suffix.lower()
                if ext == ".psd":
                    parser = psd_parser
                elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"):
                    parser = img_parser
                else:
                    skipped += 1
                    _update_progress(skipped=skipped)
                    continue

                result = parser.parse(p)
                if not result.success or result.asset_meta is None:
                    logger.warning(f"[repair-parse] FAIL {display}: {result.errors}")
                    failed += 1
                    _update_progress(failed=failed)
                    continue

                meta_dict = meta_to_dict(result.asset_meta)

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

                done += 1
                _update_progress(done=done)

                if done % 100 == 0:
                    logger.info(f"[repair-parse] {done}/{total} done, {failed} failed, {skipped} skipped")

            except Exception as e:
                logger.warning(f"[repair-parse] ERROR {display}: {e}")
                try:
                    db.conn.rollback()
                except Exception:
                    pass
                failed += 1
                _update_progress(failed=failed)

        logger.info(
            f"[repair-parse] Complete: {done} repaired, {failed} failed, "
            f"{skipped} skipped (WebDAV/missing), {total} total"
        )
        _update_progress(phase="done")

    except Exception as e:
        logger.error(f"[repair-parse] Fatal error: {e}", exc_info=True)
        _update_progress(phase="error")
    finally:
        with _repair_lock:
            _repair_state["running"] = False
            _repair_state["progress"]["current_file"] = ""
        try:
            db.close()
        except Exception:
            pass
        logger.info("[repair-parse] Thread finished")


@router.post("/admin/tools/repair-parse")
def start_repair_parse(
    _user: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Re-parse all accessible files and repair metadata JSON + FTS."""
    with _repair_lock:
        if _repair_state["running"]:
            return {
                "success": False,
                "detail": "Repair already running",
                "progress": dict(_repair_state["progress"]),
            }

    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM files WHERE file_path LIKE 'webdav://%'")
    webdav_count = cursor.fetchone()[0]
    local_count = total - webdav_count

    if total == 0:
        return {"success": False, "detail": "No files in database"}

    logger.info(f"[repair-parse] Starting: {total} total, {local_count} local, {webdav_count} WebDAV (skip)")

    from backend.server.config import get_db_path
    db_path = get_db_path()

    with _repair_lock:
        _repair_state["running"] = True
        _repair_state["progress"] = {
            "done": 0,
            "failed": 0,
            "skipped": 0,
            "total": total,
            "current_file": "",
            "phase": "starting",
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
        "local_count": local_count,
        "webdav_count": webdav_count,
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
