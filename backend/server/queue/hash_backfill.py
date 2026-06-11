"""Content-hash backfill (CAS M2) — pre-burn preparation.

Fills files.content_hash for rows that predate hash computation, so the
derivation cache (M1 shadow write, M3 reads) covers the whole library.

The hash is a boundary hash (size + first/last 8KB), so remote WebDAV
files need only two small Range reads (~16KB/file) instead of a full
re-download — the entire backlog (~14k files, ~1TB of originals) costs
roughly 230MB of transfer.

Runs as a single background thread inside the server (admin-triggered).
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_state = {
    "running": False,
    "total": 0,
    "done": 0,
    "failed": 0,
    "skipped": 0,
    "last_error": None,
}


def get_status() -> dict:
    with _lock:
        return dict(_state)


def start_backfill(db_factory) -> dict:
    """Start the backfill thread. db_factory: callable returning a SQLiteDB
    (each thread needs its own thread-local connection entry point)."""
    with _lock:
        if _state["running"]:
            return {"success": False, "error": "Backfill already running"}
        _state.update(running=True, total=0, done=0, failed=0,
                      skipped=0, last_error=None)

    threading.Thread(target=_run, args=(db_factory,), daemon=True,
                     name="hash-backfill").start()
    return {"success": True}


def _run(db_factory):
    try:
        db = db_factory()
        cursor = db.conn.cursor()
        cursor.execute(
            """SELECT id, file_path, file_size FROM files
               WHERE content_hash IS NULL OR content_hash = ''"""
        )
        rows = cursor.fetchall()
        with _lock:
            _state["total"] = len(rows)

        clients = {}  # source_id → WebDAVClient
        pending_commit = 0
        for file_id, file_path, file_size in rows:
            if not _state["running"]:
                break
            try:
                content_hash = _hash_one(file_path, file_size, clients)
                if content_hash:
                    cursor.execute(
                        "UPDATE files SET content_hash = ? WHERE id = ?",
                        (content_hash, file_id),
                    )
                    pending_commit += 1
                    with _lock:
                        _state["done"] += 1
                    if pending_commit >= 50:
                        db.conn.commit()
                        pending_commit = 0
                else:
                    with _lock:
                        _state["skipped"] += 1
            except Exception as e:
                with _lock:
                    _state["failed"] += 1
                    _state["last_error"] = str(e)[:200]
        db.conn.commit()

        for c in clients.values():
            try:
                c.close()
            except Exception:
                pass
        logger.info(
            f"Hash backfill finished: done={_state['done']} "
            f"skipped={_state['skipped']} failed={_state['failed']}"
        )
    except Exception as e:
        logger.error(f"Hash backfill crashed: {e}")
        with _lock:
            _state["last_error"] = str(e)[:200]
    finally:
        with _lock:
            _state["running"] = False


def _hash_one(file_path: str, file_size, clients: dict):
    """Hash a single file — local read or remote Range reads."""
    from backend.utils.content_hash import (
        compute_content_hash, compute_content_hash_from_parts, split_points,
    )

    if not file_path.startswith("webdav://"):
        p = Path(file_path)
        if not p.exists():
            return None  # missing local file — skip, audit handles it
        return compute_content_hash(p)

    # Remote: two Range reads against the registered source
    from backend.server.queue.download_ahead import (
        parse_webdav_path, get_webdav_source,
    )
    source_id, remote_rel = parse_webdav_path(file_path)
    if source_id not in clients:
        cfg = get_webdav_source(source_id)
        if not cfg:
            return None  # source not registered on this server — skip
        from backend.remote.webdav_client import WebDAVClient
        clients[source_id] = WebDAVClient(
            base_url=cfg["url"],
            username=cfg["username"],
            password=cfg["password"],
            remote_path="/",
            verify_ssl=cfg.get("verify_ssl", True),
        )
    client = clients[source_id]

    size = int(file_size or 0)
    if size <= 0:
        return None  # no recorded size — needs a PROPFIND pass; skip for now

    head_range, tail_range = split_points(size)
    if head_range is None:
        return None
    head = client.read_range(remote_rel, *head_range)
    if head is None:
        raise RuntimeError(f"range read failed: {remote_rel}")
    tail = b""
    if tail_range:
        tail = client.read_range(remote_rel, *tail_range)
        if tail is None:
            raise RuntimeError(f"range read failed (tail): {remote_rel}")
    return compute_content_hash_from_parts(size, head, tail)


def stop_backfill():
    with _lock:
        _state["running"] = False
