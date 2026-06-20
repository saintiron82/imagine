"""
In-memory ring buffer of recent server log records + a logging.Handler (IMGV2-26).

Feeds the admin live-log view via a cursor-polling endpoint (GET /admin/logs).
Thread-safe, bounded, and non-blocking for the emitting thread — log records
arrive from many worker/pool threads, so emit() only does a cheap append.
"""
import logging
import threading
from collections import deque
from typing import Optional

_MAX = 800
_lock = threading.Lock()
_buf: deque = deque(maxlen=_MAX)
_seq = 0

_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

_NETWORK_HINTS = ("download", "webdav", "http", "relay", "tunnel", "connection", "nas", "sync")
_JOB_HINTS = ("job", "task", "phase", "pipeline", "worker", "scheduler", "analysis", "vision", "embed", "parser")


def _category(name: str) -> str:
    """Coarse bucket so the UI can filter network vs. job vs. general lines."""
    n = (name or "").lower()
    if any(h in n for h in _NETWORK_HINTS):
        return "network"
    if any(h in n for h in _JOB_HINTS):
        return "job"
    return "general"


class RingBufferLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        global _seq
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(getattr(record, "msg", ""))
        if len(msg) > 1000:
            msg = msg[:1000] + "…"
        entry = {
            "ts": record.created,
            "level": record.levelname,
            "name": record.name,
            "category": _category(record.name),
            "message": msg,
        }
        with _lock:
            _seq += 1
            entry["seq"] = _seq
            _buf.append(entry)


def get_since(after: int = 0, limit: int = 300, level: Optional[str] = None) -> dict:
    """Return log entries with seq > after (capped at `limit` most-recent), plus last_seq."""
    with _lock:
        items = [e for e in _buf if e["seq"] > after]
        last = _seq
    if level:
        wanted = _LEVEL_ORDER.get(level.upper(), 0)
        items = [e for e in items if _LEVEL_ORDER.get(e["level"], 0) >= wanted]
    if len(items) > limit:
        items = items[-limit:]
    return {"entries": items, "last_seq": last}


_installed = False


def install(level: int = logging.INFO) -> None:
    """Attach the ring-buffer handler to the root logger (idempotent).

    Also ensures the root logger's effective level admits INFO — otherwise
    INFO records are filtered at the logger before ever reaching any handler.
    """
    global _installed
    root = logging.getLogger()
    # Truly idempotent: drop any prior ring handler before attaching. All handlers
    # share the module-global _buf/_seq, so two attached handlers double-count
    # every record (app-import install + a test fixture re-install would do this).
    for h in list(root.handlers):
        if isinstance(h, RingBufferLogHandler):
            root.removeHandler(h)
    if root.getEffectiveLevel() > level:
        root.setLevel(level)
    handler = RingBufferLogHandler()
    handler.setLevel(level)
    root.addHandler(handler)
    _installed = True


def _reset_for_tests() -> None:
    global _seq
    with _lock:
        _buf.clear()
        _seq = 0
