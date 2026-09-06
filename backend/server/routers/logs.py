"""Admin live-log view (IMGV2-26) — cursor-polling over the in-memory ring buffer.

EventSource(SSE) can't carry the Authorization header our JWT auth needs, so the
client polls GET /admin/logs?after=<seq> instead. Near-real-time (~1.5s) and
robust with the existing Bearer auth path.
"""
from typing import Optional

from fastapi import APIRouter, Depends

from backend.server import log_buffer
from backend.server.deps import require_admin

router = APIRouter(prefix="/admin/logs", tags=["logs"])


@router.get("")
def get_logs(
    after: int = 0,
    limit: int = 300,
    level: Optional[str] = None,
    _admin: dict = Depends(require_admin),
):
    """Return log entries newer than `after` (cursor) + the current last_seq."""
    limit = min(max(limit, 1), 800)
    return log_buffer.get_since(after=after, limit=limit, level=level)
