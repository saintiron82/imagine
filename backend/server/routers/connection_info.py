"""Connection info router.

Admin-only — exposes the connection candidate inventory so the admin UI
and worker command generator stay in sync.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Request

from backend.db.sqlite_client import SQLiteDB
from backend.server import connection_info as _ci
from backend.server.deps import get_db_safe, require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["connection-info"])

# Re-exposed so tests can monkeypatch network detection without touching
# the helper module.
_detect_lan_ip = _ci.detect_lan_ip
_detect_public_ip = _ci.detect_public_ip


def _port_from_request(request: Request, fallback: int = 8000) -> int:
    parsed = urlparse(str(request.base_url))
    if parsed.port:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return fallback


@router.get("/server/connection-info")
def get_connection_info(
    request: Request,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    request_origin = str(request.base_url).rstrip("/")
    return _ci.build_connection_info(
        db=db,
        request_origin=request_origin,
        port=_port_from_request(request),
        lan_ip=_detect_lan_ip(),
        public_ip=_detect_public_ip(),
        # relay endpoint/online wired in Phase 5
        relay_endpoint="",
        relay_online=False,
    )
