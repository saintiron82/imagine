"""Connection info router.

Admin-only — exposes the connection candidate inventory so the admin UI
and worker command generator stay in sync. Phase 3/4 completion:
the response now reflects the *actual* relay state (configured endpoint
+ live connector status) instead of the previous hardcoded zeros.
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


def _resolve_relay_state() -> tuple[str, bool]:
    """Return (relay_endpoint, relay_online) from current server state.

    Reads config first; then asks the live RelayClient singleton (if any)
    whether the connection is actually established. The latter prevents
    the UI from claiming "online" when only a stale env var exists.
    """
    endpoint = ""
    try:
        from backend.server.config import get_relay_config

        endpoint = (get_relay_config().get("endpoint") or "").strip()
    except Exception:
        endpoint = ""

    online = False
    try:
        from backend.server import relay_client as _relay

        singleton = getattr(_relay, "_singleton", None)
        if singleton is not None:
            online = bool(singleton.is_connected)
            # Fall back to the singleton's endpoint when config is empty
            # (e.g., started via env var without writing config.yaml).
            if not endpoint:
                endpoint = singleton.endpoint or ""
    except Exception:
        online = False

    return endpoint, online


@router.get("/server/connection-info")
def get_connection_info(
    request: Request,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    request_origin = str(request.base_url).rstrip("/")
    relay_endpoint, relay_online = _resolve_relay_state()
    return _ci.build_connection_info(
        db=db,
        request_origin=request_origin,
        port=_port_from_request(request),
        lan_ip=_detect_lan_ip(),
        public_ip=_detect_public_ip(),
        relay_endpoint=relay_endpoint,
        relay_online=relay_online,
    )


@router.get("/admin/relay/status")
def get_relay_status(
    admin: dict = Depends(require_admin),
):
    """Live status of the local relay connector.

    Separate endpoint so the UI can poll relay state without rebuilding
    the full connect_modes inventory.
    """
    endpoint, online = _resolve_relay_state()
    configured = bool(endpoint)
    secret_configured = False
    try:
        from backend.server.config import get_relay_config

        secret_configured = bool((get_relay_config().get("server_secret") or "").strip())
    except Exception:
        secret_configured = False

    return {
        "configured": configured,
        "secret_configured": secret_configured,
        "online": online,
        "endpoint": endpoint,
    }
