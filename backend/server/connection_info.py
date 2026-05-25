"""Connection candidate inventory.

The admin UI and worker command generator share this single source of truth
so they advertise the same connection modes.

The module deliberately depends only on the request URL plus locally
detectable network info; secrets (tokens, password hashes, etc.) never
flow through these helpers.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.request
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def detect_lan_ip() -> str:
    """Best-effort LAN IP (UDP socket trick — no packets sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return ""


def detect_public_ip(timeout: float = 5.0) -> str:
    """Public IP from ipify; empty string on any failure."""
    try:
        req = urllib.request.Request(
            "https://api.ipify.org?format=json",
            headers={"User-Agent": "Imagine-Server/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()).get("ip", "")
    except Exception:
        return ""


def get_or_create_server_id(db) -> str:
    """Lazy-allocate a stable server_id in system_meta."""
    cur = db.conn.cursor()
    cur.execute("SELECT value FROM system_meta WHERE key = 'server_id'")
    row = cur.fetchone()
    if row and row[0]:
        return row[0]
    new_id = f"imagine-{uuid.uuid4().hex[:12]}"
    cur.execute(
        "INSERT OR REPLACE INTO system_meta (key, value) VALUES ('server_id', ?)",
        (new_id,),
    )
    db.conn.commit()
    return new_id


def get_group_name(db) -> str:
    cur = db.conn.cursor()
    cur.execute("SELECT value FROM system_meta WHERE key = 'group_name'")
    row = cur.fetchone()
    return row[0] if row and row[0] else ""


def build_connect_modes(
    *,
    port: int,
    lan_ip: str,
    public_ip: str,
    relay_endpoint: str = "",
    relay_online: bool = False,
) -> list[dict]:
    """Return the ordered candidate list.

    Direct local is always available. LAN is available when a non-loopback
    LAN IP is detected. Manual external is listed when a public IP is
    known but marked `available=False` — Imagine never auto-trusts a raw
    public port. Relay session is listed even when offline so the UI can
    advertise the recommended path.
    """

    safe_lan_ip = lan_ip if lan_ip and lan_ip != "127.0.0.1" else ""

    modes: list[dict] = [
        {
            "mode": "direct_local",
            "url": f"http://localhost:{port}",
            "reachable_scope": "same-machine",
            "security": "safe-local",
            "available": True,
        },
        {
            "mode": "direct_lan",
            "url": f"http://{safe_lan_ip}:{port}" if safe_lan_ip else "",
            "reachable_scope": "same-lan",
            "security": "lan-only",
            "available": bool(safe_lan_ip),
        },
        {
            "mode": "manual_external",
            "url": f"http://{public_ip}:{port}" if public_ip else "",
            "reachable_scope": "external",
            "security": "public-not-verified",
            "available": False,  # never auto-trust raw public port
        },
        {
            "mode": "relay_session",
            "url": relay_endpoint or "",
            "reachable_scope": "external",
            "security": "recommended-external",
            "available": bool(relay_online and relay_endpoint),
        },
    ]
    return modes


def build_connection_info(
    *,
    db,
    request_origin: str,
    port: int,
    lan_ip: Optional[str] = None,
    public_ip: Optional[str] = None,
    relay_endpoint: str = "",
    relay_online: bool = False,
) -> dict:
    if lan_ip is None:
        lan_ip = detect_lan_ip()
    if public_ip is None:
        public_ip = detect_public_ip()

    return {
        "server_id": get_or_create_server_id(db),
        "group_name": get_group_name(db),
        "request_origin": request_origin,
        "connect_modes": build_connect_modes(
            port=port,
            lan_ip=lan_ip,
            public_ip=public_ip,
            relay_endpoint=relay_endpoint,
            relay_online=relay_online,
        ),
    }
