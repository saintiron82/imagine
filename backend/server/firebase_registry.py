"""
Firestore server group registration.

On server init, register group info (group_name, LAN IP, port) to Firestore
so external clients can discover the server by group name.

Phase 1 of the secure external worker access plan extends the payload with
connection-mode metadata so clients/workers can pick a candidate URL without
guessing public-IP reachability.

Uses Firestore REST API (no SDK dependency) — same collection as
frontend/src/api/firebase.js.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

FIRESTORE_BASE = (
    "https://firestore.googleapis.com/v1/projects/imagine-b1e9c"
    "/databases/(default)/documents/groups"
)

CONNECT_MODES = (
    "direct_local",
    "direct_lan",
    "manual_external",
    "relay_session",
    "cloud_server",
)

REACHABLE_STATES = ("unknown", "verified", "unreachable")


def _to_key(group_name: str) -> str:
    """Normalize group name → Firestore document key (same as frontend toKey)."""
    return group_name.strip().lower().replace(" ", "_")


def _get_lan_ip() -> str:
    """Get the LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _get_public_ip() -> str:
    """Get public IP via external service (best-effort)."""
    try:
        req = urllib.request.Request(
            "https://api.ipify.org?format=json",
            headers={"User-Agent": "Imagine-Server/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("ip", "")
    except Exception:
        return ""


def _default_connect_mode(lan_ip: str, external_url: str, relay_online: bool) -> str:
    if relay_online:
        return "relay_session"
    if external_url:
        return "manual_external"
    if lan_ip and lan_ip != "127.0.0.1":
        return "direct_lan"
    return "direct_local"


def register_group(
    group_name: str,
    port: int,
    *,
    connect_mode: Optional[str] = None,
    external_url: str = "",
    tunnel_url: str = "",
    relay_endpoint: str = "",
    relay_online: bool = False,
    reachable_status: str = "unknown",
) -> bool:
    """Register server group to Firestore (best-effort).

    Returns True if successful, False otherwise.
    Does NOT raise exceptions — server must continue regardless.

    `connect_mode` and the *_url/relay_* arguments let an admin/relay
    connector override the published candidate set. When omitted, the
    function publishes the best guess for direct discovery.
    """
    try:
        from datetime import datetime, timezone

        key = _to_key(group_name)
        lan_ip = _get_lan_ip()
        public_ip = _get_public_ip()
        lan_url = f"http://{lan_ip}:{port}" if lan_ip else ""

        if connect_mode is None:
            connect_mode = _default_connect_mode(lan_ip, external_url, relay_online)
        if connect_mode not in CONNECT_MODES:
            logger.warning("Unknown connect_mode=%r; falling back to direct_lan", connect_mode)
            connect_mode = "direct_lan"
        if reachable_status not in REACHABLE_STATES:
            reachable_status = "unknown"

        fields = {
            "group_name": {"stringValue": group_name},
            "lan_ip": {"stringValue": lan_ip},
            "public_ip": {"stringValue": public_ip},
            "port": {"integerValue": str(port)},
            "connect_mode": {"stringValue": connect_mode},
            "lan_url": {"stringValue": lan_url},
            "external_url": {"stringValue": external_url or ""},
            "tunnel_url": {"stringValue": tunnel_url or ""},
            "relay_endpoint": {"stringValue": relay_endpoint or ""},
            "relay_online": {"booleanValue": bool(relay_online)},
            "reachable_status": {"stringValue": reachable_status},
            "updated_at": {"stringValue": datetime.now(timezone.utc).isoformat()},
        }

        url = f"{FIRESTORE_BASE}/{key}"
        data = json.dumps({"fields": fields}).encode("utf-8")

        req = urllib.request.Request(
            url, data=data, method="PATCH",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status == 200:
                logger.info(
                    "Registered group %r to Firestore "
                    "(mode=%s LAN=%s public=%s port=%d relay=%s)",
                    group_name, connect_mode, lan_ip, public_ip, port, relay_online,
                )
                return True
            logger.warning("Firestore registration returned HTTP %s", resp.status)
            return False

    except Exception as e:
        logger.warning(f"Firestore registration failed (non-critical): {e}")
        return False
