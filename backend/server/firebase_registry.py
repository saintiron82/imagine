"""
Firestore server group registration.

On server init, register group info (group_name, LAN IP, port, tunnel_url)
to Firestore so external clients can discover the server by group name.

Uses Firestore REST API (no SDK dependency) — same collection as
frontend/src/api/firebase.js.
"""

import logging
import socket
import urllib.parse
import urllib.request
import json

logger = logging.getLogger(__name__)

FIRESTORE_BASE = (
    "https://firestore.googleapis.com/v1/projects/imagine-b1e9c"
    "/databases/(default)/documents/groups"
)


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


def register_group(group_name: str, port: int, tunnel_url: str = "") -> bool:
    """Register server group to Firestore (best-effort).

    Returns True if successful, False otherwise.
    Does NOT raise exceptions — server must continue regardless.
    """
    try:
        from datetime import datetime, timezone

        key = _to_key(group_name)
        lan_ip = _get_lan_ip()
        public_ip = _get_public_ip()

        fields = {
            "group_name": {"stringValue": group_name},
            "lan_ip": {"stringValue": lan_ip},
            "public_ip": {"stringValue": public_ip},
            "port": {"integerValue": str(port)},
            "tunnel_url": {"stringValue": tunnel_url},
            "updated_at": {"stringValue": datetime.now(timezone.utc).isoformat()},
        }

        # PATCH to create or update document (same as frontend registerGroup)
        url = f"{FIRESTORE_BASE}/{key}"
        data = json.dumps({"fields": fields}).encode("utf-8")

        req = urllib.request.Request(
            url, data=data, method="PATCH",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status == 200:
                tunnel_info = f", tunnel={tunnel_url}" if tunnel_url else ""
                logger.info(
                    f"Registered group '{group_name}' to Firestore "
                    f"(LAN={lan_ip}, Public={public_ip}, port={port}{tunnel_info})"
                )
                return True
            else:
                logger.warning(f"Firestore registration returned HTTP {resp.status}")
                return False

    except Exception as e:
        logger.warning(f"Firestore registration failed (non-critical): {e}")
        return False
