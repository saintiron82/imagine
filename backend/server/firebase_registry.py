"""
Firebase Realtime DB server registration.

On server init, register group info (group_name, LAN IP, port) to Firebase
so workers can discover the server by group name.
"""

import logging
import socket
import urllib.parse
import urllib.request
import json

logger = logging.getLogger(__name__)

FIREBASE_BASE = "https://imagine-b1e9c-default-rtdb.firebaseio.com/groups"


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


def register_group(group_name: str, port: int) -> bool:
    """Register server group to Firebase Realtime DB (best-effort).

    Returns True if successful, False otherwise.
    Does NOT raise exceptions — server must continue regardless.
    """
    try:
        from datetime import datetime, timezone

        key = group_name.strip().lower().replace(" ", "_")
        key = urllib.parse.quote(key, safe="")

        lan_ip = _get_lan_ip()
        public_ip = _get_public_ip()

        payload = {
            "group_name": group_name,
            "lan_ip": lan_ip,
            "public_ip": public_ip,
            "port": port,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        data = json.dumps(payload).encode("utf-8")
        url = f"{FIREBASE_BASE}/{key}.json"

        req = urllib.request.Request(
            url, data=data, method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status == 200:
                logger.info(f"Registered group '{group_name}' to Firebase (LAN={lan_ip}, Public={public_ip}, port={port})")
                return True
            else:
                logger.warning(f"Firebase registration returned HTTP {resp.status}")
                return False

    except Exception as e:
        logger.warning(f"Firebase registration failed (non-critical): {e}")
        return False
