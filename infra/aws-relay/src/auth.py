"""Imagine relay protocol — shared validation and auth helpers.

This module is the single source of truth for the relay control plane.
It is imported both by the AWS Lambda handlers (connect/disconnect/
message) and by the local contract tests under tests/. Keep it free of
AWS-specific imports so it can be unit-tested with plain Python.

See docs/relay_protocol_contract_ko.md for the human-readable spec.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional


@dataclass(frozen=True)
class RelayLimits:
    """Hard caps the relay enforces on every envelope.

    The values are also documented in docs/relay_protocol_contract_ko.md
    — if you change one here, change the doc and the contract test.
    """

    MAX_MESSAGE_BYTES: int = 16 * 1024
    HEARTBEAT_INTERVAL_SECONDS: int = 60
    IDLE_TIMEOUT_SECONDS: int = 600  # 10 minutes
    SESSION_MAX_DURATION_SECONDS: int = 2 * 60 * 60  # 2 hours
    MAX_TIMESTAMP_SKEW_SECONDS: int = 5 * 60


PROTOCOL_VERSION = 1

ALLOWED_TYPES = frozenset(
    {
        "server.register",
        "server.heartbeat",
        "client.attach",
        "worker.attach",
        "worker.claim",
        "worker.complete",
        "session.close",
        "error",
    }
)

ERROR_CODES = frozenset(
    {
        "BAD_ENVELOPE",
        "UNSUPPORTED_V",
        "UNKNOWN_TYPE",
        "AUTH_FAILED",
        "PAYLOAD_TOO_BIG",
        "RATE_LIMITED",
        "NO_SUCH_SERVER",
        "FORBIDDEN_TYPE",
        "IDLE_TIMEOUT",
        "MAX_DURATION",
    }
)

# Anything that smells like raw payload bytes or a high-trust secret is
# refused at the envelope level — relay never carries this content.
FORBIDDEN_BODY_KEYS = frozenset(
    {
        "file_bytes",
        "image_bytes",
        "thumbnail_bytes",
        "raw_bytes",
        "payload_bytes",
        "weights",
        "model_weights",
        "db_bytes",
        "server_password",
        "server_password_hash",
        "admin_refresh_token",
        "admin_token",
        "firebase_id_token",
    }
)

_REQUIRED_ENVELOPE_KEYS = ("v", "type", "msg_id", "server_id", "ts", "body")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate_envelope(envelope: dict) -> tuple[bool, Optional[str]]:
    """Return (ok, error_code) for a single decoded envelope."""

    if not isinstance(envelope, dict):
        return False, "BAD_ENVELOPE"

    for key in _REQUIRED_ENVELOPE_KEYS:
        if key not in envelope:
            return False, "BAD_ENVELOPE"

    if envelope["v"] != PROTOCOL_VERSION:
        return False, "UNSUPPORTED_V"

    msg_type = envelope["type"]
    if not isinstance(msg_type, str):
        return False, "BAD_ENVELOPE"
    if msg_type not in ALLOWED_TYPES:
        return False, "UNKNOWN_TYPE"

    msg_id = envelope["msg_id"]
    server_id = envelope["server_id"]
    if not isinstance(msg_id, str) or not msg_id:
        return False, "BAD_ENVELOPE"
    if not isinstance(server_id, str) or not server_id:
        return False, "BAD_ENVELOPE"

    ts_value = envelope["ts"]
    if not isinstance(ts_value, str):
        return False, "BAD_ENVELOPE"
    ts = _parse_ts(ts_value)
    if ts is None:
        return False, "BAD_ENVELOPE"
    if abs((_now() - ts).total_seconds()) > RelayLimits.MAX_TIMESTAMP_SKEW_SECONDS:
        return False, "BAD_ENVELOPE"

    body = envelope["body"]
    if not isinstance(body, dict):
        return False, "BAD_ENVELOPE"

    bad = FORBIDDEN_BODY_KEYS.intersection(body.keys())
    if bad:
        return False, "FORBIDDEN_TYPE"

    return True, None


def validate_envelope_bytes(raw: bytes) -> tuple[bool, Optional[str]]:
    """Convenience wrapper that also enforces the byte size cap."""
    if len(raw) > RelayLimits.MAX_MESSAGE_BYTES:
        return False, "PAYLOAD_TOO_BIG"
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "BAD_ENVELOPE"
    return validate_envelope(envelope)


def authorize_register(
    envelope: dict, *, expected_secret: str
) -> tuple[bool, Optional[str]]:
    """`server.register` must carry the server's pre-shared secret."""
    body = envelope.get("body") or {}
    presented = body.get("server_secret") or ""
    if not expected_secret or not presented:
        return False, "AUTH_FAILED"
    if not hmac.compare_digest(presented, expected_secret):
        return False, "AUTH_FAILED"
    return True, None


def authorize_attach(
    envelope: dict, *, expected_kind: str
) -> tuple[bool, Optional[str]]:
    """`client.attach` / `worker.attach` must carry a non-empty token."""
    body = envelope.get("body") or {}
    if expected_kind == "client":
        token = body.get("attach_token") or ""
    elif expected_kind == "worker":
        token = body.get("enrollment_token") or ""
    else:
        return False, "AUTH_FAILED"
    if not isinstance(token, str) or not token.strip():
        return False, "AUTH_FAILED"
    return True, None


def route(
    envelope: dict,
    *,
    known_servers: Iterable[str],
) -> dict:
    """Decide how to forward a validated envelope.

    Returns a small descriptor instead of side-effecting so this stays
    unit-testable. Lambdas wrap it with the actual API Gateway call.
    """
    server_id = envelope.get("server_id") or ""
    msg_id = envelope.get("msg_id")
    known = set(known_servers or [])

    if server_id not in known:
        return {
            "action": "error",
            "code": "NO_SUCH_SERVER",
            "msg_id": msg_id,
        }

    return {
        "action": "forward",
        "target": server_id,
        "msg_id": msg_id,
    }


def make_error(code: str, *, msg_id: Optional[str] = None, detail: str = "") -> dict:
    """Build a fully-formed `error` envelope ready to send back."""
    if code not in ERROR_CODES:
        code = "BAD_ENVELOPE"
    return {
        "v": PROTOCOL_VERSION,
        "type": "error",
        "msg_id": msg_id or "",
        "server_id": "",
        "session_id": None,
        "ts": _now().isoformat(),
        "body": {"code": code, "detail": detail},
    }
