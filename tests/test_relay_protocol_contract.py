"""Phase 4: Relay protocol contract.

These tests cover the parts of the protocol that the local server and
worker depend on. The actual AWS Lambda runtime is exercised via the
same module so the contract stays single-source.
"""

from __future__ import annotations

import importlib
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAMBDA_SRC = REPO_ROOT / "infra" / "aws-relay" / "src"


@pytest.fixture(scope="module")
def relay_auth():
    """Import infra/aws-relay/src/auth.py without packaging.

    Lambdas live outside the python package tree on purpose — they ship
    in their own deployment artefact. The contract test still treats
    them as the canonical source.
    """
    if str(LAMBDA_SRC) not in sys.path:
        sys.path.insert(0, str(LAMBDA_SRC))
    if "auth" in sys.modules:
        return importlib.reload(sys.modules["auth"])
    return importlib.import_module("auth")


def _envelope(msg_type: str, **body) -> dict:
    return {
        "v": 1,
        "type": msg_type,
        "msg_id": str(uuid.uuid4()),
        "server_id": "imagine-test",
        "session_id": None,
        "ts": datetime.now(timezone.utc).isoformat(),
        "body": body,
    }


# ── Hard limits stay in sync with the contract document ─────────────


def test_limits_match_documented_hard_caps(relay_auth):
    limits = relay_auth.RelayLimits
    assert limits.MAX_MESSAGE_BYTES == 16 * 1024
    assert limits.HEARTBEAT_INTERVAL_SECONDS == 60
    assert limits.IDLE_TIMEOUT_SECONDS == 600
    assert limits.SESSION_MAX_DURATION_SECONDS == 2 * 60 * 60


def test_message_types_are_the_documented_set(relay_auth):
    expected = {
        "server.register",
        "server.heartbeat",
        "client.attach",
        "worker.attach",
        "worker.claim",
        "worker.complete",
        "session.close",
        "error",
    }
    assert relay_auth.ALLOWED_TYPES == expected


def test_error_codes_are_complete(relay_auth):
    expected = {
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
    assert expected.issubset(relay_auth.ERROR_CODES)


# ── Envelope validation ────────────────────────────────────────────


def test_validate_envelope_accepts_well_formed(relay_auth):
    ok, err = relay_auth.validate_envelope(_envelope("server.heartbeat", online=True))
    assert ok is True
    assert err is None


def test_validate_envelope_rejects_missing_fields(relay_auth):
    env = _envelope("server.heartbeat", online=True)
    env.pop("type")
    ok, err = relay_auth.validate_envelope(env)
    assert ok is False
    assert err == "BAD_ENVELOPE"


def test_validate_envelope_rejects_unknown_type(relay_auth):
    env = _envelope("server.heartbeat", online=True)
    env["type"] = "server.exfiltrate"
    ok, err = relay_auth.validate_envelope(env)
    assert ok is False
    assert err == "UNKNOWN_TYPE"


def test_validate_envelope_rejects_old_version(relay_auth):
    env = _envelope("server.heartbeat", online=True)
    env["v"] = 0
    ok, err = relay_auth.validate_envelope(env)
    assert ok is False
    assert err == "UNSUPPORTED_V"


def test_validate_envelope_rejects_stale_timestamp(relay_auth):
    env = _envelope("server.heartbeat", online=True)
    drifted = datetime.now(timezone.utc) - timedelta(minutes=10)
    env["ts"] = drifted.isoformat()
    ok, err = relay_auth.validate_envelope(env)
    assert ok is False
    assert err == "BAD_ENVELOPE"


def test_validate_envelope_rejects_oversized_payload(relay_auth):
    env = _envelope("server.heartbeat", online=True)
    env["body"] = {"blob": "x" * (relay_auth.RelayLimits.MAX_MESSAGE_BYTES)}
    encoded = json.dumps(env).encode("utf-8")
    ok, err = relay_auth.validate_envelope_bytes(encoded)
    assert ok is False
    assert err == "PAYLOAD_TOO_BIG"


# ── Forbidden content (data plane policy) ──────────────────────────


@pytest.mark.parametrize(
    "forbidden_key",
    ["file_bytes", "image_bytes", "thumbnail_bytes", "raw_bytes", "weights"],
)
def test_validate_envelope_rejects_file_bytes(relay_auth, forbidden_key):
    env = _envelope("worker.complete", task_id=1, **{forbidden_key: "AAAA"})
    ok, err = relay_auth.validate_envelope(env)
    assert ok is False
    assert err == "FORBIDDEN_TYPE"


def test_validate_envelope_rejects_secret_leakage(relay_auth):
    """relay must never accept server password / admin refresh tokens."""
    env = _envelope("server.register", server_secret="ok", server_password="nope")
    ok, err = relay_auth.validate_envelope(env)
    assert ok is False
    assert err == "FORBIDDEN_TYPE"

    env = _envelope("client.attach", attach_token="ok", admin_refresh_token="nope")
    ok, err = relay_auth.validate_envelope(env)
    assert ok is False
    assert err == "FORBIDDEN_TYPE"


# ── Auth helpers ───────────────────────────────────────────────────


def test_server_register_requires_secret(relay_auth):
    env = _envelope("server.register")
    ok, err = relay_auth.authorize_register(env, expected_secret="topsecret")
    assert ok is False
    assert err == "AUTH_FAILED"

    env = _envelope("server.register", server_secret="topsecret", group_name="studio")
    ok, err = relay_auth.authorize_register(env, expected_secret="topsecret")
    assert ok is True
    assert err is None


def test_client_attach_requires_token(relay_auth):
    env = _envelope("client.attach", attach_token="")
    ok, err = relay_auth.authorize_attach(env, expected_kind="client")
    assert ok is False
    assert err == "AUTH_FAILED"

    env = _envelope("client.attach", attach_token="abc")
    ok, err = relay_auth.authorize_attach(env, expected_kind="client")
    assert ok is True


def test_worker_attach_requires_enrollment_token(relay_auth):
    env = _envelope("worker.attach")
    ok, err = relay_auth.authorize_attach(env, expected_kind="worker")
    assert ok is False
    assert err == "AUTH_FAILED"

    env = _envelope("worker.attach", enrollment_token="abc")
    ok, err = relay_auth.authorize_attach(env, expected_kind="worker")
    assert ok is True


# ── Routing ────────────────────────────────────────────────────────


def test_route_message_requires_known_server(relay_auth):
    env = _envelope("client.attach", attach_token="abc")
    decision = relay_auth.route(env, known_servers=set())
    assert decision["action"] == "error"
    assert decision["code"] == "NO_SUCH_SERVER"


def test_route_message_with_registered_server(relay_auth):
    env = _envelope("client.attach", attach_token="abc")
    decision = relay_auth.route(env, known_servers={"imagine-test"})
    assert decision["action"] == "forward"
    assert decision["target"] == "imagine-test"
