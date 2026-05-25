"""Phase 5: server-side relay connector contract.

These tests exercise the local state machine and Firebase update flow
without opening a real WebSocket. The relay client accepts a transport
factory so tests can plug in a stub that records what was sent.
"""

from __future__ import annotations

import json
import time
from typing import Callable

import pytest

from backend.server import relay_client as rc


class FakeTransport:
    """Records all outbound envelopes and lets the test drive callbacks."""

    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False
        self._on_message: Callable[[dict], None] | None = None
        self.connect_attempts = 0

    def connect(self):
        self.connect_attempts += 1

    def send(self, envelope: dict):
        self.sent.append(envelope)

    def close(self):
        self.closed = True

    def set_on_message(self, cb):
        self._on_message = cb

    def deliver(self, envelope: dict):
        if self._on_message:
            self._on_message(envelope)


class FailingTransport(FakeTransport):
    """First N connect attempts raise — used to test backoff."""

    def __init__(self, fail_times: int):
        super().__init__()
        self._fail_remaining = fail_times

    def connect(self):
        self.connect_attempts += 1
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise ConnectionError("fake failure")


def _config(**overrides):
    base = {
        "relay.enabled": True,
        "relay.endpoint": "wss://relay.example.com/prod",
        "relay.server_id": "imagine-test",
        "relay.server_secret": "topsecret",
        "relay.auto_connect": True,
        "relay.heartbeat_interval_seconds": 60,
        "relay.idle_timeout_seconds": 600,
    }
    base.update(overrides)
    return base


def _make_client(transport, *, config=None, registry_log=None):
    cfg = config or _config()
    if registry_log is None:
        registry_log = []

    def fake_register(group_name, port, **fields):
        registry_log.append({"group_name": group_name, "port": port, **fields})
        return True

    return rc.RelayClient(
        config=cfg,
        group_name="studio",
        port=8000,
        transport_factory=lambda endpoint: transport,
        firebase_register=fake_register,
        clock=lambda: time.time(),
    ), registry_log


# ── Config / activation ────────────────────────────────────────────


def test_disabled_relay_does_not_connect():
    transport = FakeTransport()
    client, log = _make_client(transport, config=_config(**{"relay.enabled": False}))

    client.start()

    assert transport.connect_attempts == 0
    assert client.is_connected is False
    assert log == []


def test_missing_endpoint_blocks_connect():
    transport = FakeTransport()
    client, log = _make_client(transport, config=_config(**{"relay.endpoint": ""}))

    client.start()

    assert transport.connect_attempts == 0
    assert client.is_connected is False


def test_missing_secret_blocks_connect():
    transport = FakeTransport()
    client, log = _make_client(transport, config=_config(**{"relay.server_secret": ""}))

    client.start()

    assert transport.connect_attempts == 0
    assert client.is_connected is False


# ── Connect flow ───────────────────────────────────────────────────


def test_start_sends_register_envelope():
    transport = FakeTransport()
    client, _ = _make_client(transport)

    client.start()

    assert transport.connect_attempts == 1
    assert len(transport.sent) == 1
    env = transport.sent[0]
    assert env["type"] == "server.register"
    assert env["server_id"] == "imagine-test"
    assert env["body"]["server_secret"] == "topsecret"
    assert env["body"]["group_name"] == "studio"


def test_register_ack_publishes_relay_online_to_firebase():
    transport = FakeTransport()
    client, log = _make_client(transport)

    client.start()
    # Lambda would echo a session.close ack on failure — here we simulate
    # the happy-path register-ok message.
    transport.deliver({
        "v": 1,
        "type": "server.heartbeat",
        "msg_id": "ack-1",
        "server_id": "imagine-test",
        "ts": "2026-05-25T00:00:00+00:00",
        "body": {"ok": True},
    })

    assert client.is_connected is True
    # Firebase should have received an update marking relay_online=true.
    online_updates = [e for e in log if e.get("relay_online") is True]
    assert online_updates
    last = online_updates[-1]
    assert last["connect_mode"] == "relay_session"
    assert last["relay_endpoint"] == "wss://relay.example.com/prod"


def test_stop_marks_relay_offline_in_firebase():
    transport = FakeTransport()
    client, log = _make_client(transport)

    client.start()
    transport.deliver({
        "v": 1,
        "type": "server.heartbeat",
        "msg_id": "ack-1",
        "server_id": "imagine-test",
        "ts": "2026-05-25T00:00:00+00:00",
        "body": {"ok": True},
    })
    client.stop()

    assert transport.closed is True
    offline = [e for e in log if e.get("relay_online") is False]
    assert offline


# ── Heartbeat / send envelope shape ────────────────────────────────


def test_heartbeat_sends_envelope_when_triggered():
    transport = FakeTransport()
    client, _ = _make_client(transport)

    client.start()
    transport.sent.clear()

    client._tick_heartbeat(force=True)

    assert len(transport.sent) == 1
    env = transport.sent[0]
    assert env["type"] == "server.heartbeat"
    assert env["server_id"] == "imagine-test"
    assert env["body"]["online"] is True


def test_envelope_validates_against_protocol_contract():
    """Every outbound envelope must pass the shared relay contract."""
    import sys
    from pathlib import Path

    relay_src = Path(__file__).resolve().parents[1] / "infra" / "aws-relay" / "src"
    if str(relay_src) not in sys.path:
        sys.path.insert(0, str(relay_src))
    import auth as relay_auth  # type: ignore

    transport = FakeTransport()
    client, _ = _make_client(transport)

    client.start()
    client._tick_heartbeat(force=True)

    for env in transport.sent:
        ok, err = relay_auth.validate_envelope(env)
        assert ok, f"envelope {env['type']} failed contract: {err}"
        # And the raw bytes still pass the size cap.
        raw = json.dumps(env).encode("utf-8")
        ok, err = relay_auth.validate_envelope_bytes(raw)
        assert ok, f"raw envelope failed: {err}"


# ── Reconnect / backoff ────────────────────────────────────────────


def test_failed_connect_does_not_publish_relay_online():
    transport = FailingTransport(fail_times=1)
    client, log = _make_client(transport)

    client.start()

    online_updates = [e for e in log if e.get("relay_online") is True]
    assert online_updates == []
    assert client.is_connected is False
    assert transport.connect_attempts == 1


def test_request_reconnect_re_attempts_connection():
    transport = FailingTransport(fail_times=1)
    client, _ = _make_client(transport)

    client.start()
    # Drop the failure marker and retry — emulates backoff fire.
    transport._fail_remaining = 0
    client.request_reconnect()

    assert transport.connect_attempts == 2
    assert len(transport.sent) == 1
    assert transport.sent[0]["type"] == "server.register"
