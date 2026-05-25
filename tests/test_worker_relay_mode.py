"""Phase 6: worker relay mode contract.

Covers:
  - CLI accepts IMAGINE_CONNECT_MODE=relay only when the relay env set
    is complete.
  - RelayTransport produces protocol-compliant envelopes for the control
    operations a worker actually uses (attach/heartbeat/claim/complete).
  - File-byte methods (thumbnail, mc data, save_*) refuse to go over
    the relay — they raise an explicit error so the daemon falls back
    to direct mode or surfaces the problem clearly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAMBDA_SRC = REPO_ROOT / "infra" / "aws-relay" / "src"


@pytest.fixture(scope="module")
def relay_auth():
    if str(LAMBDA_SRC) not in sys.path:
        sys.path.insert(0, str(LAMBDA_SRC))
    import importlib
    if "auth" in sys.modules:
        return importlib.reload(sys.modules["auth"])
    return importlib.import_module("auth")


# ── CLI ────────────────────────────────────────────────────────────


def test_cli_relay_mode_requires_endpoint(monkeypatch):
    from backend.worker import cli

    monkeypatch.delenv("IMAGINE_RELAY_ENDPOINT", raising=False)
    monkeypatch.delenv("IMAGINE_SERVER_URL", raising=False)
    with pytest.raises(SystemExit):
        cli.load_headless_config(
            [
                "--connect-mode",
                "relay",
                "--server-id",
                "imagine-test",
                "--access-token",
                "tok",
                "--enrollment-token",
                "etok",
            ]
        )


def test_cli_relay_mode_requires_server_id(monkeypatch):
    from backend.worker import cli

    monkeypatch.delenv("IMAGINE_SERVER_URL", raising=False)
    with pytest.raises(SystemExit):
        cli.load_headless_config(
            [
                "--connect-mode",
                "relay",
                "--relay-endpoint",
                "wss://relay.example.com",
                "--access-token",
                "tok",
                "--enrollment-token",
                "etok",
            ]
        )


def test_cli_relay_mode_requires_enrollment_token(monkeypatch):
    from backend.worker import cli

    monkeypatch.delenv("IMAGINE_SERVER_URL", raising=False)
    with pytest.raises(SystemExit):
        cli.load_headless_config(
            [
                "--connect-mode",
                "relay",
                "--relay-endpoint",
                "wss://relay.example.com",
                "--server-id",
                "imagine-test",
                "--access-token",
                "tok",
            ]
        )


def test_cli_relay_mode_happy_path(monkeypatch):
    from backend.worker import cli

    monkeypatch.delenv("IMAGINE_SERVER_URL", raising=False)
    cfg = cli.load_headless_config(
        [
            "--connect-mode",
            "relay",
            "--relay-endpoint",
            "wss://relay.example.com",
            "--server-id",
            "imagine-test",
            "--access-token",
            "tok",
            "--enrollment-token",
            "etok",
            "--worker-name",
            "cloud-1",
        ]
    )
    assert cfg.connect_mode == "relay"
    assert cfg.relay_endpoint == "wss://relay.example.com"
    assert cfg.server_id == "imagine-test"
    assert cfg.enrollment_token == "etok"


def test_cli_direct_mode_still_requires_server_url(monkeypatch):
    from backend.worker import cli

    monkeypatch.delenv("IMAGINE_SERVER_URL", raising=False)
    with pytest.raises(SystemExit):
        cli.load_headless_config(["--access-token", "tok"])


# ── RelayTransport ────────────────────────────────────────────────


class _RecordingSocket:
    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False
        self._on_message = None
        self.connected = False

    def connect(self):
        self.connected = True

    def send(self, env):
        self.sent.append(env)

    def close(self):
        self.closed = True

    def set_on_message(self, cb):
        self._on_message = cb

    def deliver(self, env):
        if self._on_message:
            self._on_message(env)


def test_relay_transport_emits_worker_attach_on_connect(relay_auth):
    from backend.worker.relay_transport import RelayTransport

    sock = _RecordingSocket()
    transport = RelayTransport(
        endpoint="wss://relay.example.com",
        server_id="imagine-test",
        enrollment_token="etok",
        socket_factory=lambda endpoint: sock,
    )
    transport.connect()

    assert sock.connected
    assert len(sock.sent) == 1
    env = sock.sent[0]
    assert env["type"] == "worker.attach"
    assert env["server_id"] == "imagine-test"
    assert env["body"]["enrollment_token"] == "etok"

    ok, err = relay_auth.validate_envelope(env)
    assert ok, err


def test_relay_transport_heartbeat_and_claim_envelopes(relay_auth):
    from backend.worker.relay_transport import RelayTransport

    sock = _RecordingSocket()
    transport = RelayTransport(
        endpoint="wss://relay.example.com",
        server_id="imagine-test",
        enrollment_token="etok",
        socket_factory=lambda endpoint: sock,
    )
    transport.connect()
    sock.sent.clear()

    transport.heartbeat({"phase": "mc"})
    transport.claim(phase="mc", max_batch=4)
    transport.complete(task_id=42, phase="mc", status="done")

    assert [e["type"] for e in sock.sent] == [
        "server.heartbeat",
        "worker.claim",
        "worker.complete",
    ]
    for env in sock.sent:
        ok, err = relay_auth.validate_envelope(env)
        assert ok, f"{env['type']}: {err}"


def test_relay_transport_rejects_file_byte_methods():
    from backend.worker.relay_transport import RelayNotSupported, RelayTransport

    sock = _RecordingSocket()
    transport = RelayTransport(
        endpoint="wss://relay.example.com",
        server_id="imagine-test",
        enrollment_token="etok",
        socket_factory=lambda endpoint: sock,
    )
    transport.connect()

    with pytest.raises(RelayNotSupported):
        transport.get_thumbnail(task_id=1)
    with pytest.raises(RelayNotSupported):
        transport.get_mc_data(task_id=1)
    with pytest.raises(RelayNotSupported):
        transport.save_vision(task_id=1, data=b"x")
    with pytest.raises(RelayNotSupported):
        transport.save_vv(task_id=1, data=b"x")
    with pytest.raises(RelayNotSupported):
        transport.save_mv(task_id=1, data=b"x")


def test_relay_transport_disconnect_closes_socket():
    from backend.worker.relay_transport import RelayTransport

    sock = _RecordingSocket()
    transport = RelayTransport(
        endpoint="wss://relay.example.com",
        server_id="imagine-test",
        enrollment_token="etok",
        socket_factory=lambda endpoint: sock,
    )
    transport.connect()
    transport.disconnect()

    assert sock.closed
    # session.close envelope should be sent before close
    assert sock.sent[-1]["type"] == "session.close"
