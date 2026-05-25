"""Worker-side relay transport (Phase 6).

Provides the WorkerTransport-compatible surface that WorkerDaemon needs,
but everything flows over a WebSocket to the AWS relay. File bytes
(thumbnails, MC source data, vision/VV/MV blobs) cannot ride the relay —
those methods raise :class:`RelayNotSupported` so the daemon either
falls back to direct mode or surfaces the issue to the user.

Same contract that infra/aws-relay/src/auth.py validates is used here,
so envelopes built locally are also accepted by the Lambda handler.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class RelayNotSupported(RuntimeError):
    """Raised when an operation cannot be carried out over the relay.

    The relay is a control plane only: it never carries image bytes,
    bulk vectors, or DB blobs (see docs/relay_protocol_contract_ko.md).
    """


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _envelope(msg_type: str, *, server_id: str, body: dict) -> dict:
    return {
        "v": 1,
        "type": msg_type,
        "msg_id": str(uuid.uuid4()),
        "server_id": server_id,
        "session_id": None,
        "ts": _utcnow_iso(),
        "body": body,
    }


class RelayTransport:
    """Talk to the Imagine server via the AWS control relay.

    Constructor parameters are kept transport-agnostic so unit tests can
    inject a fake socket. Production usage gets a real WebSocket via
    `default_socket_factory`.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        server_id: str,
        enrollment_token: str,
        socket_factory: Callable[[str], Any] | None = None,
    ):
        if not endpoint:
            raise ValueError("relay endpoint is required")
        if not server_id:
            raise ValueError("server_id is required for relay mode")
        if not enrollment_token:
            raise ValueError("enrollment_token is required for relay mode")

        self._endpoint = endpoint
        self._server_id = server_id
        self._enrollment_token = enrollment_token
        self._socket_factory = socket_factory or default_socket_factory
        self._socket: Any | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._inbox: list[dict] = []

    # ── Lifecycle ─────────────────────────────────────────────────

    def connect(self) -> None:
        sock = self._socket_factory(self._endpoint)
        sock.set_on_message(self._on_message)
        sock.connect()
        self._socket = sock
        self._send_envelope(
            _envelope(
                "worker.attach",
                server_id=self._server_id,
                body={"enrollment_token": self._enrollment_token},
            )
        )
        self._connected = True

    def disconnect(self) -> None:
        if self._socket is None:
            return
        try:
            self._send_envelope(
                _envelope(
                    "session.close",
                    server_id=self._server_id,
                    body={"reason": "worker-disconnect"},
                )
            )
        finally:
            try:
                self._socket.close()
            except Exception:  # pragma: no cover - close best-effort
                pass
            self._socket = None
            self._connected = False

    # ── Control plane (allowed) ────────────────────────────────────

    def heartbeat(self, stats: Optional[dict] = None) -> None:
        self._send_envelope(
            _envelope(
                "server.heartbeat",
                server_id=self._server_id,
                body={"online": True, "stats": stats or {}},
            )
        )

    def claim(self, *, phase: str, max_batch: int) -> None:
        self._send_envelope(
            _envelope(
                "worker.claim",
                server_id=self._server_id,
                body={"phase": phase, "max_batch": int(max_batch)},
            )
        )

    def complete(self, *, task_id: int, phase: str, status: str, summary: Optional[dict] = None) -> None:
        body = {"task_id": int(task_id), "phase": phase, "status": status}
        if summary:
            body["summary"] = summary
        self._send_envelope(
            _envelope("worker.complete", server_id=self._server_id, body=body)
        )

    # ── Data plane (rejected) ─────────────────────────────────────

    def get_thumbnail(self, *_args, **_kwargs):
        raise RelayNotSupported(
            "Thumbnail bytes cannot be fetched through the relay. "
            "Use direct/manual_external mode for asset transfers."
        )

    def get_mc_data(self, *_args, **_kwargs):
        raise RelayNotSupported(
            "MC source data is not relay-eligible (see data-plane policy)."
        )

    def save_vision(self, *_args, **_kwargs):
        raise RelayNotSupported(
            "Vision blobs cannot be uploaded through the relay."
        )

    def save_vv(self, *_args, **_kwargs):
        raise RelayNotSupported(
            "VV vectors cannot be uploaded through the relay."
        )

    def save_mv(self, *_args, **_kwargs):
        raise RelayNotSupported(
            "MV vectors cannot be uploaded through the relay."
        )

    # ── Internals ─────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _send_envelope(self, env: dict) -> None:
        if self._socket is None:
            raise RuntimeError("RelayTransport.connect() must be called first")
        self._socket.send(env)

    def _on_message(self, env: dict) -> None:
        with self._lock:
            self._inbox.append(env)


# ── Production WebSocket ──────────────────────────────────────────


class _WebSocketAdapter:
    """Minimal adapter around `websocket-client` used in production."""

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._ws = None
        self._thread: threading.Thread | None = None
        self._on_message: Optional[Callable[[dict], None]] = None

    def set_on_message(self, cb):
        self._on_message = cb

    def connect(self):
        try:
            import websocket  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "websocket-client is required for relay; install with `pip install websocket-client`"
            ) from exc

        import json as _json

        def _on_message(_ws, message):
            try:
                env = _json.loads(message)
            except Exception:
                return
            if self._on_message:
                self._on_message(env)

        self._ws = websocket.WebSocketApp(self._endpoint, on_message=_on_message)
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True, name="relay-worker-ws")
        self._thread.start()

    def send(self, env: dict):
        import json as _json

        if self._ws is None:
            return
        self._ws.send(_json.dumps(env))

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # pragma: no cover
                pass


def default_socket_factory(endpoint: str) -> _WebSocketAdapter:
    return _WebSocketAdapter(endpoint)


# ── Convenience constructor for the CLI ───────────────────────────


def from_env() -> RelayTransport | None:
    """Build a RelayTransport from IMAGINE_* env vars; return None if missing."""
    endpoint = os.environ.get("IMAGINE_RELAY_ENDPOINT", "").strip()
    server_id = os.environ.get("IMAGINE_SERVER_ID", "").strip()
    token = os.environ.get("IMAGINE_WORKER_ENROLLMENT_TOKEN", "").strip()
    if not endpoint or not server_id or not token:
        return None
    return RelayTransport(
        endpoint=endpoint,
        server_id=server_id,
        enrollment_token=token,
    )
