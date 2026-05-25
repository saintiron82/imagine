"""Server-side relay connector (Phase 5).

The connector opens an outbound WebSocket to the AWS relay, registers
the local server, and keeps the session alive with heartbeats. It is
intentionally transport-agnostic so contract tests can drive it with a
fake socket. The production transport (websocket-client) is selected
when the relay actually needs to attach.

Whenever the connection state flips, the connector pushes an updated
record to Firestore via `firebase_register` so external clients can see
`connect_mode=relay_session` / `relay_online=true` in discovery.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


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


def normalize_relay_settings(cfg) -> dict:
    """Read relay.* keys with defaults — accepts the Config object or a dict.

    Always returns a flat dict keyed by the dotted relay.* names so
    callers don't have to know which container they were handed. Public
    name avoids collision with `backend.server.config.get_relay_config`,
    which loads from config.yaml + env vars.
    """
    def _read(key: str, default: Any):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return cfg.get(key, default)

    return {
        "relay.enabled": bool(_read("relay.enabled", False)),
        "relay.endpoint": (_read("relay.endpoint", "") or "").strip(),
        "relay.server_id": (_read("relay.server_id", "") or "").strip(),
        "relay.server_secret": (_read("relay.server_secret", "") or "").strip(),
        "relay.auto_connect": bool(_read("relay.auto_connect", True)),
        "relay.heartbeat_interval_seconds": int(
            _read("relay.heartbeat_interval_seconds", 60) or 60
        ),
        "relay.idle_timeout_seconds": int(
            _read("relay.idle_timeout_seconds", 600) or 600
        ),
    }


class RelayClient:
    """Manages a single relay session for the local server."""

    def __init__(
        self,
        *,
        config: Any,
        group_name: str,
        port: int,
        transport_factory: Callable[[str], Any],
        firebase_register: Callable[..., bool],
        clock: Callable[[], float] = time.time,
        audit_hook: Optional[Callable[[str], None]] = None,
    ):
        self._cfg = normalize_relay_settings(config)
        self._group_name = group_name
        self._port = port
        self._transport_factory: Callable[[str], Any] = transport_factory
        self._firebase_register = firebase_register
        self._clock = clock
        self._audit_hook = audit_hook

        self._transport: Any | None = None
        self._connected: bool = False
        self._stop_requested = False
        self._last_heartbeat: float = 0.0
        self._lock = threading.Lock()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()

    # ── Properties ─────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def endpoint(self) -> str:
        return self._cfg["relay.endpoint"]

    @property
    def server_id(self) -> str:
        return self._cfg["relay.server_id"]

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> bool:
        """Attempt one connect+register pass. Returns True on TCP success."""
        if not self._cfg["relay.enabled"]:
            logger.info("Relay disabled — skipping connector start")
            return False
        if not self._cfg["relay.endpoint"]:
            logger.warning("Relay endpoint missing — connector idle")
            return False
        if not self._cfg["relay.server_secret"]:
            logger.warning("Relay server_secret missing — connector idle")
            return False
        if not self._cfg["relay.server_id"]:
            logger.warning("Relay server_id missing — connector idle")
            return False

        return self._open()

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            self._heartbeat_stop.set()
            if self._transport is not None:
                try:
                    self._transport.close()
                except Exception:  # pragma: no cover - close best-effort
                    pass
            was_connected = self._connected
            self._connected = False
            self._transport = None

        thread = self._heartbeat_thread
        self._heartbeat_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        if was_connected:
            self._publish_relay_state(online=False)

    def _start_heartbeat_thread(self) -> None:
        if self._heartbeat_thread is not None:
            return
        self._heartbeat_stop.clear()

        def _loop():
            interval = max(5, int(self._cfg["relay.heartbeat_interval_seconds"]))
            while not self._heartbeat_stop.wait(interval):
                try:
                    self._tick_heartbeat()
                except Exception as exc:  # pragma: no cover - heartbeat is best-effort
                    logger.warning("Relay heartbeat failed: %s", exc)

        thread = threading.Thread(
            target=_loop, daemon=True, name="relay-heartbeat"
        )
        self._heartbeat_thread = thread
        thread.start()

    def request_reconnect(self) -> bool:
        """Tear down any existing transport and run start() again."""
        with self._lock:
            if self._transport is not None:
                try:
                    self._transport.close()
                except Exception:  # pragma: no cover
                    pass
            self._transport = None
            self._connected = False
        return self._open()

    # ── Outbound ───────────────────────────────────────────────────

    def _tick_heartbeat(self, *, force: bool = False) -> None:
        # Heartbeats fire as long as the transport is open. We keep them
        # flowing even before the relay acks the register because the
        # relay treats them as liveness regardless of session phase.
        if self._transport is None:
            return
        interval = self._cfg["relay.heartbeat_interval_seconds"]
        now = self._clock()
        if not force and (now - self._last_heartbeat) < interval:
            return
        self._send(
            _envelope(
                "server.heartbeat",
                server_id=self.server_id,
                body={"online": True},
            )
        )
        self._last_heartbeat = now

    def _send(self, envelope: dict) -> None:
        if not self._transport:
            return
        self._transport.send(envelope)

    # ── Inbound ────────────────────────────────────────────────────

    def _on_message(self, envelope: dict) -> None:
        # The Lambda echos a `server.heartbeat` style ack when register
        # succeeds. We treat receipt of any non-error envelope as
        # confirmation that the session is live.
        if not isinstance(envelope, dict):
            return
        msg_type = envelope.get("type")
        if msg_type == "error":
            logger.warning("Relay returned error: %s", envelope.get("body"))
            return
        if not self._connected:
            self._connected = True
            self._publish_relay_state(online=True)

    # ── Internals ──────────────────────────────────────────────────

    def _open(self) -> bool:
        transport = None
        try:
            transport = self._transport_factory(self._cfg["relay.endpoint"])
            transport.set_on_message(self._on_message)
            transport.connect()
        except Exception as exc:
            logger.warning("Relay connect failed: %s", exc)
            self._connected = False
            self._transport = None
            return False

        with self._lock:
            self._transport = transport
            self._stop_requested = False

        self._send(
            _envelope(
                "server.register",
                server_id=self.server_id,
                body={
                    "server_secret": self._cfg["relay.server_secret"],
                    "group_name": self._group_name,
                },
            )
        )
        self._last_heartbeat = self._clock()
        self._start_heartbeat_thread()
        return True

    def _publish_relay_state(self, *, online: bool) -> None:
        try:
            self._firebase_register(
                self._group_name,
                self._port,
                connect_mode="relay_session" if online else "direct_lan",
                relay_endpoint=self.endpoint if online else "",
                relay_online=online,
                reachable_status="verified" if online else "unknown",
            )
        except Exception as exc:  # pragma: no cover - registry is best-effort
            logger.warning("Failed to publish relay state to Firestore: %s", exc)
        if self._audit_hook is not None:
            try:
                self._audit_hook("relay_connected" if online else "relay_disconnected")
            except Exception as exc:  # pragma: no cover
                logger.warning("Audit hook failed: %s", exc)


# ── Production transport ───────────────────────────────────────────


class WebSocketRelayTransport:
    """Thin wrapper around the `websocket-client` library.

    Not used by tests — they substitute a fake transport. The class is
    instantiated by `default_transport_factory` so production deployments
    get the real socket.
    """

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
        except ImportError as exc:  # pragma: no cover - optional dependency
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

        self._ws = websocket.WebSocketApp(
            self._endpoint,
            on_message=_on_message,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever, daemon=True, name="relay-ws"
        )
        self._thread.start()

    def send(self, envelope: dict):
        import json as _json

        if self._ws is None:
            return
        self._ws.send(_json.dumps(envelope))

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # pragma: no cover
                pass


def default_transport_factory(endpoint: str) -> WebSocketRelayTransport:
    return WebSocketRelayTransport(endpoint)


# ── App wiring ─────────────────────────────────────────────────────


_singleton: RelayClient | None = None


def maybe_start(*, app, db) -> Optional[RelayClient]:
    """Start a relay client if config / env enables it.

    Reads:
      - `relay.*` from utils.config
      - falls back to env: IMAGINE_RELAY_ENDPOINT, IMAGINE_RELAY_SERVER_SECRET
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    try:
        from backend.server.config import get_relay_config as _settings
        from backend.server.config import get_server_config as _server_cfg

        relay_settings = _settings()
        server_settings = _server_cfg()
    except Exception:  # pragma: no cover - bootstrapping safety
        relay_settings, server_settings = {}, {}

    endpoint = (relay_settings.get("endpoint") or "").strip()
    if not endpoint:
        return None

    secret = (relay_settings.get("server_secret") or "").strip()
    if not secret:
        return None

    from backend.server.connection_info import get_or_create_server_id, get_group_name
    from backend.server.firebase_registry import register_group
    from backend.server.security import audit_log as _audit

    server_id = get_or_create_server_id(db)
    group = get_group_name(db) or "imagine"

    def _audit_hook(event_name: str) -> None:
        _audit.record(
            db,
            event_name,
            target_kind="relay",
            target_id=server_id,
            detail=endpoint,
        )

    flat_cfg = {
        "relay.enabled": True,
        "relay.endpoint": endpoint,
        "relay.server_id": server_id,
        "relay.server_secret": secret,
        "relay.auto_connect": bool(relay_settings.get("auto_connect", True)),
        "relay.heartbeat_interval_seconds": int(
            relay_settings.get("heartbeat_interval_seconds", 60)
        ),
        "relay.idle_timeout_seconds": int(
            relay_settings.get("idle_timeout_seconds", 600)
        ),
    }
    port = int(server_settings.get("port", 8000))

    client = RelayClient(
        config=flat_cfg,
        group_name=group,
        port=port,
        transport_factory=default_transport_factory,
        firebase_register=register_group,
        audit_hook=_audit_hook,
    )
    if client.start():
        _singleton = client
        return client
    return None


def stop_singleton() -> None:
    global _singleton
    if _singleton is not None:
        try:
            _singleton.stop()
        finally:
            _singleton = None
