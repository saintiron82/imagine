"""Phase 2: GET /api/v1/server/connection-info exposes connection candidates.

The endpoint is the single source of truth used by the admin UI and the
worker command generator. Tokens or password hashes must never appear in
the response.
"""

from __future__ import annotations

import sqlite3
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server import deps


@pytest.fixture
def in_memory_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(
        "CREATE TABLE system_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO system_meta (key, value) VALUES (?, ?)",
        ("group_name", "studio"),
    )
    conn.commit()
    return types.SimpleNamespace(conn=conn)


@pytest.fixture
def client(in_memory_db, monkeypatch):
    from backend.server.routers import connection_info

    monkeypatch.setattr(connection_info, "_detect_lan_ip", lambda: "192.168.0.8")
    monkeypatch.setattr(connection_info, "_detect_public_ip", lambda: "")

    app = FastAPI()
    app.include_router(connection_info.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: in_memory_db
    app.dependency_overrides[deps.require_admin] = lambda: {
        "id": 1,
        "username": "admin",
        "role": "admin",
    }
    return TestClient(app, base_url="http://localhost:8000")


def test_request_origin_and_group_are_returned(client):
    response = client.get("/api/v1/server/connection-info")
    assert response.status_code == 200
    body = response.json()
    assert body["request_origin"] == "http://localhost:8000"
    assert body["group_name"] == "studio"
    assert body["server_id"]  # generated and persisted, non-empty


def test_localhost_mode_is_always_present(client):
    body = client.get("/api/v1/server/connection-info").json()
    modes = {m["mode"]: m for m in body["connect_modes"]}
    assert modes["direct_local"]["url"] == "http://localhost:8000"
    assert modes["direct_local"]["reachable_scope"] == "same-machine"
    assert modes["direct_local"]["security"] == "safe-local"
    assert modes["direct_local"]["available"] is True


def test_lan_mode_uses_detected_lan_ip(client):
    body = client.get("/api/v1/server/connection-info").json()
    modes = {m["mode"]: m for m in body["connect_modes"]}
    lan = modes["direct_lan"]
    assert lan["url"] == "http://192.168.0.8:8000"
    assert lan["reachable_scope"] == "same-lan"
    assert lan["security"] == "lan-only"
    assert lan["available"] is True


def test_relay_session_is_listed_but_unavailable_until_configured(client):
    body = client.get("/api/v1/server/connection-info").json()
    modes = {m["mode"]: m for m in body["connect_modes"]}
    relay = modes["relay_session"]
    assert relay["available"] is False
    assert relay["url"] == ""
    assert relay["reachable_scope"] == "external"
    assert relay["security"] == "recommended-external"


def test_direct_public_is_marked_not_verified(in_memory_db, monkeypatch):
    from backend.server.routers import connection_info

    monkeypatch.setattr(connection_info, "_detect_lan_ip", lambda: "192.168.0.8")
    monkeypatch.setattr(connection_info, "_detect_public_ip", lambda: "203.0.113.7")

    app = FastAPI()
    app.include_router(connection_info.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: in_memory_db
    app.dependency_overrides[deps.require_admin] = lambda: {
        "id": 1, "username": "admin", "role": "admin",
    }
    test_client = TestClient(app, base_url="http://localhost:8000")

    body = test_client.get("/api/v1/server/connection-info").json()
    modes = {m["mode"]: m for m in body["connect_modes"]}
    pub = modes["manual_external"]
    assert pub["url"] == "http://203.0.113.7:8000"
    assert pub["security"] == "public-not-verified"
    assert pub["reachable_scope"] == "external"
    assert pub["available"] is False  # never auto-trusted


def test_response_carries_no_tokens(client):
    body = client.get("/api/v1/server/connection-info").json()
    flat = repr(body).lower()
    for forbidden in (
        "access_token",
        "refresh_token",
        "password",
        "token_hash",
        "secret",
    ):
        assert forbidden not in flat, f"connection-info leaked {forbidden}"


def test_endpoint_requires_admin(in_memory_db, monkeypatch):
    from backend.server.routers import connection_info

    monkeypatch.setattr(connection_info, "_detect_lan_ip", lambda: "192.168.0.8")
    monkeypatch.setattr(connection_info, "_detect_public_ip", lambda: "")

    from fastapi import HTTPException

    app = FastAPI()
    app.include_router(connection_info.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: in_memory_db

    def deny():
        raise HTTPException(status_code=403, detail="admin required")

    app.dependency_overrides[deps.require_admin] = deny
    test_client = TestClient(app, base_url="http://localhost:8000")

    resp = test_client.get("/api/v1/server/connection-info")
    assert resp.status_code == 403


def test_server_id_is_persisted_across_calls(client):
    first = client.get("/api/v1/server/connection-info").json()["server_id"]
    second = client.get("/api/v1/server/connection-info").json()["server_id"]
    assert first == second


# ── Phase 3/4 completion: relay state surfaces through the endpoint ──


def test_relay_session_reflects_configured_endpoint(in_memory_db, monkeypatch):
    """When relay is configured + connector online, the candidate becomes available."""
    from backend.server.routers import connection_info

    monkeypatch.setattr(connection_info, "_detect_lan_ip", lambda: "192.168.0.8")
    monkeypatch.setattr(connection_info, "_detect_public_ip", lambda: "")

    # Stub config + live singleton: relay endpoint set, connector online.
    class _FakeSingleton:
        is_connected = True
        endpoint = "wss://relay.example.com/prod"

    fake_relay_cfg = {
        "endpoint": "wss://relay.example.com/prod",
        "server_secret": "topsecret",
        "enabled": True,
    }

    import backend.server.relay_client as _rc
    monkeypatch.setattr(_rc, "_singleton", _FakeSingleton(), raising=False)

    import backend.server.config as _cfg
    monkeypatch.setattr(_cfg, "get_relay_config", lambda: dict(fake_relay_cfg))

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.server import deps

    app = FastAPI()
    app.include_router(connection_info.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: in_memory_db
    app.dependency_overrides[deps.require_admin] = lambda: {
        "id": 1, "username": "admin", "role": "admin",
    }
    test_client = TestClient(app, base_url="http://localhost:8000")

    body = test_client.get("/api/v1/server/connection-info").json()
    modes = {m["mode"]: m for m in body["connect_modes"]}
    relay = modes["relay_session"]
    assert relay["available"] is True
    assert relay["url"] == "wss://relay.example.com/prod"


def test_relay_session_remains_unavailable_when_endpoint_configured_but_offline(
    in_memory_db, monkeypatch,
):
    """Endpoint configured, but connector not online — UI must not claim available."""
    from backend.server.routers import connection_info

    monkeypatch.setattr(connection_info, "_detect_lan_ip", lambda: "192.168.0.8")
    monkeypatch.setattr(connection_info, "_detect_public_ip", lambda: "")

    import backend.server.relay_client as _rc
    monkeypatch.setattr(_rc, "_singleton", None, raising=False)

    import backend.server.config as _cfg
    monkeypatch.setattr(
        _cfg, "get_relay_config",
        lambda: {"endpoint": "wss://relay.example.com/prod", "server_secret": ""},
    )

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.server import deps

    app = FastAPI()
    app.include_router(connection_info.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: in_memory_db
    app.dependency_overrides[deps.require_admin] = lambda: {
        "id": 1, "username": "admin", "role": "admin",
    }
    test_client = TestClient(app, base_url="http://localhost:8000")

    body = test_client.get("/api/v1/server/connection-info").json()
    relay = next(m for m in body["connect_modes"] if m["mode"] == "relay_session")
    assert relay["available"] is False
    # Endpoint string is still surfaced so the admin can see what's configured.
    assert relay["url"] == "wss://relay.example.com/prod"


def test_admin_relay_status_reports_configuration_and_liveness(
    in_memory_db, monkeypatch,
):
    """/admin/relay/status: configured? secret? live?"""
    from backend.server.routers import connection_info

    class _FakeSingleton:
        is_connected = True
        endpoint = "wss://relay.example.com/prod"

    import backend.server.relay_client as _rc
    monkeypatch.setattr(_rc, "_singleton", _FakeSingleton(), raising=False)

    import backend.server.config as _cfg
    monkeypatch.setattr(
        _cfg, "get_relay_config",
        lambda: {
            "endpoint": "wss://relay.example.com/prod",
            "server_secret": "topsecret",
        },
    )

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.server import deps

    app = FastAPI()
    app.include_router(connection_info.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: in_memory_db
    app.dependency_overrides[deps.require_admin] = lambda: {
        "id": 1, "username": "admin", "role": "admin",
    }
    test_client = TestClient(app, base_url="http://localhost:8000")

    body = test_client.get("/api/v1/admin/relay/status").json()
    assert body["configured"] is True
    assert body["secret_configured"] is True
    assert body["online"] is True
    assert body["endpoint"] == "wss://relay.example.com/prod"


def test_admin_relay_status_when_nothing_configured(in_memory_db, monkeypatch):
    from backend.server.routers import connection_info

    import backend.server.relay_client as _rc
    monkeypatch.setattr(_rc, "_singleton", None, raising=False)

    import backend.server.config as _cfg
    monkeypatch.setattr(_cfg, "get_relay_config", lambda: {})

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.server import deps

    app = FastAPI()
    app.include_router(connection_info.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: in_memory_db
    app.dependency_overrides[deps.require_admin] = lambda: {
        "id": 1, "username": "admin", "role": "admin",
    }
    test_client = TestClient(app, base_url="http://localhost:8000")

    body = test_client.get("/api/v1/admin/relay/status").json()
    assert body == {
        "configured": False,
        "secret_configured": False,
        "online": False,
        "endpoint": "",
    }
