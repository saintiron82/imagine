"""
Tests for the folder-browse + WebDAV source router (IMGV2-13).

Uses a minimal FastAPI app with the browse router and dependency overrides
(no full server lifespan / DB needed). The in-memory source registry and
WebDAVClient are real / monkeypatched.
"""
import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server.routers import browse as browse_mod
from backend.server.deps import get_current_user, require_admin
from backend.server.queue import download_ahead


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate persistence to a temp file (never touch project root).
    monkeypatch.setenv("IMAGINE_WEBDAV_SOURCES_FILE", str(tmp_path / "sources.json"))
    # Clear the shared in-memory registry between tests.
    with download_ahead._sources_lock:
        download_ahead._webdav_sources.clear()

    app = FastAPI()
    app.include_router(browse_mod.router, prefix="/api/v1")
    fake_admin = {"id": 1, "username": "admin", "role": "admin", "is_active": True}
    app.dependency_overrides[get_current_user] = lambda: fake_admin
    app.dependency_overrides[require_admin] = lambda: fake_admin
    return TestClient(app)


def test_add_list_source_hides_password(client):
    r = client.post("/api/v1/sources", json={
        "id": "nas1", "url": "https://nas.local:5006",
        "username": "bob", "password": "secret", "verify_ssl": False,
    })
    assert r.status_code == 200 and r.json()["success"]

    r = client.get("/api/v1/sources")
    sources = r.json()["sources"]
    assert len(sources) == 1
    s = sources[0]
    assert s["id"] == "nas1" and s["username"] == "bob"
    assert "password" not in s          # secret never leaves the server
    assert s["verify_ssl"] is False


def test_add_source_requires_id_and_url(client):
    r = client.post("/api/v1/sources", json={"id": "", "url": ""})
    assert r.status_code == 400


def test_delete_source(client):
    client.post("/api/v1/sources", json={"id": "nas1", "url": "http://x"})
    assert client.delete("/api/v1/sources/nas1").status_code == 200
    assert client.delete("/api/v1/sources/nas1").status_code == 404  # gone


def test_browse_webdav_uses_client(client, monkeypatch):
    client.post("/api/v1/sources", json={
        "id": "nas1", "url": "https://nas.local:5006", "username": "u", "password": "p",
    })

    class FakeClient:
        def __init__(self, **kw):
            self.kw = kw
        def list_dir(self, path):
            assert path == "/작업분"
            return {"folders": [{"name": "2026", "path": "/작업분/2026"}], "files": []}

    monkeypatch.setattr("backend.remote.webdav_client.WebDAVClient", FakeClient)
    r = client.get("/api/v1/browse/webdav/nas1", params={"path": "/작업분"})
    assert r.status_code == 200
    body = r.json()
    assert body["folders"][0]["name"] == "2026"
    assert body["source_id"] == "nas1"


def test_browse_webdav_unknown_source_404(client):
    assert client.get("/api/v1/browse/webdav/nope").status_code == 404


def test_browse_local_disabled_by_default(client, monkeypatch):
    monkeypatch.setattr(browse_mod, "get_browse_config", lambda: {"allowed_roots": []})
    assert client.get("/api/v1/browse/local", params={"path": "/etc"}).status_code == 403


def test_browse_local_lists_whitelisted(client, tmp_path, monkeypatch):
    root = tmp_path / "assets"
    (root / "characters").mkdir(parents=True)
    (root / "backgrounds").mkdir()
    (root / ".hidden").mkdir()
    monkeypatch.setattr(browse_mod, "get_browse_config", lambda: {"allowed_roots": [str(root)]})

    # No path → returns the whitelisted root(s)
    r = client.get("/api/v1/browse/local")
    assert r.status_code == 200
    assert any(f["path"] == os.path.realpath(str(root)) for f in r.json()["folders"])

    # Inside the root → lists subdirs, hides dotfiles
    r = client.get("/api/v1/browse/local", params={"path": str(root)})
    names = [f["name"] for f in r.json()["folders"]]
    assert names == ["backgrounds", "characters"]  # sorted, .hidden excluded


def test_browse_local_blocks_traversal(client, tmp_path, monkeypatch):
    root = tmp_path / "assets"
    root.mkdir()
    monkeypatch.setattr(browse_mod, "get_browse_config", lambda: {"allowed_roots": [str(root)]})
    # Escape attempt via .. → outside allowed root → 403
    r = client.get("/api/v1/browse/local", params={"path": str(root / ".." / "..")})
    assert r.status_code == 403


def test_persistence_roundtrip(client, monkeypatch):
    client.post("/api/v1/sources", json={"id": "nas1", "url": "http://x", "username": "u", "password": "p"})
    # Simulate restart: clear memory, reload from disk.
    with download_ahead._sources_lock:
        download_ahead._webdav_sources.clear()
    n = download_ahead.load_persisted_sources()
    assert n == 1
    assert download_ahead.get_webdav_source("nas1")["password"] == "p"
