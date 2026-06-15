"""
Tests for DB backup/restore (IMGV2-48) — export_snapshot / import_snapshot
on SQLiteDB plus the admin router password gate.

Roundtrip uses real SQLite files in tmp_path (never the project DB).
"""
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.db.sqlite_client import SQLiteDB
from backend.server.routers import database as db_router
from backend.server.deps import require_admin, get_db_safe


def _marker(db, value):
    db._set_system_meta("backup_test_marker", value)


def _read_marker(path):
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT value FROM system_meta WHERE key='backup_test_marker'"
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def test_export_creates_valid_sqlite(tmp_path):
    db = SQLiteDB(str(tmp_path / "src.db"))
    _marker(db, "alpha")
    snap = str(tmp_path / "snap.db")

    result = db.export_snapshot(snap)
    assert result["success"], result
    assert result["bytes"] > 0
    # The snapshot is a standalone valid SQLite DB carrying our schema + data.
    assert _read_marker(snap) == "alpha"
    conn = sqlite3.connect(snap)
    try:
        has_files = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='files'"
        ).fetchone()
        assert has_files is not None
    finally:
        conn.close()


def test_export_import_roundtrip(tmp_path):
    src = SQLiteDB(str(tmp_path / "src.db"))
    _marker(src, "from-source")
    snap = str(tmp_path / "snap.db")
    assert src.export_snapshot(snap)["success"]

    # A different live DB with its own content gets fully replaced on import.
    dest = SQLiteDB(str(tmp_path / "dest.db"))
    _marker(dest, "original-dest")
    assert _read_marker(str(tmp_path / "dest.db")) == "original-dest"

    result = dest.import_snapshot(snap)
    assert result["success"], result
    # Live connection now reflects the snapshot's content.
    row = dest.conn.execute(
        "SELECT value FROM system_meta WHERE key='backup_test_marker'"
    ).fetchone()
    assert row[0] == "from-source"


def test_import_rejects_non_imagine_db(tmp_path):
    # A valid SQLite file but without our schema (no `files` table).
    foreign = str(tmp_path / "foreign.db")
    conn = sqlite3.connect(foreign)
    conn.execute("CREATE TABLE notes (id INTEGER)")
    conn.commit()
    conn.close()

    dest = SQLiteDB(str(tmp_path / "dest.db"))
    result = dest.import_snapshot(foreign)
    assert result["success"] is False
    assert "files" in result["error"]


def test_import_rejects_garbage_file(tmp_path):
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not a sqlite database")
    dest = SQLiteDB(str(tmp_path / "dest.db"))
    result = dest.import_snapshot(str(junk))
    assert result["success"] is False


@pytest.fixture
def client(tmp_path, monkeypatch):
    import bcrypt
    db = SQLiteDB(str(tmp_path / "live.db"))
    # admin user with a known bcrypt password hash
    pw_hash = bcrypt.hashpw(b"correct-horse", bcrypt.gensalt()).decode()
    db.conn.execute(
        "INSERT INTO users (id, username, password_hash, role, is_active) VALUES (1,'admin',?, 'admin', 1)",
        (pw_hash,),
    )
    db.conn.commit()

    app = FastAPI()
    app.include_router(db_router.router, prefix="/api/v1")
    fake_admin = {"id": 1, "username": "admin", "role": "admin", "is_active": True}
    app.dependency_overrides[require_admin] = lambda: fake_admin
    app.dependency_overrides[get_db_safe] = lambda: db
    return TestClient(app), db, str(tmp_path)


def test_import_wrong_password_403(client):
    c, db, root = client
    files = {"file": ("x.db", b"\x00\x01", "application/x-sqlite3")}
    r = c.post("/api/v1/admin/database/import", data={"password": "wrong"}, files=files)
    assert r.status_code == 403


def test_export_then_import_via_http(client, tmp_path):
    c, db, root = client
    db._set_system_meta("backup_test_marker", "http-roundtrip")

    # export → bytes of a valid sqlite snapshot
    r = c.get("/api/v1/admin/database/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-sqlite3")
    snap_bytes = r.content
    assert snap_bytes[:16].startswith(b"SQLite format 3")

    # import the same snapshot back with the correct password
    files = {"file": ("backup.db", snap_bytes, "application/x-sqlite3")}
    r = c.post("/api/v1/admin/database/import", data={"password": "correct-horse"}, files=files)
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True
