"""CAS M1: derivation cache shadow write (2026-06-11).

Results saved for a hashed file land in derivations keyed by
(content_hash, phase, model_version); legacy rows without a hash are
skipped silently. Read paths are untouched in M1.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.db.sqlite_client import SQLiteDB
from backend.server import deps
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.queue.derivations import record_derivation
from backend.server.routers import analysis
from backend.utils.model_version import get_model_version


@pytest.fixture()
def db(tmp_path):
    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager(db)
    return db


def _seed_file(db, content_hash="abc123def456"):
    cur = db.conn.cursor()
    cur.execute(
        "INSERT INTO files (file_path, file_name, content_hash) VALUES (?,?,?)",
        ("/x/a.png", "a.png", content_hash))
    db.conn.commit()
    return cur.lastrowid


def test_migration_creates_tables(db):
    tables = {r[0] for r in db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "derivations" in tables
    assert "model_registry" in tables


def test_model_version_deterministic_and_phase_specific():
    v1, v2 = get_model_version("vv"), get_model_version("vv")
    assert v1 == v2 and v1
    assert get_model_version("mc") != get_model_version("vv")
    with pytest.raises(ValueError):
        get_model_version("parse")


def test_record_derivation_writes_cache_row(db):
    fid = _seed_file(db)
    ok = record_derivation(db, fid, "mc",
                           result_json=json.dumps({"caption": "x"}),
                           created_by="worker-1")
    db.conn.commit()
    assert ok

    row = db.conn.execute(
        "SELECT content_hash, phase, model_version, status, result_json, created_by "
        "FROM derivations").fetchone()
    assert row[0] == "abc123def456"
    assert row[1] == "mc"
    assert row[2] == get_model_version("mc")
    assert row[3] == "done"
    assert json.loads(row[4]) == {"caption": "x"}
    assert row[5] == "worker-1"

    # Registry knows the active version
    reg = db.conn.execute(
        "SELECT model_version, is_active FROM model_registry WHERE phase='mc'"
    ).fetchone()
    assert reg[0] == get_model_version("mc") and reg[1] == 1


def test_record_derivation_skips_unhashed_legacy_rows(db):
    fid = _seed_file(db, content_hash=None)
    assert record_derivation(db, fid, "vv", vector_blob=b"\x00\x01") is False
    db.conn.commit()
    count = db.conn.execute("SELECT COUNT(*) FROM derivations").fetchone()[0]
    assert count == 0


def test_record_derivation_idempotent_replace(db):
    fid = _seed_file(db)
    record_derivation(db, fid, "vv", vector_blob=b"\x00\x01")
    record_derivation(db, fid, "vv", vector_blob=b"\x02\x03")  # re-run
    db.conn.commit()
    rows = db.conn.execute(
        "SELECT vector_blob FROM derivations WHERE phase='vv'").fetchall()
    assert len(rows) == 1
    assert bytes(rows[0][0]) == b"\x02\x03"


# ── Endpoint-level: shadow write rides the primary save ──────


def _client(db, monkeypatch):
    app = FastAPI()
    app.include_router(analysis.router)
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.get_current_user] = (
        lambda: {"id": 1, "username": "w"})
    monkeypatch.setattr(analysis, "_require_file_phase_assignment",
                        lambda *a, **k: None)
    return TestClient(app)


def test_vv_endpoint_shadow_writes(db, monkeypatch):
    fid = _seed_file(db)
    client = _client(db, monkeypatch)

    resp = client.patch(f"/api/v1/files/{fid}/vv",
                        json={"vector": __import__("conftest").vv_list(0.1)})  # real vec0 dimension
    assert resp.status_code == 200

    # Primary save (read path unchanged)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM vec_files WHERE file_id=?", (fid,)
    ).fetchone()[0] == 1
    # Shadow write
    row = db.conn.execute(
        "SELECT content_hash, vector_blob FROM derivations WHERE phase='vv'"
    ).fetchone()
    assert row[0] == "abc123def456" and row[1] is not None


def test_vision_endpoint_shadow_writes(db, monkeypatch):
    fid = _seed_file(db)
    client = _client(db, monkeypatch)

    payload = {"mc_caption": "a red chair", "ai_tags": ["chair"]}
    resp = client.patch(f"/api/v1/files/{fid}/vision", json=payload)
    assert resp.status_code == 200

    row = db.conn.execute(
        "SELECT result_json FROM derivations WHERE phase='mc'").fetchone()
    assert json.loads(row[0])["mc_caption"] == "a red chair"
