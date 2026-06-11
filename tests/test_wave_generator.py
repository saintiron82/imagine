"""CAS M4: model-version reprocessing waves (2026-06-11).

A wave is a normal analysis job targeting files that lack a 'done'
derivation under the phase's ACTIVE model version.
"""

import struct

import pytest

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.queue.derivations import record_derivation
from backend.server.queue.waves import create_wave_job


@pytest.fixture()
def db(tmp_path):
    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager(db)
    return db


def _seed_file(db, path, *, content_hash, caption="cap", thumb="/t.png"):
    cur = db.conn.cursor()
    cur.execute(
        "INSERT INTO files (file_path, file_name, content_hash, mc_caption, "
        "thumbnail_url) VALUES (?,?,?,?,?)",
        (path, path.rsplit("/", 1)[-1], content_hash, caption, thumb))
    db.conn.commit()
    return cur.lastrowid


def test_vv_wave_targets_only_stale_files(db):
    stale = _seed_file(db, "/a.png", content_hash="h_stale")
    fresh = _seed_file(db, "/b.png", content_hash="h_fresh")
    unhashed = _seed_file(db, "/c.png", content_hash=None)

    # fresh already has a current-version vv derivation
    record_derivation(db, fresh, "vv",
                      vector_blob=struct.pack("<1152f", *([0.1] * 1152)))
    db.conn.commit()

    result = create_wave_job(db, "vv")
    assert result["success"] and result["candidates"] == 1

    rows = db.conn.execute(
        "SELECT file_id, mc_status, vv_status, mv_status, parse_status "
        "FROM file_tasks WHERE analysis_job_id=?",
        (result["job_id"],)).fetchall()
    assert len(rows) == 1
    fid, mc, vv, mv, parse = rows[0]
    assert fid == stale
    # vv-only wave: target pending, others 'done' (old results stand)
    assert (mc, vv, mv, parse) == ("done", "pending", "done", "done")

    job = db.conn.execute(
        "SELECT name, source_path, status FROM analysis_jobs WHERE id=?",
        (result["job_id"],)).fetchone()
    assert job[1] == "wave://vv" and job[2] == "active"


def test_mc_wave_recomputes_mv_too(db):
    _seed_file(db, "/a.png", content_hash="h1")
    result = create_wave_job(db, "mc")
    row = db.conn.execute(
        "SELECT mc_status, vv_status, mv_status FROM file_tasks "
        "WHERE analysis_job_id=?", (result["job_id"],)).fetchone()
    assert tuple(row) == ("pending", "done", "pending")


def test_dry_run_counts_without_creating(db):
    _seed_file(db, "/a.png", content_hash="h1")
    result = create_wave_job(db, "vv", dry_run=True)
    assert result["candidates"] == 1 and result["job_id"] is None
    assert db.conn.execute(
        "SELECT COUNT(*) FROM analysis_jobs").fetchone()[0] == 0


def test_files_in_active_jobs_are_excluded(db):
    fid = _seed_file(db, "/a.png", content_hash="h1")
    cur = db.conn.cursor()
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files) "
        "VALUES ('running','/x','active',1)")
    cur.execute(
        "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
        "download_status, parse_status) VALUES (?,?,?,'n/a','pending')",
        (cur.lastrowid, fid, "/a.png"))
    db.conn.commit()

    result = create_wave_job(db, "vv", dry_run=True)
    assert result["candidates"] == 0


def test_mv_wave_requires_caption(db):
    _seed_file(db, "/no_cap.png", content_hash="h1", caption=None)
    _seed_file(db, "/with_cap.png", content_hash="h2", caption="ok")
    result = create_wave_job(db, "mv", dry_run=True)
    assert result["candidates"] == 1


def test_wave_completes_through_normal_pipeline(db):
    """Completing the single recompute phase completes the wave job."""
    _seed_file(db, "/a.png", content_hash="h1")
    result = create_wave_job(db, "vv")
    job_id = result["job_id"]
    task_id = db.conn.execute(
        "SELECT id FROM file_tasks WHERE analysis_job_id=?",
        (job_id,)).fetchone()[0]

    mgr = AnalysisJobManager(db)
    mgr.complete_task_phase(task_id, "vv", success=True)

    status = db.conn.execute(
        "SELECT status FROM analysis_jobs WHERE id=?", (job_id,)).fetchone()[0]
    assert status == "completed"
