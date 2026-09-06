"""Job-fair claim interleaving (M0, 2026-06-11).

A large job must not starve a later small job: claims round-robin across
active jobs (ROW_NUMBER per job), with explicit priority still trumping
fairness.
"""

import pytest

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.analysis_manager import AnalysisJobManager


@pytest.fixture()
def db(tmp_path):
    AnalysisJobManager._initialized = False  # fresh temp DB needs full schema
    db = SQLiteDB(str(tmp_path / "test.db"))
    return db


@pytest.fixture()
def mgr(db):
    manager = AnalysisJobManager(db)
    cur = db.conn.cursor()
    cur.execute("INSERT INTO users (username, password_hash) VALUES ('w','x')")
    cur.execute(
        "INSERT INTO worker_sessions (user_id, worker_name, status) "
        "VALUES (1,'w','online')")
    manager._wid = cur.lastrowid
    db.conn.commit()
    return manager


def _make_job(db, name, n_tasks, created_at, priority=0):
    cur = db.conn.cursor()
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files) "
        "VALUES (?,?,'active',?)", (name, f"/{name}", n_tasks))
    jid = cur.lastrowid
    for i in range(n_tasks):
        cur.execute("INSERT INTO files (file_path, file_name) VALUES (?,?)",
                    (f"/{name}/{i}.png", f"{i}.png"))
        cur.execute(
            "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
            "download_status, parse_status, mc_status, priority, created_at) "
            "VALUES (?,?,?,'n/a','done','pending',?,?)",
            (jid, cur.lastrowid, f"/{name}/{i}.png", priority, created_at))
    db.conn.commit()
    return jid


def test_small_late_job_not_starved(db, mgr):
    big = _make_job(db, "big", 20, "2026-06-01 00:00:00")
    small = _make_job(db, "small", 4, "2026-06-02 00:00:00")  # created later

    tasks = mgr.claim_tasks(phase="mc", worker_id=mgr._wid, count=8)
    jobs = [t["job_id"] for t in tasks]

    # Old behavior: all 8 from 'big' (FIFO). Fair: both jobs interleaved.
    assert small in jobs, f"small job starved: {jobs}"
    assert big in jobs
    # Round-robin gives the small job its full remaining share
    assert jobs.count(small) == 4


def test_priority_still_trumps_fairness(db, mgr):
    _make_job(db, "normal", 10, "2026-06-01 00:00:00", priority=0)
    urgent = _make_job(db, "urgent", 3, "2026-06-02 00:00:00", priority=10)

    tasks = mgr.claim_tasks(phase="mc", worker_id=mgr._wid, count=3)
    assert all(t["job_id"] == urgent for t in tasks)


def test_single_job_unaffected(db, mgr):
    jid = _make_job(db, "only", 5, "2026-06-01 00:00:00")
    tasks = mgr.claim_tasks(phase="mc", worker_id=mgr._wid, count=10)
    assert len(tasks) == 5
    assert all(t["job_id"] == jid for t in tasks)
