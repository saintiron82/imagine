"""Scheduler assignment: admin pin enforcement, EMA speed updates,
MV completion-bonus cap (2026-06-11).
"""

import pytest

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.queue.scheduler import WorkerScheduler


@pytest.fixture()
def db(tmp_path):
    return SQLiteDB(str(tmp_path / "test.db"))


@pytest.fixture()
def scheduler(db):
    AnalysisJobManager._initialized = False  # fresh temp DB needs full schema
    AnalysisJobManager(db)  # ensures file_tasks/analysis_jobs columns
    return WorkerScheduler(db)


def _seed_worker(db, **overrides):
    cur = db.conn.cursor()
    cur.execute("INSERT INTO users (username, password_hash) VALUES ('w','x')")
    fields = {
        "user_id": 1, "worker_name": "monster", "status": "online",
        "gpu_class": "strong", "mc_capable": 1,
        "mc_speed": 80.0, "vv_speed": 200.0, "mv_speed": 300.0,
    }
    fields.update(overrides)
    cols = ", ".join(fields)
    ph = ", ".join("?" * len(fields))
    cur.execute(f"INSERT INTO worker_sessions ({cols}) VALUES ({ph})",
                list(fields.values()))
    db.conn.commit()
    return cur.lastrowid


def _seed_tasks(db, mc=0, vv=0, mv=0):
    cur = db.conn.cursor()
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files) "
        "VALUES ('t','/x','active',?)", (mc + vv + mv,))
    jid = cur.lastrowid
    rows = (
        [("done", "pending", "pending", "pending")] * mc
        + [("done", "done", "pending", "pending")] * vv
        + [("done", "done", "done", "pending")] * mv
    )
    for i, st in enumerate(rows):
        cur.execute("INSERT INTO files (file_path, file_name) VALUES (?,?)",
                    (f"/x/{jid}_{i}.png", f"{jid}_{i}.png"))
        cur.execute(
            "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
            "download_status, parse_status, mc_status, vv_status, mv_status) "
            "VALUES (?,?,?,'n/a',?,?,?,?)",
            (jid, cur.lastrowid, f"/x/{jid}_{i}.png", *st))
    db.conn.commit()


def test_assign_prefers_mc_naturally(db, scheduler):
    """MC dominates pressure (PHASE_TIME weight) — no pin needed."""
    wid = _seed_worker(db)
    _seed_tasks(db, mc=10, vv=10, mv=10)
    decision = scheduler.assign(wid)
    assert decision["phase"] == "mc"
    assert decision["count"] >= 1


def test_assign_honors_admin_pin(db, scheduler):
    wid = _seed_worker(db, processing_mode_override="mv")
    _seed_tasks(db, mc=10, vv=10, mv=10)
    assert scheduler.assign(wid)["phase"] == "mv"


def test_pinned_worker_idles_when_phase_drained(db, scheduler):
    """A pinned worker must not poach other phases."""
    wid = _seed_worker(db, processing_mode_override="mc")
    _seed_tasks(db, mc=0, vv=10, mv=10)
    decision = scheduler.assign(wid)
    assert decision["phase"] is None
    assert decision["count"] == 0


def test_mc_only_legacy_alias(db, scheduler):
    wid = _seed_worker(db, processing_mode_override="mc_only")
    _seed_tasks(db, mc=5, vv=10, mv=10)
    assert scheduler.assign(wid)["phase"] == "mc"


def test_update_speed_ema(db, scheduler):
    wid = _seed_worker(db, mc_speed=None)
    scheduler.update_speed(wid, "mc", 8.0)
    scheduler.update_speed(wid, "mc", 12.0)  # 8*0.7 + 12*0.3 = 9.2
    row = db.conn.execute(
        "SELECT mc_speed FROM worker_sessions WHERE id=?", (wid,)
    ).fetchone()
    assert abs(row[0] - 9.2) < 0.01


def test_mv_bonus_is_capped():
    """A huge MV backlog must not outweigh a large MC backlog.

    Before the cap, pending×10 made any big MV queue dominate MC pressure
    for every worker (cluster-wide mode flapping).
    """
    s = WorkerScheduler.__new__(WorkerScheduler)  # _pick_best_phase is pure
    profile = {"gpu_class": "strong", "mc_speed": 8.0, "vv_speed": 80.0,
               "mv_speed": 120.0}

    phase = s._pick_best_phase(
        claimable={"mc": 1000, "mv": 10000},
        workers_on={},
        current_phase=None,
        profile=profile,
    )
    assert phase == "mc"

    # Near the MC tail, draining MVs wins (completion bonus intent preserved)
    phase = s._pick_best_phase(
        claimable={"mc": 3, "mv": 500},
        workers_on={},
        current_phase=None,
        profile=profile,
    )
    assert phase == "mv"
