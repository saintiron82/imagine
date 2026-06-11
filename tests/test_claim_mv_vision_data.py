"""MV claim inlines MC text per task (2026-06-11) — kills the per-file
GET /files/{id}/mc round trip (N+1) for MV batches.
"""

import json

import pytest

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.analysis_manager import AnalysisJobManager


@pytest.fixture()
def mgr(tmp_path):
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager._initialized = False  # fresh temp DB needs full schema
    manager = AnalysisJobManager(db)
    cur = db.conn.cursor()
    cur.execute("INSERT INTO users (username, password_hash) VALUES ('w','x')")
    cur.execute(
        "INSERT INTO worker_sessions (user_id, worker_name, status) "
        "VALUES (1,'w','online')")
    manager._test_worker_id = cur.lastrowid
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files) "
        "VALUES ('t','/x','active',2)")
    jid = cur.lastrowid
    for i in range(2):
        cur.execute(
            "INSERT INTO files (file_path, file_name, mc_caption, ai_tags, image_type) "
            "VALUES (?,?,?,?,?)",
            (f"/x/f{i}.png", f"f{i}.png", f"caption {i}",
             json.dumps(["tag_a", "tag_b"]), "character"))
        cur.execute(
            "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
            "download_status, parse_status, mc_status, vv_status, mv_status) "
            "VALUES (?,?,?,'n/a','done','done','pending','pending')",
            (jid, cur.lastrowid, f"/x/f{i}.png"))
    db.conn.commit()
    manager._test_db = db
    return manager


def test_mv_claim_inlines_vision_data(mgr):
    tasks = mgr.claim_tasks(phase="mv", worker_id=mgr._test_worker_id, count=10)
    assert len(tasks) == 2
    for t in tasks:
        vd = t["vision_data"]
        assert vd["mc_caption"].startswith("caption")
        assert vd["ai_tags"] == ["tag_a", "tag_b"]
        assert vd["image_type"] == "character"


def test_non_mv_claim_has_no_vision_data(mgr):
    tasks = mgr.claim_tasks(phase="vv", worker_id=mgr._test_worker_id, count=10)
    assert tasks
    assert all("vision_data" not in t for t in tasks)
