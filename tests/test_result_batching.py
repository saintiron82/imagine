"""Result ingest batching (pre-burn prep, 2026-06-11).

Worker buffers completion reports and sends them in one
POST /tasks/complete-batch per ~10 results; the server processes items
independently. Protects against per-file HTTP/commit storms from
high-throughput workers.
"""

import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.db.sqlite_client import SQLiteDB
from backend.server import deps
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.routers import analysis
from backend.worker.worker_daemon import WorkerDaemon


# ── Worker side ──────────────────────────────────────────────


def _daemon():
    d = WorkerDaemon.__new__(WorkerDaemon)
    d.transport = None
    d.server_url = "http://test"
    d._report_buffer = []
    d._report_lock = threading.Lock()
    d._batch_report_supported = True
    return d


def _ok(status=200):
    return SimpleNamespace(status_code=status, json=lambda: {"success": True})


def test_reports_flush_in_one_batch_at_threshold():
    d = _daemon()
    calls = []
    d._authed_request = lambda m, url, **kw: (calls.append((url, kw["json"])), _ok())[1]

    for i in range(10):
        d._report_task_phase(i + 1, "vv", True, elapsed_s=0.5)

    assert len(calls) == 1
    url, payload = calls[0]
    assert url.endswith("/tasks/complete-batch")
    assert len(payload["results"]) == 10
    assert d._report_buffer == []


def test_partial_buffer_flushes_at_batch_boundary():
    d = _daemon()
    calls = []
    d._authed_request = lambda m, url, **kw: (calls.append(url), _ok())[1]

    d._report_task_phase(1, "mv", True)
    d._report_task_phase(2, "mv", False, error="boom")
    assert calls == []          # below threshold — still buffered

    d._flush_reports()          # process_batch_phased finally-clause
    assert len(calls) == 1
    assert d._report_buffer == []


def test_404_falls_back_to_per_item_reports():
    d = _daemon()
    calls = []

    def fake(method, url, **kw):
        calls.append(url)
        if url.endswith("/complete-batch"):
            return _ok(404)
        return _ok()

    d._authed_request = fake
    for i in range(10):
        d._report_task_phase(i + 1, "vv", True)

    assert d._batch_report_supported is False
    per_item = [u for u in calls if u.endswith("/tasks/complete")]
    assert len(per_item) == 10
    assert d._report_buffer == []


def test_network_failure_keeps_buffer_for_retry():
    d = _daemon()

    def boom(method, url, **kw):
        raise ConnectionError("down")

    d._authed_request = boom
    for i in range(10):
        d._report_task_phase(i + 1, "vv", True)
    assert len(d._report_buffer) == 10  # nothing lost

    sent = []
    d._authed_request = lambda m, url, **kw: (sent.append(kw["json"]), _ok())[1]
    d._flush_reports()
    assert len(sent) == 1 and len(sent[0]["results"]) == 10
    assert d._report_buffer == []


# ── Server side ──────────────────────────────────────────────


@pytest.fixture()
def server_env(tmp_path):
    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager(db)
    cur = db.conn.cursor()
    cur.execute("INSERT INTO users (username, password_hash) VALUES ('w','x')")
    cur.execute(
        "INSERT INTO worker_sessions (user_id, worker_name, status) "
        "VALUES (1,'w','online')")
    wid = cur.lastrowid
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files) "
        "VALUES ('t','/x','active',2)")
    jid = cur.lastrowid
    from conftest import vec_blob
    vec = vec_blob(db)   # width from the schema, not a hardcoded 1152
    tasks = []
    for i in range(2):
        cur.execute("INSERT INTO files (file_path, file_name) VALUES (?,?)",
                    (f"/x/{i}.png", f"{i}.png"))
        fid = cur.lastrowid
        # The server VERIFIES completion against real data (vec_files row
        # must exist for a vv 'done') — store the vector like a worker would
        cur.execute("INSERT INTO vec_files (file_id, embedding) VALUES (?,?)",
                    (fid, vec))
        cur.execute(
            "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
            "download_status, parse_status, mc_status, vv_status, mv_status, "
            "vv_assigned_to) "
            "VALUES (?,?,?,'n/a','done','done','assigned','done',?)",
            (jid, fid, f"/x/{i}.png", wid))
        tasks.append(cur.lastrowid)
    db.conn.commit()

    app = FastAPI()
    app.include_router(analysis.router)
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.get_current_user] = (
        lambda: {"id": 1, "username": "w"})
    return db, TestClient(app), tasks, jid


def test_batch_endpoint_completes_tasks_and_job(server_env):
    db, client, tasks, jid = server_env
    resp = client.post("/api/v1/tasks/complete-batch", json={
        "results": [
            {"task_id": tasks[0], "phase": "vv", "success": True, "elapsed_s": 0.4},
            {"task_id": tasks[1], "phase": "vv", "success": True},
        ]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] == 2 and body["errors"] == []

    statuses = [r[0] for r in db.conn.execute(
        "SELECT vv_status FROM file_tasks WHERE id IN (?,?)",
        tasks).fetchall()]
    assert statuses == ["done", "done"]
    # All phases done across the job → job completed
    assert db.conn.execute(
        "SELECT status FROM analysis_jobs WHERE id=?", (jid,)
    ).fetchone()[0] == "completed"


def test_batch_endpoint_isolates_bad_items(server_env):
    db, client, tasks, _ = server_env
    resp = client.post("/api/v1/tasks/complete-batch", json={
        "results": [
            {"task_id": tasks[0], "phase": "vv", "success": True},
            {"task_id": 999999, "phase": "vv", "success": True},  # not assigned
        ]})
    body = resp.json()
    assert body["accepted"] == 1
    assert len(body["errors"]) == 1 and body["errors"][0]["task_id"] == 999999


def test_batch_endpoint_rejects_oversize(server_env):
    _, client, tasks, _ = server_env
    resp = client.post("/api/v1/tasks/complete-batch", json={
        "results": [{"task_id": tasks[0], "phase": "vv", "success": True}] * 51})
    assert resp.status_code == 400
