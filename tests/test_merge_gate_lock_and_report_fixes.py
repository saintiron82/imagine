"""Regression locks for the merge-gate audit of PR #1 (feat/app-v2).

Five defects on the write-serialization / completion-reporting path, each of
which had NO test before this file:

C1  worker dropped completion reports the server had rejected per-item
C2  hash backfill held the process-wide write gate across WebDAV network I/O
I1  a reclaimed write-gate hold was never re-acquired (serialization broke)
I4  complete-batch 500'd the whole batch on any non-HTTPException/ValueError
I5  the backfill background thread had no `finally: rollback()` safety net
"""

import sqlite3
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.db.sqlite_client import _WriteGate, _SerializedConnection, SQLiteDB
from backend.server import deps
from backend.server.queue import hash_backfill
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.routers import analysis
from backend.worker.worker_daemon import WorkerDaemon


# ── I1: a reclaimed hold must be re-acquired before the next write ──────────

def test_reclaimed_hold_is_not_live_for_the_stale_owner():
    gate = _WriteGate()
    a, b = object(), object()

    gen_a = gate.acquire(a, timeout=0.05)
    assert gate.is_live(a, gen_a)

    # b waits past the bound and reclaims from a.
    gen_b = gate.acquire(b, timeout=0.05)

    assert gate.is_live(b, gen_b)
    assert not gate.is_live(a, gen_a), (
        "the stale owner must not still read as the live holder — that is what "
        "let its remaining writes bypass the gate")


def test_stale_owner_reacquires_instead_of_writing_unserialized():
    """The real bug: `_held` stayed True after a reclaim, so `_acquire_write()`
    returned early and the rest of that transaction wrote with no gate at all,
    concurrently with the new owner."""
    import backend.db.sqlite_client as mod

    gate = _WriteGate()
    original, mod._WRITE_GATE = mod._WRITE_GATE, gate
    try:
        conn = _SerializedConnection(sqlite3.connect(":memory:"))
        conn.execute("CREATE TABLE t (v INTEGER)")   # write → takes the gate
        assert conn._held

        # Someone else reclaims the gate out from under this connection.
        thief = object()
        gate.acquire(thief, timeout=0.05)
        assert not gate.is_live(conn, conn._gen)

        # The connection still thinks it holds it. Its next write must block on
        # a real acquire rather than proceeding unserialized.
        started = threading.Event()
        acquired = threading.Event()

        def _next_write():
            started.set()
            conn._acquire_write()
            acquired.set()

        t = threading.Thread(target=_next_write, daemon=True)
        t.start()
        started.wait(1.0)
        assert not acquired.wait(0.02), (
            "a stale holder re-acquired instantly — it never actually waited "
            "for the gate, so two writers were live at once")

        gate.release(thief, gate._gen)
        assert acquired.wait(1.0), "re-acquire never completed after release"
        assert gate.is_live(conn, conn._gen)
    finally:
        mod._WRITE_GATE = original


# ── C2 + I5: the backfill must not hold the gate across network I/O ─────────

class _GateSpyDB:
    """Records whether a write txn is open at each _hash_one() call."""

    def __init__(self, rows=None):
        self.conn = self
        self.open_txn = False
        self.txn_open_during_io = []
        self.rollbacks = 0
        self.updates = 0
        self.fail_commit_on = None
        self._rows = rows if rows is not None else [
            (1, "/a.psd", 100), (2, "/b.psd", 200), (3, "/c.psd", 300)]

    # -- SQLiteDB surface used by hash_backfill._run --
    def cursor(self):
        return self

    def execute(self, sql, params=()):
        if sql.strip().upper().startswith("SELECT"):
            pass          # rows are fixed at construction
        else:
            self.updates += 1
            self.open_txn = True
        return self

    def fetchall(self):
        return self._rows

    def commit(self):
        if self.fail_commit_on is not None and self.updates == self.fail_commit_on:
            raise sqlite3.OperationalError("database is locked")
        self.open_txn = False

    def rollback(self):
        self.rollbacks += 1
        self.open_txn = False


@pytest.fixture
def _reset_backfill_state():
    # `_state` is module-global, so reset on BOTH sides: another test in the
    # suite can leave counters behind, and these tests assert on exact counts.
    def _clear():
        with hash_backfill._lock:
            hash_backfill._state.update(running=False, total=0, done=0,
                                        failed=0, skipped=0, last_error=None)
    _clear()
    yield
    _clear()


def _run_backfill(db, monkeypatch, hash_side_effect=None):
    def fake_hash_one(file_path, file_size, clients):
        # Stands in for the WebDAV Range reads — this is the window in which
        # the write gate must NOT be held.
        db.txn_open_during_io.append(db.open_txn)
        if hash_side_effect:
            hash_side_effect(file_path)
        return f"hash-{file_path}"

    monkeypatch.setattr(hash_backfill, "_hash_one", fake_hash_one)
    with hash_backfill._lock:
        hash_backfill._state["running"] = True
    hash_backfill._run(lambda: db)


def test_backfill_never_holds_a_write_txn_across_network_io(
        monkeypatch, _reset_backfill_state):
    db = _GateSpyDB()
    _run_backfill(db, monkeypatch)

    assert db.txn_open_during_io, "hash lookups never ran"
    assert not any(db.txn_open_during_io), (
        "a write txn was open while _hash_one() did network I/O — this pins the "
        "process-wide write gate and stalls every other writer until reclaim")
    assert hash_backfill.get_status()["done"] == 3


def test_backfill_rolls_back_after_a_failed_item(monkeypatch, _reset_backfill_state):
    db = _GateSpyDB()

    def boom(file_path):
        if file_path == "/b.psd":
            raise RuntimeError("range read failed")

    _run_backfill(db, monkeypatch, hash_side_effect=boom)

    assert hash_backfill.get_status()["failed"] == 1
    assert db.rollbacks >= 1, "a failed item left its txn open for the next file"
    assert not db.open_txn


def test_backfill_rolls_back_when_a_commit_fails(monkeypatch, _reset_backfill_state):
    """I5: without the `finally: rollback()`, a failing commit left an open
    write txn pinning the gate until the 20s reclaim bound."""
    db = _GateSpyDB()
    db.fail_commit_on = 2   # second UPDATE's commit raises 'database is locked'

    _run_backfill(db, monkeypatch)

    assert db.rollbacks >= 1, "no rollback safety net ran after the commit failure"
    assert not db.open_txn, "the backfill thread exited holding a write txn"
    assert hash_backfill.get_status()["running"] is False


# ── C1: a 200 with per-item rejections must not be read as full success ─────

def _daemon():
    d = WorkerDaemon.__new__(WorkerDaemon)
    d.transport = None
    d.server_url = "http://test"
    d._report_buffer = []
    d._report_lock = threading.Lock()
    d._batch_report_supported = True
    return d


def _resp(status=200, body=None):
    return SimpleNamespace(status_code=status, text="",
                           json=lambda: (body if body is not None
                                         else {"success": True, "errors": []}))


def test_rejected_completions_are_surfaced_not_silently_dropped(caplog):
    """The server returns per-item rejections in `errors[]` of a 200 response.
    The worker clears the buffer either way, so an unread rejection means the
    file's phase status never advances while the worker looks healthy."""
    d = _daemon()
    body = {"success": True, "accepted": 8, "errors": [
        {"task_id": 41, "error": "Task is not assigned to current user's worker"},
        {"task_id": 42, "error": "Task is not assigned to current user's worker"},
    ]}
    d._authed_request = lambda m, url, **kw: _resp(200, body)

    with caplog.at_level("ERROR"):
        for i in range(10):
            d._report_task_phase(i + 1, "mc", True)

    assert d._report_buffer == []
    assert d._report_rejected == 2, "rejected completions were not counted"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "41" in logged and "42" in logged, (
        "rejected task ids never reached the log — the loss stayed silent")


def test_clean_batch_records_no_rejections(caplog):
    d = _daemon()
    d._authed_request = lambda m, url, **kw: _resp(200)
    with caplog.at_level("ERROR"):
        for i in range(10):
            d._report_task_phase(i + 1, "mc", True)
    assert getattr(d, "_report_rejected", 0) == 0
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_per_item_fallback_surfaces_a_rejection():
    """The 404 fallback posted each item and ignored the status entirely."""
    d = _daemon()

    def fake(method, url, **kw):
        if url.endswith("/complete-batch"):
            return _resp(404)
        return _resp(403)          # every per-item post is refused

    d._authed_request = fake
    for i in range(10):
        d._report_task_phase(i + 1, "mc", True)

    assert d._batch_report_supported is False
    assert d._report_rejected == 10, (
        "per-item fallback discarded refused completions without a trace")


# ── I4: complete-batch must isolate ANY item failure, not just two types ────

@pytest.fixture()
def server_env(tmp_path):
    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager(db)
    cur = db.conn.cursor()
    cur.execute("INSERT INTO users (username, password_hash) VALUES ('w','x')")
    cur.execute("INSERT INTO worker_sessions (user_id, worker_name, status) "
                "VALUES (1,'w','online')")
    wid = cur.lastrowid
    cur.execute("INSERT INTO analysis_jobs (name, source_path, status, total_files) "
                "VALUES ('t','/x','active',2)")
    jid = cur.lastrowid
    from conftest import vec_blob
    vec = vec_blob(db)
    tasks = []
    for i in range(2):
        cur.execute("INSERT INTO files (file_path, file_name) VALUES (?,?)",
                    (f"/x/{i}.png", f"{i}.png"))
        fid = cur.lastrowid
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
    app.dependency_overrides[deps.get_current_user] = lambda: {"id": 1, "username": "w"}
    return db, TestClient(app), tasks, jid


def test_unexpected_item_error_does_not_500_the_whole_batch(server_env, monkeypatch):
    """complete_task_phase() commits per item. Before the fix, anything that
    was not HTTPException/ValueError (a DB lock, say) escaped the loop and
    500'd the request with the earlier items ALREADY committed — the worker
    then re-sent all 50 and the tail was never reported."""
    db, client, tasks, _ = server_env
    real = AnalysisJobManager.complete_task_phase
    calls = {"n": 0}

    def flaky(self, task_id, phase, success, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return real(self, task_id=task_id, phase=phase, success=success, **kw)

    monkeypatch.setattr(AnalysisJobManager, "complete_task_phase", flaky)

    resp = client.post("/api/v1/tasks/complete-batch", json={
        "results": [
            {"task_id": tasks[0], "phase": "vv", "success": True},
            {"task_id": tasks[1], "phase": "vv", "success": True},
        ]})

    assert resp.status_code == 200, "one bad item still failed the whole batch"
    body = resp.json()
    assert body["accepted"] == 1
    assert len(body["errors"]) == 1
    assert body["errors"][0]["task_id"] == tasks[1]
    assert "locked" in body["errors"][0]["error"]


# ── I2: every completed file must be counted exactly once ──────────────────

def _mc_daemon(tmp_path, save_result=True):
    """A daemon wired just enough to run _process_batch_mc's upload loop."""
    d = WorkerDaemon.__new__(WorkerDaemon)
    d.transport = SimpleNamespace(
        save_vision=lambda file_id, fields: save_result,
        report_complete=lambda *a, **k: None,
    )
    d.uploader = None
    d.server_url = "http://test"
    d.storage_mode = "shared_fs"
    d._total_completed = 0
    d._total_failed = 0
    d._phase_counts = {"mc": 0, "vv": 0, "mv": 0}
    d._phase_throughput = {"mc": 0.0, "vv": 0.0, "mv": 0.0}
    d._batch_throughput = 0.0
    d._report_buffer = []
    d._report_lock = threading.Lock()
    d._batch_report_supported = True
    d._io_thread = None
    d._result_queue = None
    d._authed_request = lambda m, url, **kw: _resp(200)
    d._report_task_start = lambda *a, **k: None
    d._report_task_phase = lambda *a, **k: None
    d._get_downloaded = lambda job: None
    d._resolve_thumbnail = lambda job: None
    d._clear_current = lambda: None

    thumb = tmp_path / "t.png"
    thumb.write_bytes(b"x")
    job = {"job_id": 1, "task_id": 11, "file_id": 21,
           "file_path": "/x/a.psd", "thumb_path": str(thumb), "metadata": {}}
    return d, [job]


def _stub_vision(d, monkeypatch, mark):
    """Replace the VLM phase; `mark(ctx)` sets up the post-vision state."""
    def fake(active, progress_callback=None):
        for ctx in active:
            ctx.vision_fields = {"mc_caption": "a caption"}
            # Mirrors the real vision loop, which counts the phase here.
            d._phase_counts["mc"] += 1
            mark(ctx)
        return 1.0
    monkeypatch.setattr(d, "_run_vision_phase", fake)


def test_file_saved_at_its_own_save_point_is_counted_once(tmp_path, monkeypatch):
    """Both save points (IO thread `_save_result`, inline CLI save) already
    increment _total_completed. The batch-end loop incremented again for every
    ctx marked `_saved` — so each file counted twice, and the heartbeat ships
    the delta as jobs_completed into the scheduler's batch sizing."""
    d, jobs = _mc_daemon(tmp_path)

    def mark(ctx):
        d._total_completed += 1     # what the real save point does
        ctx._save_ok = True
        ctx._saved = True

    _stub_vision(d, monkeypatch, mark)
    results = d._process_batch_mc(jobs)

    assert results == [(1, True, "")]
    assert d._total_completed == 1, (
        f"one file counted {d._total_completed}x — jobs_completed is inflated")
    assert d._phase_counts["mc"] == 1


def test_failed_inline_save_is_not_reported_as_success(tmp_path, monkeypatch):
    """`ctx._saved = True` was set even when the inline save FAILED, and the
    batch-end loop appended (job_id, True, "") for anything marked saved."""
    d, jobs = _mc_daemon(tmp_path)

    def mark(ctx):
        ctx._save_ok = False        # save was attempted and refused
        ctx._save_error = "MC save rejected"
        ctx._saved = True

    _stub_vision(d, monkeypatch, mark)
    results = d._process_batch_mc(jobs)

    assert results == [(1, False, "MC save rejected")]
    assert d._total_completed == 0, "a failed save was counted as completed"
    assert d._total_failed == 1


def test_batch_end_save_counts_the_phase_once(tmp_path, monkeypatch):
    """The batch-end save path incremented _phase_counts['mc'] a second time,
    on top of the vision loop's increment — so it disagreed with the queued
    and inline paths, which count once."""
    d, jobs = _mc_daemon(tmp_path, save_result=True)
    _stub_vision(d, monkeypatch, lambda ctx: None)   # nothing pre-saved

    results = d._process_batch_mc(jobs)

    assert results == [(1, True, "")]
    assert d._total_completed == 1
    assert d._phase_counts["mc"] == 1, (
        f"phase count is {d._phase_counts['mc']} for one file")


def test_batch_end_save_failure_counts_as_failed(tmp_path, monkeypatch):
    d, jobs = _mc_daemon(tmp_path, save_result="quota exceeded")
    _stub_vision(d, monkeypatch, lambda ctx: None)

    results = d._process_batch_mc(jobs)

    assert results[0][1] is False
    assert d._total_completed == 0
    assert d._total_failed == 1


# ── I3: connect-time reclaim must not touch a live sibling worker ───────────

@pytest.fixture()
def worker_env(tmp_path):
    """Two workers on ONE account, each holding an assigned task."""
    from backend.server.routers import workers as workers_router

    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "w.db"))
    AnalysisJobManager(db)
    cur = db.conn.cursor()
    cur.execute("INSERT INTO users (username, password_hash) VALUES ('acct','x')")
    cur.execute("INSERT INTO analysis_jobs (name, source_path, status, total_files) "
                "VALUES ('t','/x','active',2)")
    jid = cur.lastrowid

    sessions = {}
    tasks = {}
    for name in ("gpu-a", "gpu-b"):
        cur.execute("INSERT INTO worker_sessions (user_id, worker_name, status, "
                    "last_heartbeat) VALUES (1,?, 'online', datetime('now'))", (name,))
        sessions[name] = cur.lastrowid
        cur.execute("INSERT INTO files (file_path, file_name) VALUES (?,?)",
                    (f"/x/{name}.png", f"{name}.png"))
        fid = cur.lastrowid
        cur.execute(
            "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
            "download_status, parse_status, mc_status, vv_status, mv_status, "
            "mc_assigned_to) "
            "VALUES (?,?,?,'n/a','done','assigned','pending','pending',?)",
            (jid, fid, f"/x/{name}.png", sessions[name]))
        tasks[name] = cur.lastrowid
    db.conn.commit()

    app = FastAPI()
    app.include_router(workers_router.router)
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.get_current_user] = (
        lambda: {"id": 1, "username": "acct"})
    return db, TestClient(app), sessions, tasks


def _mc_state(db, task_id):
    row = db.conn.execute(
        "SELECT mc_status, mc_assigned_to FROM file_tasks WHERE id = ?",
        (task_id,)).fetchone()
    return tuple(row)   # sqlite3.Row does not compare equal to a plain tuple


def test_connect_does_not_reclaim_a_live_sibling_workers_tasks(worker_env):
    """gpu-b connecting must not disturb gpu-a. Scoping the reclaim by user_id
    alone reset the sibling's in-flight task, so a second worker redid the work
    and gpu-a's later /tasks/complete 403'd — its GPU work was thrown away."""
    db, client, sessions, tasks = worker_env

    resp = client.post("/workers/connect", json={
        "worker_name": "gpu-b", "hostname": "h", "origin": "headless",
        "launcher": "cli", "batch_capacity": 5})
    assert resp.status_code == 200

    status, assigned = _mc_state(db, tasks["gpu-a"])
    assert (status, assigned) == ("assigned", sessions["gpu-a"]), (
        "a live sibling's in-flight task was reclaimed by another worker's connect")
    assert db.conn.execute(
        "SELECT status FROM worker_sessions WHERE id = ?",
        (sessions["gpu-a"],)).fetchone()[0] == "online", (
        "a live sibling session was marked offline")


def test_connect_reclaims_the_same_workers_own_previous_session(worker_env):
    """The case the reclaim exists for: gpu-a crashed and reconnects under the
    same name — its stranded task must go back to pending immediately."""
    db, client, sessions, tasks = worker_env

    resp = client.post("/workers/connect", json={
        "worker_name": "gpu-a", "hostname": "h", "origin": "headless",
        "launcher": "cli", "batch_capacity": 5})
    assert resp.status_code == 200

    status, assigned = _mc_state(db, tasks["gpu-a"])
    assert (status, assigned) == ("pending", None), (
        "the worker's own stranded task was not reclaimed on reconnect")
    assert db.conn.execute(
        "SELECT status FROM worker_sessions WHERE id = ?",
        (sessions["gpu-a"],)).fetchone()[0] == "offline"
    # gpu-b untouched
    assert _mc_state(db, tasks["gpu-b"]) == ("assigned", sessions["gpu-b"])


# ── I7: a short Range read must fail, not produce a wrong hash ─────────────

class _ShortRangeSession:
    """Returns fewer bytes than the requested range asked for."""

    def __init__(self, status, payload):
        self.status, self.payload = status, payload

    def get(self, url, **kw):
        payload, status = self.payload, self.status

        class _R:
            status_code = status
            content = payload

            def raise_for_status(self): pass
            def iter_content(self, chunk_size=65536): yield payload
            def close(self): pass

        return _R()


def _client_with(session):
    from backend.remote.webdav_client import WebDAVClient
    c = WebDAVClient.__new__(WebDAVClient)
    c.session = session
    c.timeout = 5
    c._url = lambda p: f"http://dav/{p}"
    return c


@pytest.mark.parametrize("status", [206, 200])
def test_short_range_read_returns_none_instead_of_partial_bytes(status):
    """The bytes feed the boundary content_hash, which is the CAS cache key.
    Hashing a short read stored a hash that can never match the local
    computation — the cache missed forever and the backfill counted it done."""
    c = _client_with(_ShortRangeSession(status, b"only-8b"))   # 7 bytes
    assert c.read_range("f.psd", 0, 8191) is None


@pytest.mark.parametrize("status", [206, 200])
def test_full_range_read_still_returns_the_bytes(status):
    payload = b"z" * 8192
    c = _client_with(_ShortRangeSession(status, payload))
    assert c.read_range("f.psd", 0, 8191) == payload


def test_short_range_read_is_counted_as_failed_by_the_backfill(
        monkeypatch, _reset_backfill_state):
    """End of the chain: read_range None → _hash_one raises → the item is
    recorded as failed with a reason, instead of silently 'done'."""
    from backend.server.queue import hash_backfill as hb

    class _NullRangeClient:
        def read_range(self, *a, **k): return None
        def close(self): pass

    monkeypatch.setattr(hb, "get_webdav_source", lambda sid: {
        "url": "http://dav", "username": "u", "password": "p"}, raising=False)

    db = _GateSpyDB(rows=[(1, "webdav://s1/a.psd", 100000)])

    def fake_hash_one(file_path, file_size, clients):
        clients["s1"] = _NullRangeClient()
        head = clients["s1"].read_range(file_path, 0, 8191)
        if head is None:
            raise RuntimeError(f"range read failed: {file_path}")
        return "unreachable"

    monkeypatch.setattr(hb, "_hash_one", fake_hash_one)
    with hb._lock:
        hb._state["running"] = True
    hb._run(lambda: db)

    st = hb.get_status()
    assert st["failed"] == 1 and st["done"] == 0
    assert "range read failed" in (st["last_error"] or "")


# ── I14: the local-worker toggle must actually toggle ──────────────────────

def test_worker_toggle_writes_the_key_the_getter_and_gate_read(monkeypatch):
    """PATCH used to write `server.embedded_worker.enabled` — a key nothing in
    the repo reads — via `_set_dotted`, which does not persist. Enabling was a
    complete no-op and the GET reported an unrelated key."""
    from backend.server.routers import workers as w

    saved, started, stopped = {}, [], []

    class _Cfg:
        def get(self, key, default=None):
            return saved.get(key, default)

        def save_user_setting(self, key, value):
            saved[key] = value

        def _set_dotted(self, key, value):   # must NOT be used here
            raise AssertionError(f"non-persistent _set_dotted used for {key}")

    import backend.utils.config as cfgmod
    monkeypatch.setattr(cfgmod, "get_config", lambda: _Cfg())
    monkeypatch.setattr(w, "_start_local_worker", lambda app: started.append(True))
    monkeypatch.setattr(w, "_stop_local_worker", lambda: stopped.append(True))
    monkeypatch.setattr(w, "get_status", lambda: {"running": False}, raising=False)

    import backend.server.local_worker as lw
    monkeypatch.setattr(lw, "get_status", lambda: {"running": False})

    w.admin_update_embedded_worker(
        req=w.EmbeddedWorkerUpdate(enabled=True), request=None, admin={"id": 1})

    assert saved.get("server.auto_processing.enabled") is True, (
        f"enable wrote nothing readable — saved keys: {list(saved)}")
    assert started, "enable did not start the local worker"

    w.admin_update_embedded_worker(
        req=w.EmbeddedWorkerUpdate(enabled=False), request=None, admin={"id": 1})
    assert saved["server.auto_processing.enabled"] is False
    assert stopped, "disable did not stop the local worker"


# ── I9: a failed phase-resume after restore must be visible ────────────────

def test_restore_surfaces_a_failed_phase_resume():
    """import paused every phase; swallowing the resume failure left the whole
    pipeline stopped behind a 200 with nothing in the log."""
    import inspect
    from backend.server.routers import database

    src = inspect.getsource(database)
    assert "phases_resumed" in src, (
        "the restore path does not report whether phases were resumed")
    # The bare `except Exception: pass` around set_paused_phases must be gone.
    resume_block = src[src.index("set_paused_phases(db, {p: False"):]
    resume_block = resume_block[:400]
    assert "except Exception:\n                pass" not in resume_block, (
        "the phase-resume failure is still swallowed silently")
