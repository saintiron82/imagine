"""Parallel FileTaskParsePool (2026-06-11): pipelined thread pool replaces
the single-threaded batch-barrier loop. Verifies real files parse
concurrently and statuses/thumbnails land correctly.
"""

import time

import pytest
from PIL import Image

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.queue.file_task_parse_pool import FileTaskParsePool


@pytest.fixture()
def env(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (64 + i, 64), (i * 40, 100, 150)).save(p)
        paths.append(str(p))

    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager._initialized = False  # fresh temp DB needs full schema
    AnalysisJobManager(db)  # ensures elapsed/snapshot columns
    cur = db.conn.cursor()
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files) "
        "VALUES ('t', ?, 'active', 6)", (str(tmp_path),))
    jid = cur.lastrowid
    for p in paths:
        cur.execute("INSERT INTO files (file_path, file_name) VALUES (?,?)",
                    (p, p.rsplit("/", 1)[-1]))
        cur.execute(
            "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
            "download_status, parse_status) VALUES (?,?,?,'n/a','pending')",
            (jid, cur.lastrowid, p))
    db.conn.commit()
    return db


def test_parallel_parse_completes_all_tasks(env):
    db = env
    pool = FileTaskParsePool(db)
    assert pool._parse_workers >= 1

    pool.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            done = db.conn.execute(
                "SELECT COUNT(*) FROM file_tasks WHERE parse_status='done'"
            ).fetchone()[0]
            db.conn.rollback()
            if done == 6:
                break
            time.sleep(0.3)
    finally:
        pool.stop()

    statuses = dict(db.conn.execute(
        "SELECT parse_status, COUNT(*) FROM file_tasks GROUP BY parse_status"
    ).fetchall())
    assert statuses == {"done": 6}, statuses

    # Every file got parse_elapsed recorded and a thumbnail
    elapsed = db.conn.execute(
        "SELECT COUNT(*) FROM file_tasks WHERE parse_elapsed_s IS NOT NULL"
    ).fetchone()[0]
    assert elapsed == 6
    thumbs = db.conn.execute(
        "SELECT COUNT(*) FROM files WHERE thumbnail_url IS NOT NULL "
        "AND thumbnail_url != ''"
    ).fetchone()[0]
    assert thumbs == 6


def test_submit_respects_capacity(env):
    """Claim volume is bounded by 2× worker threads (pipelined, no barrier)."""
    pool = FileTaskParsePool(env)
    pool._running = True  # claim loop guards on this; no loop thread needed here
    try:
        submitted = pool._submit_tasks()
        assert 0 < submitted <= pool._parse_workers * 2
    finally:
        pool._running = False
        pool._executor.shutdown(wait=True)
