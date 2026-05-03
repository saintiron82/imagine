import sqlite3
from types import SimpleNamespace

import pytest

from backend.server.queue.analysis_manager import AnalysisJobManager


def test_worker_phase_validation_runs_before_sql():
    mgr = AnalysisJobManager.__new__(AnalysisJobManager)

    with pytest.raises(ValueError, match="invalid worker phase"):
        mgr.start_task_phase(task_id=1, phase="parse; DROP TABLE file_tasks")

    with pytest.raises(ValueError, match="invalid worker phase"):
        mgr.complete_task_phase(task_id=1, phase="download", success=True)


def test_retry_phase_validation_runs_before_sql():
    mgr = AnalysisJobManager.__new__(AnalysisJobManager)

    with pytest.raises(ValueError, match="invalid retry phase"):
        mgr.retry_failed(job_id=1, phase="bad_phase")


def test_retry_failed_returns_total_across_all_phases():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE analysis_jobs (id INTEGER PRIMARY KEY, status TEXT, completed_at TEXT)")
    conn.execute("""
        CREATE TABLE file_tasks (
            analysis_job_id INTEGER,
            download_status TEXT,
            parse_status TEXT,
            mc_status TEXT,
            vv_status TEXT,
            mv_status TEXT,
            retry_count INTEGER,
            max_retries INTEGER,
            error_message TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("INSERT INTO analysis_jobs (id, status, completed_at) VALUES (1, 'active', NULL)")
    conn.execute("""
        INSERT INTO file_tasks (
            analysis_job_id, download_status, parse_status, mc_status,
            vv_status, mv_status, retry_count, max_retries
        ) VALUES (1, 'failed', 'pending', 'failed', 'failed', 'pending', 0, 3)
    """)

    mgr = AnalysisJobManager.__new__(AnalysisJobManager)
    mgr.db = SimpleNamespace(conn=conn)

    assert mgr.retry_failed(job_id=1) == 3
    row = conn.execute("""
        SELECT download_status, mc_status, vv_status
        FROM file_tasks WHERE analysis_job_id = 1
    """).fetchone()
    assert row == ("pending", "pending", "pending")
