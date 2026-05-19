from pathlib import Path


def test_worker_daemon_no_longer_exposes_single_job_parse_pipeline():
    source = Path("backend/worker/worker_daemon.py").read_text(encoding="utf-8")

    assert "def process_job(" not in source
    assert "def _run_parse(" not in source
