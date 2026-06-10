from pathlib import Path


def test_worker_daemon_no_longer_exposes_single_job_parse_pipeline():
    source = Path("backend/worker/worker_daemon.py").read_text(encoding="utf-8")

    assert "def process_job(" not in source
    assert "def _run_parse(" not in source


def test_worker_no_longer_calls_legacy_jobs_endpoints():
    """The server has no /api/v1/jobs/* routes — workers must not call them.

    Task lifecycle goes through /api/v1/tasks/*, payloads through
    /api/v1/files/{id}/(vision|vv|mv).
    """
    for module in ("worker_daemon.py", "worker_ipc.py", "result_uploader.py"):
        source = Path(f"backend/worker/{module}").read_text(encoding="utf-8")
        assert "/api/v1/jobs/" not in source, module
        assert "fail_job(" not in source, module
        assert "complete_job(" not in source, module


def test_result_uploader_exposes_only_files_api_writers():
    source = Path("backend/worker/result_uploader.py").read_text(encoding="utf-8")

    assert "def save_vision_fields(" in source
    assert "def save_vv_vector(" in source
    assert "def save_mv_vector(" in source
    assert "def complete_mc(" not in source
    assert "def report_progress(" not in source
