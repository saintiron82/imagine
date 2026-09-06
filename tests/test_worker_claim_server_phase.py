"""Worker applies the server-assigned phase from the claim response
(2026-06-11) — the scheduler is authoritative.

Before this fix, claim_jobs_count never updated processing_mode, so a
worker stayed locked to its connect-time mode and could run tasks
through the wrong pipeline when the scheduler assigned a different
phase.
"""

from types import SimpleNamespace

from backend.worker.worker_daemon import WorkerDaemon


def _daemon(mode="mc"):
    d = WorkerDaemon.__new__(WorkerDaemon)  # skip heavy __init__
    d.processing_mode = mode
    d.session_id = 42
    d.server_url = "http://test"
    d.transport = None
    d._log_prefix = "[test]"
    return d


def _resp(payload, status=200):
    return SimpleNamespace(status_code=status, json=lambda: payload)


def test_claim_applies_server_assigned_phase():
    d = _daemon(mode="mc")
    requests_seen = []

    def fake(method, url, **kw):
        requests_seen.append(kw["json"])
        return _resp({
            "success": True, "phase": "vv",
            "tasks": [{"task_id": 1, "file_id": 9, "file_path": "/a.png",
                       "job_id": 3}],
        })

    d._authed_request = fake
    jobs = d.claim_jobs_count(10)

    assert len(jobs) == 1
    assert d.processing_mode == "vv"          # server phase applied
    # Server decides phase+count — request carries only the worker id
    assert requests_seen == [{"worker_id": 42}]


def test_empty_claim_keeps_current_mode():
    d = _daemon(mode="vv")
    d._authed_request = lambda m, u, **k: _resp(
        {"success": True, "phase": None, "tasks": []})
    assert d.claim_jobs_count(10) == []
    assert d.processing_mode == "vv"


def test_claim_passes_vision_data_through():
    d = _daemon(mode="mv")
    d._authed_request = lambda m, u, **k: _resp({
        "success": True, "phase": "mv",
        "tasks": [{"task_id": 1, "file_id": 9, "file_path": "/a.png",
                   "job_id": 3,
                   "vision_data": {"mc_caption": "a chair"}}],
    })
    jobs = d.claim_jobs_count(10)
    assert jobs[0]["vision_data"] == {"mc_caption": "a chair"}


def test_idle_worker_can_still_claim():
    """The old phase gate blocked idle workers from ever claiming."""
    d = _daemon(mode="idle")
    d._authed_request = lambda m, u, **k: _resp({
        "success": True, "phase": "mc",
        "tasks": [{"task_id": 1, "file_id": 9, "file_path": "/a.png",
                   "job_id": 3}],
    })
    jobs = d.claim_jobs_count(10)
    assert len(jobs) == 1
    assert d.processing_mode == "mc"
