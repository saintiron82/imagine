"""
Tests for the admin live-log ring buffer + cursor-polling endpoint (IMGV2-26).
"""
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server import log_buffer
from backend.server.routers import logs as logs_mod
from backend.server.deps import require_admin


@pytest.fixture(autouse=True)
def fresh_buffer():
    log_buffer._reset_for_tests()
    log_buffer._installed = False
    log_buffer.install()
    yield
    log_buffer._reset_for_tests()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(logs_mod.router, prefix="/api/v1")
    app.dependency_overrides[require_admin] = lambda: {"id": 1, "username": "admin"}
    return TestClient(app)


def test_buffer_captures_and_cursors():
    log = logging.getLogger("backend.server.queue.scheduler")  # → job category
    log.info("scheduling batch 1")
    log.warning("worker slow")

    first = log_buffer.get_since(after=0)
    assert len(first["entries"]) == 2
    assert first["entries"][0]["message"] == "scheduling batch 1"
    assert first["entries"][0]["category"] == "job"
    assert first["entries"][1]["level"] == "WARNING"
    last = first["last_seq"]

    # cursor: nothing new yet
    assert log_buffer.get_since(after=last)["entries"] == []

    log.info("batch 2 done")
    nxt = log_buffer.get_since(after=last)
    assert len(nxt["entries"]) == 1
    assert nxt["entries"][0]["message"] == "batch 2 done"


def test_category_network():
    logging.getLogger("backend.server.queue.download_ahead").info("fetching from nas")
    e = log_buffer.get_since(after=0)["entries"][-1]
    assert e["category"] == "network"


def test_level_filter():
    log = logging.getLogger("backend.test")
    log.info("info line")
    log.error("boom")
    only_err = log_buffer.get_since(after=0, level="WARNING")
    assert all(x["level"] in ("WARNING", "ERROR", "CRITICAL") for x in only_err["entries"])
    assert any(x["message"] == "boom" for x in only_err["entries"])
    assert not any(x["message"] == "info line" for x in only_err["entries"])


def test_endpoint_returns_entries_and_cursor(client):
    logging.getLogger("backend.server.queue.phase_runner").info("phase mc start")
    r = client.get("/api/v1/admin/logs?after=0")
    assert r.status_code == 200
    body = r.json()
    assert "last_seq" in body and body["last_seq"] >= 1
    assert any(e["message"] == "phase mc start" for e in body["entries"])
