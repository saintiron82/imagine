"""DB write-lock contention → 503 (retryable), not 500.

Under heavy concurrent writes SQLite raises OperationalError('database is locked').
The app exception handler must map that to 503 + Retry-After so frontend apiClient
and worker _authed_request retry instead of hard-failing (which marked workers
offline and surfaced 'Internal server error' on login).
"""
import asyncio
import sqlite3

from backend.server.app import unhandled_exception_handler


class _URL:
    path = "/api/v1/test"


class _Req:
    method = "POST"
    url = _URL()


def _handle(exc):
    return asyncio.run(unhandled_exception_handler(_Req(), exc))


def test_database_locked_maps_to_503_retryable():
    resp = _handle(sqlite3.OperationalError("database is locked"))
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "1"


def test_database_busy_maps_to_503():
    resp = _handle(sqlite3.OperationalError("database is busy"))
    assert resp.status_code == 503


def test_non_lock_operational_error_stays_500():
    resp = _handle(sqlite3.OperationalError("no such table: files"))
    assert resp.status_code == 500


def test_generic_exception_stays_500():
    resp = _handle(ValueError("boom"))
    assert resp.status_code == 500
