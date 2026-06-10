"""Phase 7: worker data-plane access scope.

Covers the four invariants from docs/data_plane_policy_ko.md:

1. worker can download only assigned file
2. worker cannot download another user's file
3. worker cannot update task not assigned to itself
4. blocked worker cannot claim
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.server.routers.analysis import (
    _require_file_phase_assignment,
    _require_task_assignment,
    _require_worker_session,
)


def _db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE worker_sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            status TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE file_tasks (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            mc_status TEXT DEFAULT 'pending',
            mc_assigned_to INTEGER,
            vv_status TEXT DEFAULT 'pending',
            vv_assigned_to INTEGER,
            mv_status TEXT DEFAULT 'pending',
            mv_assigned_to INTEGER
        )"""
    )
    # user 1 owns worker 10 (online); user 2 owns worker 20 (online);
    # worker 30 is blocked (user 1).
    conn.execute("INSERT INTO worker_sessions VALUES (10, 1, 'online')")
    conn.execute("INSERT INTO worker_sessions VALUES (20, 2, 'online')")
    conn.execute("INSERT INTO worker_sessions VALUES (30, 1, 'blocked')")

    # Two file_tasks:
    #   100 (file 500): MC assigned to worker 10 (user 1)
    #   200 (file 600): MC assigned to worker 20 (user 2)
    conn.execute(
        """INSERT INTO file_tasks
           (id, file_id, mc_status, mc_assigned_to)
           VALUES (100, 500, 'assigned', 10),
                  (200, 600, 'assigned', 20)"""
    )
    conn.commit()
    return SimpleNamespace(conn=conn)


# ── 1. Worker can download only assigned file ──────────────────────


def test_worker_can_access_its_own_assigned_file():
    db = _db()
    # user 1 fetching the MC of file 500 (assigned to its worker 10) — OK
    # via the file-phase helper that backs /api/v1/files/{id}/mc.
    # The helper actually only allows MV phase access for MC reads; we use
    # MC here directly to verify the underlying ownership check.
    _require_task_assignment(db, {"id": 1}, 100, "mc")


# ── 2. Worker cannot download another user's file ──────────────────


def test_worker_cannot_access_another_users_file():
    db = _db()
    with pytest.raises(HTTPException) as exc:
        _require_task_assignment(db, {"id": 1}, 200, "mc")
    assert exc.value.status_code == 403


def test_worker_cannot_access_via_file_phase_helper_when_unassigned():
    db = _db()
    with pytest.raises(HTTPException) as exc:
        _require_file_phase_assignment(db, {"id": 1}, 600, "mc")
    assert exc.value.status_code == 403


# ── 3. Worker cannot update task not assigned to itself ────────────


def test_worker_cannot_update_task_assigned_to_other_user_worker():
    db = _db()
    with pytest.raises(HTTPException) as exc:
        _require_task_assignment(db, {"id": 1}, 200, "mc")
    assert exc.value.status_code == 403


# ── 4. Blocked worker cannot claim ────────────────────────────────


def test_blocked_worker_session_is_rejected():
    db = _db()
    # worker 30 belongs to user 1 but is blocked.
    with pytest.raises(HTTPException) as exc:
        _require_worker_session(db, {"id": 1}, 30)
    assert exc.value.status_code == 403


def test_other_users_worker_session_is_rejected():
    db = _db()
    # user 1 trying to act as user 2's worker 20.
    with pytest.raises(HTTPException) as exc:
        _require_worker_session(db, {"id": 1}, 20)
    assert exc.value.status_code == 403
