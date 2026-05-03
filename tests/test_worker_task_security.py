import sqlite3
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.modules.setdefault(
    "jwt",
    types.SimpleNamespace(
        ExpiredSignatureError=Exception,
        InvalidTokenError=Exception,
        decode=lambda *args, **kwargs: {},
        encode=lambda *args, **kwargs: "",
    ),
)

from backend.server.routers.analysis import (
    _require_file_phase_assignment,
    _require_task_assignment,
    _require_worker_session,
)


def _db():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE worker_sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            status TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE file_tasks (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            mc_status TEXT DEFAULT 'pending',
            mc_assigned_to INTEGER,
            vv_status TEXT DEFAULT 'pending',
            vv_assigned_to INTEGER,
            mv_status TEXT DEFAULT 'pending',
            mv_assigned_to INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO worker_sessions (id, user_id, status) VALUES (10, 1, 'online')"
    )
    conn.execute(
        "INSERT INTO worker_sessions (id, user_id, status) VALUES (20, 2, 'online')"
    )
    conn.execute("""
        INSERT INTO file_tasks
            (id, file_id, mc_status, mc_assigned_to, vv_status, vv_assigned_to)
        VALUES
            (100, 500, 'assigned', 10, 'assigned', 20)
    """)
    conn.commit()
    return SimpleNamespace(conn=conn)


def test_worker_session_must_belong_to_current_user():
    db = _db()
    _require_worker_session(db, {"id": 1}, 10)

    with pytest.raises(HTTPException) as exc:
        _require_worker_session(db, {"id": 1}, 20)
    assert exc.value.status_code == 403


def test_task_assignment_must_belong_to_current_users_worker():
    db = _db()
    _require_task_assignment(db, {"id": 1}, 100, "mc")

    with pytest.raises(HTTPException) as exc:
        _require_task_assignment(db, {"id": 1}, 100, "vv")
    assert exc.value.status_code == 403


def test_file_phase_assignment_must_belong_to_current_users_worker():
    db = _db()
    _require_file_phase_assignment(db, {"id": 1}, 500, "mc")

    with pytest.raises(HTTPException) as exc:
        _require_file_phase_assignment(db, {"id": 1}, 500, "vv")
    assert exc.value.status_code == 403
