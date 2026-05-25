"""Phase 8: audit log captures security-relevant events."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from backend.server.security import audit_log


def _db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            actor_user_id INTEGER,
            actor_username TEXT,
            target_kind TEXT,
            target_id TEXT,
            ip_address TEXT,
            detail TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )"""
    )
    conn.commit()
    return SimpleNamespace(conn=conn)


def test_record_inserts_row_with_documented_fields():
    db = _db()
    audit_log.record(
        db,
        "worker_connected",
        actor_user_id=1,
        actor_username="admin",
        target_kind="worker_session",
        target_id=42,
        ip_address="192.168.0.8",
        detail="name=cloud-1 origin=headless launcher=cloud",
    )
    row = db.conn.execute("SELECT * FROM audit_log").fetchone()
    assert row[1] == "worker_connected"
    assert row[2] == 1
    assert row[3] == "admin"
    assert row[4] == "worker_session"
    assert row[5] == "42"
    assert row[6] == "192.168.0.8"
    assert "cloud-1" in row[7]


def test_record_silently_drops_unknown_events():
    db = _db()
    audit_log.record(db, "definitely_not_a_real_event")
    assert db.conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0


def test_record_never_raises_when_db_breaks():
    bad_db = SimpleNamespace(conn=None)  # any attribute access on conn explodes
    # Must not raise — audit failures cannot block the actual operation.
    audit_log.record(bad_db, "admin_login", actor_username="admin")


@pytest.mark.parametrize(
    "event",
    sorted(audit_log.ALLOWED_EVENTS),
)
def test_every_documented_event_is_accepted(event):
    db = _db()
    audit_log.record(db, event, detail="smoke")
    count = db.conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE event = ?", (event,)
    ).fetchone()[0]
    assert count == 1


def test_recent_returns_descending_order():
    db = _db()
    for i in range(5):
        audit_log.record(db, "admin_login", detail=f"login-{i}")
    rows = audit_log.recent(db, limit=3)
    assert len(rows) == 3
    assert rows[0]["detail"] == "login-4"
    assert rows[1]["detail"] == "login-3"
    assert rows[2]["detail"] == "login-2"


def test_recent_filters_by_event():
    db = _db()
    audit_log.record(db, "admin_login", detail="x")
    audit_log.record(db, "failed_login", detail="y")
    audit_log.record(db, "admin_login", detail="z")

    only_admin = audit_log.recent(db, event="admin_login")
    assert [r["detail"] for r in only_admin] == ["z", "x"]
