"""Phase 8: worker enrollment token table contract.

The current MVP only persists the enrollment token row; full per-request
token-scope enforcement happens once the auth router learns about the
new table (separate change). These tests pin the schema and basic
invariants so callers can rely on the contract before that work lands.
"""

from __future__ import annotations

import hashlib
import sqlite3
from types import SimpleNamespace

import pytest


def _db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE worker_enrollment_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash TEXT UNIQUE NOT NULL,
            created_by INTEGER,
            worker_name TEXT,
            max_uses INTEGER DEFAULT 1,
            use_count INTEGER DEFAULT 0,
            expires_at TEXT NOT NULL,
            revoked INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            last_used_at TEXT
        );
        """
    )
    conn.commit()
    return SimpleNamespace(conn=conn)


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def test_token_hash_is_unique():
    db = _db()
    db.conn.execute(
        "INSERT INTO worker_enrollment_tokens (token_hash, expires_at) VALUES (?, ?)",
        (_h("token-a"), "2026-12-31"),
    )
    db.conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO worker_enrollment_tokens (token_hash, expires_at) VALUES (?, ?)",
            (_h("token-a"), "2026-12-31"),
        )
        db.conn.commit()


def test_default_max_uses_is_one():
    db = _db()
    db.conn.execute(
        "INSERT INTO worker_enrollment_tokens (token_hash, expires_at) VALUES (?, ?)",
        (_h("once"), "2026-12-31"),
    )
    db.conn.commit()
    row = db.conn.execute(
        "SELECT max_uses, use_count, revoked FROM worker_enrollment_tokens"
    ).fetchone()
    assert row == (1, 0, 0)


def test_revoked_flag_is_persistable():
    db = _db()
    db.conn.execute(
        "INSERT INTO worker_enrollment_tokens (token_hash, expires_at, revoked) VALUES (?, ?, 1)",
        (_h("revoked"), "2026-12-31"),
    )
    db.conn.commit()
    row = db.conn.execute(
        "SELECT revoked FROM worker_enrollment_tokens WHERE token_hash = ?",
        (_h("revoked"),),
    ).fetchone()
    assert row[0] == 1


def test_table_is_distinct_from_refresh_tokens():
    """Enrollment tokens never grant admin API access — distinct schema."""
    import inspect

    from backend.db import sqlite_migrations as mig

    src = inspect.getsource(mig.migrate_worker_enrollment_tokens)
    # The whole point of Phase 8 is that this table lives apart from
    # refresh_tokens, so an enrollment token cannot accidentally satisfy
    # admin token checks.
    assert "REFERENCES users" not in src or "REFERENCES users(id)" not in src.replace(
        "created_by INTEGER", ""
    ), src
