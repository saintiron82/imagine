"""Phase D: search_feedback table schema."""
from __future__ import annotations

import sqlite3
import types


def _db():
    conn = sqlite3.connect(":memory:")
    return types.SimpleNamespace(
        conn=conn,
        _table_exists=lambda name: conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None,
    )


def test_migration_creates_search_feedback_table():
    from backend.db.sqlite_migrations import migrate_search_feedback

    db = _db()
    assert db._table_exists("search_feedback") is False
    migrate_search_feedback(db)
    assert db._table_exists("search_feedback") is True

    cols = {row[1] for row in db.conn.execute(
        "PRAGMA table_info(search_feedback)"
    ).fetchall()}
    assert {"id", "query", "file_id", "label", "user_id", "created_at"}.issubset(cols)


def test_migration_is_idempotent():
    from backend.db.sqlite_migrations import migrate_search_feedback

    db = _db()
    migrate_search_feedback(db)
    migrate_search_feedback(db)  # second call must not raise
    assert db._table_exists("search_feedback")


def test_label_check_constraint_only_allows_irrelevant():
    """Schema enforces label = 'irrelevant' (forward-compat: other labels
    require an explicit migration to widen the constraint)."""
    from backend.db.sqlite_migrations import migrate_search_feedback
    import pytest

    db = _db()
    migrate_search_feedback(db)
    # Valid insert
    db.conn.execute(
        "INSERT INTO search_feedback (query, file_id, label, user_id) "
        "VALUES (?, ?, ?, ?)",
        ("query-a", 7, "irrelevant", 1),
    )
    db.conn.commit()
    # Invalid label must raise
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO search_feedback (query, file_id, label, user_id) "
            "VALUES (?, ?, ?, ?)",
            ("query-a", 7, "love-it", 1),
        )
        db.conn.commit()


def test_indices_on_query_and_file_id_exist_for_lookup():
    """Soft-demotion query at search time hits (query, file_id) — both
    columns must be indexed so feedback lookups stay O(log n)."""
    from backend.db.sqlite_migrations import migrate_search_feedback

    db = _db()
    migrate_search_feedback(db)
    indices = {
        row[1] for row in db.conn.execute(
            "SELECT * FROM sqlite_master WHERE type='index' "
            "AND tbl_name='search_feedback'"
        ).fetchall()
    }
    assert "idx_search_feedback_query" in indices
    assert "idx_search_feedback_file" in indices
