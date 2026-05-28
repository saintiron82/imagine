"""Sprint 2 γ4: auto-update user_tags when a file is repeatedly flagged."""
from __future__ import annotations

import sqlite3
import types


def _db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            user_tags TEXT
        );
        CREATE TABLE search_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            user_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    conn.executemany(
        "INSERT INTO files (id, user_tags) VALUES (?, ?)",
        [(1, ""), (2, "existing"), (3, ""), (4, None)],
    )
    conn.commit()
    return types.SimpleNamespace(conn=conn)


def _seed(db, file_id, n):
    for _ in range(n):
        db.conn.execute(
            "INSERT INTO search_feedback (query, file_id, label, user_id) "
            "VALUES ('q', ?, 'irrelevant', 1)",
            (file_id,),
        )
    db.conn.commit()


def test_no_op_below_threshold():
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=1, n=2)
    n_updated = apply_feedback_to_user_tags(db, threshold=3)
    assert n_updated == 0
    assert db.conn.execute("SELECT user_tags FROM files WHERE id=1").fetchone()[0] == ""


def test_adds_low_relevance_tag_when_threshold_reached():
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=1, n=3)
    n_updated = apply_feedback_to_user_tags(db, threshold=3)
    assert n_updated == 1
    row = db.conn.execute("SELECT user_tags FROM files WHERE id=1").fetchone()[0]
    assert "low-relevance" in row


def test_does_not_duplicate_tag_on_re_run():
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=1, n=3)
    apply_feedback_to_user_tags(db, threshold=3)
    second = apply_feedback_to_user_tags(db, threshold=3)
    assert second == 0
    row = db.conn.execute("SELECT user_tags FROM files WHERE id=1").fetchone()[0]
    assert row.count("low-relevance") == 1


def test_preserves_existing_user_tags():
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=2, n=4)
    apply_feedback_to_user_tags(db, threshold=3)
    row = db.conn.execute("SELECT user_tags FROM files WHERE id=2").fetchone()[0]
    assert "existing" in row
    assert "low-relevance" in row


def test_handles_null_user_tags_column():
    """File row with NULL user_tags must still get the new tag."""
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=4, n=5)
    n_updated = apply_feedback_to_user_tags(db, threshold=3)
    assert n_updated == 1
    row = db.conn.execute("SELECT user_tags FROM files WHERE id=4").fetchone()[0]
    assert "low-relevance" in row


def test_skips_files_that_dont_exist_in_files_table():
    """If search_feedback references a file_id no longer in files, skip silently."""
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=99999, n=5)   # no such file row
    n_updated = apply_feedback_to_user_tags(db, threshold=3)
    assert n_updated == 0
