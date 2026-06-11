"""Derivation cache write path (CAS M1 — shadow write).

Results saved for a file are also recorded under
(content_hash, phase, model_version) so identical content can later be
materialized without recomputation (reads activate in M3). Files without
a content_hash (pre-backfill legacy rows) are skipped silently.

Shadow writes must never fail the primary save — callers wrap in
record_derivation()'s own error handling and commit as part of their
own transaction.
"""

import logging

from backend.utils.model_version import get_model_version

logger = logging.getLogger(__name__)


def _ensure_active_registry(cursor, phase: str, version: str):
    """Make sure the registry knows this version and marks it active.

    M1 keeps this passive (no wave generation on change — that is M4):
    the currently-produced version is simply the active one.
    """
    cursor.execute(
        """INSERT OR IGNORE INTO model_registry (phase, model_version, is_active, activated_at)
           VALUES (?, ?, 1, datetime('now'))""",
        (phase, version),
    )
    cursor.execute(
        "UPDATE model_registry SET is_active = (model_version = ?) WHERE phase = ?",
        (version, phase),
    )


def record_derivation(db, file_id: int, phase: str, *,
                      result_json: str = None,
                      vector_blob: bytes = None,
                      created_by: str = None) -> bool:
    """Record a completed phase result in the derivation cache.

    Uses the caller's connection/transaction — the caller commits.
    Returns True if a cache row was written, False if skipped (no hash).
    Never raises: shadow write failures are logged and swallowed.
    """
    try:
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT content_hash FROM files WHERE id = ?", (file_id,)
        )
        row = cursor.fetchone()
        content_hash = row[0] if row else None
        if not content_hash:
            return False  # legacy row — resolved by the M2 backfill

        version = get_model_version(phase)
        _ensure_active_registry(cursor, phase, version)
        cursor.execute(
            """INSERT OR REPLACE INTO derivations
               (content_hash, phase, model_version, status,
                result_json, vector_blob, created_at, created_by)
               VALUES (?, ?, ?, 'done', ?, ?, datetime('now'), ?)""",
            (content_hash, phase, version, result_json, vector_blob, created_by),
        )
        return True
    except Exception as e:
        logger.warning(f"Derivation shadow write skipped (file={file_id}, {phase}): {e}")
        return False
