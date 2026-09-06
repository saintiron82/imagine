"""Derivation cache (CAS) — shadow write (M1) + lookup/materialize (M3).

Results saved for a file are also recorded under
(content_hash, phase, model_version) so identical content is materialized
without recomputation. The read side lives here too: lookup_done(),
materialize() and apply_cache_hits(). Files without
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


# ── M3: cache lookup + materialization ───────────────────────


def lookup_done(db, content_hash: str) -> dict:
    """Cached 'done' derivations for this content under the ACTIVE model
    versions. Returns {phase: row_dict}."""
    if not content_hash:
        return {}
    cursor = db.conn.cursor()
    hits = {}
    for phase in ("mc", "vv", "mv"):
        try:
            version = get_model_version(phase)
        except Exception:
            continue
        cursor.execute(
            """SELECT result_json, vector_blob FROM derivations
               WHERE content_hash = ? AND phase = ? AND model_version = ?
                 AND status = 'done'""",
            (content_hash, phase, version),
        )
        row = cursor.fetchone()
        if row:
            hits[phase] = {"result_json": row[0], "vector_blob": row[1]}
    return hits


def materialize(db, file_id: int, phase: str, derivation: dict) -> bool:
    """Apply a cached derivation to the search plane for one file.

    Reuses the same storage paths the worker save endpoints use — no new
    write code (CAS design §4.1). Caller commits.
    """
    try:
        if phase == "mc":
            import json as _json
            from backend.server.routers.analysis import _save_vision_fields_for_file
            body = _json.loads(derivation.get("result_json") or "{}")
            return bool(body) and _save_vision_fields_for_file(db, file_id, body)

        blob = derivation.get("vector_blob")
        if not blob:
            return False
        table = "vec_files" if phase == "vv" else "vec_text"
        cursor = db.conn.cursor()
        cursor.execute(f"DELETE FROM {table} WHERE file_id = ?", (file_id,))
        cursor.execute(
            f"INSERT INTO {table} (file_id, embedding) VALUES (?, ?)",
            (file_id, blob),
        )
        return True
    except Exception as e:
        logger.warning(f"Materialize failed (file={file_id}, {phase}): {e}")
        return False


def apply_cache_hits(db, task_id: int, file_id: int, content_hash: str) -> list:
    """After parse: materialize every cached phase and mark the ledger.

    Returns the list of phases satisfied from cache. Phases NOT in the
    list stay pending and flow to workers as usual. Caller commits.
    """
    hits = lookup_done(db, content_hash)

    # MV embeds the MC caption: an MV hit is only coherent when MC is also
    # served from cache (otherwise a fresh MC could produce a different
    # caption than the one the cached MV vector encodes).
    if "mv" in hits and "mc" not in hits:
        del hits["mv"]

    applied = []
    cursor = db.conn.cursor()
    for phase, derivation in hits.items():
        if not materialize(db, file_id, phase, derivation):
            continue
        cursor.execute(
            f"""UPDATE file_tasks
                SET {phase}_status = 'done',
                    {phase}_cache_hit = 1,
                    {phase}_completed_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE id = ? AND {phase}_status = 'pending'""",
            (task_id,),
        )
        if cursor.rowcount:
            applied.append(phase)
    return applied
