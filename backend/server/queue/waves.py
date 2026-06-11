"""Model-version reprocessing waves (CAS M4).

A wave is just a normal analysis job whose tasks target files lacking a
'done' derivation under the ACTIVE model version of a phase. The regular
pipeline (scheduler, workers, shadow write) does the rest; search keeps
serving the old materialized results until each file is replaced.

Phase semantics:
- mc wave  → recompute mc AND mv (the MV vector embeds the MC caption)
- vv wave  → recompute vv only
- mv wave  → recompute mv only (requires an existing caption)

Non-target phases are marked 'done' (nothing to do — old results stand),
which keeps the job-completion predicate (all three done) working.
"""

import logging

from backend.utils.model_version import get_model_version

logger = logging.getLogger(__name__)

_WAVE_TARGETS = {
    "mc": ("mc", "mv"),
    "vv": ("vv",),
    "mv": ("mv",),
}


def _wave_candidates(cursor, phase: str, version: str):
    """Files that need recomputation for (phase, version).

    Requirements:
    - hashed (backfill first — unhashed rows can't use the cache)
    - no 'done' derivation under the active version
    - not already part of an active job (no double processing)
    - has the inputs the wave needs (thumbnail for mc/vv, caption for mv)
    """
    input_filter = (
        "AND f.mc_caption IS NOT NULL AND f.mc_caption != ''"
        if phase == "mv"
        else "AND f.thumbnail_url IS NOT NULL AND f.thumbnail_url != ''"
    )
    cursor.execute(f"""
        SELECT f.id, f.file_path FROM files f
        WHERE f.content_hash IS NOT NULL AND f.content_hash != ''
          {input_filter}
          AND NOT EXISTS (
              SELECT 1 FROM derivations d
              WHERE d.content_hash = f.content_hash
                AND d.phase = ? AND d.model_version = ?
                AND d.status = 'done'
          )
          AND NOT EXISTS (
              SELECT 1 FROM file_tasks ft
              JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
              WHERE ft.file_id = f.id AND aj.status = 'active'
          )
    """, (phase, version))
    return cursor.fetchall()


def create_wave_job(db, phase: str, *, dry_run: bool = False,
                    created_by: int = None) -> dict:
    """Create a reprocessing wave for a phase's active model version."""
    if phase not in _WAVE_TARGETS:
        return {"success": False, "error": f"invalid phase: {phase}"}

    version = get_model_version(phase)
    targets = _WAVE_TARGETS[phase]
    cursor = db.conn.cursor()
    rows = _wave_candidates(cursor, phase, version)

    if dry_run or not rows:
        db.conn.commit()  # release any read txn state
        return {
            "success": True, "dry_run": dry_run, "phase": phase,
            "model_version": version, "candidates": len(rows),
            "job_id": None,
        }

    cursor.execute(
        """INSERT INTO analysis_jobs (name, source_path, status, total_files, created_by)
           VALUES (?, ?, 'active', ?, ?)""",
        (f"재처리 파도: {phase} → {version}", f"wave://{phase}", len(rows), created_by),
    )
    job_id = cursor.lastrowid

    status = {p: ("pending" if p in targets else "done") for p in ("mc", "vv", "mv")}
    cursor.executemany(
        """INSERT INTO file_tasks
           (analysis_job_id, file_id, file_path,
            download_status, parse_status, mc_status, vv_status, mv_status)
           VALUES (?, ?, ?, 'n/a', 'done', ?, ?, ?)""",
        [(job_id, fid, fpath, status["mc"], status["vv"], status["mv"])
         for fid, fpath in rows],
    )
    db.conn.commit()

    logger.info(
        f"Wave created: job={job_id} phase={phase} version={version} "
        f"files={len(rows)} (recompute: {targets})"
    )
    return {
        "success": True, "dry_run": False, "phase": phase,
        "model_version": version, "candidates": len(rows), "job_id": job_id,
    }
