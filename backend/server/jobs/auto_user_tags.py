"""Sprint 2 γ4: when a file accumulates N 'irrelevant' labels, add a
low-relevance user_tag. Idempotent — re-running doesn't duplicate.

Demotion is handled downstream (Phase D soft demotion uses rrf_score
penalty; this job adds the persistent user_tags signal so ranking
and downstream consumers also see it).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

LOW_RELEVANCE_TAG = "low-relevance"


def _split_tags(text):
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]


def _join_tags(tags):
    return ", ".join(tags)


def apply_feedback_to_user_tags(db, *, threshold: int = 3) -> int:
    cur = db.conn.cursor()
    rows = cur.execute(
        """SELECT file_id, COUNT(*) FROM search_feedback
           WHERE label = 'irrelevant'
           GROUP BY file_id
           HAVING COUNT(*) >= ?""",
        (int(threshold),),
    ).fetchall()

    updated = 0
    for file_id, _count in rows:
        existing = cur.execute(
            "SELECT user_tags FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        if existing is None:
            # search_feedback referenced a file that no longer exists.
            continue
        tags = _split_tags(existing[0])
        if LOW_RELEVANCE_TAG in tags:
            continue
        tags.append(LOW_RELEVANCE_TAG)
        cur.execute(
            "UPDATE files SET user_tags = ? WHERE id = ?",
            (_join_tags(tags), file_id),
        )
        updated += 1
    db.conn.commit()
    if updated:
        logger.info("auto_user_tags: tagged %d file(s) as low-relevance", updated)
    return updated
