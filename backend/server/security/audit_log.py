"""Phase 8 audit log.

Append-only ledger of security-relevant events. Every call is
best-effort: an audit failure must never block the actual operation
(we use a defensive try/except per call site). Reads go through the
admin-only API that we'll wire when the UI catches up.

The set of allowed event names lives here so callers can't typo a name
and create silent observability holes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# These are the only event names accepted. Keep alphabetised so additions
# stay easy to review.
ALLOWED_EVENTS = frozenset(
    {
        "admin_login",
        "database_exported",
        "database_imported",
        "database_reset",
        "external_url_changed",
        "failed_login",
        "firebase_group_registered",
        "relay_connected",
        "relay_disconnected",
        "server_initialized",
        "worker_blocked",
        "worker_connected",
        "worker_enrollment_token_created",
        "worker_enrollment_token_revoked",
        "worker_token_revoked",
    }
)


def record(
    db,
    event: str,
    *,
    actor_user_id: Optional[int] = None,
    actor_username: Optional[str] = None,
    target_kind: Optional[str] = None,
    target_id: Any = None,
    ip_address: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    """Insert a row into audit_log. Never raises."""
    if event not in ALLOWED_EVENTS:
        logger.warning("audit_log: refusing unknown event %r", event)
        return
    try:
        conn = db.conn
        conn.execute(
            """INSERT INTO audit_log
               (event, actor_user_id, actor_username,
                target_kind, target_id, ip_address, detail)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event,
                actor_user_id,
                actor_username,
                target_kind,
                str(target_id) if target_id is not None else None,
                ip_address,
                detail,
            ),
        )
        conn.commit()
    except Exception as exc:  # pragma: no cover - audit must never crash callers
        logger.warning("audit_log insert failed (event=%s): %s", event, exc)


def recent(db, *, limit: int = 100, event: Optional[str] = None) -> list[dict]:
    """Read most-recent audit events (admin debug helper)."""
    sql = (
        "SELECT id, event, actor_user_id, actor_username, target_kind, "
        "target_id, ip_address, detail, created_at "
        "FROM audit_log"
    )
    params: list[Any] = []
    if event:
        sql += " WHERE event = ?"
        params.append(event)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cur = db.conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
