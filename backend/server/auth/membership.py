"""
Membership: email-bound invites, seat accounting, and entry gating (IMGV2-14).

Decisions (user):
- No auto-signup: an uninvited Firebase user is rejected at /auth/connect
  (except first-admin bootstrap when no Firebase users exist yet).
- Email invite reserves a seat; accepting converts the reservation into a user.
- Hard expiry block: when the server license is expired, non-admins are denied
  entry (admins are exempt so they can renew). This is additive (server_expires_at
  in system_meta) and independent of the licensing grace subsystem.

Access is governed by the `users` table (the same table /auth/connect keys on).
Seat cap reuses the existing system_meta `max_users` (0 = unlimited).
"""
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException


def ensure_invites_table(db) -> None:
    """Idempotently create the email-bound invites table (works on existing DBs)."""
    db.conn.execute(
        """
        CREATE TABLE IF NOT EXISTS member_invites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            role TEXT DEFAULT 'user' CHECK (role IN ('admin','user')),
            status TEXT DEFAULT 'pending' CHECK (status IN ('pending','accepted','revoked')),
            invited_by INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            accepted_at TEXT,
            accepted_user_id INTEGER
        )
        """
    )
    db.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_member_invites_email ON member_invites(email)"
    )
    db.conn.commit()


# ── Seats ────────────────────────────────────────────────────
def seat_limit(db) -> int:
    row = db.conn.execute(
        "SELECT value FROM system_meta WHERE key='max_users'"
    ).fetchone()
    try:
        return int(row[0]) if row and row[0] is not None else 0
    except (TypeError, ValueError):
        return 0


def count_active_users(db) -> int:
    return db.conn.execute(
        "SELECT COUNT(*) FROM users WHERE is_active=1"
    ).fetchone()[0]


def count_pending_invites(db) -> int:
    ensure_invites_table(db)
    return db.conn.execute(
        "SELECT COUNT(*) FROM member_invites WHERE status='pending'"
    ).fetchone()[0]


def seats_used(db) -> int:
    return count_active_users(db) + count_pending_invites(db)


def assert_seat_available(db, n: int = 1) -> None:
    limit = seat_limit(db)
    if limit and seats_used(db) + n > limit:
        raise HTTPException(status_code=409, detail="좌석이 가득 찼습니다 (플랜 한도 초과)")


# ── Invites ──────────────────────────────────────────────────
def find_pending_invite(db, email: str):
    """Return (id, role) of a pending invite for this email, or None."""
    if not email:
        return None
    ensure_invites_table(db)
    return db.conn.execute(
        "SELECT id, role FROM member_invites WHERE lower(email)=lower(?) AND status='pending'",
        (email,),
    ).fetchone()


def create_invite(db, email: str, role: str = "user", invited_by: Optional[int] = None) -> dict:
    """Create a pending invite (reserving a seat). Idempotent per pending email."""
    ensure_invites_table(db)
    existing = find_pending_invite(db, email)
    if existing:
        row = db.conn.execute(
            "SELECT token FROM member_invites WHERE id=?", (existing[0],)
        ).fetchone()
        return {"email": email, "token": row[0], "role": existing[1], "reused": True}
    assert_seat_available(db, 1)
    token = secrets.token_urlsafe(24)
    role = role if role in ("admin", "user") else "user"
    db.conn.execute(
        "INSERT INTO member_invites (email, token, role, invited_by) VALUES (?, ?, ?, ?)",
        (email, token, role, invited_by),
    )
    db.conn.commit()
    return {"email": email, "token": token, "role": role, "reused": False}


def mark_invite_accepted(db, invite_id: int, user_id: int) -> None:
    db.conn.execute(
        "UPDATE member_invites SET status='accepted', accepted_at=datetime('now'), accepted_user_id=? WHERE id=?",
        (user_id, invite_id),
    )
    db.conn.commit()


def revoke_invite(db, invite_id: int) -> bool:
    ensure_invites_table(db)
    cur = db.conn.execute(
        "UPDATE member_invites SET status='revoked' WHERE id=? AND status='pending'",
        (invite_id,),
    )
    db.conn.commit()
    return cur.rowcount > 0


# ── Expiry (hard entry block) ────────────────────────────────
def server_expires_at(db) -> Optional[str]:
    row = db.conn.execute(
        "SELECT value FROM system_meta WHERE key='server_expires_at'"
    ).fetchone()
    return row[0] if row and row[0] else None


def is_expired(db) -> bool:
    exp = server_expires_at(db)
    if not exp:
        return False
    try:
        dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) > dt


def assert_entry_allowed(db, role: str) -> None:
    """Hard block: expired license denies entry to non-admins (admins renew)."""
    if role != "admin" and is_expired(db):
        raise HTTPException(
            status_code=403,
            detail="라이선스가 만료되어 서버 입장이 차단되었습니다. 갱신 후 다시 시도하세요.",
        )
