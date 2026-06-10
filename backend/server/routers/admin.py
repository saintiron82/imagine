"""
Admin router — user management, invite codes, embedded worker control.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_db_safe, require_admin, get_current_user
from backend.server.auth.schemas import (
    InviteCodeCreate, InviteCodeResponse,
    UserResponse, UserUpdateRequest,
    MemberResponse, MemberRoleUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Invite Codes ─────────────────────────────────────────────

@router.post("/invite-codes", response_model=InviteCodeResponse)
def create_invite_code(
    req: InviteCodeCreate,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Create a new invite code (admin only)."""
    code = secrets.token_urlsafe(16)[:16].upper()
    expires_at = None
    if req.expires_in_days:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=req.expires_in_days)
        ).isoformat()

    cursor = db.conn.cursor()
    cursor.execute(
        """INSERT INTO invite_codes (code, created_by, max_uses, expires_at)
           VALUES (?, ?, ?, ?)""",
        (code, admin["id"], req.max_uses, expires_at)
    )
    db.conn.commit()
    invite_id = cursor.lastrowid

    logger.info(f"Admin {admin['username']} created invite code: {code}")
    return InviteCodeResponse(
        id=invite_id, code=code, created_by=admin["id"],
        max_uses=req.max_uses, use_count=0, expires_at=expires_at,
    )


@router.get("/invite-codes", response_model=List[InviteCodeResponse])
def list_invite_codes(
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List all invite codes (admin only)."""
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT id, code, created_by, max_uses, use_count, expires_at, created_at FROM invite_codes ORDER BY created_at DESC"
    )
    return [
        InviteCodeResponse(
            id=row[0], code=row[1], created_by=row[2],
            max_uses=row[3], use_count=row[4],
            expires_at=row[5], created_at=row[6],
        )
        for row in cursor.fetchall()
    ]


# ── User Management ──────────────────────────────────────────

@router.get("/users", response_model=List[UserResponse])
def list_users(
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List all users (admin only)."""
    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT id, username, email, role, is_active, created_at,
                  last_login_at, quota_files_per_day, quota_search_per_min
           FROM users ORDER BY created_at DESC"""
    )
    return [
        UserResponse(
            id=row[0], username=row[1], email=row[2], role=row[3],
            is_active=bool(row[4]), created_at=row[5], last_login_at=row[6],
            quota_files_per_day=row[7], quota_search_per_min=row[8],
        )
        for row in cursor.fetchall()
    ]


@router.patch("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    req: UserUpdateRequest,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Update a user (admin only)."""
    cursor = db.conn.cursor()

    # Build dynamic UPDATE
    updates = {}
    if req.role is not None:
        if req.role not in ("admin", "user"):
            raise HTTPException(status_code=400, detail="Invalid role")
        updates["role"] = req.role
    if req.is_active is not None:
        updates["is_active"] = 1 if req.is_active else 0
    if req.quota_files_per_day is not None:
        updates["quota_files_per_day"] = req.quota_files_per_day
    if req.quota_search_per_min is not None:
        updates["quota_search_per_min"] = req.quota_search_per_min

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [user_id]
    cursor.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="User not found")

    db.conn.commit()
    logger.info(f"Admin {admin['username']} updated user {user_id}: {updates}")

    # Return updated user
    cursor.execute(
        """SELECT id, username, email, role, is_active, created_at,
                  last_login_at, quota_files_per_day, quota_search_per_min
           FROM users WHERE id = ?""",
        (user_id,)
    )
    row = cursor.fetchone()
    return UserResponse(
        id=row[0], username=row[1], email=row[2], role=row[3],
        is_active=bool(row[4]), created_at=row[5], last_login_at=row[6],
        quota_files_per_day=row[7], quota_search_per_min=row[8],
    )


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Delete a user (admin only). Cannot delete yourself."""
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="User not found")

    db.conn.commit()
    logger.info(f"Admin {admin['username']} deleted user {user_id}")
    return {"success": True, "deleted_user_id": user_id}


# ── Member Management (Firebase Auth) ────────────────────────

@router.get("/members", response_model=List[MemberResponse])
def list_members(
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List all group members (admin only)."""
    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT id, firebase_uid, email, display_name, role, is_active,
                  joined_at, last_seen_at, quota_files_per_day, quota_search_per_min
           FROM members ORDER BY joined_at DESC"""
    )
    return [
        MemberResponse(
            id=row[0], firebase_uid=row[1], email=row[2],
            display_name=row[3], role=row[4], is_active=bool(row[5]),
            joined_at=row[6], last_seen_at=row[7],
            quota_files_per_day=row[8], quota_search_per_min=row[9],
        )
        for row in cursor.fetchall()
    ]


@router.patch("/members/{member_id}/role", response_model=MemberResponse)
def update_member_role(
    member_id: int,
    req: MemberRoleUpdate,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Change a member's role (admin only)."""
    if member_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    cursor = db.conn.cursor()
    cursor.execute(
        "UPDATE members SET role = ? WHERE id = ?",
        (req.role, member_id)
    )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Member not found")

    db.conn.commit()
    logger.info(f"Admin {admin['username']} changed member {member_id} role to {req.role}")

    cursor.execute(
        """SELECT id, firebase_uid, email, display_name, role, is_active,
                  joined_at, last_seen_at, quota_files_per_day, quota_search_per_min
           FROM members WHERE id = ?""",
        (member_id,)
    )
    row = cursor.fetchone()
    return MemberResponse(
        id=row[0], firebase_uid=row[1], email=row[2],
        display_name=row[3], role=row[4], is_active=bool(row[5]),
        joined_at=row[6], last_seen_at=row[7],
        quota_files_per_day=row[8], quota_search_per_min=row[9],
    )


@router.delete("/members/{member_id}")
def remove_member(
    member_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Remove a member from the group (admin only). Cannot remove yourself."""
    if member_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot remove yourself")

    cursor = db.conn.cursor()

    # Revoke all refresh tokens for this member
    cursor.execute("DELETE FROM refresh_tokens WHERE user_id = ?", (member_id,))

    cursor.execute("DELETE FROM members WHERE id = ?", (member_id,))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Member not found")

    db.conn.commit()
    logger.info(f"Admin {admin['username']} removed member {member_id}")
    return {"success": True, "removed_member_id": member_id}


@router.patch("/members/{member_id}/deactivate")
def deactivate_member(
    member_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Deactivate a member (admin only). They can't login but data is preserved."""
    if member_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")

    cursor = db.conn.cursor()
    cursor.execute(
        "UPDATE members SET is_active = 0 WHERE id = ?",
        (member_id,)
    )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Member not found")

    # Revoke refresh tokens
    cursor.execute("DELETE FROM refresh_tokens WHERE user_id = ?", (member_id,))

    db.conn.commit()
    logger.info(f"Admin {admin['username']} deactivated member {member_id}")
    return {"success": True}


@router.patch("/members/{member_id}/activate")
def activate_member(
    member_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Re-activate a deactivated member (admin only)."""
    cursor = db.conn.cursor()
    cursor.execute(
        "UPDATE members SET is_active = 1 WHERE id = ?",
        (member_id,)
    )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Member not found")

    db.conn.commit()
    logger.info(f"Admin {admin['username']} activated member {member_id}")
    return {"success": True}


# ── Local Worker Control ─────────────────────────────────────

@router.post("/worker/start")
def start_local_worker(
    admin: dict = Depends(require_admin),
):
    """Start the server machine's local worker process (admin only).

    The worker runs as a separate process and connects over localhost HTTP,
    identical to an external worker. Token minting is handled internally.
    """
    from backend.server.local_worker import start_worker

    result = start_worker()
    if result.get("success"):
        logger.info(f"Admin {admin['username']} started local worker")
    return result


@router.post("/worker/stop")
def stop_local_worker(
    admin: dict = Depends(require_admin),
):
    """Stop the local worker process (admin only)."""
    from backend.server.local_worker import stop_worker

    result = stop_worker()
    logger.info(f"Admin {admin['username']} stopped local worker")
    return result


@router.get("/worker/status")
def get_local_worker_status(
    _user: dict = Depends(get_current_user),
):
    """Get local worker status (any authenticated user)."""
    from backend.server.local_worker import get_status
    return get_status()
