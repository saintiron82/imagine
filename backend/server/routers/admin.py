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
from backend.server.deps import get_db, require_admin, get_current_user
from backend.server.auth.schemas import (
    InviteCodeCreate, InviteCodeResponse,
    UserResponse, UserUpdateRequest,
)
from backend.server.auth.jwt import create_access_token, create_refresh_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Invite Codes ─────────────────────────────────────────────

@router.post("/invite-codes", response_model=InviteCodeResponse)
def create_invite_code(
    req: InviteCodeCreate,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db),
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
    db: SQLiteDB = Depends(get_db),
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
    db: SQLiteDB = Depends(get_db),
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
    db: SQLiteDB = Depends(get_db),
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
    db: SQLiteDB = Depends(get_db),
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


# ── Embedded Worker Control ──────────────────────────────────

@router.post("/worker/start")
def start_embedded_worker(
    request: Request,
    admin: dict = Depends(require_admin),
):
    """Start the embedded worker (admin only).

    Creates a long-lived JWT token for the worker thread and starts it.
    The worker uses HTTP loopback to claim/complete jobs from this server.
    """
    from backend.server.embedded_worker import start_worker

    # Create a long-lived token for the worker (24h)
    worker_token = create_access_token(
        admin["id"], admin["username"], admin["role"],
        expires_minutes=1440,
    )
    worker_refresh = create_refresh_token()

    # Determine server URL (loopback)
    server_url = str(request.base_url).rstrip("/")

    result = start_worker(server_url, worker_token, worker_refresh)
    if result.get("success"):
        logger.info(f"Admin {admin['username']} started embedded worker")
    return result


@router.post("/worker/stop")
def stop_embedded_worker(
    admin: dict = Depends(require_admin),
):
    """Stop the embedded worker (admin only)."""
    from backend.server.embedded_worker import stop_worker

    result = stop_worker()
    logger.info(f"Admin {admin['username']} stopped embedded worker")
    return result


@router.get("/worker/status")
def get_embedded_worker_status(
    _user: dict = Depends(get_current_user),
):
    """Get embedded worker status (any authenticated user)."""
    from backend.server.embedded_worker import get_status
    return get_status()
