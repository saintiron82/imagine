"""
Server initialization and info endpoints.

POST /server/init — First-time server setup (group name, password, admin account)
GET  /server/info — Public server info (no auth required)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db
from backend.server.auth.schemas import ServerInitRequest, TokenResponse
from backend.server.auth.jwt import (
    create_access_token, create_refresh_token,
    hash_refresh_token, get_refresh_token_expiry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/server", tags=["server"])


def _hash_password(password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


@router.post("/init", response_model=TokenResponse)
def init_server(req: ServerInitRequest, db: SQLiteDB = Depends(get_db)):
    """Initialize server with group name, server password, and admin account.
    Can only be called once (before any admin exists)."""
    try:
        cursor = db.conn.cursor()

        # Check if already initialized
        cursor.execute("SELECT value FROM system_meta WHERE key = 'group_name'")
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="Server already initialized")

        # Clean up residual data from partial init attempts
        # Order matters: child tables first due to FK constraints
        cursor.execute("DELETE FROM worker_sessions")
        cursor.execute("DELETE FROM worker_tokens")
        cursor.execute("DELETE FROM invite_uses")
        cursor.execute("DELETE FROM invite_codes")
        cursor.execute("DELETE FROM refresh_tokens")
        cursor.execute("DELETE FROM users")

        # Store group config
        password_hash = _hash_password(req.server_password)

        cursor.execute(
            "INSERT OR REPLACE INTO system_meta (key, value) VALUES (?, ?)",
            ("group_name", req.group_name)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO system_meta (key, value) VALUES (?, ?)",
            ("server_password_hash", password_hash)
        )

        # Create admin user
        admin_hash = _hash_password(req.admin_password)
        cursor.execute(
            """INSERT INTO users (username, password_hash, role, is_active, email)
               VALUES (?, ?, 'admin', 1, '')""",
            (req.admin_username, admin_hash)
        )
        user_id = cursor.lastrowid

        # Generate tokens for auto-login
        access_token = create_access_token(user_id, req.admin_username, "admin")
        refresh_token = create_refresh_token()

        cursor.execute(
            """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
               VALUES (?, ?, ?)""",
            (user_id, hash_refresh_token(refresh_token),
             get_refresh_token_expiry().isoformat())
        )

        db.conn.commit()
        logger.info(f"Server initialized: group='{req.group_name}', admin='{req.admin_username}'")

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Server init failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
def server_info(db: SQLiteDB = Depends(get_db)):
    """Public server info — no authentication required."""
    cursor = db.conn.cursor()

    cursor.execute("SELECT value FROM system_meta WHERE key = 'group_name'")
    row = cursor.fetchone()
    group_name = row[0] if row else None

    return {
        "group_name": group_name,
        "initialized": group_name is not None,
        "version": "0.6.3",
    }
