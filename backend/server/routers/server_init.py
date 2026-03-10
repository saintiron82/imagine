"""
Server initialization and info endpoints.

POST /server/init        — First-time server setup (group name, password, admin account)
POST /server/reset-group — Reset group (auth data) while preserving file data
GET  /server/info        — Public server info (no auth required)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db
from backend.server.auth.schemas import (
    ServerInitRequest, ResetGroupRequest, TokenResponse,
    FirebaseServerInitRequest,
)
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

        # Full reset: new group starts with clean slate
        # Data tables (FK order: children first)
        cursor.execute("DELETE FROM vec_files")
        cursor.execute("DELETE FROM vec_text")
        cursor.execute("DELETE FROM files_fts")
        cursor.execute("DELETE FROM job_queue")
        cursor.execute("DELETE FROM layers")
        cursor.execute("DELETE FROM files")
        # Auth tables
        cursor.execute("DELETE FROM worker_sessions")
        cursor.execute("DELETE FROM worker_tokens")
        cursor.execute("DELETE FROM invite_uses")
        cursor.execute("DELETE FROM invite_codes")
        cursor.execute("DELETE FROM refresh_tokens")
        cursor.execute("DELETE FROM users")
        # Reset system meta
        cursor.execute("DELETE FROM system_meta")

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

        # Create admin user (with Firebase UID if provided)
        admin_hash = _hash_password(req.admin_password)
        firebase_uid = getattr(req, 'firebase_uid', None) or None
        firebase_email = getattr(req, 'firebase_email', None) or ''
        cursor.execute(
            """INSERT INTO users (username, password_hash, role, is_active, email, firebase_uid)
               VALUES (?, ?, 'admin', 1, ?, ?)""",
            (req.admin_username, admin_hash, firebase_email, firebase_uid)
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

        # Best-effort: register group to Firebase for name-based discovery
        try:
            from backend.server.firebase_registry import register_group
            import threading
            port = 8000  # default; could be read from config
            threading.Thread(
                target=register_group,
                args=(req.group_name, port),
                daemon=True,
            ).start()
        except Exception as e:
            logger.debug(f"Firebase registration skipped: {e}")

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Server init failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/firebase-init", response_model=TokenResponse)
def firebase_init_server(req: FirebaseServerInitRequest, db: SQLiteDB = Depends(get_db)):
    """Initialize server with Firebase Auth (no server password needed).

    The first user (from Firebase ID Token) becomes the admin.
    """
    from backend.server.firebase_auth import verify_firebase_token

    decoded = verify_firebase_token(req.id_token)
    if decoded is None:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    uid = decoded['uid']
    email = decoded['email']
    display_name = decoded.get('name', '') or email.split('@')[0]

    try:
        cursor = db.conn.cursor()

        # Check if already initialized
        cursor.execute("SELECT value FROM system_meta WHERE key = 'group_name'")
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="Server already initialized")

        # Full reset: new group starts with clean slate
        cursor.execute("DELETE FROM vec_files")
        cursor.execute("DELETE FROM vec_text")
        cursor.execute("DELETE FROM files_fts")
        cursor.execute("DELETE FROM job_queue")
        cursor.execute("DELETE FROM layers")
        cursor.execute("DELETE FROM files")
        cursor.execute("DELETE FROM worker_sessions")
        cursor.execute("DELETE FROM worker_tokens")
        if db._table_exists('invite_uses'):
            cursor.execute("DELETE FROM invite_uses")
        cursor.execute("DELETE FROM invite_codes")
        cursor.execute("DELETE FROM refresh_tokens")
        cursor.execute("DELETE FROM users")
        if db._table_exists('members'):
            cursor.execute("DELETE FROM members")
        cursor.execute("DELETE FROM system_meta")

        # Store group name
        cursor.execute(
            "INSERT OR REPLACE INTO system_meta (key, value) VALUES (?, ?)",
            ("group_name", req.group_name)
        )
        # Mark as Firebase Auth mode
        cursor.execute(
            "INSERT OR REPLACE INTO system_meta (key, value) VALUES (?, ?)",
            ("auth_mode", "firebase")
        )

        # Create admin member
        cursor.execute(
            """INSERT INTO members (firebase_uid, email, display_name, role, is_active)
               VALUES (?, ?, ?, 'admin', 1)""",
            (uid, email, display_name)
        )
        member_id = cursor.lastrowid

        # Generate tokens for auto-login
        access_token = create_access_token(member_id, display_name, "admin")
        refresh_token = create_refresh_token()

        cursor.execute(
            """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
               VALUES (?, ?, ?)""",
            (member_id, hash_refresh_token(refresh_token),
             get_refresh_token_expiry().isoformat())
        )

        db.conn.commit()
        logger.info(f"Server initialized (Firebase): group='{req.group_name}', admin='{email}'")

        # Best-effort: register group to Firebase RTDB
        try:
            from backend.server.firebase_registry import register_group
            import threading
            port = 8000
            threading.Thread(
                target=register_group,
                args=(req.group_name, port),
                daemon=True,
            ).start()
        except Exception as e:
            logger.debug(f"Firebase registration skipped: {e}")

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Firebase server init failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-group")
def reset_group(req: ResetGroupRequest, db: SQLiteDB = Depends(get_db)):
    """Reset group and auth data. File data is preserved.
    Requires server password to prevent unauthorized resets."""
    try:
        import bcrypt

        cursor = db.conn.cursor()

        # Check if there's a group to reset
        cursor.execute("SELECT value FROM system_meta WHERE key = 'group_name'")
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No group to reset")

        group_name = row[0]

        # Verify server password
        cursor.execute("SELECT value FROM system_meta WHERE key = 'server_password_hash'")
        pw_row = cursor.fetchone()
        if not pw_row or not bcrypt.checkpw(req.server_password.encode(), pw_row[0].encode()):
            raise HTTPException(status_code=403, detail="Invalid server password")

        # Clear auth tables
        cursor.execute("DELETE FROM worker_sessions")
        cursor.execute("DELETE FROM worker_tokens")
        if db._table_exists('invite_uses'):
            cursor.execute("DELETE FROM invite_uses")
        cursor.execute("DELETE FROM invite_codes")
        cursor.execute("DELETE FROM refresh_tokens")
        cursor.execute("DELETE FROM users")

        # Clear group info from system_meta
        cursor.execute("DELETE FROM system_meta WHERE key IN ('group_name', 'server_password_hash')")

        db.conn.commit()
        logger.info(f"Group reset: '{group_name}' removed (file data preserved)")

        return {"success": True, "group_name": group_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Group reset failed: {e}", exc_info=True)
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
        "version": "0.6.4",
    }
