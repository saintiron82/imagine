"""
FastAPI dependency injection — DB session, auth, etc.
"""

import logging
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.db.sqlite_client import SQLiteDB
from backend.server.auth.jwt import decode_access_token
from backend.server.config import get_db_path

logger = logging.getLogger(__name__)

# ── Shared singleton DB instance ─────────────────────────────

_db_instance: Optional[SQLiteDB] = None

security = HTTPBearer(auto_error=False)

# Localhost auto-admin: virtual user returned for tokenless localhost requests.
_LOCALHOST_ADMIN = {
    "id": 0,
    "username": "local",
    "email": "",
    "role": "admin",
    "is_active": True,
}

_LOCALHOST_HOSTS = frozenset(("127.0.0.1", "::1", "localhost"))


def get_db() -> SQLiteDB:
    """Get shared SQLiteDB instance (singleton)."""
    global _db_instance
    if _db_instance is None:
        _db_instance = SQLiteDB(get_db_path())
    return _db_instance


def close_db():
    """Close shared DB instance (call on shutdown)."""
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
        _db_instance = None


# ── Auth dependencies ────────────────────────────────────────

def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: SQLiteDB = Depends(get_db),
) -> Dict[str, Any]:
    """Extract and validate current user from JWT token.

    Localhost auto-admin: requests from 127.0.0.1/::1 without a Bearer
    token are granted admin access automatically. This covers:
    - Electron embedded server (host is always admin)
    - Local development/testing

    When a token IS provided, it is validated normally regardless of source.
    """
    if credentials is None:
        # Localhost auto-admin: no token + localhost = admin
        client_host = request.client.host if request.client else None
        if client_host in _LOCALHOST_HOSTS:
            return _LOCALHOST_ADMIN
        logger.warning("[AUTH] 401 — No Bearer token in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_preview = credentials.credentials[:20] + "..." if len(credentials.credentials) > 20 else credentials.credentials
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        # Token exists but invalid/expired — try localhost fallback
        client_host = request.client.host if request.client else None
        if client_host in _LOCALHOST_HOSTS:
            return _LOCALHOST_ADMIN
        logger.warning(f"[AUTH] 401 — Token decode failed (token={token_preview})")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_raw = payload.get("sub")
    if user_id_raw is None:
        logger.warning(f"[AUTH] 401 — No 'sub' in payload (token={token_preview})")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    try:
        user_id = int(user_id_raw)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    # Verify user still exists and is active
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT id, username, email, role, is_active FROM users WHERE id = ?",
        (user_id,)
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not row[4]:  # is_active
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return {
        "id": row[0],
        "username": row[1],
        "email": row[2],
        "role": row[3],
        "is_active": bool(row[4]),
    }


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require admin role."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
