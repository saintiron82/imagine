"""
Authentication router — register, login, refresh, me.
"""

import logging
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_current_user
from backend.server.auth.schemas import (
    RegisterRequest, LoginRequest, RefreshRequest,
    TokenResponse, UserResponse, WorkerTokenExchange,
)
from backend.server.auth.jwt import (
    create_access_token, create_refresh_token,
    hash_refresh_token, get_refresh_token_expiry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    import bcrypt
    return bcrypt.checkpw(password.encode(), hashed.encode())


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: SQLiteDB = Depends(get_db)):
    """Register a new user with an invite code."""
    cursor = db.conn.cursor()

    # Validate invite code
    cursor.execute(
        "SELECT id, max_uses, use_count, expires_at FROM invite_codes WHERE code = ?",
        (req.invite_code,)
    )
    invite = cursor.fetchone()
    if invite is None:
        raise HTTPException(status_code=400, detail="Invalid invite code")

    invite_id, max_uses, use_count, expires_at = invite
    if use_count >= max_uses:
        raise HTTPException(status_code=400, detail="Invite code has been fully used")

    if expires_at:
        exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if exp < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Invite code has expired")

    # Check username uniqueness
    cursor.execute("SELECT id FROM users WHERE username = ?", (req.username,))
    if cursor.fetchone():
        raise HTTPException(status_code=409, detail="Username already taken")

    # Check email uniqueness (only if provided)
    if req.email:
        cursor.execute("SELECT id FROM users WHERE email = ?", (req.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="Email already registered")

    # Create user
    password_hash = _hash_password(req.password)
    cursor.execute(
        """INSERT INTO users (username, email, password_hash, role, is_active)
           VALUES (?, ?, ?, 'user', 1)""",
        (req.username, req.email, password_hash)
    )
    user_id = cursor.lastrowid

    # Update invite usage
    cursor.execute(
        "UPDATE invite_codes SET use_count = use_count + 1 WHERE id = ?",
        (invite_id,)
    )
    cursor.execute(
        "INSERT INTO invite_uses (invite_id, user_id) VALUES (?, ?)",
        (invite_id, user_id)
    )

    # Generate tokens
    access_token = create_access_token(user_id, req.username, "user")
    refresh_token = create_refresh_token()

    # Store refresh token
    cursor.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
           VALUES (?, ?, ?)""",
        (user_id, hash_refresh_token(refresh_token),
         get_refresh_token_expiry().isoformat())
    )

    db.conn.commit()
    logger.info(f"New user registered: {req.username} (ID: {user_id})")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: SQLiteDB = Depends(get_db)):
    """Login with username or email and password."""
    cursor = db.conn.cursor()

    # Try username first, then email
    identifier = req.username or req.email
    if not identifier:
        raise HTTPException(status_code=400, detail="Username or email required")

    cursor.execute(
        "SELECT id, username, email, password_hash, role, is_active FROM users WHERE username = ? OR email = ?",
        (identifier, identifier)
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id, username, email, password_hash, role, is_active = row

    if not is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    if not _verify_password(req.password, password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Update last login
    cursor.execute(
        "UPDATE users SET last_login_at = datetime('now') WHERE id = ?",
        (user_id,)
    )

    # Generate tokens
    access_token = create_access_token(user_id, username, role)
    refresh_token = create_refresh_token()

    # Store refresh token
    cursor.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
           VALUES (?, ?, ?)""",
        (user_id, hash_refresh_token(refresh_token),
         get_refresh_token_expiry().isoformat())
    )

    db.conn.commit()
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest, db: SQLiteDB = Depends(get_db)):
    """Refresh access token using a valid refresh token."""
    cursor = db.conn.cursor()
    token_hash = hash_refresh_token(req.refresh_token)
    token_preview = req.refresh_token[:16] + "..." if len(req.refresh_token) > 16 else req.refresh_token

    cursor.execute(
        """SELECT rt.id, rt.user_id, rt.expires_at, rt.revoked,
                  u.username, u.role, u.is_active
           FROM refresh_tokens rt
           JOIN users u ON u.id = rt.user_id
           WHERE rt.token_hash = ?""",
        (token_hash,)
    )
    row = cursor.fetchone()
    if row is None:
        logger.warning(f"[REFRESH] 401 — Token not found in DB (token={token_preview})")
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    rt_id, user_id, expires_at, revoked, username, role, is_active = row

    if revoked:
        logger.warning(f"[REFRESH] 401 — Token REVOKED (user={username}, rt_id={rt_id}, token={token_preview})")
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")

    if not is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    if exp < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token has expired")

    # Revoke old refresh token (rotation)
    cursor.execute(
        "UPDATE refresh_tokens SET revoked = 1 WHERE id = ?",
        (rt_id,)
    )

    # Issue new tokens
    access_token = create_access_token(user_id, username, role)
    new_refresh_token = create_refresh_token()

    cursor.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
           VALUES (?, ?, ?)""",
        (user_id, hash_refresh_token(new_refresh_token),
         get_refresh_token_expiry().isoformat())
    )

    db.conn.commit()
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )


@router.post("/worker-token", response_model=TokenResponse)
def exchange_worker_token(req: WorkerTokenExchange, db: SQLiteDB = Depends(get_db)):
    """Exchange a worker token secret for JWT access/refresh tokens."""
    import hashlib

    token_hash = hashlib.sha256(req.token.encode()).hexdigest()

    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT wt.id, wt.is_active, wt.expires_at, wt.created_by,
                  u.id, u.username, u.role, u.is_active
           FROM worker_tokens wt
           JOIN users u ON u.id = wt.created_by
           WHERE wt.token_hash = ?""",
        (token_hash,)
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid worker token")

    wt_id, wt_active, wt_expires, created_by, user_id, username, role, user_active = row

    if not wt_active:
        raise HTTPException(status_code=401, detail="Worker token has been revoked")

    if not user_active:
        raise HTTPException(status_code=403, detail="Token owner account is deactivated")

    if wt_expires:
        from datetime import datetime, timezone
        exp = datetime.fromisoformat(wt_expires.replace("Z", "+00:00"))
        if exp < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Worker token has expired")

    # Update last_used_at
    cursor.execute(
        "UPDATE worker_tokens SET last_used_at = datetime('now') WHERE id = ?",
        (wt_id,)
    )

    # Generate JWT tokens (1 hour access, standard refresh)
    access_token = create_access_token(user_id, username, role, expires_minutes=60)
    refresh_token = create_refresh_token()

    cursor.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
           VALUES (?, ?, ?)""",
        (user_id, hash_refresh_token(refresh_token),
         get_refresh_token_expiry().isoformat())
    )

    db.conn.commit()
    logger.info(f"Worker token exchanged for user {username} (token ID: {wt_id})")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user), db: SQLiteDB = Depends(get_db)):
    """Get current user info."""
    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT id, username, email, role, is_active, created_at,
                  last_login_at, quota_files_per_day, quota_search_per_min
           FROM users WHERE id = ?""",
        (current_user["id"],)
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        id=row[0], username=row[1], email=row[2], role=row[3],
        is_active=bool(row[4]), created_at=row[5], last_login_at=row[6],
        quota_files_per_day=row[7], quota_search_per_min=row[8],
    )
