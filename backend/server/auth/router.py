"""
Authentication router — register, login, refresh, me.
"""

import logging
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_current_user
from backend.server.rate_limit import (
    check_login_rate, check_register_rate,
    check_refresh_rate, check_worker_token_rate,
)
from backend.server.auth.schemas import (
    RegisterRequest, LoginRequest, RefreshRequest,
    TokenResponse, UserResponse, WorkerTokenExchange,
    FirebaseConnectRequest,
    FirebaseLoginRequest, JoinGroupRequest, MemberResponse,
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


@router.post("/register", response_model=TokenResponse,
              dependencies=[Depends(check_register_rate)])
def register(req: RegisterRequest, db: SQLiteDB = Depends(get_db)):
    """Register a new user with server password verification."""
    cursor = db.conn.cursor()

    # Validate server password
    cursor.execute("SELECT value FROM system_meta WHERE key = 'server_password_hash'")
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    if not _verify_password(req.server_password, row[0]):
        raise HTTPException(status_code=403, detail="Invalid server password")

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


@router.post("/login", response_model=TokenResponse,
              dependencies=[Depends(check_login_rate)])
def login(req: LoginRequest, db: SQLiteDB = Depends(get_db)):
    """Login with server password + username/email + password."""
    import bcrypt
    cursor = db.conn.cursor()

    # Verify server password first
    cursor.execute("SELECT value FROM system_meta WHERE key = 'server_password_hash'")
    sp_row = cursor.fetchone()
    if sp_row is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    if not bcrypt.checkpw(req.server_password.encode(), sp_row[0].encode()):
        raise HTTPException(status_code=403, detail="Invalid server password")

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


@router.post("/refresh", response_model=TokenResponse,
              dependencies=[Depends(check_refresh_rate)])
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

    # Reuse existing refresh token (no rotation) — avoids conflict when
    # browser and embedded worker share the same refresh token.
    access_token = create_access_token(user_id, username, role)

    db.conn.commit()
    return TokenResponse(
        access_token=access_token,
        refresh_token=req.refresh_token,  # Return same refresh token
    )


@router.post("/worker-token", response_model=TokenResponse,
              dependencies=[Depends(check_worker_token_rate)])
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

    # Try members table first (Firebase Auth), fall back to users table (legacy)
    cursor.execute(
        """SELECT id, display_name, email, role, is_active, joined_at,
                  last_seen_at, quota_files_per_day, quota_search_per_min
           FROM members WHERE id = ?""",
        (current_user["id"],)
    )
    row = cursor.fetchone()
    if row:
        return UserResponse(
            id=row[0], username=row[1] or row[2],  # display_name or email
            email=row[2], role=row[3],
            is_active=bool(row[4]), created_at=row[5], last_login_at=row[6],
            quota_files_per_day=row[7], quota_search_per_min=row[8],
        )

    # Legacy users table fallback
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


# ── 2-Layer Auth: Firebase Identity + Server Password ────────

@router.post("/connect", response_model=TokenResponse,
              dependencies=[Depends(check_login_rate)])
def connect_with_firebase(req: FirebaseConnectRequest, db: SQLiteDB = Depends(get_db)):
    """Connect to server using Firebase ID token + server password.

    2-layer auth:
    1. Firebase Auth verifies identity (ID token)
    2. Server verifies authorization (server_password)

    Auto-creates user on first connection.
    """
    cursor = db.conn.cursor()

    # 1. Verify server password
    cursor.execute("SELECT value FROM system_meta WHERE key = 'server_password_hash'")
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    if not _verify_password(req.server_password, row[0]):
        raise HTTPException(status_code=403, detail="Invalid server password")

    # 2. Verify Firebase ID token
    from backend.server.firebase_auth import verify_firebase_token

    decoded = verify_firebase_token(req.firebase_id_token)
    if decoded is None:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    firebase_uid = decoded['uid']
    email = decoded.get('email', '')
    display_name = decoded.get('name', '') or email.split('@')[0]

    # 3. Find or create user by firebase_uid
    cursor.execute(
        "SELECT id, username, role, is_active FROM users WHERE firebase_uid = ?",
        (firebase_uid,)
    )
    user_row = cursor.fetchone()

    if user_row:
        user_id, username, role, is_active = user_row
        if not is_active:
            raise HTTPException(status_code=403, detail="Account is deactivated")
        # Update last login
        cursor.execute(
            "UPDATE users SET last_login_at = datetime('now') WHERE id = ?",
            (user_id,)
        )
    else:
        # Check max_users limit (0 = unlimited)
        cursor.execute("SELECT value FROM system_meta WHERE key = 'max_users'")
        max_row = cursor.fetchone()
        max_users = int(max_row[0]) if max_row else 0

        if max_users > 0:
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            current_count = cursor.fetchone()[0]
            if current_count >= max_users:
                raise HTTPException(status_code=403, detail="Server user limit reached")

        # Auto-create user (first user gets admin if no users exist)
        cursor.execute("SELECT COUNT(*) FROM users")
        is_first = cursor.fetchone()[0] == 0
        role = 'admin' if is_first else 'user'

        # Use display_name as username, ensure uniqueness
        username = display_name
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            # Append short uid suffix for uniqueness
            username = f"{display_name}_{firebase_uid[:6]}"

        cursor.execute(
            """INSERT INTO users (username, email, password_hash, role, is_active, firebase_uid)
               VALUES (?, ?, '', ?, 1, ?)""",
            (username, email, role, firebase_uid)
        )
        user_id = cursor.lastrowid
        logger.info(f"Auto-created user: {username} (firebase_uid={firebase_uid[:8]}..., role={role})")

    # 4. Issue JWT
    access_token = create_access_token(user_id, username, role)
    refresh_token = create_refresh_token()

    cursor.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
           VALUES (?, ?, ?)""",
        (user_id, hash_refresh_token(refresh_token),
         get_refresh_token_expiry().isoformat())
    )

    db.conn.commit()
    logger.info(f"Firebase connect: {username} (uid={firebase_uid[:8]}..., role={role})")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


# ── Firebase Auth endpoints ──────────────────────────────────

@router.post("/firebase-login", response_model=TokenResponse,
              dependencies=[Depends(check_login_rate)])
def firebase_login(req: FirebaseLoginRequest, db: SQLiteDB = Depends(get_db)):
    """Login to group server using Firebase ID Token.

    Flow:
    1. Verify Firebase ID Token → get uid, email
    2. Check members table for membership
    3. If member → issue server JWT (access + refresh)
    4. If not member → 403
    """
    from backend.server.firebase_auth import verify_firebase_token

    decoded = verify_firebase_token(req.id_token)
    if decoded is None:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    uid = decoded['uid']
    email = decoded['email']

    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT id, display_name, email, role, is_active
           FROM members WHERE firebase_uid = ?""",
        (uid,)
    )
    row = cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=403,
            detail="Not a member of this group"
        )

    member_id, display_name, db_email, role, is_active = row

    if not is_active:
        raise HTTPException(status_code=403, detail="Membership is deactivated")

    # Update last_seen and email if changed
    cursor.execute(
        "UPDATE members SET last_seen_at = datetime('now'), email = ? WHERE id = ?",
        (email, member_id)
    )

    # Generate server session tokens
    username = display_name or email
    access_token = create_access_token(member_id, username, role)
    refresh_token = create_refresh_token()

    cursor.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
           VALUES (?, ?, ?)""",
        (member_id, hash_refresh_token(refresh_token),
         get_refresh_token_expiry().isoformat())
    )

    db.conn.commit()
    logger.info(f"Firebase login: {email} (member_id={member_id}, role={role})")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/join", response_model=TokenResponse,
              dependencies=[Depends(check_register_rate)])
def join_group(req: JoinGroupRequest, db: SQLiteDB = Depends(get_db)):
    """Join a group using invite code + Firebase ID Token.

    Flow:
    1. Verify Firebase ID Token → get uid, email, display_name
    2. Validate invite code (active, not expired, uses remaining)
    3. Check not already a member
    4. Register as member (role: user)
    5. Issue server JWT
    """
    from backend.server.firebase_auth import verify_firebase_token

    decoded = verify_firebase_token(req.id_token)
    if decoded is None:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    uid = decoded['uid']
    email = decoded['email']
    display_name = decoded.get('name', '')

    cursor = db.conn.cursor()

    # Validate invite code
    cursor.execute(
        """SELECT id, max_uses, use_count, expires_at, is_active
           FROM invite_codes WHERE code = ?""",
        (req.invite_code,)
    )
    code_row = cursor.fetchone()
    if code_row is None:
        raise HTTPException(status_code=404, detail="Invalid invite code")

    code_id, max_uses, use_count, expires_at, code_active = code_row

    if not code_active:
        raise HTTPException(status_code=410, detail="Invite code is no longer active")

    if use_count >= max_uses:
        raise HTTPException(status_code=410, detail="Invite code has been fully used")

    if expires_at:
        from datetime import datetime, timezone
        exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if exp < datetime.now(timezone.utc):
            raise HTTPException(status_code=410, detail="Invite code has expired")

    # Check if already a member
    cursor.execute("SELECT id FROM members WHERE firebase_uid = ?", (uid,))
    if cursor.fetchone():
        raise HTTPException(status_code=409, detail="Already a member of this group")

    # Register as member
    cursor.execute(
        """INSERT INTO members (firebase_uid, email, display_name, role, is_active)
           VALUES (?, ?, ?, 'user', 1)""",
        (uid, email, display_name)
    )
    member_id = cursor.lastrowid

    # Update invite code usage
    cursor.execute(
        "UPDATE invite_codes SET use_count = use_count + 1 WHERE id = ?",
        (code_id,)
    )

    # Generate server session tokens
    username = display_name or email
    access_token = create_access_token(member_id, username, "user")
    refresh_token = create_refresh_token()

    cursor.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
           VALUES (?, ?, ?)""",
        (member_id, hash_refresh_token(refresh_token),
         get_refresh_token_expiry().isoformat())
    )

    db.conn.commit()
    logger.info(f"New member joined: {email} (member_id={member_id}, invite_code={req.invite_code})")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )
