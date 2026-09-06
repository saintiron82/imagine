"""
Authentication router — Firebase-based connect, refresh, me, group join.
"""

import logging
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_db_safe, get_current_user
from backend.server.rate_limit import (
    check_login_rate, check_register_rate, check_refresh_rate,
)
from backend.server.auth.schemas import (
    RefreshRequest, TokenResponse, UserResponse,
    FirebaseConnectRequest,
    FirebaseLoginRequest, JoinGroupRequest, MemberResponse,
)
from backend.server.auth.jwt import (
    create_access_token, create_refresh_token,
    hash_refresh_token, get_refresh_token_expiry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _verify_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    import bcrypt
    return bcrypt.checkpw(password.encode(), hashed.encode())


@router.post("/refresh", response_model=TokenResponse,
              dependencies=[Depends(check_refresh_rate)])
def refresh(req: RefreshRequest, db: SQLiteDB = Depends(get_db_safe)):
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


@router.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user), db: SQLiteDB = Depends(get_db_safe)):
    """Get current user info."""
    cursor = db.conn.cursor()

    # Try members table first (group-based Firebase Auth), fall back to users table
    try:
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
    except Exception:
        pass  # members table may not exist (2-layer auth uses users table only)

    # Users table (2-layer auth / legacy)
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
def connect_with_firebase(req: FirebaseConnectRequest, db: SQLiteDB = Depends(get_db_safe)):
    """Connect to server using Firebase ID token + server password.

    2-layer auth:
    1. Firebase Auth verifies identity (ID token)
    2. Server verifies authorization (server_password)

    Auto-creates user on first connection.
    """
    logger.info("[AUTH:connect] Starting Firebase connect...")
    cursor = db.conn.cursor()

    # 1. Verify server password
    cursor.execute("SELECT value FROM system_meta WHERE key = 'server_password_hash'")
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    if not _verify_password(req.server_password, row[0]):
        from backend.server.security import audit_log as _audit
        _audit.record(db, "failed_login", detail="bad server password")
        raise HTTPException(status_code=403, detail="Invalid server password")

    # 2. Verify Firebase ID token
    logger.info("[AUTH:connect] Server password OK, verifying Firebase token...")
    from backend.server.firebase_auth import verify_firebase_token

    try:
        decoded = verify_firebase_token(req.firebase_id_token)
    except Exception as e:
        logger.error(f"[AUTH:connect] Firebase token verification crashed: {type(e).__name__}: {e}")
        raise
    if decoded is None:
        logger.warning("[AUTH:connect] Firebase token verification returned None")
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

    from backend.server.auth import membership as _m

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
        # No auto-signup (IMGV2-14): an uninvited Firebase user is rejected,
        # except first-admin bootstrap when no Firebase users exist yet.
        cursor.execute("SELECT COUNT(*) FROM users WHERE firebase_uid IS NOT NULL")
        no_firebase_users = cursor.fetchone()[0] == 0
        invite = _m.find_pending_invite(db, email) if email else None

        if no_firebase_users:
            role = 'admin'  # bootstrap: first Firebase identity owns the server
        elif invite:
            role = invite[1] if invite[1] in ('admin', 'user') else 'user'
        else:
            from backend.server.security import audit_log as _audit
            _audit.record(db, "failed_login", detail="uninvited firebase user")
            raise HTTPException(status_code=403, detail="초대가 필요합니다 — 관리자에게 문의하세요")

        # Seat cap (max_users, 0 = unlimited). Invited users already hold a
        # reserved seat (pending invite) — only guard the non-invite path.
        if not invite:
            limit = _m.seat_limit(db)
            if limit and _m.count_active_users(db) >= limit:
                raise HTTPException(status_code=403, detail="좌석이 가득 찼습니다")

        # Use display_name as username, ensure uniqueness
        username = display_name
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            username = f"{display_name}_{firebase_uid[:6]}"

        cursor.execute(
            """INSERT INTO users (username, email, password_hash, role, is_active, firebase_uid)
               VALUES (?, ?, '', ?, 1, ?)""",
            (username, email, role, firebase_uid)
        )
        user_id = cursor.lastrowid
        if invite:
            _m.mark_invite_accepted(db, invite[0], user_id)
        logger.info(f"Created user via {'invite' if invite else 'bootstrap'}: {username} (role={role})")

    # Hard expiry block: expired license denies entry to non-admins.
    _m.assert_entry_allowed(db, role)

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
    if role == "admin":
        from backend.server.security import audit_log as _audit
        _audit.record(
            db,
            "admin_login",
            actor_user_id=user_id,
            actor_username=username,
            detail="firebase connect",
        )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


# ── Firebase Auth endpoints ──────────────────────────────────

@router.post("/firebase-login", response_model=TokenResponse,
              dependencies=[Depends(check_login_rate)])
def firebase_login(req: FirebaseLoginRequest, db: SQLiteDB = Depends(get_db_safe)):
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
def join_group(req: JoinGroupRequest, db: SQLiteDB = Depends(get_db_safe)):
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
