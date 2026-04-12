"""
Pydantic schemas for authentication requests/responses.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ── Request schemas ──────────────────────────────────────────

class ServerInitRequest(BaseModel):
    group_name: str = Field(..., min_length=1, max_length=100)
    server_password: str = Field(..., min_length=1, max_length=128)
    admin_username: str = Field(..., min_length=2, max_length=50)
    admin_password: str = Field(..., min_length=4, max_length=128)
    # Optional: Firebase UID for 2-layer auth (server creator = admin)
    firebase_uid: Optional[str] = None
    firebase_email: Optional[str] = None


class RefreshRequest(BaseModel):
    refresh_token: str


# ── Response schemas ─────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[str] = None
    last_login_at: Optional[str] = None
    quota_files_per_day: int = 1000
    quota_search_per_min: int = 60


class InviteCodeCreate(BaseModel):
    max_uses: int = Field(1, ge=1, le=100)
    expires_in_days: Optional[int] = Field(7, ge=1, le=365)


class InviteCodeResponse(BaseModel):
    id: int
    code: str
    created_by: Optional[int] = None
    max_uses: int
    use_count: int
    expires_at: Optional[str] = None
    created_at: Optional[str] = None


class UserUpdateRequest(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None
    quota_files_per_day: Optional[int] = None
    quota_search_per_min: Optional[int] = None


class ResetGroupRequest(BaseModel):
    server_password: str = Field(..., min_length=1, max_length=128)


class FirebaseConnectRequest(BaseModel):
    """Connect to server using Firebase ID token + server password."""
    firebase_id_token: str = Field(..., min_length=1)
    server_password: str = Field(..., min_length=1, max_length=128)


# ── Firebase Auth schemas ────────────────────────────────────

class FirebaseLoginRequest(BaseModel):
    """Login to group server using Firebase ID Token."""
    id_token: str = Field(..., min_length=1)


class JoinGroupRequest(BaseModel):
    """Join a group using invite code + Firebase ID Token."""
    id_token: str = Field(..., min_length=1)
    invite_code: str = Field(..., min_length=1, max_length=20)


class FirebaseServerInitRequest(BaseModel):
    """Initialize server with Firebase Auth (replaces ServerInitRequest)."""
    group_name: str = Field(..., min_length=1, max_length=100)
    id_token: str = Field(..., min_length=1)


class MemberResponse(BaseModel):
    """Response for a group member."""
    id: int
    firebase_uid: str
    email: str
    display_name: Optional[str] = None
    role: str
    is_active: bool
    joined_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    quota_files_per_day: int = 1000
    quota_search_per_min: int = 60


class MemberRoleUpdate(BaseModel):
    """Update a member's role."""
    role: str = Field(..., pattern=r'^(admin|user)$')
