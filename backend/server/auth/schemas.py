"""
Pydantic schemas for authentication requests/responses.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ── Request schemas ──────────────────────────────────────────

class RegisterRequest(BaseModel):
    invite_code: str = Field(..., min_length=4, max_length=32)
    username: str = Field(..., min_length=2, max_length=50)
    email: Optional[str] = Field(None, max_length=254)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str


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


class WorkerTokenCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(30, ge=1, le=365)


class WorkerTokenResponse(BaseModel):
    id: int
    name: str
    token: Optional[str] = None  # Only shown on creation
    created_by: Optional[int] = None
    is_active: bool = True
    expires_at: Optional[str] = None
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None


class WorkerTokenExchange(BaseModel):
    token: str
