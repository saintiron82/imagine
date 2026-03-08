"""
License data models — Pydantic schemas for license info and status.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime


class LicenseStatus(str, Enum):
    ACTIVE = "active"
    GRACE_PERIOD = "grace_period"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    FREE = "free"


class LicenseInfo(BaseModel):
    """Current license state for a group server."""
    group_name: str = ""
    plan_id: str = "free"
    status: LicenseStatus = LicenseStatus.FREE
    max_users: int = 1
    max_files: int = 1000
    current_users: int = 0
    current_files: int = 0
    features: List[str] = []
    expires_at: Optional[str] = None
    days_remaining: Optional[int] = None
    is_writable: bool = True
    message: str = ""


class LicenseActivateRequest(BaseModel):
    """Request to activate an offline license key."""
    license_key: str


class LicenseInfoResponse(BaseModel):
    """Response for GET /license/info."""
    license: LicenseInfo
    auth_mode: str = "firebase"  # "firebase" or "legacy"
