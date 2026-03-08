"""
License enforcement — FastAPI dependencies for access control.

Usage in routers:
    @router.post("/upload", dependencies=[Depends(require_writable)])
    @router.post("/admin/members", dependencies=[Depends(check_user_limit)])
"""

import logging
from fastapi import Depends, HTTPException

from backend.server.deps import get_db
from .license_schemas import LicenseStatus

logger = logging.getLogger(__name__)

# Global reference to LicenseManager (set by app.py on startup)
_license_manager = None


def set_license_manager(manager):
    """Called by app.py to set the global license manager."""
    global _license_manager
    _license_manager = manager


def get_license_info():
    """Get current license info (cached, no network call)."""
    if _license_manager is None:
        # No license manager = Electron local mode or not initialized
        from .license_schemas import LicenseInfo, LicenseStatus
        return LicenseInfo(status=LicenseStatus.ACTIVE, is_writable=True, max_users=999, max_files=999999)
    return _license_manager.get_cached()


def require_writable():
    """Dependency: require license to allow write operations (pipeline, upload)."""
    info = get_license_info()
    if not info.is_writable:
        status_msg = {
            LicenseStatus.EXPIRED: "License expired — read-only mode",
            LicenseStatus.SUSPENDED: "License suspended — read-only mode",
        }.get(info.status, "Write operations not available")
        raise HTTPException(status_code=403, detail=status_msg)
    return info


def require_active_license():
    """Dependency: require an active (non-free) license."""
    info = get_license_info()
    if info.status == LicenseStatus.FREE:
        raise HTTPException(status_code=403, detail="Active license required")
    if info.status in (LicenseStatus.EXPIRED, LicenseStatus.SUSPENDED):
        raise HTTPException(status_code=403, detail="License is not active")
    return info


def check_user_limit():
    """Dependency: check if adding another user would exceed max_users."""
    info = get_license_info()
    if info.current_users >= info.max_users:
        raise HTTPException(
            status_code=403,
            detail=f"User limit reached ({info.current_users}/{info.max_users})"
        )
    return info


def check_file_limit():
    """Dependency: check if adding files would exceed max_files."""
    info = get_license_info()
    if info.current_files >= info.max_files:
        raise HTTPException(
            status_code=403,
            detail=f"File limit reached ({info.current_files}/{info.max_files})"
        )
    return info
