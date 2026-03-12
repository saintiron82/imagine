"""
License router — license info, activation, and manual refresh.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_db_safe, require_admin
from backend.server.licensing.license_schemas import (
    LicenseInfoResponse, LicenseActivateRequest,
)
from backend.server.licensing.enforcement import get_license_info, _license_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/license", tags=["license"])


@router.get("/info", response_model=LicenseInfoResponse)
def license_info(db: SQLiteDB = Depends(get_db_safe)):
    """Get current license status. No admin required — all users can see."""
    info = get_license_info()

    # Determine auth mode
    cursor = db.conn.cursor()
    cursor.execute("SELECT value FROM system_meta WHERE key = 'auth_mode'")
    row = cursor.fetchone()
    auth_mode = row[0] if row else "legacy"

    return LicenseInfoResponse(license=info, auth_mode=auth_mode)


@router.post("/activate")
def activate_license(
    req: LicenseActivateRequest,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Activate an offline license key (admin only)."""
    from backend.server.licensing.license_keys import verify_offline_token

    # Verify the key first
    decoded = verify_offline_token(req.license_key)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Invalid license key")

    # Store it
    if _license_manager is None:
        raise HTTPException(status_code=503, detail="License manager not initialized")

    if not _license_manager.save_offline_key(req.license_key):
        raise HTTPException(status_code=500, detail="Failed to save license key")

    logger.info(f"Admin {admin['username']} activated offline license: {decoded.get('plan_id', 'unknown')}")
    return {"success": True, "plan_id": decoded.get("plan_id", "unknown")}


@router.post("/refresh")
async def refresh_license(
    admin: dict = Depends(require_admin),
):
    """Force re-verify license from Firebase (admin only)."""
    if _license_manager is None:
        raise HTTPException(status_code=503, detail="License manager not initialized")

    info = await _license_manager.verify()
    logger.info(f"Admin {admin['username']} refreshed license: {info.plan_id} / {info.status}")
    return {"success": True, "license": info.model_dump()}
