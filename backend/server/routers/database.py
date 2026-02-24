"""Database admin operations — reset with password verification."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.server.deps import require_admin, get_db

router = APIRouter(prefix="/admin/database", tags=["database"])


class ResetRequest(BaseModel):
    password: str


@router.post("/reset")
def reset_database(req: ResetRequest, admin: dict = Depends(require_admin)):
    """Reset all file data. Requires admin password re-verification."""
    from backend.server.auth.router import _verify_password

    db = get_db()
    cursor = db.conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE id = ?", (admin["id"],))
    row = cursor.fetchone()
    if not row or not _verify_password(req.password, row[0]):
        raise HTTPException(status_code=403, detail="Invalid password")

    result = db.reset_file_data()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Reset failed"))

    return result
