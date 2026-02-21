"""
Sync router — folder ↔ DB reconciliation for server mode.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.server.deps import get_current_user, require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sync", tags=["sync"])


class SyncScanRequest(BaseModel):
    folder_path: str


class MoveItem(BaseModel):
    id: int
    new_path: str


class ApplyMovesRequest(BaseModel):
    moves: List[MoveItem]


class DeleteMissingRequest(BaseModel):
    file_ids: List[int]


@router.post("/scan")
def sync_scan(
    req: SyncScanRequest,
    _user: dict = Depends(require_admin),
):
    """Scan a folder and compare disk state with DB records."""
    from backend.api_sync import sync_folder
    result = sync_folder(req.folder_path)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Scan failed"))
    return result


@router.post("/apply-moves")
def sync_apply_moves(
    req: ApplyMovesRequest,
    _user: dict = Depends(require_admin),
):
    """Apply path updates for moved files."""
    from backend.api_sync import apply_moves
    moves = [{"id": m.id, "new_path": m.new_path} for m in req.moves]
    result = apply_moves(moves)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Apply failed"))
    return result


@router.post("/delete-missing")
def sync_delete_missing(
    req: DeleteMissingRequest,
    _user: dict = Depends(require_admin),
):
    """Delete DB records for files no longer on disk."""
    from backend.api_sync import delete_missing
    result = delete_missing(req.file_ids)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Delete failed"))
    return result
