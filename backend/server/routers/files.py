"""
Files router — file metadata CRUD, thumbnail serving.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_current_user, require_admin
from backend.server.auth.jwt import decode_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])

_security = HTTPBearer(auto_error=False)


def _get_user_or_query_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
    db: SQLiteDB = Depends(get_db),
) -> dict:
    """Auth via header OR ?token= query param (for img src tags)."""
    # 1) Try Authorization header
    if credentials:
        return get_current_user(credentials, db)

    # 2) Try ?token= query param
    token = request.query_params.get("token")
    if token:
        payload = decode_access_token(token)
        if payload:
            user_id_raw = payload.get("sub")
            if user_id_raw is not None:
                return {"id": int(user_id_raw), "role": payload.get("role", "user")}

    raise HTTPException(status_code=401, detail="Not authenticated")


# ── Schemas ──────────────────────────────────────────────────

class UserMetaUpdate(BaseModel):
    user_note: Optional[str] = None
    user_tags: Optional[List[str]] = None
    user_category: Optional[str] = None
    user_rating: Optional[int] = None


class FileResponse_(BaseModel):
    """File metadata response (subset of DB fields)."""
    id: int
    file_path: str
    file_name: str
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mc_caption: Optional[str] = None
    ai_tags: Optional[list] = None
    image_type: Optional[str] = None
    art_style: Optional[str] = None
    user_note: Optional[str] = None
    user_tags: Optional[list] = None
    user_category: Optional[str] = None
    user_rating: Optional[int] = None
    thumbnail_url: Optional[str] = None
    storage_root: Optional[str] = None
    relative_path: Optional[str] = None
    mode_tier: Optional[str] = None
    parsed_at: Optional[str] = None


def _row_to_file_response(row) -> dict:
    """Convert sqlite3.Row to FileResponse dict."""
    d = dict(row)
    # Parse JSON text fields
    for key in ("ai_tags", "user_tags", "folder_tags"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                d[key] = []
    return d


# ── Endpoints ────────────────────────────────────────────────

@router.get("")
def list_files(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    format: Optional[str] = None,
    image_type: Optional[str] = None,
    storage_root: Optional[str] = None,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """List files with pagination and optional filters."""
    cursor = db.conn.cursor()

    conditions = []
    params = []
    if format:
        conditions.append("format = ?")
        params.append(format)
    if image_type:
        conditions.append("image_type = ?")
        params.append(image_type)
    if storage_root:
        conditions.append("storage_root = ?")
        params.append(storage_root)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Total count
    cursor.execute(f"SELECT COUNT(*) FROM files {where}", params)
    total = cursor.fetchone()[0]

    # Fetch page
    cursor.execute(
        f"""SELECT id, file_path, file_name, format, width, height,
                   mc_caption, ai_tags, image_type, art_style,
                   user_note, user_tags, user_category, user_rating,
                   thumbnail_url, storage_root, relative_path, mode_tier, parsed_at
            FROM files {where}
            ORDER BY parsed_at DESC
            LIMIT ? OFFSET ?""",
        params + [limit, offset]
    )
    files = [_row_to_file_response(row) for row in cursor.fetchall()]

    return {"success": True, "total": total, "files": files}


@router.get("/{file_id}")
def get_file(
    file_id: int,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Get file details by ID."""
    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT id, file_path, file_name, file_size, format, width, height,
                  mc_caption, ai_tags, ocr_text, dominant_color, ai_style,
                  image_type, art_style, color_palette, scene_type,
                  user_note, user_tags, user_category, user_rating,
                  thumbnail_url, storage_root, relative_path,
                  folder_path, folder_depth, folder_tags,
                  mode_tier, embedding_model, caption_model,
                  created_at, modified_at, parsed_at, content_hash
           FROM files WHERE id = ?""",
        (file_id,)
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")

    return {"success": True, "file": _row_to_file_response(row)}


@router.patch("/{file_id}/user-meta")
def update_user_meta(
    file_id: int,
    req: UserMetaUpdate,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Update user metadata for a file."""
    cursor = db.conn.cursor()
    cursor.execute("SELECT file_path FROM files WHERE id = ?", (file_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = row[0]
    success = db.update_user_metadata(
        file_path=file_path,
        user_note=req.user_note,
        user_tags=req.user_tags,
        user_category=req.user_category,
        user_rating=req.user_rating,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update metadata")

    return {"success": True}


@router.delete("/{file_id}")
def delete_file(
    file_id: int,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Delete a file (admin or uploader only)."""
    cursor = db.conn.cursor()
    cursor.execute("SELECT uploaded_by FROM files WHERE id = ?", (file_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")

    # Allow admin or file owner
    if user["role"] != "admin" and row[0] != user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to delete this file")

    success = db.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete file")

    return {"success": True}


@router.get("/{file_id}/thumbnail")
def get_thumbnail(
    file_id: int,
    token: Optional[str] = Query(None, description="JWT token for img tag auth"),
    _user: dict = Depends(_get_user_or_query_token),
    db: SQLiteDB = Depends(get_db),
):
    """Serve thumbnail image for a file.

    Supports both Authorization header and ?token= query param,
    because <img src> tags cannot send headers.
    """
    cursor = db.conn.cursor()
    cursor.execute("SELECT thumbnail_url, file_name FROM files WHERE id = ?", (file_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")

    thumb_path = row[0]
    file_name = row[1]

    # Resolve thumbnail path
    resolved = _resolve_thumbnail(thumb_path, file_name)
    if not resolved:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(str(resolved), media_type="image/png")


# Project root for resolving relative paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _resolve_thumbnail(thumb_path: Optional[str], file_name: Optional[str]) -> Optional[Path]:
    """Resolve thumbnail path: try DB value, then infer from file name."""
    # 1) Try DB thumbnail_url (may be relative or absolute)
    if thumb_path:
        p = Path(thumb_path)
        if p.is_absolute() and p.exists():
            return p
        # Relative → resolve from project root
        resolved = _PROJECT_ROOT / thumb_path
        if resolved.exists():
            return resolved

    # 2) Infer from file_name: {stem}_thumb.png in output/thumbnails/
    if file_name:
        stem = Path(file_name).stem
        inferred = _PROJECT_ROOT / "output" / "thumbnails" / f"{stem}_thumb.png"
        if inferred.exists():
            return inferred

    return None


@router.get("/{file_id}/download")
def download_original(
    file_id: int,
    token: Optional[str] = Query(None, description="JWT token for download auth"),
    _user: dict = Depends(_get_user_or_query_token),
    db: SQLiteDB = Depends(get_db),
):
    """Download original image file.

    Supports both Authorization header and ?token= query param.
    Returns the original file as an attachment for browser download.
    """
    cursor = db.conn.cursor()
    cursor.execute("SELECT file_path, file_name FROM files WHERE id = ?", (file_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = row[0]
    file_name = row[1]

    # Resolve file path
    p = Path(file_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / file_path
    if not p.exists():
        raise HTTPException(status_code=404, detail="Original file not found on disk")

    # Determine media type
    suffix = p.suffix.lower()
    media_types = {
        '.psd': 'application/octet-stream',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }
    media_type = media_types.get(suffix, 'application/octet-stream')

    return FileResponse(
        str(p),
        media_type=media_type,
        filename=file_name or p.name,
    )
