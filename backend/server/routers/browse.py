"""
Folder browse + WebDAV source management.

Enables the web client (no Electron IPC) to visually pick folders without
typing paths:
  GET    /api/v1/sources                       list registered WebDAV sources (no secrets)
  POST   /api/v1/sources               (admin) register/update a WebDAV source
  POST   /api/v1/sources/test          (admin) test a WebDAV connection
  DELETE /api/v1/sources/{source_id}   (admin) remove a source
  GET    /api/v1/browse/webdav/{id}?path=      list immediate folders+files under a WebDAV path
  GET    /api/v1/browse/local?path=    (admin) list immediate subfolders under a whitelisted root

Security:
  - Local browse is opt-in: only directories under server.browse.allowed_roots
    (or IMAGINE_BROWSE_ROOTS) may be enumerated. Empty whitelist → 403.
  - Path traversal is blocked by realpath + commonpath containment checks.
  - Source secrets are never returned to clients.
"""
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.server.deps import get_current_user, require_admin
from backend.server.config import get_browse_config
from backend.server.queue.download_ahead import (
    register_webdav_source, remove_webdav_source, list_webdav_sources, get_webdav_source,
)

router = APIRouter(tags=["browse"])


def _sanitize(src: dict) -> dict:
    """Strip secrets from a source config before returning to clients."""
    return {
        "id": src.get("id"),
        "url": src.get("url"),
        "username": src.get("username"),
        "verify_ssl": src.get("verify_ssl", True),
    }


class SourceIn(BaseModel):
    id: str
    url: str
    username: str = ""
    password: str = ""
    verify_ssl: bool = True


class SourceTestIn(BaseModel):
    url: str
    username: str = ""
    password: str = ""
    verify_ssl: bool = True
    remote_path: str = "/"


# ── Sources ──────────────────────────────────────────────────
@router.get("/sources")
def list_sources(_user: dict = Depends(get_current_user)):
    return {"sources": [_sanitize(s) for s in list_webdav_sources()]}


@router.post("/sources")
def add_source(body: SourceIn, _admin: dict = Depends(require_admin)):
    if not body.id.strip() or not body.url.strip():
        raise HTTPException(status_code=400, detail="id and url are required")
    register_webdav_source(body.model_dump(), persist=True)
    return {"success": True, "source": _sanitize(body.model_dump())}


@router.post("/sources/test")
def test_source(body: SourceTestIn, _admin: dict = Depends(require_admin)):
    from backend.remote.webdav_client import WebDAVClient
    try:
        client = WebDAVClient(
            base_url=body.url, username=body.username, password=body.password,
            remote_path=body.remote_path or "/", verify_ssl=body.verify_ssl,
        )
        return client.test_connection()
    except Exception as e:
        return {"success": False, "message": str(e), "file_count": 0}


@router.delete("/sources/{source_id}")
def delete_source(source_id: str, _admin: dict = Depends(require_admin)):
    existed = remove_webdav_source(source_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"success": True}


# ── WebDAV browse ────────────────────────────────────────────
@router.get("/browse/webdav/{source_id}")
def browse_webdav(
    source_id: str,
    path: Optional[str] = Query(None),
    _user: dict = Depends(get_current_user),
):
    src = get_webdav_source(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found")
    from backend.remote.webdav_client import WebDAVClient
    client = WebDAVClient(
        base_url=src.get("url", ""), username=src.get("username", ""),
        password=src.get("password", ""), remote_path="/",
        verify_ssl=src.get("verify_ssl", True),
    )
    try:
        result = client.list_dir(path or "/")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"WebDAV browse failed: {e}")
    return {"source_id": source_id, "path": path or "/", **result}


# ── Local browse (whitelisted, admin) ────────────────────────
def _allowed_roots() -> list:
    roots = []
    for r in get_browse_config().get("allowed_roots", []):
        try:
            roots.append(os.path.realpath(r))
        except Exception:
            continue
    return roots


def _within_allowed(target: str, roots: list) -> bool:
    rt = os.path.realpath(target)
    for root in roots:
        try:
            if os.path.commonpath([root, rt]) == root:
                return True
        except ValueError:
            continue  # different drive / relative — not contained
    return False


@router.get("/browse/local")
def browse_local(
    path: Optional[str] = Query(None),
    _admin: dict = Depends(require_admin),
):
    roots = _allowed_roots()
    if not roots:
        raise HTTPException(
            status_code=403,
            detail="Local browse is disabled. Set server.browse.allowed_roots to enable.",
        )
    # No path → list the whitelisted roots themselves as top-level entries.
    if not path:
        return {"path": "", "folders": [
            {"name": os.path.basename(r) or r, "path": r} for r in roots
        ]}
    if not _within_allowed(path, roots):
        raise HTTPException(status_code=403, detail="Path is outside allowed roots")
    target = os.path.realpath(path)
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail="Not a directory")
    folders = []
    try:
        with os.scandir(target) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False) and not entry.name.startswith('.'):
                    folders.append({"name": entry.name, "path": entry.path})
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    folders.sort(key=lambda f: f["name"].lower())
    return {"path": target, "folders": folders}
