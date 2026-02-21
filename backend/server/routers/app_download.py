"""
App download router — serves Electron installer files for web clients.

Public endpoints (no auth required) so users can download from the login page.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/app", tags=["app"])

# Default: look for installers in frontend/dist-electron/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DOWNLOADS_DIR = PROJECT_ROOT / "frontend" / "dist-electron"

# Allowed installer extensions
INSTALLER_EXTS = {".dmg", ".exe", ".msi", ".appimage", ".deb", ".rpm", ".zip", ".pkg"}


def _detect_platform(filename: str) -> str:
    """Detect target platform from filename extension."""
    name = filename.lower()
    if name.endswith(".dmg") or name.endswith(".pkg"):
        return "mac"
    elif name.endswith(".exe") or name.endswith(".msi"):
        return "win"
    elif name.endswith((".appimage", ".deb", ".rpm")):
        return "linux"
    return "unknown"


def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


@router.get("/downloads")
def list_downloads():
    """List available Electron installer files. No auth required."""
    if not DOWNLOADS_DIR.exists():
        return {"files": [], "available": False}

    files = []
    for f in DOWNLOADS_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in INSTALLER_EXTS:
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "size_display": _format_size(f.stat().st_size),
                "platform": _detect_platform(f.name),
            })

    files.sort(key=lambda x: x["name"])
    return {"files": files, "available": len(files) > 0}


@router.get("/downloads/{filename}")
def download_file(filename: str):
    """Download a specific installer file. No auth required."""
    file_path = (DOWNLOADS_DIR / filename).resolve()

    # Security: prevent path traversal
    if not file_path.is_relative_to(DOWNLOADS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if file_path.suffix.lower() not in INSTALLER_EXTS:
        raise HTTPException(status_code=400, detail="Not an installer file")

    return FileResponse(
        str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )
