"""
Thumbnail path resolver — shared logic for api_search.py and server/routers/files.py.

Resolves thumbnail paths from DB values or infers from file_name on disk.
"""

from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def resolve_thumbnail(
    thumb_path: Optional[str],
    file_name: Optional[str],
    project_root: Path = _PROJECT_ROOT,
) -> Optional[Path]:
    """Resolve thumbnail to an existing Path on disk.

    Strategy:
      1) DB thumbnail_url — absolute path check
      2) DB thumbnail_url — relative path resolved from project_root
      3) Infer from file_name: output/thumbnails/{stem}_thumb.png

    Args:
        thumb_path: DB thumbnail_url value (may be absolute, relative, or empty).
        file_name: Original file name (e.g. "hero.psd") for stem-based inference.
        project_root: Project root directory for resolving relative paths.

    Returns:
        Resolved Path if found on disk, else None.
    """
    # 1) Try DB value (absolute)
    if thumb_path:
        p = Path(thumb_path)
        if p.is_absolute() and p.exists():
            return p
        # 2) Relative → resolve from project root
        resolved = project_root / thumb_path
        if resolved.exists():
            return resolved

    # 3) Infer from file_name
    if file_name:
        stem = Path(file_name).stem
        inferred = project_root / "output" / "thumbnails" / f"{stem}_thumb.png"
        if inferred.exists():
            return inferred

    return None


def resolve_thumbnail_str(result: dict, project_root: Path = _PROJECT_ROOT) -> str:
    """Compatibility wrapper for api_search.py — returns str path or empty string.

    Args:
        result: Search result dict with 'thumbnail_url' and 'file_name' keys.
        project_root: Project root directory.

    Returns:
        Resolved thumbnail path as string, or empty string if not found.
    """
    thumb = result.get("thumbnail_url") or ""
    file_name = result.get("file_name", "")
    resolved = resolve_thumbnail(thumb, file_name, project_root)
    return str(resolved) if resolved else ""
