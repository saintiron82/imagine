"""Shared helpers for AssetMeta conversion."""

from pathlib import Path


def meta_to_dict(meta) -> dict:
    """Convert AssetMeta dataclass to dict for storage/API.

    Removes None values and converts non-serializable types (Path) to str.
    """
    from dataclasses import asdict
    try:
        d = asdict(meta)
        clean = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, dict)):
                clean[k] = v
            elif isinstance(v, Path):
                clean[k] = str(v)
        return clean
    except Exception:
        return {
            "file_name": getattr(meta, "file_name", ""),
            "file_path": getattr(meta, "file_path", ""),
            "format": getattr(meta, "format", ""),
            "file_size": getattr(meta, "file_size", 0),
        }
