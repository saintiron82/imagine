"""Shared helpers for AssetMeta conversion."""

from pathlib import Path


def meta_to_dict(meta) -> dict:
    """Convert AssetMeta (Pydantic BaseModel or dataclass) to dict for storage/API.

    Removes None values and converts non-serializable types (Path) to str.
    """
    # Pydantic BaseModel → model_dump(), dataclass → asdict()
    try:
        if hasattr(meta, "model_dump"):
            d = meta.model_dump()
        else:
            from dataclasses import asdict
            d = asdict(meta)
    except Exception:
        return {
            "file_name": getattr(meta, "file_name", ""),
            "file_path": getattr(meta, "file_path", ""),
            "format": getattr(meta, "format", ""),
            "file_size": getattr(meta, "file_size", 0),
        }

    clean = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, dict, tuple)):
            clean[k] = v
        elif isinstance(v, Path):
            clean[k] = str(v)
    return clean
