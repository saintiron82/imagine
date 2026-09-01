"""Shared test helpers.

Embedding widths come from ai_mode tier config, and config.yaml is gitignored —
so they are 1152/1024 on a configured dev machine and fall back to 768/1024
wherever that config is absent (CI, a fresh clone). Tests that hardcoded 1152
passed locally and failed in CI with:

    sqlite3.OperationalError: Dimension mismatch for inserted vector for the
    "embedding" column. Expected 768 dimensions but received 1152.

Use these helpers instead of a literal width.
"""

import re
import struct

import pytest


def active_dims() -> tuple[int, int]:
    """(visual_dim, text_dim) as the schema builder resolves them.

    Mirrors backend/db/sqlite_client.py's tier lookup, including its fallback,
    so a test blob always matches the table that will be created.
    """
    try:
        from backend.utils.tier_config import get_active_tier
        _, tier_config = get_active_tier()
        return (tier_config.get("visual", {}).get("dimensions", 768),
                tier_config.get("text_embed", {}).get("dimensions", 1024))
    except Exception:
        return 768, 1024


def vv_blob(value: float = 0.5) -> bytes:
    """A dummy vec_files (visual) embedding sized for this environment."""
    dim = active_dims()[0]
    return struct.pack(f"<{dim}f", *([value] * dim))


def mv_blob(value: float = 0.5) -> bytes:
    """A dummy vec_text (meaning) embedding sized for this environment."""
    dim = active_dims()[1]
    return struct.pack(f"<{dim}f", *([value] * dim))


def vv_list(value: float = 0.1) -> list:
    """Same as vv_blob but as a JSON-serializable list (for API payloads)."""
    return [value] * active_dims()[0]


def vec_dim(db) -> int:
    """The width THIS database actually declares — authoritative check."""
    row = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_files'").fetchone()
    if not row:
        raise AssertionError("vec_files table is missing — cannot derive dimension")
    m = re.search(r"FLOAT\[(\d+)\]", row[0])
    if not m:
        raise AssertionError(f"cannot parse vec_files dimension from: {row[0]!r}")
    return int(m.group(1))


def vec_blob(db, value: float = 0.5) -> bytes:
    """A dummy embedding sized from the given database's own schema."""
    dim = vec_dim(db)
    return struct.pack(f"<{dim}f", *([value] * dim))


@pytest.fixture
def vec_factory():
    """Fixture form: `vec_factory(db)` → correctly sized embedding blob."""
    return vec_blob
