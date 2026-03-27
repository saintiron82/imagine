"""
File content hash utility.

Computes SHA-256 hash of file contents for unique file identification.
Used as the primary file identifier — path-independent.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Read full file for hash (PSD files can be large, but hash must be exact)
_HASH_CHUNK_SIZE = 64 * 1024  # 64KB read chunks


def compute_file_hash(file_path: str) -> Optional[str]:
    """Compute SHA-256 hash of entire file content.

    Args:
        file_path: Absolute path to file (local or already downloaded)

    Returns:
        Hex digest string (64 chars) or None on error
    """
    try:
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to hash {file_path}: {e}")
        return None


def compute_bytes_hash(data: bytes) -> str:
    """Compute SHA-256 hash of bytes (for in-memory data)."""
    return hashlib.sha256(data).hexdigest()
