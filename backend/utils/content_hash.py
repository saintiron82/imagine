"""
Content hash utility for path-independent file identification.

Uses SHA256(file_size + first_8KB + last_8KB) for fast hashing
even on large PSD files (hundreds of MB).
"""

import hashlib
from pathlib import Path

CHUNK_SIZE = 8192  # 8KB


def compute_content_hash(file_path) -> str:
    """
    Compute SHA256 content hash from file size and boundary bytes.

    Algorithm: SHA256(file_size_8bytes_LE + first_8KB + last_8KB)

    - Reads at most 16KB regardless of file size
    - file_size prefix distinguishes files with identical headers/footers
    - Returns 64-char lowercase hex string

    Args:
        file_path: Path to the file (str or Path)

    Returns:
        64-character hex SHA256 hash string
    """
    p = Path(file_path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(size.to_bytes(8, 'little'))

    with open(p, 'rb') as f:
        head = f.read(CHUNK_SIZE)
        h.update(head)

        if size > CHUNK_SIZE * 2:
            f.seek(-CHUNK_SIZE, 2)
            tail = f.read(CHUNK_SIZE)
        elif size > CHUNK_SIZE:
            f.seek(-min(CHUNK_SIZE, size - CHUNK_SIZE), 2)
            tail = f.read()
        else:
            tail = b''
        h.update(tail)

    return h.hexdigest()
