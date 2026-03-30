"""
Path utilities — shared path normalization functions.
"""

import unicodedata


def nfc_path(p: str) -> str:
    """Normalize file path to NFC form and convert backslashes to forward slashes.

    Prevents macOS NFD/NFC duplicates in DB and ensures consistent path separators.

    Args:
        p: File path string (may contain backslashes or NFD-encoded characters).

    Returns:
        NFC-normalized path with forward slashes.
    """
    return unicodedata.normalize('NFC', str(p).replace('\\', '/'))
