"""Security utilities — path validation, SQL escaping, error sanitization."""

import ipaddress
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


# ── Filename Sanitization ─────────────────────────────────────


def sanitize_filename(filename: str) -> str:
    """Remove path separators and dangerous characters from upload filename.

    Uses ``Path.name`` to strip directory components, then removes any
    remaining characters that are not word-chars, spaces, hyphens, or dots.
    """
    # Strip directory traversal: ../../evil.psd -> evil.psd
    name = Path(filename).name
    # Remove non-safe characters (keep unicode letters, digits, -, ., spaces)
    name = re.sub(r'[^\w\s\-\.]', '_', name)
    # Reject empty or hidden-file results
    if not name or name.startswith('.'):
        ext = Path(filename).suffix or '.bin'
        name = f"upload{ext}"
    return name


# ── Server URL Validation (SSRF defense) ──────────────────────

_ALLOWED_SCHEMES = {"http", "https"}


def validate_server_url(url: str) -> str:
    """Validate a server URL against SSRF attacks.

    Rejects:
    - Non-http(s) schemes
    - Loopback / private / link-local IPs
    - Format-string injection chars ``{`` ``}``

    Returns the validated URL unchanged.

    Raises:
        ValueError: if the URL is invalid or blocked.
    """
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Missing hostname")

    # Block private / loopback IPs
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError(f"Private/loopback IP not allowed: {hostname}")
    except ValueError as orig:
        # Not an IP literal — that's fine (domain name), unless it was our own raise
        if "not allowed" in str(orig):
            raise
        # Also block well-known loopback hostnames
        if hostname in ("localhost",):
            raise ValueError(f"Loopback hostname not allowed: {hostname}")

    # Block Python format-string injection
    if '{' in url or '}' in url:
        raise ValueError("URL contains invalid characters")

    return url


# ── SQL LIKE Escape ───────────────────────────────────────────


def escape_like(value: str, escape_char: str = "\\") -> str:
    r"""Escape ``%`` and ``_`` wildcards for SQL LIKE patterns.

    The caller must append ``ESCAPE '\\'`` to the LIKE clause::

        WHERE col LIKE ? ESCAPE '\\'
    """
    return (
        value
        .replace(escape_char, escape_char * 2)
        .replace("%", escape_char + "%")
        .replace("_", escape_char + "_")
    )


# ── Error Message Sanitization ────────────────────────────────


def safe_error_detail(prefix: str, exc: Optional[Exception] = None) -> str:
    """Return a sanitized error message suitable for HTTP responses.

    Includes only the exception *type* name — never the full message,
    which may leak file paths, SQL, or stack traces.
    """
    if exc is None:
        return prefix
    exc_type = type(exc).__name__
    return f"{prefix} ({exc_type})"
