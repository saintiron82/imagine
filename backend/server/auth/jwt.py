"""
JWT token creation and verification.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt

from backend.server.config import (
    get_jwt_secret,
    get_access_token_minutes,
    get_refresh_token_days,
)

ALGORITHM = "HS256"


def create_access_token(user_id: int, username: str, role: str, expires_minutes: int = None) -> str:
    """Create an access token. Default expiry from config, or override with expires_minutes."""
    minutes = expires_minutes or get_access_token_minutes()
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def create_refresh_token() -> str:
    """Create a random refresh token string."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    """SHA256 hash of refresh token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def get_refresh_token_expiry() -> datetime:
    """Get refresh token expiry datetime."""
    return datetime.now(timezone.utc) + timedelta(days=get_refresh_token_days())


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify an access token. Returns payload or None."""
    import logging
    _logger = logging.getLogger(__name__)
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            _logger.warning(f"[JWT] Token type mismatch: expected 'access', got '{payload.get('type')}'")
            return None
        return payload
    except jwt.ExpiredSignatureError:
        _logger.warning(f"[JWT] Token EXPIRED (token={token[:20]}...)")
        return None
    except jwt.InvalidTokenError as e:
        _logger.warning(f"[JWT] Token INVALID: {e} (token={token[:20]}...)")
        return None
