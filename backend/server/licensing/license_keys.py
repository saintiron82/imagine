"""
RSA public key for offline license JWT verification.

The private key is held by the Cloud Function that signs license tokens.
This module only verifies — it cannot create tokens.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Placeholder RSA public key — replace with actual key from Cloud Function deployment
_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA0000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000
000000000000000000000000000000000000000000000000000000000000000000AQAB
-----END PUBLIC KEY-----"""


def verify_offline_token(token: str) -> Optional[dict]:
    """
    Verify an offline license JWT signed with RS256.

    Returns decoded claims if valid, None otherwise.
    Claims expected: group_name, plan_id, max_users, max_files, features, expires_at
    """
    try:
        import jwt
        decoded = jwt.decode(
            token,
            _PUBLIC_KEY_PEM,
            algorithms=["RS256"],
            options={"verify_exp": True},
        )
        return decoded
    except Exception as e:
        logger.warning(f"Offline license token verification failed: {e}")
        return None


def set_public_key(pem: str):
    """Update the RSA public key (for testing or key rotation)."""
    global _PUBLIC_KEY_PEM
    _PUBLIC_KEY_PEM = pem
