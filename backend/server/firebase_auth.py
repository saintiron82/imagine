"""
Firebase Admin SDK — ID Token verification for group server authentication.

Initializes firebase-admin lazily (first call to verify_firebase_token).
Supports two modes:
  1. Service account key file (FIREBASE_SERVICE_ACCOUNT_KEY env var or config.yaml path)
  2. Application Default Credentials (GCP environments)

If initialization fails (e.g., no credentials), the server still runs
but Firebase token verification returns None (allowing localhost auto-admin to work).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_initialized = False
_available = False


def _init_firebase():
    """Initialize Firebase Admin SDK (once)."""
    global _initialized, _available
    if _initialized:
        return
    _initialized = True

    try:
        import firebase_admin
        from firebase_admin import credentials
        import os

        # Try service account key file first
        key_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY', '')
        if key_path and os.path.isfile(key_path):
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            logger.info(f"Firebase Admin initialized with service account: {key_path}")
            _available = True
            return

        # Try config.yaml path
        try:
            from backend.server.config import get_config
            cfg = get_config()
            cfg_key_path = cfg.get('server', {}).get('firebase', {}).get('service_account_key', '')
            if cfg_key_path and os.path.isfile(cfg_key_path):
                cred = credentials.Certificate(cfg_key_path)
                firebase_admin.initialize_app(cred)
                logger.info(f"Firebase Admin initialized from config.yaml: {cfg_key_path}")
                _available = True
                return
        except Exception:
            pass

        # Try Application Default Credentials
        firebase_admin.initialize_app()
        logger.info("Firebase Admin initialized with default credentials")
        _available = True

    except Exception as e:
        logger.warning(f"Firebase Admin SDK not available: {e}")
        logger.warning("Firebase token verification disabled — localhost auto-admin still works")
        _available = False


def verify_firebase_token(id_token: str) -> Optional[dict]:
    """
    Verify a Firebase ID token and return decoded claims.

    Returns:
        dict with 'uid', 'email', 'name' (display_name), 'email_verified'
        or None if verification fails or Firebase is not available.
    """
    _init_firebase()
    if not _available:
        return None

    try:
        from firebase_admin import auth
        decoded = auth.verify_id_token(id_token)
        return {
            'uid': decoded['uid'],
            'email': decoded.get('email', ''),
            'name': decoded.get('name', ''),
            'email_verified': decoded.get('email_verified', False),
        }
    except Exception as e:
        logger.warning(f"Firebase token verification failed: {e}")
        return None


def is_firebase_available() -> bool:
    """Check if Firebase Admin SDK is initialized and available."""
    _init_firebase()
    return _available
