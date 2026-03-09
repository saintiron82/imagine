"""
Firebase Admin SDK — ID Token verification for group server authentication.

Initializes firebase-admin lazily (first call to verify_firebase_token).
Credential resolution order:
  1. Service account key file (FIREBASE_SERVICE_ACCOUNT_KEY env var)
  2. Service account key from config.yaml (server.firebase.service_account_key)
  3. firebase-tools CLI refresh token (~/.config/configstore/firebase-tools.json)
  4. Application Default Credentials (GCP environments)

If initialization fails (e.g., no credentials), the server still runs
but Firebase token verification returns None (allowing localhost auto-admin to work).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_initialized = False
_available = False

PROJECT_ID = 'imagine-b1e9c'


def _try_firebase_tools_credential():
    """Try to build a firebase-admin credential from firebase-tools CLI login token."""
    import json
    from pathlib import Path

    tools_path = Path.home() / '.config' / 'configstore' / 'firebase-tools.json'
    if not tools_path.is_file():
        return None

    try:
        from firebase_admin import credentials

        data = json.loads(tools_path.read_text())
        tokens = data.get('tokens', {})
        refresh_token = tokens.get('refresh_token')
        if not refresh_token:
            return None

        # RefreshToken accepts a dict with client_id, client_secret, refresh_token
        # Firebase CLI uses the same OAuth client as gcloud firebase
        return credentials.RefreshToken({
            'type': 'authorized_user',
            'client_id': '563584335869-fgrhgmd47bqnekij5i8b5pr03ho849e6.apps.googleusercontent.com',
            'client_secret': 'FhBEHKIHP56VBO1YJyjDcOKz',
            'refresh_token': refresh_token,
        })
    except Exception as e:
        logger.debug(f"firebase-tools credential failed: {e}")
        return None


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

        # 1. Service account key file (env var)
        key_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY', '')
        if key_path and os.path.isfile(key_path):
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            logger.info(f"Firebase Admin initialized with service account: {key_path}")
            _available = True
            return

        # 2. Service account key from config.yaml
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

        # 3. firebase-tools CLI token
        tools_cred = _try_firebase_tools_credential()
        if tools_cred is not None:
            firebase_admin.initialize_app(tools_cred, options={'projectId': PROJECT_ID})
            logger.info("Firebase Admin initialized with firebase-tools CLI credentials")
            _available = True
            return

        # 4. Application Default Credentials
        firebase_admin.initialize_app(options={'projectId': PROJECT_ID})
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
