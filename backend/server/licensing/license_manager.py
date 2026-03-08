"""
LicenseManager — hybrid license verification core.

Verification chain:
1. Firebase RTDB GET /licenses/{group_key} → success → cache + return
2. Fail (offline) → load from system_meta.license_cache → if valid → return
3. No cache → verify system_meta.license_offline_key JWT → return
4. All fail → return free-tier defaults (server never blocked)

Grace period: 7 days after expiry — full features with warning banner.
After grace: read-only (search OK, pipeline/upload blocked).
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from .license_schemas import LicenseInfo, LicenseStatus

logger = logging.getLogger(__name__)

# Firebase RTDB base URL
FIREBASE_LICENSES_BASE = "https://imagine-b1e9c-default-rtdb.firebaseio.com/licenses"

GRACE_PERIOD_DAYS = 7

# Free-tier defaults (when no license exists)
FREE_TIER = LicenseInfo(
    plan_id="free",
    status=LicenseStatus.FREE,
    max_users=1,
    max_files=1000,
    features=[],
    is_writable=True,
    message="Free tier — 1 user, 1000 files",
)


def _group_key(group_name: str) -> str:
    """Convert group name to Firebase RTDB key."""
    return group_name.strip().lower().replace(" ", "_")


def _compute_status(expires_at_str: Optional[str], raw_status: str) -> tuple:
    """Compute effective status, days_remaining, is_writable from expiry."""
    if raw_status == "suspended":
        return LicenseStatus.SUSPENDED, None, False

    if not expires_at_str:
        return LicenseStatus.ACTIVE, None, True

    try:
        expires = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return LicenseStatus.ACTIVE, None, True

    now = datetime.now(timezone.utc)
    delta = expires - now
    days_remaining = delta.days

    if days_remaining > 0:
        return LicenseStatus.ACTIVE, days_remaining, True
    elif days_remaining >= -GRACE_PERIOD_DAYS:
        return LicenseStatus.GRACE_PERIOD, days_remaining, True
    else:
        return LicenseStatus.EXPIRED, days_remaining, False


def _firebase_data_to_info(data: dict, current_users: int = 0, current_files: int = 0) -> LicenseInfo:
    """Convert Firebase RTDB data to LicenseInfo."""
    raw_status = data.get("status", "active")
    expires_at = data.get("expires_at")
    status, days_remaining, is_writable = _compute_status(expires_at, raw_status)

    return LicenseInfo(
        group_name=data.get("group_name", ""),
        plan_id=data.get("plan_id", "unknown"),
        status=status,
        max_users=data.get("max_users", 1),
        max_files=data.get("max_files", 1000),
        current_users=current_users,
        current_files=current_files,
        features=data.get("features", []),
        expires_at=expires_at,
        days_remaining=days_remaining,
        is_writable=is_writable,
    )


class LicenseManager:
    """Manages license verification with hybrid fallback chain."""

    def __init__(self, db):
        self.db = db
        self._cached_info: Optional[LicenseInfo] = None

    def _get_group_name(self) -> Optional[str]:
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT value FROM system_meta WHERE key = 'group_name'")
        row = cursor.fetchone()
        return row[0] if row else None

    def _count_users(self) -> int:
        cursor = self.db.conn.cursor()
        # Try members table first, fall back to users
        try:
            cursor.execute("SELECT COUNT(*) FROM members WHERE is_active = 1")
            return cursor.fetchone()[0]
        except Exception:
            try:
                cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
                return cursor.fetchone()[0]
            except Exception:
                return 0

    def _count_files(self) -> int:
        cursor = self.db.conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM files")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def _save_cache(self, data: dict):
        """Save license data to system_meta for offline use."""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO system_meta (key, value) VALUES (?, ?)",
                ("license_cache", json.dumps(data))
            )
            cursor.execute(
                "INSERT OR REPLACE INTO system_meta (key, value) VALUES (?, ?)",
                ("license_cache_at", datetime.now(timezone.utc).isoformat())
            )
            self.db.conn.commit()
        except Exception as e:
            logger.warning(f"Failed to save license cache: {e}")

    def _load_cache(self) -> Optional[dict]:
        """Load cached license data from system_meta."""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT value FROM system_meta WHERE key = 'license_cache'")
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        except Exception as e:
            logger.warning(f"Failed to load license cache: {e}")
        return None

    def _load_offline_key(self) -> Optional[dict]:
        """Load and verify offline license JWT from system_meta."""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT value FROM system_meta WHERE key = 'license_offline_key'")
            row = cursor.fetchone()
            if not row:
                return None

            from .license_keys import verify_offline_token
            return verify_offline_token(row[0])
        except Exception as e:
            logger.warning(f"Offline key verification failed: {e}")
        return None

    async def _fetch_firebase(self, group_name: str) -> Optional[dict]:
        """Fetch license data from Firebase RTDB."""
        import asyncio

        key = _group_key(group_name)
        url = f"{FIREBASE_LICENSES_BASE}/{key}.json"

        try:
            # Use synchronous request in thread pool (no httpx dependency)
            import urllib.request
            import ssl

            def _fetch():
                ctx = ssl.create_default_context()
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode())
                return None

            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, _fetch)
            return data
        except Exception as e:
            logger.warning(f"Firebase license fetch failed: {e}")
            return None

    async def verify(self) -> LicenseInfo:
        """
        Run the full verification chain and return current license info.
        Never blocks the server — always returns at least free-tier.
        """
        group_name = self._get_group_name()
        if not group_name:
            self._cached_info = FREE_TIER
            return FREE_TIER

        current_users = self._count_users()
        current_files = self._count_files()

        # 1. Try Firebase RTDB
        try:
            data = await self._fetch_firebase(group_name)
            if data and isinstance(data, dict) and data.get("plan_id"):
                self._save_cache(data)
                info = _firebase_data_to_info(data, current_users, current_files)
                self._cached_info = info
                logger.info(f"License verified (Firebase): {info.plan_id} / {info.status}")
                return info
        except Exception as e:
            logger.warning(f"Firebase verification failed: {e}")

        # 2. Try local cache
        cached = self._load_cache()
        if cached:
            info = _firebase_data_to_info(cached, current_users, current_files)
            self._cached_info = info
            logger.info(f"License verified (cache): {info.plan_id} / {info.status}")
            return info

        # 3. Try offline key
        offline = self._load_offline_key()
        if offline:
            info = _firebase_data_to_info(offline, current_users, current_files)
            self._cached_info = info
            logger.info(f"License verified (offline key): {info.plan_id} / {info.status}")
            return info

        # 4. Free tier fallback
        free = FREE_TIER.model_copy()
        free.group_name = group_name
        free.current_users = current_users
        free.current_files = current_files
        self._cached_info = free
        logger.info("License: free tier (no license found)")
        return free

    def get_cached(self) -> LicenseInfo:
        """Get the last verified license info (synchronous, no network)."""
        return self._cached_info or FREE_TIER

    def save_offline_key(self, key: str) -> bool:
        """Store an offline license key in system_meta."""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO system_meta (key, value) VALUES (?, ?)",
                ("license_offline_key", key)
            )
            self.db.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save offline key: {e}")
            return False
