"""
Result uploader — writes worker analysis results to the server files API.

Used by external workers in HTTP mode (transport=None). Task lifecycle
(claim/start/complete) goes through /api/v1/tasks/* in worker_daemon;
this module only persists the actual payloads:
  - PATCH /api/v1/files/{id}/vision  (MC vision fields)
  - PATCH /api/v1/files/{id}/vv      (visual vector)
  - PATCH /api/v1/files/{id}/mv      (meaning vector)
"""

import logging
from typing import Dict, Any

# numpy is imported lazily — only when vectors are actually encoded
# This avoids blocking the worker IPC startup (numpy DLL loading can hang on Windows)

logger = logging.getLogger(__name__)


class ResultUploader:
    """Uploads processing results to the Imagine server."""

    def __init__(self, session, server_url: str, authed_request_fn=None):
        """
        Args:
            session: requests.Session with Authorization header set
            server_url: Base URL (e.g. "http://localhost:8000")
            authed_request_fn: Optional callable(method, url, **kwargs) that
                handles 401 auto-refresh. If None, uses session directly.
        """
        self.session = session
        self.base = server_url
        self._authed_request = authed_request_fn

    def _request(self, method: str, url: str, **kwargs):
        """Make HTTP request with optional auth-retry."""
        if self._authed_request:
            return self._authed_request(method, url, **kwargs)
        return getattr(self.session, method)(url, **kwargs)

    def save_vision_fields(self, file_id: int, vision_fields: Dict[str, Any]):
        """Save MC results to the files table via /api/v1/files/{id}/vision.

        Returns True on success, or error string on failure.
        """
        try:
            resp = self._request('patch',
                f"{self.base}/api/v1/files/{file_id}/vision",
                json=vision_fields,
            )
            if resp.status_code == 200:
                return True
            else:
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.error(f"save_vision_fields file {file_id} failed: {err}")
                return err
        except Exception as e:
            err = f"Request failed: {e}"
            logger.error(f"save_vision_fields file {file_id}: {err}")
            return err

    def save_vv_vector(self, file_id: int, vec):
        """Save VV vector to vec_files via /api/v1/files/{id}/vv."""
        import numpy as np
        try:
            vec_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
            resp = self._request('patch',
                f"{self.base}/api/v1/files/{file_id}/vv",
                json={"vector": vec_list},
            )
            if resp.status_code == 200:
                return True
            else:
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.error(f"save_vv_vector file {file_id} failed: {err}")
                return err
        except Exception as e:
            logger.error(f"save_vv_vector file {file_id}: {e}")
            return str(e)

    def save_mv_vector(self, file_id: int, vec):
        """Save MV vector to vec_text via /api/v1/files/{id}/mv."""
        import numpy as np
        try:
            vec_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
            resp = self._request('patch',
                f"{self.base}/api/v1/files/{file_id}/mv",
                json={"vector": vec_list},
            )
            if resp.status_code == 200:
                return True
            else:
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.error(f"save_mv_vector file {file_id} failed: {err}")
                return err
        except Exception as e:
            logger.error(f"save_mv_vector file {file_id}: {e}")
            return str(e)
