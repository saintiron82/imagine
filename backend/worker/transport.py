"""
WorkerTransport — unified communication interface for workers.

Both embedded (in-process) and external (HTTP) workers use the same
abstract interface. The only difference is transport implementation.

No legacy dependency (job_queue, manager.py, pipeline.py).
"""

import logging
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkerTransport(ABC):
    """Abstract transport layer for worker↔server communication.

    Implemented by HttpTransport (direct HTTP) and RelayTransport (AWS relay).
    WorkerDaemon calls these methods without knowing the transport type.
    """

    # ── Task Lifecycle ──────────────────────────────────────

    @abstractmethod
    def claim(self) -> Dict[str, Any]:
        """Request work from server. Server decides phase + count.

        Returns:
            {"phase": "mc"|"vv"|"mv"|None, "tasks": [...], "count": int}
            Each task: {"task_id": int, "file_id": int, "file_path": str}
        """
        ...

    @abstractmethod
    def report_start(self, task_id: int, phase: str):
        """Report actual processing start (not claim time)."""
        ...

    @abstractmethod
    def report_complete(
        self, task_id: int, phase: str, success: bool,
        error: str = None, elapsed_s: float = None,
    ):
        """Report phase completion with measured elapsed time."""
        ...

    # ── File Access ─────────────────────────────────────────

    @abstractmethod
    def get_thumbnail(self, file_id: int) -> Optional[str]:
        """Get thumbnail path for a file. Returns local path or None."""
        ...

    @abstractmethod
    def get_mc_data(self, file_id: int) -> Optional[dict]:
        """Get MC caption/tags for MV processing.

        Returns:
            {"mc_caption": str, "ai_tags": list, ...} or None
        """
        ...

    # ── Result Upload ───────────────────────────────────────

    @abstractmethod
    def save_vision(self, file_id: int, fields: dict) -> bool:
        """Save MC vision fields (caption, tags, classification)."""
        ...

    @abstractmethod
    def save_vv(self, file_id: int, vector) -> bool:
        """Save VV vector (SigLIP2 embedding)."""
        ...

    @abstractmethod
    def save_mv(self, file_id: int, vector) -> bool:
        """Save MV vector (Qwen3-Embedding)."""
        ...

    # ── Session ─────────────────────────────────────────────

    @abstractmethod
    def connect(self, worker_name: str, hostname: str,
                batch_capacity: int, resources: dict,
                origin: str = "headless", launcher: str = "cli") -> dict:
        """Connect worker session. Returns session info."""
        ...

    @abstractmethod
    def heartbeat(self, data: dict) -> dict:
        """Send heartbeat with metrics. Returns server commands."""
        ...

    @abstractmethod
    def disconnect(self, session_id: int):
        """Graceful disconnect."""
        ...


class HttpTransport(WorkerTransport):
    """HTTP transport for external workers.

    Wraps all server API calls with JWT authentication + auto-refresh.
    This is a thin adapter — the actual HTTP logic is delegated to
    the existing _authed_request() mechanism in WorkerDaemon.
    """

    def __init__(self, server_url: str, authed_request_fn):
        """
        Args:
            server_url: Base server URL (e.g. http://192.168.1.10:8000)
            authed_request_fn: Callable(method, url, **kwargs) with JWT refresh
        """
        self.server_url = server_url
        self._request = authed_request_fn
        self.session_id: Optional[int] = None

    def claim(self) -> Dict[str, Any]:
        try:
            resp = self._request(
                "post",
                f"{self.server_url}/api/v1/tasks/claim",
                json={"worker_id": self.session_id or 0},
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"HttpTransport claim failed: {e}")
        return {"phase": None, "tasks": [], "count": 0}

    def report_start(self, task_id: int, phase: str):
        try:
            self._request(
                "post",
                f"{self.server_url}/api/v1/tasks/start",
                json={"task_id": task_id, "phase": phase},
            )
        except Exception as e:
            logger.debug(f"HttpTransport report_start failed: {e}")

    def report_complete(
        self, task_id: int, phase: str, success: bool,
        error: str = None, elapsed_s: float = None,
    ):
        payload = {
            "task_id": task_id,
            "phase": phase,
            "success": success,
            "error_message": error,
        }
        if elapsed_s is not None:
            payload["elapsed_s"] = round(elapsed_s, 3)
        try:
            self._request(
                "post",
                f"{self.server_url}/api/v1/tasks/complete",
                json=payload,
            )
        except Exception as e:
            logger.debug(f"HttpTransport report_complete failed: {e}")

    def get_thumbnail(self, file_id: int) -> Optional[str]:
        """Download thumbnail from server to temp file."""
        import tempfile
        try:
            resp = self._request(
                "get",
                f"{self.server_url}/api/v1/files/{file_id}/thumbnail",
                stream=True,
            )
            if resp.status_code != 200:
                return None
            # Save to temp file
            suffix = ".png"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="thumb_"
            ) as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                return f.name
        except Exception as e:
            logger.warning(f"HttpTransport get_thumbnail failed: {e}")
        return None

    def get_mc_data(self, file_id: int) -> Optional[dict]:
        try:
            resp = self._request(
                "get",
                f"{self.server_url}/api/v1/files/{file_id}/mc",
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"HttpTransport get_mc_data failed: {e}")
        return None

    def save_vision(self, file_id: int, fields: dict) -> bool:
        try:
            resp = self._request(
                "patch",
                f"{self.server_url}/api/v1/files/{file_id}/vision",
                json=fields,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"HttpTransport save_vision failed: {e}")
            return False

    def save_vv(self, file_id: int, vector) -> bool:
        try:
            vec = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            resp = self._request(
                "patch",
                f"{self.server_url}/api/v1/files/{file_id}/vv",
                json={"vector": vec},
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"HttpTransport save_vv failed: {e}")
            return False

    def save_mv(self, file_id: int, vector) -> bool:
        try:
            vec = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            resp = self._request(
                "patch",
                f"{self.server_url}/api/v1/files/{file_id}/mv",
                json={"vector": vec},
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"HttpTransport save_mv failed: {e}")
            return False

    def connect(self, worker_name: str, hostname: str,
                batch_capacity: int, resources: dict,
                origin: str = "headless", launcher: str = "cli") -> dict:
        try:
            resp = self._request(
                "post",
                f"{self.server_url}/api/v1/workers/connect",
                json={
                    "worker_name": worker_name,
                    "hostname": hostname,
                    "batch_capacity": batch_capacity,
                    "origin": origin,
                    "launcher": launcher,
                    "resources": resources,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                self.session_id = data.get("session_id")
                return data
        except Exception as e:
            logger.error(f"HttpTransport connect failed: {e}")
        return {}

    def heartbeat(self, data: dict) -> dict:
        try:
            resp = self._request(
                "post",
                f"{self.server_url}/api/v1/workers/heartbeat",
                json=data,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"HttpTransport heartbeat failed: {e}")
        return {}

    def disconnect(self, session_id: int):
        try:
            self._request(
                "post",
                f"{self.server_url}/api/v1/workers/disconnect",
                json={"session_id": session_id},
            )
        except Exception as e:
            logger.warning(f"HttpTransport disconnect failed: {e}")
