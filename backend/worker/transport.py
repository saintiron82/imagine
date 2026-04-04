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

    Both LocalTransport (embedded) and HttpTransport (external) implement this.
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
                batch_capacity: int, resources: dict) -> dict:
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


class LocalTransport(WorkerTransport):
    """In-process transport for embedded worker.

    Calls AnalysisJobManager and WorkerScheduler directly — no HTTP.
    Accesses files and DB from local filesystem.
    """

    def __init__(self, db, scheduler, manager):
        """
        Args:
            db: SQLiteDB instance
            scheduler: WorkerScheduler instance
            manager: AnalysisJobManager instance
        """
        self.db = db
        self.scheduler = scheduler
        self.manager = manager
        self.session_id: Optional[int] = None

    def claim(self) -> Dict[str, Any]:
        if not self.session_id:
            return {"phase": None, "tasks": [], "count": 0}

        # Scheduler decides phase + count
        decision = self.scheduler.assign(self.session_id)
        phase = decision.get("phase")
        count = decision.get("count", 0)

        if not phase or count == 0:
            return {"phase": None, "tasks": [], "count": 0}

        # Claim tasks via manager
        tasks = self.manager.claim_tasks(phase, self.session_id, count)
        return {
            "phase": phase,
            "tasks": tasks,
            "count": len(tasks),
        }

    def report_start(self, task_id: int, phase: str):
        try:
            self.manager.start_task_phase(task_id, phase)
        except Exception as e:
            logger.debug(f"LocalTransport report_start failed: {e}")

    def report_complete(
        self, task_id: int, phase: str, success: bool,
        error: str = None, elapsed_s: float = None,
    ):
        try:
            self.manager.complete_task_phase(
                task_id, phase, success, error, elapsed_s
            )
            # Update scheduler with speed data
            if success and elapsed_s and elapsed_s > 0:
                files_per_min = 60.0 / elapsed_s
                self.scheduler.update_speed(self.session_id, phase, files_per_min)
        except Exception as e:
            logger.debug(f"LocalTransport report_complete failed: {e}")

    def get_thumbnail(self, file_id: int) -> Optional[str]:
        """Read thumbnail_url directly from files table."""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT thumbnail_url FROM files WHERE id = ?", (file_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                p = Path(row[0])
                if p.exists():
                    return str(p)
        except Exception as e:
            logger.debug(f"LocalTransport get_thumbnail failed: {e}")
        return None

    def get_mc_data(self, file_id: int) -> Optional[dict]:
        """Read MC fields directly from files table."""
        try:
            import json as _json
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT mc_caption, ai_tags, image_type, scene_type, art_style
                FROM files WHERE id = ?
            """, (file_id,))
            row = cursor.fetchone()
            if not row:
                return None
            ai_tags = row[1]
            if isinstance(ai_tags, str):
                try:
                    ai_tags = _json.loads(ai_tags)
                except Exception:
                    ai_tags = []
            return {
                "mc_caption": row[0] or "",
                "ai_tags": ai_tags or [],
                "image_type": row[2] or "",
                "scene_type": row[3] or "",
                "art_style": row[4] or "",
            }
        except Exception as e:
            logger.debug(f"LocalTransport get_mc_data failed: {e}")
        return None

    def save_vision(self, file_id: int, fields: dict) -> bool:
        """Write MC vision fields directly to files table."""
        try:
            import json as _json
            cursor = self.db.conn.cursor()
            safe_fields = [
                "mc_caption", "ai_tags", "image_type", "art_style",
                "color_palette", "scene_type", "time_of_day", "weather",
                "character_type", "item_type", "ui_type", "structured_meta",
            ]
            updates, values = [], []
            for f in safe_fields:
                if f in fields:
                    val = fields[f]
                    if isinstance(val, (list, dict)):
                        val = _json.dumps(val, ensure_ascii=False)
                    updates.append(f"{f} = ?")
                    values.append(val)

            if updates:
                values.append(file_id)
                cursor.execute(
                    f"UPDATE files SET {', '.join(updates)} WHERE id = ?",
                    values,
                )
                self.db.conn.commit()
            return True
        except Exception as e:
            logger.error(f"LocalTransport save_vision failed: {e}")
            return False

    def save_vv(self, file_id: int, vector) -> bool:
        """Write VV vector directly to vec_files table."""
        try:
            import numpy as np
            vec = vector if isinstance(vector, (list, tuple)) else vector.tolist()
            blob = struct.pack(f"<{len(vec)}f", *vec)
            cursor = self.db.conn.cursor()
            cursor.execute("DELETE FROM vec_files WHERE file_id = ?", (file_id,))
            cursor.execute(
                "INSERT INTO vec_files (file_id, embedding) VALUES (?, ?)",
                (file_id, blob),
            )
            self.db.conn.commit()
            return True
        except Exception as e:
            logger.error(f"LocalTransport save_vv failed: {e}")
            return False

    def save_mv(self, file_id: int, vector) -> bool:
        """Write MV vector directly to vec_text table."""
        try:
            vec = vector if isinstance(vector, (list, tuple)) else vector.tolist()
            blob = struct.pack(f"<{len(vec)}f", *vec)
            cursor = self.db.conn.cursor()
            cursor.execute("DELETE FROM vec_text WHERE file_id = ?", (file_id,))
            cursor.execute(
                "INSERT INTO vec_text (file_id, embedding) VALUES (?, ?)",
                (file_id, blob),
            )
            self.db.conn.commit()
            return True
        except Exception as e:
            logger.error(f"LocalTransport save_mv failed: {e}")
            return False

    def connect(self, worker_name: str, hostname: str,
                batch_capacity: int, resources: dict) -> dict:
        """Register embedded worker session directly in DB."""
        from datetime import datetime
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.db.conn.cursor()

        # Find actual admin user_id (not hardcoded)
        cursor.execute("SELECT id FROM users WHERE role = 'admin' AND is_active = 1 LIMIT 1")
        user_row = cursor.fetchone()
        user_id = user_row[0] if user_row else None
        if not user_id:
            # No admin user — create without FK
            cursor.execute("SELECT id FROM users LIMIT 1")
            any_user = cursor.fetchone()
            user_id = any_user[0] if any_user else None

        cursor.execute("""
            INSERT INTO worker_sessions
                (user_id, worker_name, hostname, status, batch_capacity,
                 connected_at, last_heartbeat)
            VALUES (?, ?, ?, 'online', ?, ?, ?)
        """, (user_id, worker_name, hostname, batch_capacity, now, now))
        self.session_id = cursor.lastrowid
        self.db.conn.commit()

        # Register GPU specs with scheduler
        gpu = resources.get("gpu_name", "")
        vram = resources.get("gpu_memory_total_gb", 0)
        is_metal = resources.get("is_metal", False)
        self.scheduler.register_worker(self.session_id, gpu, vram, is_metal)

        return {
            "session_id": self.session_id,
            "processing_mode": "auto",
        }

    def heartbeat(self, data: dict) -> dict:
        """Update session metrics directly in DB."""
        from datetime import datetime
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.db.conn.cursor()
        cursor.execute("""
            UPDATE worker_sessions
            SET last_heartbeat = ?,
                jobs_completed = ?,
                current_phase = ?,
                current_file = ?
            WHERE id = ?
        """, (
            now,
            data.get("jobs_completed", 0),
            data.get("current_phase"),
            data.get("current_file"),
            self.session_id,
        ))
        self.db.conn.commit()
        return {"ok": True}

    def disconnect(self, session_id: int):
        """Mark session offline."""
        from datetime import datetime
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.db.conn.cursor()
        cursor.execute("""
            UPDATE worker_sessions
            SET status = 'offline', disconnected_at = ?
            WHERE id = ?
        """, (now, session_id or self.session_id))
        self.db.conn.commit()


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
                batch_capacity: int, resources: dict) -> dict:
        try:
            resp = self._request(
                "post",
                f"{self.server_url}/api/v1/workers/connect",
                json={
                    "worker_name": worker_name,
                    "hostname": hostname,
                    "batch_capacity": batch_capacity,
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
