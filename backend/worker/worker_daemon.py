"""
Imagine Worker Daemon — headless distributed pipeline worker.

Polls the server for pending jobs, downloads images (or accesses shared FS),
runs the local pipeline (Parse → Vision → Embed), then uploads results.

Features:
    - Prefetch pool: keeps job buffer at 2x batch_capacity
    - Heartbeat: periodic status report to server (30s)
    - Server commands: responds to stop/block via heartbeat

Usage:
    python -m backend.worker.worker_daemon

Environment variables (override config.yaml):
    IMAGINE_SERVER_URL       — Server base URL (e.g. http://192.168.1.10:8000)
    IMAGINE_WORKER_EMAIL     — Worker login email
    IMAGINE_WORKER_PASSWORD  — Worker login password
"""

import gc
import logging
import signal
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.worker.config import (
    get_server_url,
    get_claim_batch_size,
    get_storage_mode,
    get_batch_capacity,
    get_heartbeat_interval,
)
from backend.worker.result_uploader import ResultUploader
from backend.worker.schedule import is_active_now
from backend.worker.worker_state import WorkerStateMachine, WorkerState

import os as _os
if not _os.isatty(2):
    from backend.utils.json_log_formatter import setup_json_logging
    setup_json_logging()
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger = logging.getLogger("ImagineWorker")

# Graceful shutdown flag
_shutdown = False


_VISION_RESULT_FIELD_KEYS = (
    "mc_caption",
    "ai_tags",
    "ocr_text",
    "dominant_color",
    "ai_style",
    "image_type",
    "art_style",
    "color_palette",
    "scene_type",
    "time_of_day",
    "weather",
    "character_type",
    "item_type",
    "ui_type",
    "structured_meta",
    "perceptual_hash",
    "dup_group_id",
    "caption_model",
    "processing_status",
    "processing_error",
)


def _vision_result_to_fields(vision_result: Any) -> Dict[str, Any]:
    """Convert VLM output into DB vision fields without losing structured data."""
    if not vision_result:
        return {}

    if isinstance(vision_result, dict):
        source = dict(vision_result)
    else:
        source = {
            key: getattr(vision_result, key)
            for key in ("caption", "tags", *_VISION_RESULT_FIELD_KEYS)
            if hasattr(vision_result, key)
        }

    fields: Dict[str, Any] = {}
    if source.get("caption") is not None:
        fields["mc_caption"] = source["caption"]
    if source.get("tags") is not None:
        fields["ai_tags"] = source["tags"]

    for key in _VISION_RESULT_FIELD_KEYS:
        if source.get(key) is not None:
            fields[key] = source[key]

    if "structured_meta" not in fields and source:
        fields["structured_meta"] = dict(source)

    return fields


def _signal_handler(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received, finishing current jobs...")
    _shutdown = True


try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except ValueError:
    # signal only works in main thread — skip when imported from embedded worker
    pass


@dataclass
class _JobContext:
    """Intermediate results for a single job during batch processing."""
    job: dict = field(default_factory=dict)
    local_path: Optional[str] = None
    metadata: Optional[dict] = None
    thumb_path: Optional[str] = None
    meta_obj: Any = None
    vision_fields: dict = field(default_factory=dict)
    vv_vec: Any = None
    mv_vec: Any = None
    structure_vec: Any = None
    failed: bool = False
    error: str = ""
    error_code: Optional[str] = None


def _notify(callback, event_type: str, data: dict):
    """Call progress callback if provided."""
    if callback:
        try:
            callback(event_type, data)
        except Exception:
            pass


class WorkerDaemon:
    """Headless worker that processes jobs from the Imagine server.

    Supports two transport modes:
    - transport=None (default): HTTP-based, for external workers
    - transport=LocalTransport: direct DB calls, for embedded worker
    """

    def __init__(self, transport=None):
        import requests

        self.transport = transport  # None = HTTP mode
        self.server_url = get_server_url()
        self.session = requests.Session()
        # Do NOT set Content-Type on session — requests sets it automatically:
        # json= param → application/json, files= param → multipart/form-data.
        # A session-level Content-Type: application/json breaks multipart uploads.
        self.access_token = None
        self.refresh_token = None
        self.uploader = ResultUploader(self.session, self.server_url, authed_request_fn=self._authed_request)
        self.storage_mode = get_storage_mode()
        self.tmp_dir = tempfile.mkdtemp(prefix="imagine_worker_")

        # Batch capacity (set per claim by worker_ipc loop)
        self.batch_capacity = get_batch_capacity()

        # Worker identity — set before connect, used in log prefix
        self.worker_name = None
        self._log_prefix = ""  # "[worker_name]" for log identification (BUG-006)

        # Session tracking
        self.session_id = None
        self.processing_mode = "idle"  # "mc" | "vv" | "mv" | "parse" | "idle" — set by server on connect/heartbeat
        self._total_completed = 0
        self._total_failed = 0
        self._phase_counts = {"mc": 0, "vv": 0, "mv": 0}
        self._last_claim_diag = None  # Last empty-claim diagnostic from server
        self._batch_throughput = 0.0  # files/min from last completed batch
        self._phase_throughput = {"mc": 0.0, "vv": 0.0, "mv": 0.0}  # per-phase speed (persists across mode switches)
        self._current_job_id = None
        self._current_file = None
        self._current_phase = None

        # Stop signal — set by IPC controller to interrupt batch mid-flight
        self._stop_requested = False

        # Verbose worker logging (toggled via config or API)
        self.verbose_log = False
        try:
            from backend.utils.config import get_config
            self.verbose_log = get_config().get("worker.verbose_log", False)
        except Exception:
            pass

        # Throttle state
        self._throttle_level = "normal"
        self._original_batch_capacity = self.batch_capacity  # preserved for restore

        # Background download pool (overlap download with GPU processing)
        self._download_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dl")
        self._download_cache: Dict[int, Future] = {}  # file_id -> Future<local_path>

        # State machine (schedule + throttle driven)
        self._state_machine = WorkerStateMachine(
            on_enter_idle=self._on_enter_idle,
            on_enter_active=self._on_enter_active,
            on_enter_resting=self._on_enter_resting,
        )

        # IO/Analysis thread infrastructure
        self._result_queue = Queue()
        self._prefetch_queue = Queue(maxsize=1)
        self._shutdown = False
        self._io_thread = None
        self._analysis_thread = None

        logger.info(
            f"Worker initialized: server={self.server_url}, mode={self.storage_mode}, "
            f"batch_capacity={self.batch_capacity}"
        )

    # ── Dual-Thread Start/Stop ────────────────────────────────

    def start(self):
        """Start IO + Analysis threads. Both embedded and external use this."""
        self._shutdown = False
        self._io_thread = threading.Thread(target=self._io_loop, daemon=True, name="worker-io")
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True, name="worker-analysis")
        self._io_thread.start()
        self._analysis_thread.start()
        logger.info("Worker started (IO + Analysis threads)")

    def stop(self):
        """Stop both threads."""
        self._shutdown = True
        self._stop_requested = True
        if self._analysis_thread:
            self._analysis_thread.join(timeout=60)
        if self._io_thread:
            self._io_thread.join(timeout=10)
        self._io_thread = None
        self._analysis_thread = None
        logger.info("Worker stopped")

    def wait(self):
        """Block until analysis thread finishes (for CLI mode)."""
        if self._analysis_thread:
            self._analysis_thread.join()

    def is_running(self):
        return self._analysis_thread is not None and self._analysis_thread.is_alive()

    def _io_loop(self):
        """IO thread: heartbeat, result upload, batch prefetch."""
        heartbeat_interval = 5
        # Embedded worker holds a direct DB handle via LocalTransport; any
        # leaked implicit transaction would wedge the entire process with
        # "database is locked". Every iteration must end with an explicit
        # rollback as a safety net (CLAUDE.md SQLite rule).
        local_db = getattr(self.transport, "db", None)
        while not self._shutdown:
            try:
                # 1. Heartbeat
                self.transport.heartbeat({
                    "jobs_completed": self._total_completed,
                    "current_phase": self._current_phase,
                    "current_file": self._current_file,
                    "phase_throughput": dict(self._phase_throughput),
                    "batch_throughput": self._batch_throughput,
                    "batch_capacity": self.batch_capacity,
                })

                # 2. Drain result queue → save to DB/server
                while not self._result_queue.empty():
                    try:
                        item = self._result_queue.get_nowait()
                        self._save_result(item)
                    except Exception as e:
                        logger.warning(f"[IO] save failed: {e}")
                        break

                # 3. Prefetch next batch
                if self._prefetch_queue.empty():
                    try:
                        result = self.transport.claim()
                        if result.get("phase") and result.get("tasks"):
                            self._prefetch_queue.put(result)
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"[IO] loop error: {e}")
            finally:
                if local_db is not None:
                    try:
                        local_db.conn.rollback()
                    except Exception:
                        pass

            time.sleep(heartbeat_interval)

    def _save_result(self, item):
        """Save a single analysis result via transport."""
        rtype = item.get("type")
        file_id = item.get("file_id")
        task_id = item.get("task_id")

        # Failure report
        if item.get("success") is False or (rtype == "mc" and item.get("fields") is None):
            try:
                self.transport.report_complete(task_id, rtype, False, item.get("error"))
            except Exception:
                pass
            return

        try:
            success = False
            if rtype == "mc":
                success = self.transport.save_vision(file_id, item["fields"])
            elif rtype == "vv":
                success = self.transport.save_vv(file_id, item["vector"])
            elif rtype == "mv":
                success = self.transport.save_mv(file_id, item["vector"])

            self.transport.report_complete(
                task_id, rtype, success, elapsed_s=item.get("elapsed_s"),
            )
            if success:
                self._total_completed += 1
        except Exception as e:
            # Release any implicit transaction before bubbling back to _io_loop.
            local_db = getattr(self.transport, "db", None)
            if local_db is not None:
                try:
                    local_db.conn.rollback()
                except Exception:
                    pass
            logger.warning(f"[IO] save {rtype} failed for file {file_id}: {e}")

    def _analysis_loop(self):
        """Analysis thread: GPU inference only."""
        logger.info("[Analysis] Thread started")
        consecutive_empty = 0

        try:
            while not self._shutdown:
                # Get batch from IO thread's prefetch
                batch = None
                try:
                    batch = self._prefetch_queue.get(timeout=10)
                except Exception:
                    pass

                if not batch or not batch.get("tasks"):
                    consecutive_empty += 1
                    # Unload model when idle
                    prev = self._prev_mode if hasattr(self, '_prev_mode') else None
                    if prev:
                        logger.info(f"[Analysis] Queue empty — unloading {prev}")
                        if prev == "mc": self._unload_vlm()
                        elif prev == "vv": self._unload_vv()
                        elif prev == "mv": self._unload_mv()
                        self._prev_mode = None

                    wait = min(5 * consecutive_empty, 60)
                    for _ in range(wait):
                        if self._shutdown: break
                        time.sleep(1)
                    continue

                consecutive_empty = 0
                phase = batch["phase"]
                tasks = batch["tasks"]
                self.batch_capacity = len(tasks)

                # Convert tasks to job format
                jobs = [{
                    "job_id": t["task_id"], "file_id": t["file_id"],
                    "file_path": t["file_path"], "task_id": t["task_id"],
                    "analysis_job_id": t.get("job_id"),
                    "analysis_profile": t.get("analysis_profile"),
                } for t in tasks]

                # Unload previous model if mode changed
                prev_mode = getattr(self, '_prev_mode', None)
                if prev_mode and prev_mode != phase:
                    logger.info(f"[Analysis] Mode switch: {prev_mode} → {phase}")
                    if prev_mode == "mc": self._unload_vlm()
                    elif prev_mode == "vv": self._unload_vv()
                    elif prev_mode == "mv": self._unload_mv()

                self.processing_mode = phase
                self._prev_mode = phase

                try:
                    self.process_batch_phased(jobs)
                except Exception as e:
                    logger.error(f"[Analysis] batch failed: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"[Analysis] thread error: {e}", exc_info=True)
        finally:
            try:
                self.transport.disconnect(self.session_id)
            except Exception:
                pass
            logger.info(f"[Analysis] Thread stopped (completed {self._total_completed})")

    # ── Authentication ─────────────────────────────────────────

    def set_tokens(self, access_token: str, refresh_token: str = None) -> bool:
        """Inject existing JWT tokens (skip login, reuse session from Electron).

        Empty token is allowed only for non-HTTP transports such as the
        embedded LocalTransport path.
        """
        self.access_token = access_token or ""
        self.refresh_token = refresh_token
        if self.access_token:
            self.session.headers["Authorization"] = f"Bearer {self.access_token}"
            logger.info("Tokens injected from existing session")
        else:
            # No token — valid only when the daemon is not using HTTP auth.
            self.session.headers.pop("Authorization", None)
            logger.info("No token injected")
        return True

    def _refresh_auth(self) -> bool:
        """Refresh the access token using the stored refresh token."""
        if not self.refresh_token:
            logger.warning("[REFRESH] No refresh token available")
            return False

        try:
            rt_preview = self.refresh_token[:16] + "..."
            logger.info(f"[REFRESH] Attempting token refresh (refresh={rt_preview})")
            resp = self.session.post(
                f"{self.server_url}/api/v1/auth/refresh",
                json={"refresh_token": self.refresh_token},
            )
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data["access_token"]
                self.refresh_token = data.get("refresh_token", self.refresh_token)
                self.session.headers["Authorization"] = f"Bearer {self.access_token}"
                logger.info("[REFRESH] Token refreshed OK")
                return True
            logger.warning(f"[REFRESH] Failed: {resp.status_code} {resp.text[:200]}")
            return False
        except Exception as e:
            logger.warning(f"[REFRESH] Request failed: {e}")
            return False

    def _authed_request(self, method: str, url: str, **kwargs):
        """Make request with automatic token refresh on 401.
        Embedded worker (transport set) should not call this — log and skip.
        """
        if self.transport:
            logger.debug(f"[SKIP-HTTP] {method.upper()} {url} (using LocalTransport)")
            # Return a fake 200 response to avoid crashes in HTTP-only call sites.
            import types
            fake = types.SimpleNamespace(status_code=200, text='{}', json=lambda: {})
            return fake
        import requests
        resp = getattr(self.session, method)(url, **kwargs)
        if resp.status_code == 401:
            logger.warning(f"[WORKER-AUTH] 401 on {method.upper()} {url} — attempting refresh")
            at_preview = (self.access_token[:20] + "...") if self.access_token else "(none)"
            rt_preview = (self.refresh_token[:16] + "...") if self.refresh_token else "(none)"
            logger.info(f"[WORKER-AUTH] Current tokens: access={at_preview}, refresh={rt_preview}")
            if self._refresh_auth():
                logger.info("[WORKER-AUTH] Refresh succeeded, retrying request")
                resp = getattr(self.session, method)(url, **kwargs)
            else:
                logger.error("[WORKER-AUTH] Refresh FAILED — giving up")
        return resp

    # ── Session Management ─────────────────────────────────────

    def _connect_session(self) -> bool:
        """Register worker session with server. Returns True on success."""
        try:
            # Send hardware capability spec so server scheduler can classify
            # GPU class (strong/weak/cpu) and estimate MC capability.
            try:
                from backend.worker.capability import collect_capability
                connect_resources = collect_capability()
            except Exception:
                connect_resources = None

            resp = self._authed_request(
                "post",
                f"{self.server_url}/api/v1/workers/connect",
                json={
                    "worker_name": self.worker_name or f"{socket.gethostname()}-worker",
                    "hostname": socket.gethostname(),
                    "batch_capacity": self.batch_capacity,
                    "resources": connect_resources,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                self.session_id = data["session_id"]
                if data.get("batch_hint"):
                    self.batch_capacity = data["batch_hint"]
                if data.get("processing_mode"):
                    self.processing_mode = data["processing_mode"]
                # Set log prefix for worker identification (BUG-006)
                name = self.worker_name or f"{socket.gethostname()}-worker"
                self.worker_name = name
                self._log_prefix = f"[{name}]"
                logger.info(f"{self._log_prefix} Session registered: id={self.session_id}, batch={self.batch_capacity}, mode={self.processing_mode}")
                return True
            else:
                logger.error(f"Session connect failed: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"Session connect error: {e}")
            return False

    def _heartbeat(self) -> dict:
        """Send heartbeat and receive server commands."""
        if not self.session_id:
            return {}
        try:
            # Collect system resource metrics + throttle level
            try:
                from backend.worker.resource_monitor import collect_metrics, get_throttle_level
                resources = collect_metrics()
                throttle_level = get_throttle_level(resources)
            except Exception:
                resources = None
                throttle_level = "normal"
            resp = self._authed_request(
                "post",
                f"{self.server_url}/api/v1/workers/heartbeat",
                json={
                    "session_id": self.session_id,
                    "jobs_completed": self._total_completed,
                    "jobs_failed": self._total_failed,
                    "phase_counts": self._phase_counts,
                    "current_job_id": self._current_job_id,
                    "current_file": self._current_file,
                    "current_phase": self._current_phase,
                    "resources": resources,
                    "throttle_level": throttle_level,
                    "worker_state": self._state_machine.state_name,
                    "batch_throughput": self._batch_throughput,
                    "phase_throughput": self._phase_throughput,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                # Embedded worker (__builtin__) decides its own batch size per-phase.
                # Only external workers take batch_hint from server.
                if data.get("batch_hint") and self.worker_name != "__builtin__":
                    old_cap = self.batch_capacity
                    self.batch_capacity = data["batch_hint"]
                    if old_cap != self.batch_capacity:
                        logger.info(f"Batch capacity changed: {old_cap} → {self.batch_capacity}")
                # Mode hint from heartbeat — logged but NOT applied.
                # Actual mode is set by claim response (authoritative).
                if data.get("processing_mode"):
                    hint_mode = data["processing_mode"]
                    if hint_mode != self.processing_mode:
                        reason = data.get("mode_reason", "")
                        logger.debug(
                            f"[MODE-HINT] server suggests {hint_mode} "
                            f"(current={self.processing_mode}, reason={reason})"
                        )
                    elif data.get("mode_reason"):
                        # Log reason even without mode change (periodic diagnostic)
                        logger.debug(
                            f"[MODE-KEEP] {self.processing_mode} | reason: {data['mode_reason']}"
                        )
                return data
            return {}
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return {}

    def _disconnect_session(self):
        """Notify server of graceful disconnect."""
        if not self.session_id:
            return
        try:
            self._authed_request(
                "post",
                f"{self.server_url}/api/v1/workers/disconnect",
                json={"session_id": self.session_id},
            )
            logger.info(f"Session disconnected: id={self.session_id}")
        except Exception as e:
            logger.warning(f"Disconnect failed: {e}")

    # ── Job Pool Management ────────────────────────────────────

    def claim_jobs_count(self, count: int) -> list:
        """Claim tasks from the Analysis Job system."""
        if count <= 0:
            return []

        # Try new /api/v1/tasks/claim first
        phase = self.processing_mode
        if phase in ("mc", "vv", "mv", "parse", "download"):
            try:
                resp = self._authed_request(
                    "post",
                    f"{self.server_url}/api/v1/tasks/claim",
                    json={"phase": phase, "worker_id": self.session_id or 0, "count": count},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    tasks = data.get("tasks", [])
                    if tasks:
                        # Convert API tasks to the worker batch format.
                        jobs = []
                        for t in tasks:
                            jobs.append({
                                "job_id": t["task_id"],
                                "file_id": t["file_id"],
                                "file_path": t["file_path"],
                                "task_id": t["task_id"],  # new system ID
                                "analysis_job_id": t.get("job_id"),
                                "analysis_profile": t.get("analysis_profile"),
                            })
                        logger.info(f"{self._log_prefix} Claimed {len(jobs)} {phase} tasks (new API)")
                        return jobs
            except Exception as e:
                logger.warning(f"Task claim failed: {e}")
                return []

        # No valid phase — nothing to claim
        return []

    def _report_task_start(self, task_id: int, phase: str):
        """Report actual processing start."""
        if not task_id:
            return
        if self.transport:
            self.transport.report_start(task_id, phase)
            return
        try:
            self._authed_request(
                "post",
                f"{self.server_url}/api/v1/tasks/start",
                json={"task_id": task_id, "phase": phase},
            )
        except Exception:
            pass

    def _report_task_phase(self, task_id: int, phase: str, success: bool,
                           error: str = None, elapsed_s: float = None):
        """Report phase completion."""
        if not task_id:
            return
        if self.transport:
            self.transport.report_complete(task_id, phase, success, error, elapsed_s)
            return
        try:
            payload = {
                "task_id": task_id,
                "phase": phase,
                "success": success,
                "error_message": error,
            }
            if elapsed_s is not None:
                payload["elapsed_s"] = round(elapsed_s, 3)
            self._authed_request(
                "post",
                f"{self.server_url}/api/v1/tasks/complete",
                json=payload,
            )
        except Exception as e:
            logger.debug(f"Task phase report failed: {e}")

    def claim_jobs(self) -> list:
        """Claim jobs using configured batch size (used by worker_ipc loop)."""
        return self.claim_jobs_count(get_claim_batch_size())

    def _clear_current(self):
        """Clear current job tracking."""
        self._current_job_id = None
        self._current_file = None
        self._current_phase = None

    def _resolve_file(self, job: dict) -> str:
        """Get local file path for processing.

        Workers only handle MC/VV/MV — always need THUMBNAIL only.
        Parse is server-side (FileTaskParsePool).
        """
        file_path = job["file_path"]
        file_id = job.get("file_id")
        file_name = Path(file_path).name
        is_remote_uri = file_path.startswith(("webdav://", "http://", "https://"))

        # shared_fs mode — local paths only
        if self.storage_mode == "shared_fs" and not is_remote_uri:
            if Path(file_path).exists():
                logger.info(f"[RESOLVE] LOCAL file: {file_name}")
                return file_path
            logger.warning(f"[RESOLVE] LOCAL file not found: {file_name}")
            return None

        # MC/VV/MV: needs THUMBNAIL only (never download original)
        if job.get("pre_parsed") or is_remote_uri:
            thumb = self._resolve_thumbnail(job)
            if thumb:
                logger.info(f"[RESOLVE] THUMBNAIL ({Path(thumb).stat().st_size // 1024}KB): {file_name}")
                return thumb
            logger.error(f"[RESOLVE] THUMBNAIL failed — not available on server: {file_name} (file_id={file_id})")
            return None

        # 4) Non-pre-parsed local file (should not happen for mc/vv/mv workers)
        logger.error(f"[RESOLVE] Job not pre-parsed and not remote — cannot process: {file_name}")
        return None

    # _resolve_download_ahead removed — workers don't access download pool.
    # Parse is server-side (FileTaskParsePool).

    def _resolve_thumbnail(self, job: dict) -> Optional[str]:
        """Get thumbnail path for a pre-parsed job."""
        file_id = job.get("file_id")

        # LocalTransport: read directly from DB/filesystem
        if self.transport:
            return self.transport.get_thumbnail(file_id)

        # HttpTransport fallback: download from server
        try:
            resp = self._authed_request(
                'get',
                f"{self.server_url}/api/v1/files/{file_id}/thumbnail",
                stream=True,
            )
            try:
                if resp.status_code != 200:
                    logger.warning(f"Thumbnail download failed for file_id={file_id}: HTTP {resp.status_code}")
                    return None

                dest = Path(self.tmp_dir) / f"thumb_{file_id}.png"
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            finally:
                resp.close()

            size_kb = dest.stat().st_size / 1024
            logger.info(f"Downloaded thumbnail for file_id={file_id} ({size_kb:.0f}KB)")
            return str(dest)

        except Exception as e:
            logger.warning(f"Thumbnail download failed for file_id={file_id}: {e}")
            return None

    # ── Background Download (overlap with GPU processing) ─────

    def _prefetch_downloads(self, jobs: list):
        """Submit download tasks for upcoming jobs in background threads.

        Called right after claim_jobs() so files download in parallel
        while the current batch is being processed on GPU.

        vv/mv modes: only the thumbnail is needed (VV is computed from it),
        so we skip full file download to save bandwidth.
        """
        for job in jobs:
            file_path = job.get("file_path", "")
            is_remote = file_path.startswith(("webdav://", "http://", "https://"))

            # shared_fs: skip prefetch for local paths (instant access),
            # but DO prefetch remote URIs (webdav://) via server thumbnail
            if self.storage_mode == "shared_fs" and not is_remote:
                continue

            file_id = job.get("file_id")
            if file_id is not None and file_id not in self._download_cache:
                if self.processing_mode in ("mc", "vv", "mv") or is_remote or job.get("pre_parsed"):
                    # MC/VV/MV/pre-parsed/remote: thumbnail only (never full file)
                    future = self._download_pool.submit(self._resolve_thumbnail, job)
                else:
                    future = self._download_pool.submit(self._resolve_file, job)
                self._download_cache[file_id] = future

    def _get_downloaded(self, job: dict) -> Optional[str]:
        """Get downloaded file path, waiting if download is still in progress.

        If the file was prefetched, returns the cached result.
        Otherwise falls back to synchronous _resolve_file().
        """
        file_id = job.get("file_id")
        future = self._download_cache.pop(file_id, None) if file_id is not None else None
        if future:
            try:
                return future.result(timeout=300)  # 5 min max wait
            except Exception as e:
                logger.error(f"[PREFETCH] Download failed for file_id={file_id}: {e}")
                return None
        # Fallback: not prefetched (e.g. shared_fs or cache miss)
        return self._resolve_file(job)

    def _clear_download_cache(self):
        """Cancel pending downloads and clear the cache."""
        for file_id, future in list(self._download_cache.items()):
            future.cancel()
        self._download_cache.clear()

    # ── Pipeline Phases ────────

    def _run_vision(self, file_path: Path, thumb_path: str, meta, mc_raw_override: dict = None) -> dict:
        """Phase V: Run VLM to generate MC (caption, tags, classification).

        Args:
            file_path: Original file path (for logging).
            thumb_path: Path to thumbnail image.
            meta: AssetMeta object (local parse) or None (pre-parsed mode).
            mc_raw_override: Pre-built mc_raw dict from server (pre-parsed mode).
        """
        if not thumb_path or not Path(thumb_path).exists():
            logger.warning(f"No thumbnail for vision: {file_path.name}")
            return {}

        try:
            from backend.vision.vision_factory import get_vision_analyzer
            from backend.vision.domain_loader import get_active_domain
            from PIL import Image

            analyzer = get_vision_analyzer()
            raw_img = Image.open(thumb_path)

            # Composite to RGB
            try:
                if raw_img.mode == "RGBA":
                    thumb_img = Image.new("RGB", raw_img.size, (255, 255, 255))
                    thumb_img.paste(raw_img, mask=raw_img.split()[3])
                elif raw_img.mode != "RGB":
                    thumb_img = raw_img.convert("RGB")
                else:
                    thumb_img = raw_img
                    raw_img = None  # avoid double close
            finally:
                if raw_img is not None:
                    raw_img.close()

            # Use pre-built mc_raw if provided (pre-parsed mode), otherwise build from meta
            if mc_raw_override:
                mc_raw = mc_raw_override
            elif meta is not None:
                from backend.pipeline.ingest_engine import _build_mc_raw
                mc_raw = _build_mc_raw(meta)
            else:
                # MC-only mode without mc_raw or meta — build minimal context from job
                mc_raw = {
                    "file_name": file_path.name,
                    "folder_path": "",
                    "layer_names": [],
                    "used_fonts": [],
                    "ocr_text": "",
                    "text_content": [],
                }

            if not hasattr(analyzer, "classify_and_analyze"):
                raise RuntimeError(
                    f"{type(analyzer).__name__} does not implement spatial 2-stage analysis"
                )

            # Run the canonical spatial v2 2-stage vision contract.
            try:
                vision_result = analyzer.classify_and_analyze(
                    thumb_img,
                    context=mc_raw,
                    domain=get_active_domain(),
                )
            finally:
                thumb_img.close()

            return _vision_result_to_fields(vision_result)

        except Exception as e:
            logger.warning(f"Vision failed for {file_path.name}: {e}", exc_info=True)
            return {"_error": str(e)}

    _vv_encoder = None  # Cached SigLIP2 encoder (class-level singleton)

    def _run_embed_vv(self, thumb_path: str):
        """Phase E-VV: Generate visual vector (SigLIP2)."""
        vv_vec = None
        structure_vec = None

        if thumb_path and Path(thumb_path).exists():
            try:
                from backend.vector.siglip2_encoder import SigLIP2Encoder
                from PIL import Image

                if WorkerDaemon._vv_encoder is None:
                    WorkerDaemon._vv_encoder = SigLIP2Encoder()
                encoder = WorkerDaemon._vv_encoder
                img = Image.open(thumb_path).convert("RGB")
                try:
                    vv_vec = encoder.encode_image(img)

                    if hasattr(encoder, 'encode_structure'):
                        structure_vec = encoder.encode_structure(img)
                finally:
                    img.close()

            except Exception as e:
                logger.warning(f"VV encoding failed: {e}")

        return vv_vec, structure_vec

    def _run_embed_mv(self, metadata: dict):
        """Phase E-MV: Generate meaning vector (Qwen3-Embedding)."""
        mv_vec = None

        mc_caption = metadata.get("mc_caption", "")
        ai_tags = metadata.get("ai_tags", "")
        # ai_tags may be a list from VLM — convert to string
        if isinstance(ai_tags, list):
            ai_tags = ", ".join(str(t) for t in ai_tags)
        if mc_caption or ai_tags:
            try:
                from backend.vector.text_embedding import get_text_embedding_provider

                embedder = get_text_embedding_provider()
                mv_text = f"{mc_caption} {ai_tags}".strip()
                if mv_text:
                    mv_vec = embedder.encode(mv_text)

            except Exception as e:
                logger.warning(f"MV encoding failed: {e}")

        return mv_vec

    # ── Phase Helpers ────────────────────────────────────────────

    def _run_vision_phase(self, active: list, progress_callback=None) -> float:
        """Run Vision (VLM) phase on active job contexts.

        Iterates over each context, runs VLM to generate MC (caption/tags),
        and updates context metadata and vision_fields.

        For pre-parsed jobs, uses mc_raw from server instead of building from
        meta_obj. For full-pipeline jobs, checks the pre_parsed flag first.

        Args:
            active: List of _JobContext with resolved files.
            progress_callback: Optional progress callback.

        Returns:
            Elapsed time in seconds.
        """
        t_phase = time.perf_counter()
        _notify(progress_callback, "phase_start", {"phase": "vision", "count": len(active)})

        # Log VLM load on first file
        vlm_loaded = False
        for i, ctx in enumerate(active):
            if self._stop_requested:
                logger.info(f"Stop requested during Vision phase ({i}/{len(active)})")
                break

            self._current_phase = "vision"
            self._current_file = Path(ctx.job["file_path"]).name
            self.uploader.report_progress(ctx.job["job_id"], "vision")

            # Keep-alive: update DB metrics during long processing
            if i > 0 and i % 5 == 0:
                try:
                    if self.transport:
                        self.transport.heartbeat({
                            "jobs_completed": self._total_completed,
                            "current_phase": self._current_phase,
                            "current_file": self._current_file,
                            "phase_throughput": self._phase_throughput,
                            "batch_throughput": self._batch_throughput,
                            "batch_capacity": self.batch_capacity,
                        })
                    else:
                        self._heartbeat()
                except Exception:
                    pass

            # Log VLM backend info BEFORE loading (so user sees it immediately)
            if not vlm_loaded:
                try:
                    from backend.vision.vision_factory import VisionAnalyzerFactory
                    from backend.utils.tier_config import get_active_tier
                    tier_name, tier_config = get_active_tier()
                    vlm_cfg = tier_config.get("vlm", {})
                    chain = VisionAnalyzerFactory._resolve_backend_chain(vlm_cfg, tier_name)
                    chain_str = " -> ".join(f"{e['backend']}({e.get('model','?')})" for e in chain)
                    _notify(progress_callback, "diag_log", {
                        "message": f"[MC] Loading VLM: {chain_str}", "level": "info",
                    })
                except Exception:
                    _notify(progress_callback, "diag_log", {
                        "message": f"[MC] Loading VLM...", "level": "info",
                    })

            mc_raw_override = ctx.job.get("mc_raw") if ctx.job.get("pre_parsed") else None
            # Measure actual inference time per file
            _t_file = time.perf_counter()
            vision_fields = self._run_vision(
                Path(ctx.job["file_path"]), ctx.thumb_path, ctx.meta_obj,
                mc_raw_override=mc_raw_override,
            )
            _file_elapsed = time.perf_counter() - _t_file
            ctx._inference_elapsed = _file_elapsed  # store for reporting

            if not vlm_loaded:
                elapsed_first = time.perf_counter() - t_phase
                from backend.vision.vision_factory import VisionAnalyzerFactory
                cached = VisionAnalyzerFactory._cached_analyzer
                backend_name = type(cached).__name__ if cached else "unknown"
                model_name = getattr(cached, 'model', getattr(cached, 'model_id', '?'))
                _notify(progress_callback, "diag_log", {
                    "message": f"[MC] VLM ready: {backend_name} ({model_name}) — first file {elapsed_first:.1f}s",
                    "level": "info",
                })
                vlm_loaded = True

            vlm_err = vision_fields.get("_error") if isinstance(vision_fields, dict) else None
            if vision_fields and vision_fields.get("mc_caption"):
                self._phase_counts["mc"] += 1
                elapsed_so_far = time.perf_counter() - t_phase
                if elapsed_so_far > 0:
                    self._phase_throughput["mc"] = round((i + 1) / elapsed_so_far * 60, 1)
                    self._batch_throughput = self._phase_throughput["mc"]
                if ctx.metadata:
                    ctx.metadata.update(vision_fields)
                ctx.vision_fields = vision_fields

                # Queue result for IO thread (crash-safe, non-blocking)
                task_id = ctx.job.get("task_id")
                file_id = ctx.job.get("file_id")
                if hasattr(self, '_result_queue') and self._result_queue and file_id:
                    self._result_queue.put({
                        "type": "mc", "file_id": file_id, "task_id": task_id,
                        "fields": dict(vision_fields), "elapsed_s": _file_elapsed,
                    })
                    ctx._saved = True
            else:
                ctx.failed = True
                if vlm_err:
                    ctx.error = f"VLM error for {self._current_file}: {vlm_err}"
                else:
                    ctx.error = f"VLM returned empty MC for {self._current_file}"
                logger.warning(ctx.error)
                # Queue failure report for IO thread
                task_id = ctx.job.get("task_id")
                if hasattr(self, '_result_queue') and self._result_queue and task_id:
                    self._result_queue.put({
                        "type": "mc", "file_id": ctx.job.get("file_id"),
                        "task_id": task_id, "fields": None,
                        "error": ctx.error, "success": False,
                    })
                # Surface error to UI via diag_log
                _notify(progress_callback, "diag_log", {
                    "message": f"[MC] FAIL {self._current_file}: {ctx.error}",
                    "level": "warning",
                })

            _notify(progress_callback, "file_done", {
                "phase": "vision", "file_name": self._current_file,
                "index": i + 1, "count": len(active), "success": not ctx.failed,
            })

        elapsed = time.perf_counter() - t_phase
        fpm = (len(active) / elapsed * 60) if elapsed > 0 else 0
        _notify(progress_callback, "phase_complete", {
            "phase": "vision", "count": len(active),
            "elapsed_s": round(elapsed, 2), "files_per_min": round(fpm, 1),
        })
        return elapsed

    # ── Batch Processing (Phase-Level with Sub-Batch Inference) ──

    def process_batch_phased(self, jobs: list, progress_callback=None) -> list:
        """Process a batch of jobs using phase-level sub-batch processing.

        Mirrors the local pipeline (ingest_engine.py) approach:
        - Parse: 1-by-1 (CPU-bound)
        - Vision (VLM): 1-by-1 (MLX/transformers batch_size=1)
        - VV (SigLIP2): real batch via encode_image_batch()
        - MV (Qwen3-Embedding): real batch via encode_batch()

        Mode routing:
        - "mc":    Only Vision (VLM/MC) phase. VLM stays loaded.
                   Server handles Parse (ParseAhead), VV, MV separately.
        - "vv":    Only VV (SigLIP2) phase.
        - "mv":    Only MV (Qwen3-Embedding) phase.
        - "parse": Only Parse + Thumbnail (CPU-only, no GPU).

        Each phase tracks elapsed time and reports files/min.

        Args:
            jobs: List of job dicts with job_id, file_id, file_path.
            progress_callback: Optional callback(event_type, data) for IPC progress.
        Returns:
            List of (job_id, success) tuples.
        """
        # Parse is server-only (FileTaskParsePool). Workers handle MC/VV/MV only.
        if self.processing_mode == "mc":
            return self._process_batch_mc(jobs, progress_callback)

        if self.processing_mode == "vv":
            return self._process_batch_vv_only(jobs, progress_callback)

        if self.processing_mode == "mv":
            return self._process_batch_mv_only(jobs, progress_callback)

        t_batch = time.perf_counter()

        # Build job contexts and resolve file access (uses prefetched downloads)
        contexts = []
        for job in jobs:
            ctx = _JobContext(job=job)
            file_path = job.get("file_path", "")
            is_remote = file_path.startswith(("webdav://", "http://", "https://"))

            if not job.get("pre_parsed") and not is_remote:
                # Server must always pre-parse jobs. Non-pre-parsed = error.
                ctx.failed = True
                ctx.error = f"Job not pre-parsed by server: {file_path} (file_id={job.get('file_id')})"
                logger.error(f"[RESOLVE] {ctx.error}")
                _notify(progress_callback, "file_error", {
                    "file_name": Path(file_path).name,
                    "error": ctx.error,
                })
            else:
                # Pre-parsed by server, or remote URI
                ctx.local_path = self._get_downloaded(job)
                ctx.metadata = dict(job.get("metadata", {}))
                # Use server-generated thumbnail if available (shared_fs mode
                # returns original file path, not thumbnail)
                server_thumb = job.get("thumb_path")
                if server_thumb and Path(server_thumb).exists():
                    ctx.thumb_path = server_thumb
                else:
                    ctx.thumb_path = ctx.local_path
                ctx.meta_obj = None  # No AssetMeta object (use mc_raw dict instead)
                if not ctx.local_path or not Path(ctx.local_path).exists():
                    ctx.failed = True
                    if is_remote:
                        ctx.error_code = "THUMB_MISSING"
                        ctx.error = (
                            f"No thumbnail for remote file: {file_path} "
                            f"(file_id={job.get('file_id')})"
                        )
                    else:
                        ctx.error_code = "FILE_NOT_FOUND"
                        ctx.error = (
                            f"File unavailable: {file_path} "
                            f"(file_id={job.get('file_id')})"
                        )
                    logger.error(f"[RESOLVE] [{ctx.error_code}] {ctx.error}")
                    _notify(progress_callback, "file_error", {
                        "file_name": Path(file_path).name,
                        "error": ctx.error,
                    })

            contexts.append(ctx)

        active = [c for c in contexts if not c.failed]

        # Phase P is always handled by the server (ParseAheadPool).
        if self.verbose_log:
            file_names = [Path(c.job.get("file_path","")).name for c in active]
            logger.info(f"{self._log_prefix} Batch START: {len(active)} files, chunk={self.batch_capacity}, mode={self.processing_mode}")
            logger.info(f"[WORKER] Files: {file_names}")
        logger.info(f"Phase P: {len(active)} jobs pre-parsed by server (worker skips parsing)")
        elapsed_parse = 0.0
        fpm_parse = 0.0

        # ── Phase V → VV → MV via unified PhaseRunner ──
        from backend.pipeline.protocols import PhaseItem, FixedBatchStrategy
        from backend.pipeline.model_manager import ModelManager
        from backend.pipeline.phase_runner import PhaseRunner

        # Convert _JobContext → PhaseItem
        phase_items = []
        for ctx in active:
            has_vision = bool(ctx.job.get("vision_data"))
            item = PhaseItem(
                job_id=ctx.job["job_id"],
                file_id=ctx.job["file_id"],
                file_path=ctx.job["file_path"],
                thumb_path=ctx.thumb_path,
                mc_raw=ctx.job.get("mc_raw"),
                analysis_profile=ctx.job.get("analysis_profile"),
                skip_vision=has_vision,
            )
            # If server already provided vision data (gap-fill), populate mc_raw
            if has_vision:
                vd = ctx.job["vision_data"]
                item.mc_raw = vd
                item.vision_result = vd
                # Also update context metadata with server-provided vision data
                ctx.metadata.update(vd)
                ctx.vision_fields = vd
            phase_items.append(item)

        if [it for it in phase_items if not it.skip_vision]:
            logger.info(
                f"Phase V: {sum(1 for it in phase_items if not it.skip_vision)} "
                f"need VLM, {sum(1 for it in phase_items if it.skip_vision)} "
                f"already have MC (server gap-fill)"
            )

        # No-op storage — results accumulate on PhaseItem, uploaded later
        class _AccumulatorStorage:
            def save_vision(self, item, result): pass
            def save_vv(self, item, vv_vec, structure_vec=None): pass
            def save_mv(self, item, mv_vec, text): pass
            def flush(self): pass

        # Worker progress reporter → IPC _notify bridge
        class _WorkerProgress:
            _PHASE_MAP = {"vision": "vision", "vv": "embed_vv", "mv": "embed_mv"}

            def __init__(self, cb, daemon):
                self._cb = cb
                self._daemon = daemon

            def phase_start(self, phase, count):
                mapped = self._PHASE_MAP.get(phase, phase)
                self._daemon._current_phase = mapped
                if self._daemon.verbose_log:
                    logger.info(f"{self._daemon._log_prefix} Phase {mapped} START ({count} files, batch_capacity={self._daemon.batch_capacity})")
                try:
                    self._daemon._heartbeat()
                except Exception:
                    pass
                _notify(self._cb, "phase_start", {"phase": mapped, "count": count})

            def file_done(self, phase, index, count, file_name, success):
                mapped = self._PHASE_MAP.get(phase, phase)
                self._daemon._current_phase = mapped
                self._daemon._current_file = file_name
                if self._daemon.verbose_log:
                    status = "OK" if success else "FAIL"
                    logger.info(f"[WORKER] {mapped} [{index+1}/{count}] {file_name} → {status}")
                _notify(self._cb, "file_done", {
                    "phase": mapped, "file_name": file_name,
                    "index": index + 1, "count": count, "success": success,
                })

            def phase_complete(self, phase, elapsed_s):
                mapped = self._PHASE_MAP.get(phase, phase)
                pc = self._daemon._phase_counts
                if self._daemon.verbose_log:
                    logger.info(f"{self._daemon._log_prefix} Phase {mapped} DONE in {elapsed_s:.1f}s (totals: MC:{pc['mc']} VV:{pc['vv']} MV:{pc['mv']})")
                try:
                    self._daemon._heartbeat()
                except Exception:
                    pass
                mapped = self._PHASE_MAP.get(phase, phase)
                _notify(self._cb, "phase_complete", {
                    "phase": mapped, "count": 0,
                    "elapsed_s": round(elapsed_s, 2),
                    "files_per_min": 0,
                })

        models = ModelManager()
        storage = _AccumulatorStorage()
        batch_strategy = FixedBatchStrategy(vision=1, vv=8, mv=16)
        runner = PhaseRunner(
            models=models,
            storage=storage,
            batch_strategy=batch_strategy,
            stop_check=lambda: self._stop_requested,
            progress=_WorkerProgress(progress_callback, self),
        )

        n = len(active)

        # Phase V (MC)
        t_v = time.perf_counter()
        phase_items = runner.run_vision(phase_items)
        elapsed_vision = time.perf_counter() - t_v
        fpm_vision = (n / elapsed_vision * 60) if elapsed_vision > 0 else 0
        self._phase_counts["mc"] += n
        # Update throughput: files / elapsed since batch start
        _fpm = round(n / (time.perf_counter() - t_batch) * 60, 1)
        self._batch_throughput = _fpm
        self._phase_throughput["mc"] = _fpm

        # Emit VLM model info + per-file error diagnostics to IPC
        vision_errors = [
            (it.file_name, it.error)
            for it in phase_items if it.error
        ]
        if vision_errors:
            _notify(progress_callback, "phase_errors", {
                "phase": "vision",
                "total": len(phase_items),
                "failed": len(vision_errors),
                "errors": [
                    {"file": name, "error": err}
                    for name, err in vision_errors[:20]  # cap at 20
                ],
            })

        if self._stop_requested:
            logger.info("Stop requested after Vision phase — aborting batch")
            return self._finalize_batch(contexts, progress_callback, t_batch, interrupted=True)

        # Phase VV
        t_vv = time.perf_counter()
        phase_items = runner.run_vv(phase_items)
        elapsed_vv = time.perf_counter() - t_vv
        fpm_vv = (n / elapsed_vv * 60) if elapsed_vv > 0 else 0
        self._phase_counts["vv"] += n
        self._batch_throughput = round(n / (time.perf_counter() - t_batch) * 60, 1)
        self._phase_throughput["vv"] = round(fpm_vv, 1)

        if self._stop_requested:
            logger.info("Stop requested after VV phase — aborting batch")
            return self._finalize_batch(contexts, progress_callback, t_batch, interrupted=True)

        # Phase MV
        t_mv = time.perf_counter()
        phase_items = runner.run_mv(phase_items)
        elapsed_mv = time.perf_counter() - t_mv
        self._phase_counts["mv"] += n
        fpm_mv = (n / elapsed_mv * 60) if elapsed_mv > 0 else 0
        self._batch_throughput = round(n / (time.perf_counter() - t_batch) * 60, 1)
        self._phase_throughput["mv"] = round(fpm_mv, 1)

        # Map PhaseItem results back to _JobContext for upload
        for i, item in enumerate(phase_items):
            ctx = active[i]
            if item.vision_result and not ctx.vision_fields:
                fields = _vision_result_to_fields(item.vision_result)
                if fields:
                    ctx.metadata.update(fields)
                    ctx.vision_fields = fields
            if item.vv_embedding is not None:
                ctx.vv_vec = item.vv_embedding
            if item.structure_embedding is not None:
                ctx.structure_vec = item.structure_embedding
            if item.mv_embedding is not None:
                ctx.mv_vec = item.mv_embedding
            if item.error and not ctx.failed:
                ctx.failed = True
                ctx.error = item.error

        # ── Upload all results ──
        t_phase = time.perf_counter()
        results = []
        for ctx in contexts:
            job_id = ctx.job["job_id"]
            file_id = ctx.job["file_id"]

            if ctx.failed:
                self.uploader.fail_job(job_id, ctx.error, ctx.error_code)
                self._total_failed += 1
                results.append((job_id, False, ctx.error or "unknown error"))
                continue

            success = self.uploader.complete_job(
                job_id,
                metadata=ctx.metadata,
                vv_vec=ctx.vv_vec,
                mv_vec=ctx.mv_vec,
                structure_vec=ctx.structure_vec,
            )

            # Upload thumbnail to server
            if ctx.thumb_path and Path(ctx.thumb_path).exists():
                self.uploader.upload_thumbnail(file_id, ctx.thumb_path)

            if success:
                self._total_completed += 1
            else:
                self._total_failed += 1
            results.append((job_id, success, ""))

            _notify(progress_callback, "job_upload", {
                "job_id": job_id, "success": success,
                "file_name": Path(ctx.job["file_path"]).name,
            })

            # Clean up temp files (no cache — server manages downloads)
            if self.storage_mode == "server_upload" and ctx.local_path != ctx.job["file_path"]:
                try:
                    Path(ctx.local_path).unlink(missing_ok=True)
                except Exception:
                    pass

        elapsed_upload = time.perf_counter() - t_phase

        # Emit total batch timing
        total_elapsed = elapsed_parse + elapsed_vision + elapsed_vv + elapsed_mv + elapsed_upload
        total_fpm = (len(contexts) / total_elapsed * 60) if total_elapsed > 0 else 0
        self._batch_throughput = round(total_fpm, 1)
        _notify(progress_callback, "batch_complete", {
            "count": len(contexts),
            "elapsed_s": round(total_elapsed, 2),
            "files_per_min": round(total_fpm, 1),
            "phase_times": {
                "parse": round(elapsed_parse, 2),
                "vision": round(elapsed_vision, 2),
                "embed_vv": round(elapsed_vv, 2),
                "embed_mv": round(elapsed_mv, 2),
                "upload": round(elapsed_upload, 2),
            },
            "phase_fpm": {
                "parse": round(fpm_parse, 1),
                "vision": round(fpm_vision, 1),
                "embed_vv": round(fpm_vv, 1),
                "embed_mv": round(fpm_mv, 1),
            },
        })

        self._clear_current()

        # GPU memory cleanup
        gc.collect()
        self._try_empty_gpu_cache()

        return results

    def _finalize_batch(
        self, contexts, progress_callback, t_batch, interrupted=False
    ) -> list:
        """Finalize a batch — upload completed results, fail the rest.

        Called on normal completion or when stop is requested mid-batch.
        Already-completed phase results are uploaded; incomplete jobs are
        released back to the queue so other workers can pick them up.
        """
        results = []
        for ctx in contexts:
            job_id = ctx.job["job_id"]
            if ctx.failed or (interrupted and not ctx.metadata):
                # No useful work done — fail the job so it returns to queue
                err = ctx.error or "Interrupted by stop request"
                self.uploader.fail_job(
                    job_id, err,
                    ctx.error_code)
                self._total_failed += 1
                results.append((job_id, False, err))
            elif interrupted:
                # Partial work done — fail to return to queue for re-processing
                self.uploader.fail_job(
                    job_id, "Interrupted by stop request", "INTERRUPTED")
                self._total_failed += 1
                results.append((job_id, False, "Interrupted by stop request"))
            else:
                results.append((job_id, True, ""))

        _notify(progress_callback, "batch_complete", {
            "count": len(contexts),
            "elapsed_s": round(time.perf_counter() - t_batch, 2),
            "files_per_min": 0,
            "interrupted": interrupted,
        })

        self._clear_current()
        gc.collect()
        self._try_empty_gpu_cache()

        if interrupted:
            logger.info(
                f"Batch interrupted: {len(results)} jobs returned to queue"
            )

        return results

    def _process_batch_mc(self, jobs: list, progress_callback=None) -> list:
        """MC mode: VLM stays loaded, only generate MC (caption/tags).

        Server handles Parse (ParseAheadPool); VV/MV workers handle embedding.
        Worker only runs Phase V (VLM) and uploads vision fields.

        VLM is NOT unloaded between batches — stays resident for speed.
        """
        def _log(msg, level="info"):
            getattr(logger, level)(msg)
            _notify(progress_callback, "diag_log", {"message": msg, "level": level})

        _log(f"[MC] === START batch: {len(jobs)} jobs ===")

        # Build job contexts — all jobs should be pre-parsed in mc mode
        contexts = []
        for job in jobs:
            ctx = _JobContext(job=job)
            file_id = job.get("file_id")
            file_name = Path(job.get("file_path", "")).name
            ctx.metadata = dict(job.get("metadata", {}))

            # Resolve thumbnail for VLM processing
            server_thumb = job.get("thumb_path")

            if self.storage_mode == "shared_fs":
                if server_thumb and Path(server_thumb).exists():
                    ctx.thumb_path = server_thumb
                else:
                    ctx.local_path = self._get_downloaded(job)
                    ctx.thumb_path = ctx.local_path
            else:
                if server_thumb and Path(server_thumb).exists():
                    ctx.thumb_path = server_thumb
                else:
                    # Use prefetch cache if available (downloaded in background)
                    cached = self._get_downloaded(job)
                    if cached:
                        ctx.thumb_path = cached
                    else:
                        # Fallback: synchronous thumbnail download
                        thumb = self._resolve_thumbnail(job)
                        ctx.thumb_path = thumb

            if not ctx.thumb_path or not Path(ctx.thumb_path).exists():
                ctx.failed = True
                ctx.error_code = "THUMB_MISSING"
                ctx.error = f"THUMB_MISSING: {file_name} (server_thumb={server_thumb})"
                _log(f"[MC] FAIL {file_name}: {ctx.error}", "warning")
                _notify(progress_callback, "file_error", {
                    "file_name": file_name, "error": ctx.error,
                })
            contexts.append(ctx)

        active = [c for c in contexts if not c.failed]
        _log(f"[MC] Validation: {len(active)} active, {len(contexts)-len(active)} failed / {len(contexts)} total")

        # Report actual processing start for new system tasks
        for c in active:
            t_id = c.job.get("task_id") if isinstance(getattr(c, 'job', None), dict) else None
            if t_id:
                self._report_task_start(t_id, "mc")

        # ── Phase V: Vision/MC only (VLM, 1-by-1) ──
        elapsed_vision = self._run_vision_phase(active, progress_callback)
        mc_fpm = round(len(active) / elapsed_vision * 60, 1) if elapsed_vision > 0 else 0
        self._batch_throughput = mc_fpm
        self._phase_throughput["mc"] = mc_fpm
        _log(f"[MC] Vision phase done: {elapsed_vision:.1f}s ({mc_fpm}/m)")

        # Check how many succeeded in vision
        vision_ok = sum(1 for c in active if not c.failed and c.vision_fields)
        vision_fail = len(active) - vision_ok
        if vision_fail > 0:
            _log(f"[MC] Vision results: {vision_ok} ok, {vision_fail} fail", "warning")

        # NOTE: VLM is NOT unloaded in mc mode — stays resident

        # ── Upload MC results (skip already-saved by immediate save) ──
        t_upload = time.perf_counter()
        results = []
        for ctx in contexts:
            job_id = ctx.job["job_id"]
            file_name = Path(ctx.job.get("file_path", "")).name

            # Already saved during vision phase (transport mode)
            if getattr(ctx, '_saved', False):
                self._total_completed += 1
                results.append((job_id, True, ""))
                continue

            if ctx.failed:
                err = ctx.error or "unknown error"
                self.uploader.fail_job(job_id, err, ctx.error_code)
                self._total_failed += 1
                results.append((job_id, False, err))
                continue

            if not ctx.vision_fields:
                err = f"VLM returned empty vision_fields for {file_name}"
                _log(f"[MC] FAIL {file_name}: {err}", "warning")
                self.uploader.fail_job(job_id, err, "VLM_EMPTY")
                self._total_failed += 1
                results.append((job_id, False, err))
                continue

            # Save MC vision fields
            task_id = ctx.job.get("task_id") if isinstance(getattr(ctx, 'job', None), dict) else None
            file_id = ctx.job.get("file_id")
            if self.transport and file_id:
                upload_result = self.transport.save_vision(file_id, ctx.vision_fields)
            elif task_id:
                upload_result = self.uploader.save_vision_fields(file_id, ctx.vision_fields)
            else:
                upload_result = self.uploader.complete_mc(job_id, ctx.vision_fields)
            self._report_task_phase(task_id or job_id, "mc", upload_result is True,
                                    None if upload_result is True else str(upload_result),
                                    elapsed_s=getattr(ctx, '_inference_elapsed', None))
            if upload_result is True:
                self._total_completed += 1
                self._phase_counts["mc"] += 1
                results.append((job_id, True, ""))
            else:
                err = f"MC upload failed: {upload_result}" if isinstance(upload_result, str) else "MC upload rejected"
                _log(f"[MC] UPLOAD FAIL {file_name}: {err}", "warning")
                self._total_failed += 1
                results.append((job_id, False, err))

            _notify(progress_callback, "job_upload", {
                "job_id": job_id, "success": upload_result is True,
                "file_name": Path(ctx.job["file_path"]).name,
            })

            # Clean up temp files (no cache — server manages downloads)
            if self.storage_mode == "server_upload" and ctx.local_path != ctx.job["file_path"]:
                try:
                    Path(ctx.local_path).unlink(missing_ok=True)
                except Exception:
                    pass

        elapsed_upload = time.perf_counter() - t_upload
        total_elapsed = elapsed_vision + elapsed_upload
        total_fpm = (len(contexts) / total_elapsed * 60) if total_elapsed > 0 else 0
        fpm_vision = (len(active) / elapsed_vision * 60) if elapsed_vision > 0 else 0

        _notify(progress_callback, "batch_complete", {
            "count": len(contexts),
            "elapsed_s": round(total_elapsed, 2),
            "files_per_min": round(total_fpm, 1),
            "phase_times": {
                "vision": round(elapsed_vision, 2),
                "upload": round(elapsed_upload, 2),
            },
            "phase_fpm": {
                "vision": round(fpm_vision, 1),
            },
        })

        self._clear_current()
        return results

    # ── Model Unload Helpers ──────────────────────────────────

    def _unload_vlm(self):
        """Unload VLM to free GPU memory between phases."""
        try:
            from backend.vision.vision_factory import get_vision_analyzer, VisionAnalyzerFactory
            analyzer = get_vision_analyzer()
            model_id = getattr(analyzer, 'model_id', '?')
            if hasattr(analyzer, 'unload_model'):
                analyzer.unload_model()
            VisionAnalyzerFactory.reset()
            logger.info(f"VLM unloaded ({model_id})")
        except Exception as e:
            logger.warning(f"VLM unload error: {e}")
        gc.collect()
        self._try_empty_gpu_cache()

    # ── Single-Role Processing: VV-only, MV-only ──
    # _process_batch_parse + _save_parse_results removed
    # Parse is server-only (FileTaskParsePool)

    def _process_batch_vv_only(self, jobs: list, progress_callback=None) -> list:
        """VV-only mode: SigLIP2 stays loaded, generates visual vectors only."""
        from backend.vector.siglip2_encoder import SigLIP2Encoder
        from PIL import Image as PILImage

        self._current_phase = "vv"

        def _log(msg, level="info"):
            """Log to both Python logger AND IPC (via progress_callback)."""
            getattr(logger, level)(msg)
            _notify(progress_callback, "diag_log", {"message": msg, "level": level})

        _log(f"[VV] === START batch: {len(jobs)} jobs ===")

        results = []
        active = []
        for idx, job in enumerate(jobs):
            file_id = job.get("file_id")
            job_id = job.get("job_id")
            file_name = job.get("file_path", "").split("/")[-1].split("\\")[-1]
            self._current_file = file_name
            self._current_job_id = job_id

            # VV is MC-independent — only needs thumbnail, not mc_caption
            has_thumb_path = bool(job.get("thumb_path"))
            thumb = self._resolve_thumbnail(job)
            if not thumb:
                err = f"THUMB_FAIL: file_id={file_id}, thumb_path={job.get('thumb_path')}"
                _log(f"[VV] FAIL {file_name}: {err}", "warning")
                task_id = job.get("task_id")
                if task_id:
                    self._report_task_phase(task_id, "vv", False, err)
                else:
                    self.uploader.fail_job(job_id, err)
                results.append((job_id, False, err))
                _notify(progress_callback, "file_error", {"file_name": file_name, "error": err})
                continue
            active.append({"job": job, "thumb": thumb})

        _log(f"[VV] Validation: {len(active)} active, {len(results)} pre-failed / {len(jobs)} total")

        if not active:
            _log(f"[VV] ALL {len(jobs)} jobs failed validation", "warning")
            return results

        _notify(progress_callback, "phase_start", {"phase": "embed_vv", "count": len(active)})

        # SigLIP2 stays loaded across batches (class-level singleton)
        try:
            if WorkerDaemon._vv_encoder is None:
                _log("[VV] Loading SigLIP2 model...")
                WorkerDaemon._vv_encoder = SigLIP2Encoder()
            encoder = WorkerDaemon._vv_encoder
            model_name = getattr(encoder, 'model_name', None) or type(encoder).__name__
            _log(f"[VV] SigLIP2 ready: {model_name}")
        except Exception as e:
            err = f"SigLIP2 load failed: {e}"
            _log(f"[VV] {err}", "error")
            for ctx in active:
                job_id = ctx["job"]["job_id"]
                self.uploader.fail_job(job_id, err)
                results.append((job_id, False, err))
            _notify(progress_callback, "phase_errors", {
                "phase": "embed_vv", "total": len(active), "failed": len(active),
                "errors": [{"file": Path(c["job"].get("file_path","")).name, "error": err} for c in active[:5]],
            })
            return results

        # Report actual processing start for new system tasks
        for ctx in active:
            t_id = ctx["job"].get("task_id")
            if t_id:
                self._report_task_start(t_id, "vv")

        import time as _time
        t_phase = _time.perf_counter()
        processed = 0
        vv_batch = 8

        for i in range(0, len(active), vv_batch):
            chunk = active[i:i + vv_batch]
            images = []
            valid_ctxs = []
            for ctx in chunk:
                try:
                    img = PILImage.open(ctx["thumb"]).convert("RGB")
                    images.append(img)
                    valid_ctxs.append(ctx)
                except Exception as e:
                    job_id = ctx["job"]["job_id"]
                    file_name = Path(ctx["job"].get("file_path", "")).name
                    err = f"Image open failed: {e}"
                    logger.warning(f"[VV-ONLY] {file_name}: {err}")
                    self.uploader.fail_job(job_id, err)
                    results.append((job_id, False, err))
                    _notify(progress_callback, "file_done", {
                        "phase": "embed_vv", "file_name": file_name,
                        "index": processed + 1, "count": len(active), "success": False,
                    })
                    processed += 1

            if not images:
                continue

            try:
                _t_vv_batch = time.perf_counter()
                vv_vectors = encoder.encode_image_batch(images)
                _vv_batch_elapsed = time.perf_counter() - _t_vv_batch
                _vv_per_file = _vv_batch_elapsed / len(images) if images else 0
            except Exception as e:
                err = f"SigLIP2 batch encode failed: {e}"
                logger.error(f"[VV-ONLY] {err}", exc_info=True)
                for ctx in valid_ctxs:
                    job_id = ctx["job"]["job_id"]
                    file_name = Path(ctx["job"].get("file_path", "")).name
                    self.uploader.fail_job(job_id, err)
                    results.append((job_id, False, err))
                    _notify(progress_callback, "file_done", {
                        "phase": "embed_vv", "file_name": file_name,
                        "index": processed + 1, "count": len(active), "success": False,
                    })
                    processed += 1
                # Close images
                for img in images:
                    try: img.close()
                    except: pass
                continue

            # Close images
            for img in images:
                try: img.close()
                except: pass

            # Batch upload all VV vectors at once
            batch_items = []
            failed_items = []
            for ctx, vec in zip(valid_ctxs, vv_vectors):
                job_id = ctx["job"]["job_id"]
                file_name = Path(ctx["job"].get("file_path", "")).name
                self._current_file = file_name
                if vec is not None:
                    batch_items.append({"job_id": job_id, "vec": vec, "ctx": ctx})
                else:
                    err = "VV encoding returned None"
                    logger.warning(f"[VV-ONLY] {file_name}: {err}")
                    self.uploader.fail_job(job_id, err)
                    failed_items.append((job_id, False, err))
                    _notify(progress_callback, "file_done", {
                        "phase": "embed_vv", "file_name": file_name,
                        "index": processed + 1, "count": len(active), "success": False,
                    })
                    processed += 1

            if batch_items:
                _log(f"[VV] Uploading {len(batch_items)} vectors...")
                # Save VV vectors
                batch_results = []
                for it in batch_items:
                    file_id = it["ctx"]["job"].get("file_id")
                    if self.transport:
                        ok = self.transport.save_vv(file_id, it["vec"])
                    else:
                        ok = self.uploader.save_vv_vector(file_id, it["vec"])
                    batch_results.append({"ok": ok is True})
                n_upload_ok = sum(1 for r in batch_results if (r.get("ok") if isinstance(r, dict) else bool(r)))
                _log(f"[VV] Upload: {n_upload_ok}/{len(batch_results)} ok")
                for it, batch_result in zip(batch_items, batch_results):
                    file_name = Path(it["ctx"]["job"].get("file_path", "")).name
                    if isinstance(batch_result, dict):
                        ok = bool(batch_result.get("ok"))
                        err = batch_result.get("error", "") if not ok else ""
                    else:
                        ok = bool(batch_result)
                        err = "" if ok else "VV upload failed (server rejected)"
                    if not ok:
                        _log(f"[VV] UPLOAD FAIL {file_name}: {batch_result!r}", "warning")
                    results.append((it["job_id"], ok, err))
                    # Report to Analysis Job system (with per-file inference time)
                    t_id = it["ctx"]["job"].get("task_id") or it["job_id"]
                    self._report_task_phase(t_id, "vv", ok, err if not ok else None,
                                            elapsed_s=_vv_per_file if ok else None)
                    if ok:
                        self._phase_counts["vv"] += 1
                    _notify(progress_callback, "file_done", {
                        "phase": "embed_vv",
                        "file_name": file_name,
                        "index": processed + 1,
                        "count": len(active),
                        "success": ok,
                    })
                    processed += 1
            results.extend(failed_items)

        elapsed = _time.perf_counter() - t_phase
        fpm = (len(active) / elapsed * 60) if elapsed > 0 else 0

        # Emit error summary if any failures
        errors = [(e.split(":")[0] if ":" in e else "unknown", e) for _, s, e in results if not s and e]
        if errors:
            _notify(progress_callback, "phase_errors", {
                "phase": "embed_vv", "total": len(active), "failed": len(errors),
                "errors": [{"file": f, "error": e} for f, e in errors[:20]],
            })

        # Do NOT unload — SigLIP2 stays resident for next batch
        self._batch_throughput = round(fpm, 1)
        self._phase_throughput["vv"] = round(fpm, 1)
        _notify(progress_callback, "phase_complete", {
            "phase": "embed_vv", "count": len(active),
            "elapsed_s": round(elapsed, 2), "files_per_min": round(fpm, 1),
        })
        return results

    def _process_batch_mv_only(self, jobs: list, progress_callback=None) -> list:
        """MV-only mode: Qwen3-Embedding stays loaded, generates meaning vectors only."""
        self._current_phase = "mv"
        from backend.vector.text_embedding import get_text_embedding_provider, build_document_text

        _notify = lambda cb, evt, data: cb(evt, data) if cb else None

        def _log(msg, level="info"):
            getattr(logger, level)(msg)
            _notify(progress_callback, "diag_log", {"message": msg, "level": level})

        _log(f"[MV] === START batch: {len(jobs)} jobs ===")

        results = []
        active = []
        for job in jobs:
            file_name = Path(job.get("file_path", "")).name
            job_id = job.get("job_id") or job.get("task_id")
            task_id = job.get("task_id")

            # Get MC data from the claim payload or fetch it from the server.
            vision_data = job.get("vision_data", {})
            mc_caption = vision_data.get("mc_caption", "")

            # New system: no vision_data in claim — fetch from files DB
            if not mc_caption and task_id:
                file_id = job.get("file_id")
                try:
                    if self.transport:
                        mc_data = self.transport.get_mc_data(file_id)
                    else:
                        resp = self._authed_request("get", f"{self.server_url}/api/v1/files/{file_id}/mc")
                        mc_data = resp.json() if resp.status_code == 200 else None
                    if mc_data:
                        mc_caption = mc_data.get("mc_caption", "")
                        vision_data = mc_data
                except Exception as e:
                    _log(f"[MV] Failed to fetch MC for {file_name}: {e}", "warning")

            if not mc_caption:
                err = f"NO_MC_CAPTION: file has no MC data"
                _log(f"[MV] FAIL {file_name}: {err}", "warning")
                if task_id:
                    self._report_task_phase(task_id, "mv", False, err)
                else:
                    self.uploader.fail_job(job_id, err, "MODE_MISMATCH")
                results.append((job_id, False, err))
                _notify(progress_callback, "file_error", {"file_name": file_name, "error": err})
                continue

            ai_tags = vision_data.get("ai_tags", [])
            if isinstance(ai_tags, str):
                try:
                    ai_tags = json.loads(ai_tags)
                except (json.JSONDecodeError, TypeError):
                    ai_tags = []
            doc_text = build_document_text(mc_caption, ai_tags, facts={
                "image_type": vision_data.get("image_type", ""),
                "scene_type": vision_data.get("scene_type", ""),
                "art_style": vision_data.get("art_style", ""),
            })

            if task_id:
                self._report_task_start(task_id, "mv")
            active.append({"job": job, "text": doc_text})

        _log(f"[MV] Validation: {len(active)} active, {len(results)} failed / {len(jobs)} total")

        if not active:
            _log(f"[MV] ALL jobs failed validation", "warning")
            return results

        _notify(progress_callback, "batch_phase_start", {"phase": "embed_mv", "count": len(active)})

        _log("[MV] Loading Qwen3-Embedding...")
        provider = get_text_embedding_provider()
        _log(f"[MV] Embedding provider ready: {type(provider).__name__}")
        mv_batch = 16
        for i in range(0, len(active), mv_batch):
            chunk = active[i:i + mv_batch]
            texts = [ctx["text"] for ctx in chunk]

            try:
                _t_mv_batch = time.perf_counter()
                vecs = provider.encode_batch(texts)
                _mv_batch_elapsed = time.perf_counter() - _t_mv_batch
                _mv_per_file = _mv_batch_elapsed / len(texts) if texts else 0
            except Exception:
                _t_mv_batch = time.perf_counter()
                vecs = [provider.encode(t) for t in texts]
                _mv_batch_elapsed = time.perf_counter() - _t_mv_batch
                _mv_per_file = _mv_batch_elapsed / len(texts) if texts else 0

            # Batch upload
            batch_items = []
            failed_items = []
            for ctx, vec in zip(chunk, vecs):
                job_id = ctx["job"]["job_id"]
                self._current_file = Path(ctx["job"].get("file_path", "")).name
                if vec is not None:
                    batch_items.append({"job_id": job_id, "vec": vec, "ctx": ctx})
                else:
                    err = "MV encoding failed"
                    self.uploader.fail_job(job_id, err)
                    failed_items.append((job_id, False, err))

            if batch_items:
                _log(f"[MV] Uploading {len(batch_items)} vectors...")
                batch_results = []
                for it in batch_items:
                    file_id = it["ctx"]["job"].get("file_id")
                    if self.transport:
                        ok = self.transport.save_mv(file_id, it["vec"])
                    else:
                        ok = self.uploader.save_mv_vector(file_id, it["vec"])
                    batch_results.append(ok is True)
                n_ok = sum(1 for r in batch_results if r)
                _log(f"[MV] Upload: {n_ok}/{len(batch_results)} ok")
                for it, ok in zip(batch_items, batch_results):
                    err = "" if ok else "MV upload failed (server rejected)"
                    if not ok:
                        fn = Path(it["ctx"]["job"].get("file_path", "")).name
                        _log(f"[MV] UPLOAD FAIL {fn}", "warning")
                    results.append((it["job_id"], ok, err))
                    # Report to Analysis Job system (with per-file inference time)
                    t_id = it["ctx"]["job"].get("task_id") or it["job_id"]
                    self._report_task_phase(t_id, "mv", ok, err if not ok else None,
                                            elapsed_s=_mv_per_file if ok else None)
                    if ok:
                        self._phase_counts["mv"] += 1

            results.extend(failed_items)
            for ctx in chunk:
                _notify(progress_callback, "batch_file_done", {
                    "phase": "embed_mv",
                    "file_name": Path(ctx["job"].get("file_path", "")).name,
                    "index": len(results),
                    "count": len(active),
                    "success": results[-1][1],
                })

        # Do NOT unload — Qwen3-Embedding stays resident for next batch
        if active:
            total_elapsed = sum(getattr(a.get('_mv_elapsed', 0), '__float__', lambda: 0)() for a in active) if False else 0
            # Use batch time for throughput
            try:
                mv_fpm = round(len(active) / (_mv_batch_elapsed or 1) * 60, 1)
            except Exception:
                mv_fpm = 0
            self._batch_throughput = mv_fpm
            self._phase_throughput["mv"] = mv_fpm
        _notify(progress_callback, "batch_phase_complete", {"phase": "embed_mv", "count": len(active)})
        return results

    def _unload_vv(self):
        """Unload SigLIP2 encoder to free GPU memory."""
        if WorkerDaemon._vv_encoder is not None:
            if hasattr(WorkerDaemon._vv_encoder, 'unload'):
                WorkerDaemon._vv_encoder.unload()
            WorkerDaemon._vv_encoder = None
        gc.collect()
        self._try_empty_gpu_cache()
        logger.info("SigLIP2 unloaded")

    def _unload_mv(self):
        """Unload text embedding model to free GPU memory."""
        try:
            from backend.vector.text_embedding import reset_provider
            reset_provider()
        except Exception:
            pass
        gc.collect()
        self._try_empty_gpu_cache()
        logger.info("MV embedder unloaded")

    def _try_empty_gpu_cache(self):
        """Helper to clear GPU memory cache (CUDA, MPS, and MLX Metal)."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
        # MLX uses its own Metal buffer allocator, separate from torch.mps
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    # ── State Machine Callbacks ─────────────────────────────────

    def _on_enter_idle(self):
        """Called when transitioning to IDLE (schedule inactive).
        Unload all models to free GPU memory during off-hours."""
        logger.info("Entering IDLE: unloading all models for off-schedule period")
        self._clear_download_cache()
        self._unload_vlm()
        self._unload_vv()
        self._unload_mv()

    def _on_enter_active(self):
        """Called when transitioning to ACTIVE.
        Models will reload lazily on first use — no explicit preload needed."""
        logger.info("Entering ACTIVE: ready to process jobs (models load on demand)")

    def _on_enter_resting(self):
        """Called when transitioning to RESTING (throttle critical).
        Models should already be unloaded by the throttle handler."""
        logger.info("Entering RESTING: resource pressure critical, waiting for recovery")

    # ── Throttle Logic ────────────────────────────────────────

    def _check_throttle(self) -> str:
        """Collect metrics and determine current throttle level.

        Returns:
            One of 'normal', 'warning', 'danger', 'critical'.
        """
        try:
            from backend.worker.resource_monitor import collect_metrics, get_throttle_level
            metrics = collect_metrics()
            level = get_throttle_level(metrics)
        except Exception:
            level = "normal"

        if level != self._throttle_level:
            logger.info(f"Throttle level changed: {self._throttle_level} → {level}")
        self._throttle_level = level
        return level
