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

import collections
import gc
import logging
import signal
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.worker.config import (
    get_server_url,
    get_worker_credentials,
    get_claim_batch_size,
    get_poll_interval,
    get_storage_mode,
    get_batch_capacity,
    get_heartbeat_interval,
)
from backend.utils.meta_helpers import meta_to_dict
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

        self.transport = transport  # None = HTTP mode (legacy compatible)
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

        # Prefetch pool
        self.batch_capacity = get_batch_capacity()
        self.pool_size = self.batch_capacity * 2  # Target pool size
        self._job_pool = collections.deque()
        self._pool_lock = threading.Lock()

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

        logger.info(
            f"Worker initialized: server={self.server_url}, mode={self.storage_mode}, "
            f"batch_capacity={self.batch_capacity}, pool_size={self.pool_size}"
        )

    # ── Authentication ─────────────────────────────────────────

    def set_tokens(self, access_token: str, refresh_token: str = None) -> bool:
        """Inject existing JWT tokens (skip login, reuse session from Electron).

        Empty token is allowed for embedded worker (localhost auto-admin).
        """
        self.access_token = access_token or ""
        self.refresh_token = refresh_token
        if self.access_token:
            self.session.headers["Authorization"] = f"Bearer {self.access_token}"
            logger.info("Tokens injected from existing session")
        else:
            # No token — localhost auto-admin (embedded worker)
            self.session.headers.pop("Authorization", None)
            logger.info("No token — using localhost auto-admin")
        return True

    def exchange_worker_token(self, token_secret: str) -> bool:
        """Exchange a worker token for JWT access/refresh tokens."""
        try:
            resp = self.session.post(
                f"{self.server_url}/api/v1/auth/worker-token",
                json={"token": token_secret},
            )
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data["access_token"]
                self.refresh_token = data.get("refresh_token")
                self._worker_token_secret = token_secret
                self.session.headers["Authorization"] = f"Bearer {self.access_token}"
                logger.info("Worker token exchanged successfully")
                return True
            else:
                logger.error(f"Worker token exchange failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Worker token exchange request failed: {e}")
            return False

    def login(self) -> bool:
        """Authenticate with the server and get JWT tokens."""
        creds = get_worker_credentials()
        if not (creds.get("username") or creds.get("email")) or not creds.get("password"):
            logger.error("Worker credentials not configured. Set IMAGINE_WORKER_EMAIL/PASSWORD or config.yaml worker section.")
            return False

        try:
            resp = self.session.post(
                f"{self.server_url}/api/v1/auth/login",
                json=creds,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data["access_token"]
                self.refresh_token = data.get("refresh_token")
                self.session.headers["Authorization"] = f"Bearer {self.access_token}"
                logger.info("Authenticated successfully")
                return True
            else:
                logger.error(f"Login failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Login request failed: {e}")
            return False

    def _refresh_auth(self) -> bool:
        """Refresh the access token using the refresh token or worker token."""
        if not self.refresh_token:
            logger.warning("[REFRESH] No refresh token — trying fallback login")
            wt = getattr(self, '_worker_token_secret', None)
            if wt:
                return self.exchange_worker_token(wt)
            return self.login()

        try:
            rt_preview = self.refresh_token[:16] + "..." if self.refresh_token else "(none)"
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
            else:
                logger.warning(f"[REFRESH] Failed: {resp.status_code} {resp.text[:200]}")
                wt = getattr(self, '_worker_token_secret', None)
                if wt:
                    return self.exchange_worker_token(wt)
                return self.login()
        except Exception:
            wt = getattr(self, '_worker_token_secret', None)
            if wt:
                return self.exchange_worker_token(wt)
            return self.login()

    def _authed_request(self, method: str, url: str, **kwargs):
        """Make request with automatic token refresh on 401."""
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
                if data.get("pool_hint"):
                    self.pool_size = data["pool_hint"]
                if data.get("batch_hint"):
                    self.batch_capacity = data["batch_hint"]
                if data.get("processing_mode"):
                    self.processing_mode = data["processing_mode"]
                # Set log prefix for worker identification (BUG-006)
                name = self.worker_name or f"{socket.gethostname()}-worker"
                self.worker_name = name
                self._log_prefix = f"[{name}]"
                logger.info(f"{self._log_prefix} Session registered: id={self.session_id}, pool_hint={self.pool_size}, batch={self.batch_capacity}, mode={self.processing_mode}")
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
            with self._pool_lock:
                pool_sz = len(self._job_pool)
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
                    "pool_size": pool_sz,
                    "resources": resources,
                    "throttle_level": throttle_level,
                    "worker_state": self._state_machine.state_name,
                    "batch_throughput": self._batch_throughput,
                    "phase_throughput": self._phase_throughput,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("pool_hint"):
                    self.pool_size = data["pool_hint"]
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
        """Claim tasks from new Analysis Job system, fallback to legacy."""
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
                        # Convert to legacy job format for compatibility
                        jobs = []
                        for t in tasks:
                            jobs.append({
                                "job_id": t["task_id"],
                                "file_id": t["file_id"],
                                "file_path": t["file_path"],
                                "task_id": t["task_id"],  # new system ID
                                "analysis_job_id": t.get("job_id"),
                            })
                        logger.info(f"{self._log_prefix} Claimed {len(jobs)} {phase} tasks (new API)")
                        return jobs
            except Exception as e:
                logger.warning(f"Task claim failed: {e}")
                return []

        # No valid phase — nothing to claim
        return []

    def _report_task_start(self, task_id: int, phase: str):
        """Report actual processing start to new Analysis Job system."""
        if not task_id:
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
        """Report phase completion to new Analysis Job system."""
        if not task_id:
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
        """Claim jobs using legacy batch size (for embedded worker compatibility)."""
        return self.claim_jobs_count(get_claim_batch_size())

    def _fill_pool(self):
        """Fill prefetch pool up to target size."""
        with self._pool_lock:
            current = len(self._job_pool)
        deficit = self.pool_size - current
        if deficit > 0:
            jobs = self.claim_jobs_count(deficit)
            if jobs:
                with self._pool_lock:
                    self._job_pool.extend(jobs)
                logger.info(f"Pool filled: +{len(jobs)}, total={len(self._job_pool)}/{self.pool_size}")

    def _take_batch(self) -> list:
        """Take up to batch_capacity jobs from pool."""
        batch = []
        with self._pool_lock:
            for _ in range(min(self.batch_capacity, len(self._job_pool))):
                batch.append(self._job_pool.popleft())
        return batch

    # ── Job Lifecycle ──────────────────────────────────────────

    def process_job(self, job: dict, progress_callback=None) -> bool:
        """Process a single job: parse → vision → embed → upload results.

        Args:
            job: Job dict with job_id, file_id, file_path.
            progress_callback: Optional callback(phase, file_name) for IPC progress.
                Phase values: parse, parse_done, vision, vision_done,
                              embed_vv, embed_vv_done, embed_mv, embed_mv_done.
        """
        job_id = job["job_id"]
        file_id = job["file_id"]
        file_path = job["file_path"]
        file_name = Path(file_path).name

        self._current_job_id = job_id
        self._current_file = file_name
        self._current_phase = "parse"

        logger.info(f"Processing job {job_id}: {file_path}")

        # Resolve file access
        local_path = self._resolve_file(job)
        if not local_path:
            self.uploader.fail_job(
                job_id, f"Cannot access file: {file_path}", "FILE_NOT_FOUND")
            self._total_failed += 1
            self._clear_current()
            return False

        local_file = Path(local_path)
        if not local_file.exists():
            self.uploader.fail_job(
                job_id, f"File not found: {local_path}", "FILE_NOT_FOUND")
            self._total_failed += 1
            self._clear_current()
            return False

        def _cb(phase):
            if progress_callback:
                progress_callback(phase, file_name)

        try:
            # ── Phase P: Parse ──
            self._current_phase = "parse"
            _cb("parse")
            self.uploader.report_progress(job_id, "parse")
            parse_result = self._run_parse(local_file)
            if parse_result is None:
                self.uploader.fail_job(
                    job_id, f"Parse failed for {local_file.name}", "PARSE_FAILED")
                self._total_failed += 1
                return False
            _cb("parse_done")

            metadata, thumb_path, meta_obj = parse_result

            # ── Phase V: Vision (MC generation) ──
            self._current_phase = "vision"
            _cb("vision")
            self.uploader.report_progress(job_id, "vision")
            vision_fields = self._run_vision(local_file, thumb_path, meta_obj)
            if vision_fields:
                metadata.update(vision_fields)
            self._phase_counts["mc"] += 1
            _cb("vision_done")

            # ── Phase E-VV: Visual Vector (SigLIP2) ──
            self._current_phase = "embed"
            _cb("embed_vv")
            self.uploader.report_progress(job_id, "embed")
            logger.info(f"Job {job_id}: Phase VV (SigLIP2) start")
            vv_vec, structure_vec = self._run_embed_vv(thumb_path)
            logger.info(f"Job {job_id}: Phase VV done")
            self._phase_counts["vv"] += 1
            _cb("embed_vv_done")

            # ── Phase E-MV: Meaning Vector (Qwen3-Embedding) ──
            _cb("embed_mv")
            logger.info(f"Job {job_id}: Phase MV (Qwen3-Embedding) start")
            mv_vec = self._run_embed_mv(metadata)
            logger.info(f"Job {job_id}: Phase MV done")
            self._phase_counts["mv"] += 1
            _cb("embed_mv_done")

            # ── Upload results ──
            success = self.uploader.complete_job(
                job_id,
                metadata=metadata,
                vv_vec=vv_vec,
                mv_vec=mv_vec,
                structure_vec=structure_vec,
            )

            # Upload thumbnail to server (dual storage)
            if thumb_path and Path(thumb_path).exists():
                self.uploader.upload_thumbnail(file_id, thumb_path)

            if success:
                self._total_completed += 1
                logger.info(f"Job {job_id} completed: {local_file.name}")
            else:
                self._total_failed += 1
            return success

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            self.uploader.fail_job(job_id, str(e))
            self._total_failed += 1
            return False
        finally:
            self._clear_current()
            # Clean up temp files (no cache — server manages downloads)
            if self.storage_mode == "server_upload" and local_path != file_path:
                try:
                    Path(local_path).unlink(missing_ok=True)
                except Exception:
                    pass

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
        return result

    # _resolve_download_ahead removed — workers don't access download pool.
    # Parse is server-side (FileTaskParsePool).

    def _resolve_thumbnail(self, job: dict) -> Optional[str]:
        """Download only the thumbnail for a pre-parsed job (~200KB instead of ~500MB).

        Uses _authed_request() for automatic JWT refresh on 401.
        """
        file_id = job.get("file_id")
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

    # ── Pipeline Phases (reusing existing backend code) ────────

    def _run_parse(self, file_path: Path):
        """Phase P: Parse file and extract metadata."""
        try:
            from backend.pipeline.ingest_engine import ParserFactory, _build_mc_raw, _set_tier_metadata, _normalize_paths

            parser = ParserFactory.get_parser(file_path)
            if not parser:
                logger.warning(f"No parser for: {file_path}")
                return None

            result = parser.parse(file_path)
            if not result.success:
                logger.warning(f"Parse failed: {result.errors}")
                return None

            meta = result.asset_meta

            # Content hash
            from backend.utils.content_hash import compute_content_hash
            try:
                meta.content_hash = compute_content_hash(file_path)
            except Exception:
                pass

            # Folder metadata from path
            parent = file_path.parent
            if parent.name and parent.name not in (".", ""):
                meta.folder_path = parent.name
                meta.folder_depth = 0
                meta.folder_tags = [parent.name]

            # Tier metadata
            _set_tier_metadata(meta)

            # Normalize storage_root and relative_path (required for folder stats)
            _normalize_paths(meta, file_path)

            # Resolve thumbnail
            thumb_path = None
            if meta.thumbnail_url:
                tp = Path(meta.thumbnail_url)
                if tp.exists():
                    thumb_path = str(tp)

            # Build metadata dict for API upload
            metadata = meta_to_dict(meta)
            metadata["file_path"] = str(file_path)

            return metadata, thumb_path, meta

        except Exception as e:
            logger.error(f"Parse error: {e}", exc_info=True)
            return None

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
            from backend.pipeline.ingest_engine import _build_mc_raw
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

            # Run 2-Stage vision
            try:
                vision_result = analyzer.analyze(thumb_img, mc_raw)
            finally:
                thumb_img.close()

            fields = {}
            if vision_result:
                # VLM returns 'caption' → map to 'mc_caption', 'tags' → 'ai_tags'
                # (same mapping as ingest_engine.py local pipeline)
                if isinstance(vision_result, dict):
                    if "caption" in vision_result:
                        fields["mc_caption"] = vision_result["caption"]
                    if "tags" in vision_result:
                        fields["ai_tags"] = vision_result["tags"]

                # Copy remaining vision fields directly
                for key in [
                    "mc_caption", "ai_tags", "ocr_text", "dominant_color",
                    "ai_style", "image_type", "art_style", "color_palette",
                    "scene_type", "time_of_day", "weather", "character_type",
                    "item_type", "ui_type", "structured_meta",
                ]:
                    val = vision_result.get(key) if isinstance(vision_result, dict) else getattr(vision_result, key, None)
                    if val is not None:
                        fields[key] = val

            return fields

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

            # Keep-alive heartbeat during long processing (prevents server job reclaim)
            if i > 0 and i % 5 == 0:
                try:
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
                if ctx.metadata:
                    ctx.metadata.update(vision_fields)
                ctx.vision_fields = vision_fields
            else:
                ctx.failed = True
                if vlm_err:
                    ctx.error = f"VLM error for {self._current_file}: {vlm_err}"
                else:
                    ctx.error = f"VLM returned empty MC for {self._current_file}"
                logger.warning(ctx.error)
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
                # PhaseRunner's vision_result has raw keys (caption, tags)
                # Convert to DB column names for metadata
                fields = {}
                vr = item.vision_result
                if isinstance(vr, dict):
                    if "caption" in vr:
                        fields["mc_caption"] = vr["caption"]
                    if "tags" in vr:
                        fields["ai_tags"] = vr["tags"]
                    for key in ("image_type", "art_style", "scene_type", "ocr_text",
                                "dominant_color", "character_type", "item_type", "ui_type"):
                        if vr.get(key) is not None:
                            fields[key] = vr[key]
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

        # ── Upload MC results (vision fields only, no vectors) ──
        t_upload = time.perf_counter()
        results = []
        for ctx in contexts:
            job_id = ctx.job["job_id"]
            file_name = Path(ctx.job.get("file_path", "")).name

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

            # New system (task_id) vs legacy (job_id)
            task_id = ctx.job.get("task_id") if isinstance(getattr(ctx, 'job', None), dict) else None
            if task_id:
                # New Analysis Job system: save MC to files DB directly + report
                upload_result = self.uploader.save_vision_fields(ctx.job.get("file_id"), ctx.vision_fields)
                self._report_task_phase(task_id, "mc", upload_result is True,
                                        None if upload_result is True else str(upload_result),
                                        elapsed_s=getattr(ctx, '_inference_elapsed', None))
            else:
                # Legacy: use job_queue-based complete_mc
                upload_result = self.uploader.complete_mc(job_id, ctx.vision_fields)
            if upload_result is True:
                self._total_completed += 1
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
                # Check if new system tasks (have task_id)
                has_task_ids = any(it["ctx"]["job"].get("task_id") for it in batch_items)
                if has_task_ids:
                    # New system: save vectors via /api/v1/files/{id}/vv
                    batch_results = []
                    for it in batch_items:
                        file_id = it["ctx"]["job"].get("file_id")
                        ok = self.uploader.save_vv_vector(file_id, it["vec"])
                        batch_results.append({"ok": ok is True})
                else:
                    batch_results = self.uploader.complete_vv_batch(
                        [{"job_id": it["job_id"], "vec": it["vec"]} for it in batch_items]
                    )
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

            # Get MC data: from vision_data (legacy) or fetch from server (new system)
            vision_data = job.get("vision_data", {})
            mc_caption = vision_data.get("mc_caption", "")

            # New system: no vision_data in claim — fetch from files DB via API
            if not mc_caption and task_id:
                try:
                    file_id = job.get("file_id")
                    resp = self._authed_request("get", f"{self.server_url}/api/v1/files/{file_id}/mc")
                    if resp.status_code == 200:
                        mc_data = resp.json()
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
                has_task_ids = any(it["ctx"]["job"].get("task_id") for it in batch_items)
                if has_task_ids:
                    batch_results = []
                    for it in batch_items:
                        file_id = it["ctx"]["job"].get("file_id")
                        ok = self.uploader.save_mv_vector(file_id, it["vec"])
                        batch_results.append(ok is True)
                else:
                    batch_results = self.uploader.complete_mv_batch(
                        [{"job_id": it["job_id"], "vec": it["vec"]} for it in batch_items]
                    )
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

    def _interruptible_sleep(self, seconds: float):
        """Sleep that wakes early on shutdown signal."""
        global _shutdown
        elapsed = 0.0
        while elapsed < seconds and not _shutdown:
            time.sleep(min(1.0, seconds - elapsed))
            elapsed += 1.0

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

    def _apply_throttle(self, level: str) -> bool:
        """Apply throttle actions based on the current level.

        Args:
            level: Throttle level from _check_throttle().

        Returns:
            True if processing should proceed, False if this iteration
            should be skipped (danger pause or critical halt).
        """
        if level == "normal":
            # Restore original batch capacity if it was reduced
            if self.batch_capacity < self._original_batch_capacity:
                self.batch_capacity = self._original_batch_capacity
                logger.info(f"Throttle normal: batch_capacity restored to {self.batch_capacity}")
            return True

        if level == "warning":
            # Reduce batch capacity to ~50% (minimum 1)
            reduced = max(1, self._original_batch_capacity // 2)
            if self.batch_capacity != reduced:
                logger.warning(f"Throttle warning: batch_capacity {self.batch_capacity} → {reduced}")
                self.batch_capacity = reduced
            return True

        if level == "danger":
            # Pause for N seconds, then allow retry
            try:
                from backend.utils.config import get_config
                cfg = get_config()
                wait_s = cfg.get("worker", {}).get("throttle", {}).get("danger_wait_seconds", 30)
            except Exception:
                wait_s = 30
            logger.warning(f"Throttle danger: pausing {wait_s}s before retry")
            for _ in range(wait_s):
                if _shutdown:
                    return False
                time.sleep(1)
            return False  # Skip this iteration, re-check on next loop

        if level == "critical":
            # Stop processing, unload all models, wait for recovery
            logger.error("Throttle critical: unloading all models, waiting for recovery")
            self._unload_vlm()
            self._unload_vv()
            self._unload_mv()
            # Wait in 10-second intervals, re-checking level
            for _ in range(6):  # Up to 60 seconds
                if _shutdown:
                    return False
                time.sleep(10)
                try:
                    from backend.worker.resource_monitor import collect_metrics, get_throttle_level
                    m = collect_metrics()
                    new_level = get_throttle_level(m)
                    if new_level != "critical":
                        logger.info(f"Throttle recovered from critical → {new_level}")
                        self._throttle_level = new_level
                        return False  # Re-enter loop to apply new level
                except Exception:
                    pass
            return False  # Still critical after 60s — skip this iteration

        return True

    # ── Main Loop ─────────────────────────────────────────────

    def run(self):
        """Main worker loop: login → connect → prefetch → process + heartbeat."""
        logger.info("Imagine Worker Daemon starting...")

        if not self.login():
            logger.error("Authentication failed. Exiting.")
            return

        # Register session with server
        self._connect_session()

        poll_interval = get_poll_interval()
        heartbeat_interval = get_heartbeat_interval()
        last_heartbeat = time.time()
        consecutive_empty = 0

        # Initial pool fill
        self._fill_pool()

        while not _shutdown:
            try:
                # ── Schedule + State Machine check ──
                scheduled_active = is_active_now()
                throttle = self._check_throttle()

                # Determine if there are pending jobs (pool + server)
                with self._pool_lock:
                    pool_count = len(self._job_pool)
                has_pending = pool_count > 0

                self._state_machine.update(
                    is_scheduled_active=scheduled_active,
                    throttle_level=throttle,
                    has_pending_jobs=has_pending,
                )

                current_state = self._state_machine.state
                if current_state in (WorkerState.IDLE, WorkerState.RESTING):
                    # Not active — skip job processing, just heartbeat and wait
                    if time.time() - last_heartbeat >= heartbeat_interval:
                        hb = self._heartbeat()
                        last_heartbeat = time.time()
                        cmd = hb.get("command")
                        if cmd in ("stop", "block"):
                            logger.info(f"Server command received: {cmd}")
                            break
                    state_label = current_state.value
                    logger.debug(f"Worker {state_label}: waiting {poll_interval}s")
                    for _ in range(poll_interval):
                        if _shutdown:
                            break
                        time.sleep(1)
                    continue

                # Heartbeat check
                if time.time() - last_heartbeat >= heartbeat_interval:
                    hb = self._heartbeat()
                    last_heartbeat = time.time()
                    # Process server commands
                    cmd = hb.get("command")
                    if cmd in ("stop", "block"):
                        logger.info(f"Server command received: {cmd}")
                        break

                # Take a batch from pool
                batch = self._take_batch()

                if not batch:
                    # Pool empty — try to fill
                    self._fill_pool()
                    batch = self._take_batch()

                if not batch:
                    # No jobs available anywhere
                    consecutive_empty += 1
                    wait = min(poll_interval * consecutive_empty, 300)  # Max 5 min
                    logger.info(f"No jobs available. Waiting {wait}s...")
                    for _ in range(wait):
                        if _shutdown:
                            break
                        time.sleep(1)
                        # Keep heartbeat alive during wait
                        if time.time() - last_heartbeat >= heartbeat_interval:
                            hb = self._heartbeat()
                            last_heartbeat = time.time()
                            if hb.get("command") in ("stop", "block"):
                                _shutdown_from_server = True
                                break
                    else:
                        continue
                    # Check if we broke out due to server command
                    if locals().get('_shutdown_from_server'):
                        break
                    continue

                consecutive_empty = 0

                # ── Throttle check before processing ──
                throttle = self._check_throttle()
                if not self._apply_throttle(throttle):
                    # Put jobs back into pool (not lost)
                    with self._pool_lock:
                        self._job_pool.extendleft(reversed(batch))
                    logger.info(f"Throttle ({throttle}): {len(batch)} jobs returned to pool")
                    continue

                # Start background refill while processing
                refill_thread = threading.Thread(target=self._fill_pool, daemon=True)
                refill_thread.start()

                # Prefetch file downloads for this batch
                self._prefetch_downloads(batch)

                # Phase-level batch processing (all files through each phase)
                if not _shutdown:
                    self.process_batch_phased(batch)
                    # Record job activity to reset idle timeout timer
                    self._state_machine.record_job_activity()

                # Wait for refill to complete
                refill_thread.join(timeout=30)

                # Rest after batch (configurable cooldown)
                if not _shutdown:
                    from backend.worker.config import get_rest_after_batch_s
                    rest_s = get_rest_after_batch_s()
                    if rest_s > 0:
                        logger.info(f"Resting {rest_s}s after batch")
                        self._interruptible_sleep(rest_s)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                time.sleep(poll_interval)

        # Graceful shutdown
        self._clear_download_cache()
        self._download_pool.shutdown(wait=False)
        self._disconnect_session()
        logger.info(f"Worker daemon shutting down. (completed={self._total_completed}, failed={self._total_failed})")
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        except Exception:
            pass


def main():
    """CLI entry point with optional --server/--token args."""
    import argparse

    parser = argparse.ArgumentParser(description="Imagine Worker Daemon")
    parser.add_argument("--server", type=str, help="Server URL (e.g., http://192.168.1.10:8000)")
    parser.add_argument("--token", type=str, help="Worker token (from admin panel)")
    args = parser.parse_args()

    daemon = WorkerDaemon()

    if args.server:
        daemon.server_url = args.server
        daemon.uploader.server_url = args.server

    if args.token:
        # Exchange worker token for JWT
        if not daemon.exchange_worker_token(args.token):
            logger.error("Worker token exchange failed. Exiting.")
            sys.exit(1)
        daemon.run()
    else:
        # Default: use env/config credentials
        daemon.run()


if __name__ == "__main__":
    main()
