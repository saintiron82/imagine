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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
from backend.worker.result_uploader import ResultUploader

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


def _notify(callback, event_type: str, data: dict):
    """Call progress callback if provided."""
    if callback:
        try:
            callback(event_type, data)
        except Exception:
            pass


class WorkerDaemon:
    """Headless worker that processes jobs from the Imagine server."""

    def __init__(self):
        import requests

        self.server_url = get_server_url()
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        self.access_token = None
        self.refresh_token = None
        self.uploader = ResultUploader(self.session, self.server_url)
        self.storage_mode = get_storage_mode()
        self.tmp_dir = tempfile.mkdtemp(prefix="imagine_worker_")

        # Prefetch pool
        self.batch_capacity = get_batch_capacity()
        self.pool_size = self.batch_capacity * 2  # Target pool size
        self._job_pool = collections.deque()
        self._pool_lock = threading.Lock()

        # Session tracking
        self.session_id = None
        self._total_completed = 0
        self._total_failed = 0
        self._current_job_id = None
        self._current_file = None
        self._current_phase = None

        logger.info(
            f"Worker initialized: server={self.server_url}, mode={self.storage_mode}, "
            f"batch_capacity={self.batch_capacity}, pool_size={self.pool_size}"
        )

    # ── Authentication ─────────────────────────────────────────

    def set_tokens(self, access_token: str, refresh_token: str = None) -> bool:
        """Inject existing JWT tokens (skip login, reuse session from Electron)."""
        if not access_token:
            logger.error("No access token provided")
            return False

        self.access_token = access_token
        self.refresh_token = refresh_token
        self.session.headers["Authorization"] = f"Bearer {self.access_token}"
        logger.info("Tokens injected from existing session")
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
            # Try worker token re-exchange if available
            wt = getattr(self, '_worker_token_secret', None)
            if wt:
                return self.exchange_worker_token(wt)
            return self.login()

        try:
            resp = self.session.post(
                f"{self.server_url}/api/v1/auth/refresh",
                json={"refresh_token": self.refresh_token},
            )
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data["access_token"]
                self.refresh_token = data.get("refresh_token", self.refresh_token)
                self.session.headers["Authorization"] = f"Bearer {self.access_token}"
                logger.info("Token refreshed")
                return True
            else:
                logger.warning("Token refresh failed, re-authenticating...")
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
            if self._refresh_auth():
                resp = getattr(self.session, method)(url, **kwargs)
        return resp

    # ── Session Management ─────────────────────────────────────

    def _connect_session(self):
        """Register worker session with server."""
        try:
            resp = self._authed_request(
                "post",
                f"{self.server_url}/api/v1/workers/connect",
                json={
                    "worker_name": f"{socket.gethostname()}-worker",
                    "hostname": socket.gethostname(),
                    "batch_capacity": self.batch_capacity,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                self.session_id = data["session_id"]
                if data.get("pool_hint"):
                    self.pool_size = data["pool_hint"]
                logger.info(f"Session registered: id={self.session_id}, pool_hint={self.pool_size}")
            else:
                logger.warning(f"Session connect failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Session connect error: {e}")

    def _heartbeat(self) -> dict:
        """Send heartbeat and receive server commands."""
        if not self.session_id:
            return {}
        try:
            with self._pool_lock:
                pool_sz = len(self._job_pool)
            resp = self._authed_request(
                "post",
                f"{self.server_url}/api/v1/workers/heartbeat",
                json={
                    "session_id": self.session_id,
                    "jobs_completed": self._total_completed,
                    "jobs_failed": self._total_failed,
                    "current_job_id": self._current_job_id,
                    "current_file": self._current_file,
                    "current_phase": self._current_phase,
                    "pool_size": pool_sz,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("pool_hint"):
                    self.pool_size = data["pool_hint"]
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
        """Claim up to N jobs from the server."""
        if count <= 0:
            return []
        try:
            resp = self._authed_request(
                "post",
                f"{self.server_url}/api/v1/jobs/claim",
                json={"count": count},
            )
            if resp.status_code == 200:
                data = resp.json()
                jobs = data.get("jobs", [])
                if jobs:
                    logger.info(f"Claimed {len(jobs)} jobs (requested {count})")
                return jobs
            else:
                logger.warning(f"Claim failed: {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"Claim request failed: {e}")
            return []

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
            self.uploader.fail_job(job_id, f"Cannot access file: {file_path}")
            self._total_failed += 1
            self._clear_current()
            return False

        local_file = Path(local_path)
        if not local_file.exists():
            self.uploader.fail_job(job_id, f"File not found: {local_path}")
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
                self.uploader.fail_job(job_id, f"Parse failed for {local_file.name}")
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
            _cb("vision_done")

            # ── Phase E-VV: Visual Vector (SigLIP2) ──
            self._current_phase = "embed"
            _cb("embed_vv")
            self.uploader.report_progress(job_id, "embed")
            vv_vec, structure_vec = self._run_embed_vv(thumb_path)
            _cb("embed_vv_done")

            # ── Phase E-MV: Meaning Vector (Qwen3-Embedding) ──
            _cb("embed_mv")
            mv_vec = self._run_embed_mv(metadata)
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
            # Cleanup downloaded temp files
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
        """Get local file path — either shared FS path or download from server."""
        file_path = job["file_path"]
        file_id = job["file_id"]

        if self.storage_mode == "shared_fs":
            # Direct access via shared filesystem
            if Path(file_path).exists():
                return file_path
            logger.warning(f"Shared FS file not found: {file_path}")
            return None
        else:
            # Download from server
            return self.uploader.download_file(file_id, self.tmp_dir)

    # ── Pipeline Phases (reusing existing backend code) ────────

    def _run_parse(self, file_path: Path):
        """Phase P: Parse file and extract metadata."""
        try:
            from backend.pipeline.ingest_engine import ParserFactory, _build_mc_raw, _set_tier_metadata

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

            # Resolve thumbnail
            thumb_path = None
            if meta.thumbnail_url:
                tp = Path(meta.thumbnail_url)
                if tp.exists():
                    thumb_path = str(tp)

            # Build metadata dict for API upload
            metadata = self._meta_to_dict(meta)
            metadata["file_path"] = str(file_path)

            return metadata, thumb_path, meta

        except Exception as e:
            logger.error(f"Parse error: {e}", exc_info=True)
            return None

    def _run_vision(self, file_path: Path, thumb_path: str, meta) -> dict:
        """Phase V: Run VLM to generate MC (caption, tags, classification)."""
        if not thumb_path or not Path(thumb_path).exists():
            logger.warning(f"No thumbnail for vision: {file_path.name}")
            return {}

        try:
            from backend.vision.vision_factory import get_vision_analyzer
            from backend.pipeline.ingest_engine import _build_mc_raw
            from PIL import Image

            analyzer = get_vision_analyzer()
            thumb_img = Image.open(thumb_path)

            # Composite to RGB
            if thumb_img.mode == "RGBA":
                bg = Image.new("RGB", thumb_img.size, (255, 255, 255))
                bg.paste(thumb_img, mask=thumb_img.split()[3])
                thumb_img = bg
            elif thumb_img.mode != "RGB":
                thumb_img = thumb_img.convert("RGB")

            mc_raw = _build_mc_raw(meta)

            # Run 2-Stage vision
            vision_result = analyzer.analyze(thumb_img, mc_raw)

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
            logger.warning(f"Vision failed for {file_path.name}: {e}")
            return {}

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
                vv_vec = encoder.encode_image(img)

                if hasattr(encoder, 'encode_structure'):
                    structure_vec = encoder.encode_structure(img)

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

    def _meta_to_dict(self, meta) -> dict:
        """Convert AssetMeta dataclass to dict for API submission."""
        from dataclasses import asdict
        try:
            d = asdict(meta)
            # Remove None values and non-serializable types
            clean = {}
            for k, v in d.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                elif isinstance(v, (list, dict)):
                    clean[k] = v
                elif isinstance(v, Path):
                    clean[k] = str(v)
            return clean
        except Exception:
            # Fallback: manually extract key fields
            return {
                "file_name": getattr(meta, "file_name", ""),
                "file_path": getattr(meta, "file_path", ""),
                "format": getattr(meta, "format", ""),
                "file_size": getattr(meta, "file_size", 0),
            }

    # ── Batch Processing (Phase-Level with Sub-Batch Inference) ──

    def process_batch_phased(self, jobs: list, progress_callback=None) -> list:
        """Process a batch of jobs using phase-level sub-batch processing.

        Mirrors the local pipeline (ingest_engine.py) approach:
        - Parse: 1-by-1 (CPU-bound)
        - Vision (VLM): 1-by-1 (MLX/transformers batch_size=1)
        - VV (SigLIP2): real batch via encode_image_batch()
        - MV (Qwen3-Embedding): real batch via encode_batch()

        Each phase tracks elapsed time and reports files/min.

        Args:
            jobs: List of job dicts with job_id, file_id, file_path.
            progress_callback: Optional callback(event_type, data) for IPC progress.
        Returns:
            List of (job_id, success) tuples.
        """
        # Build job contexts and resolve file access
        contexts = []
        for job in jobs:
            ctx = _JobContext(job=job)
            ctx.local_path = self._resolve_file(job)
            if not ctx.local_path or not Path(ctx.local_path).exists():
                ctx.failed = True
                ctx.error = f"Cannot access file: {job['file_path']}"
            contexts.append(ctx)

        active = [c for c in contexts if not c.failed]

        # ── Phase P: Parse all (CPU, 1-by-1) ──
        t_phase = time.perf_counter()
        _notify(progress_callback, "phase_start", {"phase": "parse", "count": len(active)})
        for i, ctx in enumerate(active):
            self._current_phase = "parse"
            self._current_file = Path(ctx.job["file_path"]).name
            self.uploader.report_progress(ctx.job["job_id"], "parse")

            result = self._run_parse(Path(ctx.local_path))
            if result is None:
                ctx.failed = True
                ctx.error = f"Parse failed: {ctx.job['file_path']}"
            else:
                ctx.metadata, ctx.thumb_path, ctx.meta_obj = result

            _notify(progress_callback, "file_done", {
                "phase": "parse", "file_name": self._current_file,
                "index": i + 1, "count": len(active), "success": not ctx.failed,
            })
        elapsed_parse = time.perf_counter() - t_phase
        fpm_parse = (len(active) / elapsed_parse * 60) if elapsed_parse > 0 else 0
        _notify(progress_callback, "phase_complete", {
            "phase": "parse", "count": len(active),
            "elapsed_s": round(elapsed_parse, 2), "files_per_min": round(fpm_parse, 1),
        })

        active = [c for c in contexts if not c.failed]

        # ── Phase V: Vision (VLM, 1-by-1 — MLX batch_size=1) ──
        t_phase = time.perf_counter()
        _notify(progress_callback, "phase_start", {"phase": "vision", "count": len(active)})
        for i, ctx in enumerate(active):
            self._current_phase = "vision"
            self._current_file = Path(ctx.job["file_path"]).name
            self.uploader.report_progress(ctx.job["job_id"], "vision")

            vision_fields = self._run_vision(
                Path(ctx.local_path), ctx.thumb_path, ctx.meta_obj
            )
            if vision_fields:
                ctx.metadata.update(vision_fields)
                ctx.vision_fields = vision_fields

            _notify(progress_callback, "file_done", {
                "phase": "vision", "file_name": self._current_file,
                "index": i + 1, "count": len(active), "success": True,
            })
        elapsed_vision = time.perf_counter() - t_phase
        fpm_vision = (len(active) / elapsed_vision * 60) if elapsed_vision > 0 else 0
        _notify(progress_callback, "phase_complete", {
            "phase": "vision", "count": len(active),
            "elapsed_s": round(elapsed_vision, 2), "files_per_min": round(fpm_vision, 1),
        })

        # Unload VLM to free GPU memory for embedding phases
        self._unload_vlm()

        # ── Phase VV: SigLIP2 (real batch via encode_image_batch) ──
        t_phase = time.perf_counter()
        vv_batch_size = 8  # SigLIP2 sub-batch size
        _notify(progress_callback, "phase_start", {"phase": "embed_vv", "count": len(active)})

        # Load SigLIP2 encoder once
        from backend.vector.siglip2_encoder import SigLIP2Encoder
        from PIL import Image as PILImage
        if WorkerDaemon._vv_encoder is None:
            WorkerDaemon._vv_encoder = SigLIP2Encoder()
        encoder = WorkerDaemon._vv_encoder

        processed_vv = 0
        for chunk_start in range(0, len(active), vv_batch_size):
            chunk = active[chunk_start:chunk_start + vv_batch_size]
            images = []
            chunk_valid = []

            for ctx in chunk:
                if ctx.thumb_path and Path(ctx.thumb_path).exists():
                    try:
                        img = PILImage.open(ctx.thumb_path).convert("RGB")
                        images.append(img)
                        chunk_valid.append(ctx)
                    except Exception as e:
                        logger.warning(f"VV load failed: {ctx.job['file_path']}: {e}")

            if images:
                try:
                    vv_vectors = encoder.encode_image_batch(images)
                    for j, vec in enumerate(vv_vectors):
                        chunk_valid[j].vv_vec = vec
                        # Structure vector (if available)
                        if hasattr(encoder, 'encode_structure'):
                            chunk_valid[j].structure_vec = encoder.encode_structure(images[j])
                except Exception as e:
                    logger.warning(f"VV batch encode failed: {e}")
                    # Fallback: individual encoding
                    for j, img in enumerate(images):
                        try:
                            chunk_valid[j].vv_vec = encoder.encode_image(img)
                        except Exception:
                            pass
                del images

            processed_vv += len(chunk)
            self._current_phase = "embed"
            last_name = Path(chunk[-1].job["file_path"]).name if chunk else ""
            self._current_file = last_name
            _notify(progress_callback, "file_done", {
                "phase": "embed_vv", "file_name": last_name,
                "index": processed_vv, "count": len(active),
                "success": True, "batch_size": len(chunk),
            })
            gc.collect()

        elapsed_vv = time.perf_counter() - t_phase
        fpm_vv = (len(active) / elapsed_vv * 60) if elapsed_vv > 0 else 0
        _notify(progress_callback, "phase_complete", {
            "phase": "embed_vv", "count": len(active),
            "elapsed_s": round(elapsed_vv, 2), "files_per_min": round(fpm_vv, 1),
        })

        # Unload SigLIP2 to free GPU memory for MV phase
        self._unload_vv()

        # ── Phase MV: Qwen3-Embedding (real batch via encode_batch) ──
        t_phase = time.perf_counter()
        mv_batch_size = 16  # Text embedding sub-batch size
        _notify(progress_callback, "phase_start", {"phase": "embed_mv", "count": len(active)})

        from backend.vector.text_embedding import get_text_embedding_provider, build_document_text
        mv_provider = get_text_embedding_provider()

        # Prepare texts for all active contexts
        mv_items = []  # (ctx_index, text)
        for i, ctx in enumerate(active):
            mc_caption = ctx.metadata.get("mc_caption", "")
            ai_tags = ctx.metadata.get("ai_tags", "")
            if isinstance(ai_tags, list):
                ai_tags_str = ", ".join(str(t) for t in ai_tags)
            else:
                ai_tags_str = ai_tags or ""

            facts = {
                "image_type": ctx.metadata.get("image_type"),
                "scene_type": ctx.metadata.get("scene_type"),
                "art_style": ctx.metadata.get("art_style"),
            }
            doc_text = build_document_text(mc_caption, ai_tags, facts=facts)
            if doc_text:
                mv_items.append((i, doc_text))

        processed_mv = 0
        for chunk_start in range(0, len(mv_items), mv_batch_size):
            chunk = mv_items[chunk_start:chunk_start + mv_batch_size]
            texts = [text for _, text in chunk]

            try:
                if hasattr(mv_provider, 'encode_batch'):
                    vecs = mv_provider.encode_batch(texts)
                else:
                    vecs = [mv_provider.encode(t) for t in texts]

                for j, vec in enumerate(vecs):
                    ctx_idx = chunk[j][0]
                    active[ctx_idx].mv_vec = vec
            except Exception as e:
                logger.warning(f"MV batch encode failed: {e}, falling back to individual")
                for j, (ctx_idx, text) in enumerate(chunk):
                    try:
                        active[ctx_idx].mv_vec = mv_provider.encode(text)
                    except Exception:
                        pass

            processed_mv += len(chunk)
            self._current_phase = "embed"
            last_name = Path(active[chunk[-1][0]].job["file_path"]).name if chunk else ""
            self._current_file = last_name
            _notify(progress_callback, "file_done", {
                "phase": "embed_mv", "file_name": last_name,
                "index": processed_mv, "count": len(mv_items),
                "success": True, "batch_size": len(chunk),
            })
            gc.collect()

        elapsed_mv = time.perf_counter() - t_phase
        fpm_mv = (len(mv_items) / elapsed_mv * 60) if elapsed_mv > 0 else 0
        _notify(progress_callback, "phase_complete", {
            "phase": "embed_mv", "count": len(mv_items),
            "elapsed_s": round(elapsed_mv, 2), "files_per_min": round(fpm_mv, 1),
        })

        # Unload MV model
        self._unload_mv()

        # ── Upload all results ──
        t_phase = time.perf_counter()
        results = []
        for ctx in contexts:
            job_id = ctx.job["job_id"]
            file_id = ctx.job["file_id"]

            if ctx.failed:
                self.uploader.fail_job(job_id, ctx.error)
                self._total_failed += 1
                results.append((job_id, False))
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
            results.append((job_id, success))

            _notify(progress_callback, "job_upload", {
                "job_id": job_id, "success": success,
                "file_name": Path(ctx.job["file_path"]).name,
            })

            # Cleanup downloaded temp files
            if self.storage_mode == "server_upload" and ctx.local_path != ctx.job["file_path"]:
                try:
                    Path(ctx.local_path).unlink(missing_ok=True)
                except Exception:
                    pass

        elapsed_upload = time.perf_counter() - t_phase

        # Emit total batch timing
        total_elapsed = elapsed_parse + elapsed_vision + elapsed_vv + elapsed_mv + elapsed_upload
        total_fpm = (len(contexts) / total_elapsed * 60) if total_elapsed > 0 else 0
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

    # ── Model Unload Helpers ──────────────────────────────────

    def _unload_vlm(self):
        """Unload VLM to free GPU memory between phases."""
        try:
            from backend.vision.vision_factory import get_vision_analyzer, VisionAnalyzerFactory
            analyzer = get_vision_analyzer()
            if hasattr(analyzer, 'unload_model'):
                analyzer.unload_model()
            VisionAnalyzerFactory.reset()
        except Exception:
            pass
        gc.collect()
        self._try_empty_gpu_cache()
        logger.info("VLM unloaded")

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
        """Helper to clear GPU memory cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

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

                # Start background refill while processing
                refill_thread = threading.Thread(target=self._fill_pool, daemon=True)
                refill_thread.start()

                # Phase-level batch processing (all files through each phase)
                if not _shutdown:
                    self.process_batch_phased(batch)

                # Wait for refill to complete
                refill_thread.join(timeout=30)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                time.sleep(poll_interval)

        # Graceful shutdown
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
