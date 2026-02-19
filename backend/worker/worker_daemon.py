"""
Imagine Worker Daemon — headless distributed pipeline worker.

Polls the server for pending jobs, downloads images (or accesses shared FS),
runs the local pipeline (Parse → Vision → Embed), then uploads results.

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
import sys
import tempfile
import time
from pathlib import Path

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

        logger.info(f"Worker initialized: server={self.server_url}, mode={self.storage_mode}")

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
        """Refresh the access token using the refresh token."""
        if not self.refresh_token:
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
                logger.warning("Token refresh failed, re-logging in...")
                return self.login()
        except Exception:
            return self.login()

    def _authed_request(self, method: str, url: str, **kwargs):
        """Make request with automatic token refresh on 401."""
        import requests
        resp = getattr(self.session, method)(url, **kwargs)
        if resp.status_code == 401:
            if self._refresh_auth():
                resp = getattr(self.session, method)(url, **kwargs)
        return resp

    # ── Job Lifecycle ──────────────────────────────────────────

    def claim_jobs(self) -> list:
        """Claim pending jobs from the server."""
        batch_size = get_claim_batch_size()
        try:
            resp = self._authed_request(
                "post",
                f"{self.server_url}/api/v1/jobs/claim",
                json={"count": batch_size},
            )
            if resp.status_code == 200:
                data = resp.json()
                jobs = data.get("jobs", [])
                if jobs:
                    logger.info(f"Claimed {len(jobs)} jobs")
                return jobs
            else:
                logger.warning(f"Claim failed: {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"Claim request failed: {e}")
            return []

    def process_job(self, job: dict) -> bool:
        """Process a single job: parse → vision → embed → upload results."""
        job_id = job["job_id"]
        file_id = job["file_id"]
        file_path = job["file_path"]

        logger.info(f"Processing job {job_id}: {file_path}")

        # Resolve file access
        local_path = self._resolve_file(job)
        if not local_path:
            self.uploader.fail_job(job_id, f"Cannot access file: {file_path}")
            return False

        local_file = Path(local_path)
        if not local_file.exists():
            self.uploader.fail_job(job_id, f"File not found: {local_path}")
            return False

        try:
            # ── Phase P: Parse ──
            self.uploader.report_progress(job_id, "parse")
            parse_result = self._run_parse(local_file)
            if parse_result is None:
                self.uploader.fail_job(job_id, f"Parse failed for {local_file.name}")
                return False

            metadata, thumb_path, meta_obj = parse_result

            # ── Phase V: Vision (MC generation) ──
            self.uploader.report_progress(job_id, "vision")
            vision_fields = self._run_vision(local_file, thumb_path, meta_obj)
            if vision_fields:
                metadata.update(vision_fields)

            # ── Phase E: Embed (VV + MV) ──
            self.uploader.report_progress(job_id, "embed")
            vv_vec, mv_vec, structure_vec = self._run_embed(thumb_path, metadata)

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
                logger.info(f"Job {job_id} completed: {local_file.name}")
            return success

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            self.uploader.fail_job(job_id, str(e))
            return False
        finally:
            # Cleanup downloaded temp files
            if self.storage_mode == "server_upload" and local_path != file_path:
                try:
                    Path(local_path).unlink(missing_ok=True)
                except Exception:
                    pass

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

    def _run_embed(self, thumb_path: str, metadata: dict):
        """Phase E: Generate VV + MV vectors."""
        import numpy as np
        vv_vec = None
        mv_vec = None
        structure_vec = None

        # VV: SigLIP2 visual embedding
        if thumb_path and Path(thumb_path).exists():
            try:
                from backend.vector.siglip2_encoder import SigLIP2Encoder
                from PIL import Image

                encoder = SigLIP2Encoder.get_instance()
                img = Image.open(thumb_path).convert("RGB")
                vv_vec = encoder.encode_image(img)

                # Structure vector (if supported)
                if hasattr(encoder, 'encode_structure'):
                    structure_vec = encoder.encode_structure(img)

            except Exception as e:
                logger.warning(f"VV encoding failed: {e}")

        # MV: Qwen3-Embedding text meaning vector
        mc_caption = metadata.get("mc_caption", "")
        ai_tags = metadata.get("ai_tags", "")
        if mc_caption or ai_tags:
            try:
                from backend.vector.text_embedding import get_text_embedder

                embedder = get_text_embedder()
                # Combine caption and tags for MV
                mv_text = f"{mc_caption} {ai_tags}".strip()
                if mv_text:
                    mv_vec = embedder.encode(mv_text)

            except Exception as e:
                logger.warning(f"MV encoding failed: {e}")

        return vv_vec, mv_vec, structure_vec

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

    # ── Main Loop ─────────────────────────────────────────────

    def run(self):
        """Main worker loop: login → claim → process → repeat."""
        logger.info("Imagine Worker Daemon starting...")

        if not self.login():
            logger.error("Authentication failed. Exiting.")
            return

        poll_interval = get_poll_interval()
        consecutive_empty = 0

        while not _shutdown:
            try:
                jobs = self.claim_jobs()

                if not jobs:
                    consecutive_empty += 1
                    wait = min(poll_interval * consecutive_empty, 300)  # Max 5 min
                    logger.info(f"No jobs available. Waiting {wait}s...")
                    for _ in range(wait):
                        if _shutdown:
                            break
                        time.sleep(1)
                    continue

                consecutive_empty = 0

                for job in jobs:
                    if _shutdown:
                        break
                    self.process_job(job)

                # Cleanup GPU memory between batches
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except ImportError:
                    pass

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                time.sleep(poll_interval)

        logger.info("Worker daemon shutting down.")
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        except Exception:
            pass


def main():
    """CLI entry point."""
    daemon = WorkerDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
