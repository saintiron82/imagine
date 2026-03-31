"""
Result uploader — sends pipeline analysis results to the server API.

Converts local pipeline output (metadata + vectors) into API calls:
  - PATCH /api/v1/jobs/{job_id}/progress  (phase updates)
  - PATCH /api/v1/jobs/{job_id}/complete  (metadata + vectors)
  - PATCH /api/v1/jobs/{job_id}/fail      (error report)
  - POST  /api/v1/upload/thumbnails/{file_id}  (thumbnail upload)
"""

import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any

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

    def report_progress(self, job_id: int, phase: str) -> bool:
        """Report phase completion (parse/vision/embed)."""
        try:
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/{job_id}/progress",
                json={"phase": phase},
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Progress report failed for job {job_id}: {e}")
            return False

    def complete_job(
        self,
        job_id: int,
        metadata: Dict[str, Any],
        vv_vec=None,
        mv_vec=None,
        structure_vec=None,
    ) -> bool:
        """Upload final results and mark job complete."""
        vectors = {}
        if vv_vec is not None:
            vectors["vv"] = _encode_vector(vv_vec)
        if mv_vec is not None:
            vectors["mv"] = _encode_vector(mv_vec)
        if structure_vec is not None:
            vectors["structure"] = _encode_vector(structure_vec)

        payload = {"metadata": metadata}
        if vectors:
            payload["vectors"] = vectors

        try:
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/{job_id}/complete",
                json=payload,
            )
            if resp.status_code == 200:
                logger.info(f"Job {job_id} completed successfully")
                return True
            else:
                logger.error(f"Job {job_id} complete failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Job {job_id} complete request failed: {e}")
            return False

    def fail_job(self, job_id: int, error_message: str,
                 error_code: str = None) -> bool:
        """Report job failure with optional structured error code."""
        try:
            payload = {"error_message": error_message}
            if error_code:
                payload["error_code"] = error_code
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/{job_id}/fail",
                json=payload,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Fail report failed for job {job_id}: {e}")
            return False

    def complete_embed(self, job_id: int, vv_vec=None, mv_vec=None) -> bool:
        """Embed-only mode: upload VV+MV vectors for vision-complete jobs.

        Server has already run Parse+Vision. Worker only generates VV+MV vectors.
        """
        vectors = {}
        if vv_vec is not None:
            vectors["vv"] = _encode_vector(vv_vec)
        if mv_vec is not None:
            vectors["mv"] = _encode_vector(mv_vec)

        try:
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/{job_id}/complete_embed",
                json={"vectors": vectors},
            )
            if resp.status_code == 200:
                logger.info(f"Embed-only job {job_id} completed successfully")
                return True
            else:
                logger.error(f"Embed-only job {job_id} failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Embed-only job {job_id} request failed: {e}")
            return False

    def complete_mc(self, job_id: int, vision_fields: Dict[str, Any]) -> bool:
        """MC-only mode: upload vision fields without vectors.

        Server handles VV (ParseAhead) and MV (EmbedAhead) separately.
        """
        try:
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/{job_id}/complete_mc",
                json={"vision_fields": vision_fields},
            )
            if resp.status_code == 200:
                logger.info(f"MC-only job {job_id} completed successfully")
                return True
            else:
                logger.error(f"MC-only job {job_id} failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"MC-only job {job_id} request failed: {e}")
            return False

    def complete_parse(self, job_id: int, metadata: dict, thumbnail_path: str = None) -> bool:
        """parse_thumb mode: upload parse results + thumbnail to server."""
        try:
            # 1. Upload metadata
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/{job_id}/complete_parse",
                json={"metadata": metadata},
            )
            if resp.status_code != 200:
                logger.error(f"Parse job {job_id} metadata upload failed: {resp.status_code} {resp.text}")
                return False

            # 2. Upload thumbnail if available
            if thumbnail_path:
                file_id = metadata.get("file_id") or resp.json().get("file_id")
                if file_id:
                    self.upload_thumbnail(file_id, thumbnail_path)

            logger.info(f"Parse job {job_id} completed successfully")
            return True
        except Exception as e:
            logger.error(f"Parse job {job_id} request failed: {e}")
            return False

    def complete_vv(self, job_id: int, vv_vec) -> bool:
        """VV-only mode: upload single VV vector."""
        try:
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/{job_id}/complete_vv",
                json={"vectors": {"vv": _encode_vector(vv_vec)}},
            )
            if resp.status_code == 200:
                return True
            logger.error(f"VV-only job {job_id} failed: {resp.status_code} {resp.text}")
            return False
        except Exception as e:
            logger.error(f"VV-only job {job_id} request failed: {e}")
            return False

    def complete_vv_batch(self, items: list) -> list:
        """Batch VV upload: [{job_id, vec}, ...] → [bool, ...]"""
        try:
            payload = [{"job_id": it["job_id"], "vv": _encode_vector(it["vec"])} for it in items]
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/batch/complete_vv",
                json={"items": payload},
            )
            if resp.status_code == 200:
                return resp.json().get("results", [True] * len(items))
            logger.error(f"VV batch failed: {resp.status_code}")
            return [False] * len(items)
        except Exception as e:
            logger.error(f"VV batch request failed: {e}")
            return [False] * len(items)

    def complete_mv(self, job_id: int, mv_vec) -> bool:
        """MV-only mode: upload single MV vector."""
        try:
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/{job_id}/complete_mv",
                json={"vectors": {"mv": _encode_vector(mv_vec)}},
            )
            if resp.status_code == 200:
                return True
            logger.error(f"MV-only job {job_id} failed: {resp.status_code} {resp.text}")
            return False
        except Exception as e:
            logger.error(f"MV-only job {job_id} request failed: {e}")
            return False

    def complete_mv_batch(self, items: list) -> list:
        """Batch MV upload: [{job_id, vec}, ...] → [bool, ...]"""
        try:
            payload = [{"job_id": it["job_id"], "mv": _encode_vector(it["vec"])} for it in items]
            resp = self._request('patch',
                f"{self.base}/api/v1/jobs/batch/complete_mv",
                json={"items": payload},
            )
            if resp.status_code == 200:
                return resp.json().get("results", [True] * len(items))
            logger.error(f"MV batch failed: {resp.status_code}")
            return [False] * len(items)
        except Exception as e:
            logger.error(f"MV batch request failed: {e}")
            return [False] * len(items)

    def upload_thumbnail(self, file_id: int, thumb_path: str) -> bool:
        """Upload thumbnail to server (dual storage)."""
        path = Path(thumb_path)
        if not path.exists():
            return False

        try:
            with open(path, "rb") as f:
                resp = self._request('post',
                    f"{self.base}/api/v1/upload/thumbnails/{file_id}",
                    files={"file": (path.name, f, "image/png")},
                )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Thumbnail upload failed for file {file_id}: {e}")
            return False

    def download_file(self, file_id: int, dest_dir: str) -> Optional[str]:
        """Download original file from server for processing."""
        try:
            resp = self._request('get',
                f"{self.base}/api/v1/upload/download/{file_id}",
                stream=True,
            )
            try:
                if resp.status_code != 200:
                    logger.error(f"Download failed for file {file_id}: {resp.status_code}")
                    return None

                # Extract filename from Content-Disposition or use file_id
                filename = f"file_{file_id}"
                cd = resp.headers.get("content-disposition", "")
                if "filename=" in cd:
                    filename = cd.split("filename=")[-1].strip('" ')

                dest = Path(dest_dir) / filename
                dest.parent.mkdir(parents=True, exist_ok=True)

                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            finally:
                resp.close()

            logger.info(f"Downloaded file {file_id} -> {dest}")
            return str(dest)
        except Exception as e:
            logger.error(f"Download failed for file {file_id}: {e}")
            return None

    def download_thumbnail(self, file_id: int, dest_dir: str) -> Optional[str]:
        """Download thumbnail only for a pre-parsed job (~200KB instead of ~500MB)."""
        try:
            resp = self._request('get',
                f"{self.base}/api/v1/upload/download/thumbnail/{file_id}",
                stream=True,
            )
            try:
                if resp.status_code != 200:
                    logger.warning(f"Thumbnail download failed for file {file_id}: {resp.status_code}")
                    return None

                dest = Path(dest_dir) / f"thumb_{file_id}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)

                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            finally:
                resp.close()

            size_kb = dest.stat().st_size / 1024
            logger.info(f"Downloaded thumbnail for file {file_id} ({size_kb:.0f}KB)")
            return str(dest)
        except Exception as e:
            logger.error(f"Thumbnail download failed for file {file_id}: {e}")
            return None


def _encode_vector(vec) -> str:
    """Encode numpy float32 vector to base64 string."""
    import numpy as np
    return base64.b64encode(vec.astype(np.float32).tobytes()).decode("ascii")
