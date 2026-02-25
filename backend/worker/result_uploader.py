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

    def __init__(self, session, server_url: str):
        """
        Args:
            session: requests.Session with Authorization header set
            server_url: Base URL (e.g. "http://localhost:8000")
        """
        self.session = session
        self.base = server_url

    def report_progress(self, job_id: int, phase: str) -> bool:
        """Report phase completion (parse/vision/embed)."""
        try:
            resp = self.session.patch(
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
            resp = self.session.patch(
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

    def fail_job(self, job_id: int, error_message: str) -> bool:
        """Report job failure."""
        try:
            resp = self.session.patch(
                f"{self.base}/api/v1/jobs/{job_id}/fail",
                json={"error_message": error_message},
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
            resp = self.session.patch(
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
            resp = self.session.patch(
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

    def upload_thumbnail(self, file_id: int, thumb_path: str) -> bool:
        """Upload thumbnail to server (dual storage)."""
        path = Path(thumb_path)
        if not path.exists():
            return False

        try:
            with open(path, "rb") as f:
                resp = self.session.post(
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
            resp = self.session.get(
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
            resp = self.session.get(
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
