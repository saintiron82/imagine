"""
FileTaskParsePool — server-side parse daemon for Analysis Job System v1.

Polls file_tasks for files where download is complete (or n/a for local)
and parse is pending. Parses each file, generates thumbnail, upserts
metadata to files table, and marks parse_status='done'.

No retired queue-manager dependency.
"""

import json
import logging
import shutil
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Optional

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.base_ahead_pool import BaseAheadPool
from backend.utils.meta_helpers import meta_to_dict

logger = logging.getLogger(__name__)


class FileTaskParsePool(BaseAheadPool):
    """Background daemon that parses files for the Analysis Job system.

    Scans file_tasks for download_status IN ('done','n/a') AND parse_status='pending'.
    Parses each file → thumbnail + metadata → updates files table + file_tasks.
    For WebDAV files, accesses DownloadAheadPool temp files and releases slots after parse.
    """

    def __init__(self, db: SQLiteDB, download_pool=None):
        """
        Args:
            db: SQLite database connection.
            download_pool: DownloadAheadPool instance for WebDAV temp file access.
                           None if no WebDAV sources configured.
        """
        super().__init__(db)
        self._download_pool = download_pool
        self._parsed_count = 0
        self._failed_count = 0
        self._stats_lock = threading.Lock()
        self._last_retry_check = time.monotonic()
        # Parallel parse workers — SQLiteDB connections are thread-local,
        # so each worker thread reads/writes through its own connection.
        self._parse_workers = max(1, int(self._get_config_value(
            "server.parse_ahead.parse_workers", 3) or 3))
        self._executor = ThreadPoolExecutor(
            max_workers=self._parse_workers,
            thread_name_prefix="parse",
        )
        self._futures = set()

    def _unload_models(self):
        """No models to unload — parse is CPU-only."""
        pass

    def _loop(self):
        """Main loop: keep the parse worker pool fed (pipelined, no batch barrier)."""
        poll_interval = self._get_config_value(
            "server.parse_ahead.poll_interval_s", 2
        )

        while self._running:
            try:
                # Drain finished futures
                self._futures = {f for f in self._futures if not f.done()}

                submitted = self._submit_tasks()

                if submitted == 0:
                    # Nothing new: idle-poll if drained, short wait if working
                    time.sleep(poll_interval if not self._futures else 0.2)

                # Periodic recovery: reset failed tasks back to pending
                now = time.monotonic()
                if now - self._last_retry_check >= 300:
                    self._retry_failed()
                    self._last_retry_check = now

            except Exception as e:
                logger.error(f"FileTaskParsePool loop error: {e}")
                time.sleep(5)

        # Let in-flight parses finish and commit their status updates
        self._executor.shutdown(wait=True)

    def _submit_tasks(self) -> int:
        """Claim pending tasks up to free worker capacity and submit them.

        Returns the number of tasks submitted.
        """
        # Check if parse is paused
        try:
            from backend.server.routers.analysis import get_paused_phases
            if get_paused_phases(self.db).get("parse"):
                return 0
        except Exception:
            pass

        # Keep up to 2× workers in flight so threads never starve between claims
        capacity = self._parse_workers * 2 - len(self._futures)
        if capacity <= 0:
            return 0

        cursor = self.db.conn.cursor()
        try:
            # Job-fair interleave (see claim_tasks) — parse feeds every
            # downstream phase, so fairness must start here.
            cursor.execute("""
                SELECT id, file_id, file_path FROM (
                    SELECT ft.id, ft.file_id, ft.file_path,
                           ft.priority, ft.created_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY ft.analysis_job_id
                               ORDER BY ft.created_at ASC
                           ) AS job_rn
                    FROM file_tasks ft
                    JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
                    WHERE aj.status = 'active'
                      AND ft.download_status IN ('done', 'n/a')
                      AND ft.parse_status = 'pending'
                )
                ORDER BY priority DESC, job_rn ASC, created_at ASC
                LIMIT ?
            """, (capacity,))
            rows = cursor.fetchall()
        except Exception as e:
            logger.warning(f"FileTaskParsePool: query failed: {e}")
            return 0

        if not rows:
            return 0

        submitted = 0
        for task_id, file_id, file_path in rows:
            if not self._running:
                break

            # Atomically claim: pending → assigned
            cursor.execute("""
                UPDATE file_tasks
                SET parse_status = 'assigned', updated_at = ?
                WHERE id = ? AND parse_status = 'pending'
            """, (_now(), task_id))
            if cursor.rowcount == 0:
                self.db.conn.commit()
                continue  # another thread/process claimed it
            self.db.conn.commit()

            self._futures.add(
                self._executor.submit(self._process_task, task_id, file_id, file_path)
            )
            submitted += 1

        return submitted

    def _process_task(self, task_id: int, file_id: int, file_path: str):
        """Parse one claimed task and record its outcome.

        Runs in an executor thread — all DB access goes through this
        thread's own connection (SQLiteDB is thread-local).
        """
        cursor = self.db.conn.cursor()
        t_start = time.perf_counter()
        try:
            error = self._parse_single_task(task_id, file_id, file_path)
        except Exception as e:
            error = f"unexpected: {e}"
        elapsed = time.perf_counter() - t_start

        try:
            if not error:
                cursor.execute("""
                    UPDATE file_tasks
                    SET parse_status = 'done',
                        parse_completed_at = ?,
                        parse_elapsed_s = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (_now(), round(elapsed, 3), _now(), task_id))
                self.db.conn.commit()
                with self._stats_lock:
                    self._parsed_count += 1
                logger.info(
                    f"FileTaskParse: task {task_id} OK "
                    f"({Path(file_path).name}, {elapsed:.1f}s)"
                )
            else:
                # Parse failures are usually permanent (corrupt file, unsupported format).
                # Set max_retries=0 so _retry_failed() won't re-queue them.
                err_msg = f"{Path(file_path).name}: {error}"
                logger.warning(f"FileTaskParse: FAIL {err_msg}")
                cursor.execute("""
                    UPDATE file_tasks
                    SET parse_status = 'failed',
                        parse_completed_at = ?,
                        parse_elapsed_s = ?,
                        retry_count = retry_count + 1,
                        max_retries = 0,
                        error_message = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (_now(), round(elapsed, 3), err_msg, _now(), task_id))
                self.db.conn.commit()
                with self._stats_lock:
                    self._failed_count += 1
        except Exception as e:
            logger.error(f"FileTaskParse: status update failed for task {task_id}: {e}")
            try:
                self.db.conn.rollback()
            except Exception:
                pass

    def _parse_single_task(
        self, task_id: int, file_id: int, file_path: str
    ) -> str:
        """Parse a single file: resolve → parse → thumbnail → DB upsert.

        Returns empty string on success, error message on failure.
        """
        from backend.pipeline.ingest_engine import (
            ParserFactory,
            _set_tier_metadata,
        )
        from backend.utils.content_hash import compute_content_hash

        # 1. Resolve file path
        file_p = Path(file_path)
        is_webdav = file_path.startswith("webdav://")

        if not file_p.exists():
            if is_webdav:
                temp = self._get_webdav_temp(file_id)
                if temp:
                    file_p = Path(temp)
                else:
                    return f"WebDAV file not yet downloaded"
            else:
                return f"file not found: {file_path}"

        # 2. Parse (fallback to thumbnail-only on failure)
        parser = ParserFactory.get_parser(file_p)
        if not parser:
            return f"no parser for extension: {file_p.suffix}"

        parse_note = None
        result = parser.parse(file_p)
        if result.success:
            meta = result.asset_meta
        else:
            # Parse failed (unsupported compression, corrupt layers, etc.)
            # → still generate thumbnail via PIL and continue to MC/VV/MV
            parse_errors = "; ".join(result.errors) if result.errors else "unknown parse error"
            logger.info(
                f"FileTaskParse: layer parse failed ({parse_errors}), "
                f"thumbnail-only mode: {file_p.name}"
            )
            meta = self._fallback_thumbnail(file_p, file_path)
            if meta is None:
                return f"parse failed ({parse_errors}) AND thumbnail fallback also failed"
            parse_note = "thumbnail_only"  # 레이어 추출 불가, 썸네일만 생성

        # 3. Content hash
        try:
            meta.content_hash = compute_content_hash(file_p)
        except Exception as e:
            logger.warning(f"FileTaskParse: content_hash failed: {e}")

        # 4. Folder metadata from original path (not temp)
        orig_parent = PurePosixPath(file_path).parent.name
        if orig_parent and orig_parent not in (".", ""):
            meta.folder_path = orig_parent
            meta.folder_depth = 0
            meta.folder_tags = [orig_parent]

        # 5. Tier metadata
        _set_tier_metadata(meta)

        # 6. Copy thumbnail to server thumbnails directory
        server_thumb_path = None
        if meta.thumbnail_url:
            src_thumb = Path(meta.thumbnail_url)
            if src_thumb.exists():
                try:
                    thumb_dir = self._get_thumbnail_dir()
                    dest_name = f"{file_p.stem}_thumb.png"
                    server_thumb_path = thumb_dir / dest_name
                    shutil.copy2(str(src_thumb), str(server_thumb_path))
                except Exception as e:
                    logger.warning(f"FileTaskParse: thumbnail copy failed: {e}")
                    server_thumb_path = None

                # Clean temp-located fallback thumbnails (and their dirs)
                import tempfile as _tf
                try:
                    if str(src_thumb).startswith(_tf.gettempdir()):
                        src_thumb.unlink(missing_ok=True)
                        if src_thumb.parent.name.startswith("imagine_fallback_thumb_"):
                            src_thumb.parent.rmdir()
                except OSError:
                    pass

        # 7. Upsert metadata to files table
        meta_dict = meta_to_dict(meta)
        canonical = file_path if is_webdav else str(file_p)
        nfc_path = unicodedata.normalize("NFC", canonical)
        meta_dict["file_path"] = nfc_path

        try:
            stored_file_id = self.db.upsert_metadata(nfc_path, meta_dict)
        except Exception as e:
            return f"metadata upsert failed: {e}"

        # P10: record parse outcome via update_vision_fields (which accepts
        # processing_status/processing_error). Lets SQL find rows that need re-parsing.
        try:
            status_fields = (
                {"processing_status": "parse_fallback",
                 "processing_error": "psd-tools could not extract layers"}
                if parse_note == "thumbnail_only"
                else {"processing_status": "parsed"}
            )
            self.db.update_vision_fields(nfc_path, status_fields)
        except Exception as e:
            logger.debug(f"FileTaskParse: status record failed: {e}")

        # Update thumbnail_url in files table
        if server_thumb_path:
            try:
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "UPDATE files SET thumbnail_url = ? WHERE id = ?",
                    (str(server_thumb_path), stored_file_id),
                )
                self.db._refresh_fts_row(cursor, stored_file_id)
                self.db.conn.commit()
            except Exception as e:
                logger.warning(f"FileTaskParse: thumbnail_url update failed: {e}")

        # 8. CAS M3: materialize cached derivations — phases already
        # computed for identical content skip the workers entirely.
        try:
            from backend.server.queue.derivations import apply_cache_hits
            applied = apply_cache_hits(
                self.db, task_id, stored_file_id, meta.content_hash)
            self.db.conn.commit()
            if applied:
                logger.info(
                    f"FileTaskParse: cache hit {applied} for {file_p.name}")
                if set(applied) >= {"mc", "vv", "mv"}:
                    # Fully served from cache — close out the job if done
                    from backend.server.queue.analysis_manager import AnalysisJobManager
                    cursor = self.db.conn.cursor()
                    cursor.execute(
                        "SELECT analysis_job_id FROM file_tasks WHERE id = ?",
                        (task_id,))
                    row = cursor.fetchone()
                    if row:
                        AnalysisJobManager(self.db)._check_job_completion(row[0])
        except Exception as e:
            logger.warning(f"FileTaskParse: cache-hit application failed: {e}")
            try:
                self.db.conn.rollback()
            except Exception:
                pass

        # 9. Release download buffer slot (WebDAV only)
        if is_webdav and self._download_pool:
            self._download_pool.release_slot(file_id)

        return ""  # success

    def _fallback_thumbnail(self, file_p: Path, file_path: str):
        """Generate minimal AssetMeta with thumbnail when full parse fails.

        Opens the file as a flat image (PIL), generates thumbnail,
        and creates basic metadata. Layer info will be empty.
        """
        try:
            from PIL import Image as PILImage
            from backend.parser.schema import AssetMeta

            img = PILImage.open(file_p)
            img = img.convert("RGB")

            # Generate thumbnail in a temp dir — never next to the user's
            # original (it would leave *_thumb.png residue in their asset
            # folder). The server copy in _parse_single_task step 6 is the
            # persistent location; the temp source is cleaned after copy.
            import tempfile
            thumb_size = (512, 512)
            thumb = img.copy()
            thumb.thumbnail(thumb_size, PILImage.LANCZOS)
            thumb_dir = Path(tempfile.mkdtemp(prefix="imagine_fallback_thumb_"))
            thumb_path = thumb_dir / f"{file_p.stem}_thumb.png"
            thumb.save(str(thumb_path), "PNG")

            # P08: lowercase canonical format
            ext = file_p.suffix.lower().lstrip(".")
            fmt = "jpg" if ext == "jpeg" else ext

            meta = AssetMeta(
                file_path=file_path,
                file_name=file_p.name,
                file_size=file_p.stat().st_size,
                format=fmt,
                resolution=(img.width, img.height),
                thumbnail_url=str(thumb_path),
                layer_count=0,
                semantic_tags="",
            )

            logger.info(
                f"FileTaskParse: fallback OK ({file_p.name}, "
                f"{img.width}x{img.height}, thumbnail generated)"
            )
            return meta
        except Exception as e:
            logger.error(f"FileTaskParse: fallback thumbnail failed: {e}")
            return None

    def _get_webdav_temp(self, file_id: int) -> Optional[str]:
        """Look up downloaded temp file from DownloadAheadPool."""
        if not self._download_pool:
            return None
        temp = self._download_pool.get_temp_path(file_id)
        if temp and Path(temp).exists():
            return temp
        return None

    def _get_thumbnail_dir(self) -> Path:
        """Get server thumbnail directory."""
        from backend.server.config import get_storage_config

        cfg = get_storage_config()
        thumb_dir = Path(cfg.get("thumbnail_dir", "./thumbnails"))
        thumb_dir.mkdir(parents=True, exist_ok=True)
        return thumb_dir

    def _retry_failed(self):
        """Reset failed parse tasks back to pending (up to max_retries)."""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("""
                UPDATE file_tasks
                SET parse_status = 'pending',
                    error_message = NULL,
                    updated_at = ?
                WHERE parse_status = 'failed'
                  AND retry_count < max_retries
                  AND datetime(updated_at) < datetime('now', '-5 minutes')
            """, (_now(),))
            recovered = cursor.rowcount
            self.db.conn.commit()
            if recovered > 0:
                logger.info(
                    f"FileTaskParse: recovered {recovered} failed tasks → pending"
                )
        except Exception as e:
            logger.warning(f"FileTaskParse: retry recovery failed: {e}")
            try:
                self.db.conn.rollback()
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Return current pool statistics."""
        return {
            "parsed_count": self._parsed_count,
            "failed_count": self._failed_count,
            "running": self._running,
        }


def _now() -> str:
    """UTC timestamp string for SQLite."""
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
