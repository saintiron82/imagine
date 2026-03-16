"""
API wrapper for job queue operations.
Used by Electron IPC in server mode to register jobs directly (bypassing HTTP auth).

Commands:
  register-paths  — Register file paths into job queue
  scan-folder     — DFS scan folder and create jobs
  stats           — Get queue statistics
"""
import sys
import json
import io
import logging
from pathlib import Path, PurePosixPath

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Force all logging to stderr so stdout stays clean for JSON IPC
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.manager import JobQueueManager


def _extract_folder_from_path(fpath):
    """Extract folder_path/folder_depth/folder_tags from file_path.

    WebDAV:  webdav://source_id/발송작품/작품/원피스/1/file.psd
             → ("발송작품/작품/원피스/1", 4, ["발송작품", "작품", "원피스", "1"])
    Local:   /Users/user/imageDB/마캬베리즈무/장소/강다외부/file.psd
             → ("마캬베리즈무/장소/강다외부", 3, ["마캬베리즈무", "장소", "강다외부"])
    """
    if fpath.startswith("webdav://"):
        without_scheme = fpath.replace("webdav://", "", 1)
        slash_idx = without_scheme.find("/")
        if slash_idx != -1:
            sub = without_scheme[slash_idx:]  # /발송작품/작품/원피스/1/file.psd
            parent = str(PurePosixPath(sub).parent).lstrip("/")
            if parent:
                tags = [p for p in parent.split("/") if p]
                return parent, len(tags), tags
    else:
        # Local: use path relative to home directory (skip system dirs)
        fp = Path(fpath)
        try:
            home = Path.home()
            rel = fp.parent.relative_to(home)
            parts = [p for p in rel.parts if p]
            if parts:
                folder_str = "/".join(parts)
                return folder_str, len(parts), parts
        except ValueError:
            pass
        # Fallback: parent folder name only
        parent = fp.parent.name
        if parent and parent not in (".", ""):
            return parent, 0, [parent]
    return None, 0, []


def register_paths(file_paths, priority=0):
    """Register file paths and create processing jobs."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        cursor = db.conn.cursor()

        file_ids = []
        registered_paths = []

        for fpath in file_paths:
            fname = fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath.rsplit("\\", 1)[-1] if "\\" in fpath else fpath
            meta = {"file_path": fpath, "file_name": fname}
            folder_path, depth, tags = _extract_folder_from_path(fpath)
            if folder_path:
                meta["folder_path"] = folder_path
                meta["folder_depth"] = depth
                meta["folder_tags"] = tags
            try:
                fid = db.upsert_metadata(fpath, meta)
                file_ids.append(fid)
                registered_paths.append(fpath)
            except Exception as e:
                pass  # skip failed files

        db.conn.commit()
        jobs_created = queue.create_jobs(file_ids, registered_paths, priority) if file_ids else 0
        db.close()

        return {
            "success": True,
            "registered": len(file_ids),
            "jobs_created": jobs_created,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def scan_folder(folder_path, priority=0, webdav_configs=None):
    """DFS scan folder and create Work Request with jobs."""
    try:
        from backend.pipeline.ingest_engine import discover_files
        from collections import defaultdict

        # WebDAV: delegate to scan_folders (reuses shared logic)
        if folder_path.startswith("webdav://"):
            return scan_folders([folder_path], priority, webdav_configs)

        folder = Path(folder_path).resolve()
        if not folder.exists() or not folder.is_dir():
            return {"success": False, "error": "Directory not found"}

        discovered = discover_files(folder)
        if not discovered:
            return {"success": True, "discovered": 0, "jobs_created": 0, "skipped": 0}

        db = SQLiteDB()
        queue = JobQueueManager(db)
        cursor = db.conn.cursor()

        # Group files by sub-folder for work_subtasks
        file_groups = defaultdict(list)  # {folder_path: [(file_id, file_path), ...]}
        skipped = 0

        for file_path, folder_str, depth, folder_tags in discovered:
            fpath_str = str(file_path)

            # Check for existing active job
            cursor.execute(
                "SELECT id FROM job_queue WHERE file_path = ? AND status NOT IN ('completed', 'failed', 'cancelled')",
                (fpath_str,)
            )
            if cursor.fetchone():
                skipped += 1
                continue

            meta = {
                "file_path": fpath_str,
                "file_name": file_path.name,
                "folder_path": folder_str,
                "folder_depth": depth,
                "folder_tags": folder_tags,
            }
            try:
                fid = db.upsert_metadata(fpath_str, meta)
                file_groups[folder_str or str(folder)].append((fid, fpath_str))
            except Exception:
                pass

        db.conn.commit()

        if file_groups:
            result = queue.create_work_request(
                name=folder.name,
                source_path=str(folder),
                file_groups=dict(file_groups),
                priority=priority,
            )
            jobs_created = result.get("jobs_created", 0)
        else:
            jobs_created = 0

        db.close()

        return {
            "success": True,
            "discovered": len(discovered),
            "jobs_created": jobs_created,
            "skipped": skipped,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _parse_webdav_path(webdav_path):
    """Parse webdav://source-id/sub/path into (source_id, sub_path)."""
    without_scheme = webdav_path.replace("webdav://", "", 1)
    slash_idx = without_scheme.find("/")
    if slash_idx == -1:
        return without_scheme, "/"
    return without_scheme[:slash_idx], without_scheme[slash_idx:] or "/"


def _get_webdav_client(source_id, webdav_configs=None):
    """Create a WebDAVClient from provided configs (Electron-decrypted)."""
    from backend.remote.webdav_client import WebDAVClient

    if not webdav_configs or source_id not in webdav_configs:
        return None, None

    cfg = webdav_configs[source_id]
    client = WebDAVClient(
        base_url=cfg["url"],
        username=cfg.get("username", ""),
        password=cfg.get("password", ""),
        remote_path=cfg.get("remote_path", "/"),
        verify_ssl=cfg.get("verify_ssl", True),
    )
    return client, cfg


def _scan_webdav_folder(folder_path, db, cursor, webdav_configs=None):
    """Scan a webdav:// folder and return (discovered, skipped, file_groups)."""
    from backend.remote.webdav_client import WebDAVClient
    from pathlib import PurePosixPath
    from collections import defaultdict

    source_id, sub_path = _parse_webdav_path(folder_path)
    client, source = _get_webdav_client(source_id, webdav_configs)
    if not client:
        return 0, 0, {}

    # Override remote_path to scan the specific subfolder
    original_remote = client.remote_path
    client.remote_path = original_remote.rstrip('/') + '/' + sub_path.lstrip('/') if sub_path != "/" else original_remote

    remote_files = client.list_files_recursive()
    file_groups = defaultdict(list)
    discovered = 0
    skipped = 0

    for rf in remote_files:
        # Build webdav:// URI for this file
        fpath_str = f"webdav://{source_id}{rf.remote_path}"
        file_name = PurePosixPath(rf.remote_path).name

        # Folder path from full remote path (not relative to scan root)
        # e.g. remote_path="/발송작품/작품/원피스/1/44-040.psd" → "발송작품/작품/원피스/1"
        full_parent = str(PurePosixPath(rf.remote_path).parent).lstrip("/")
        folder_tags = [p for p in full_parent.split("/") if p]

        discovered += 1

        # Check for existing active job
        cursor.execute(
            "SELECT id FROM job_queue WHERE file_path = ? AND status NOT IN ('completed', 'failed', 'cancelled')",
            (fpath_str,)
        )
        if cursor.fetchone():
            skipped += 1
            continue

        meta = {
            "file_path": fpath_str,
            "file_name": file_name,
            "folder_path": full_parent,
            "folder_depth": len(folder_tags),
            "folder_tags": folder_tags,
            "storage_root": f"webdav://{source_id}{client.remote_path}",
        }
        try:
            fid = db.upsert_metadata(fpath_str, meta)
            group_key = rel_parent or f"webdav://{source_id}{client.remote_path}"
            file_groups[group_key].append((fid, fpath_str))
        except Exception:
            pass

    return discovered, skipped, dict(file_groups)


def scan_folders(folder_paths, priority=0, webdav_configs=None):
    """DFS scan multiple folders and create ONE Work Request with jobs.

    Supports both local paths and webdav:// URIs.
    webdav_configs: {source_id: {url, username, password, remote_path, verify_ssl}}
                    Pre-decrypted by Electron IPC.
    """
    try:
        from backend.pipeline.ingest_engine import discover_files
        from collections import defaultdict

        db = SQLiteDB()
        queue = JobQueueManager(db)
        cursor = db.conn.cursor()

        all_file_groups = defaultdict(list)
        total_discovered = 0
        total_skipped = 0

        for folder_path in folder_paths:
            # WebDAV path: use remote file listing
            if folder_path.startswith("webdav://"):
                discovered, skipped, file_groups = _scan_webdav_folder(folder_path, db, cursor, webdav_configs)
                total_discovered += discovered
                total_skipped += skipped
                for k, v in file_groups.items():
                    all_file_groups[k].extend(v)
                continue

            # Local path: use filesystem DFS
            folder = Path(folder_path).resolve()
            if not folder.exists() or not folder.is_dir():
                continue

            discovered = discover_files(folder)
            total_discovered += len(discovered)

            for file_path, folder_str, depth, folder_tags in discovered:
                fpath_str = str(file_path)

                # Check for existing active job
                cursor.execute(
                    "SELECT id FROM job_queue WHERE file_path = ? AND status NOT IN ('completed', 'failed', 'cancelled')",
                    (fpath_str,)
                )
                if cursor.fetchone():
                    total_skipped += 1
                    continue

                meta = {
                    "file_path": fpath_str,
                    "file_name": file_path.name,
                    "folder_path": folder_str,
                    "folder_depth": depth,
                    "folder_tags": folder_tags,
                }
                try:
                    fid = db.upsert_metadata(fpath_str, meta)
                    all_file_groups[folder_str or str(folder)].append((fid, fpath_str))
                except Exception:
                    pass

        db.conn.commit()

        if all_file_groups:
            # Build name from folder list
            names = []
            for p in folder_paths[:3]:
                if p.startswith("webdav://"):
                    _, sub = _parse_webdav_path(p)
                    names.append(PurePosixPath(sub).name or sub)
                else:
                    names.append(Path(p).name)
            name = ", ".join(names)
            if len(folder_paths) > 3:
                name += f" +{len(folder_paths) - 3}"

            result = queue.create_work_request(
                name=name,
                source_path=folder_paths[0],
                file_groups=dict(all_file_groups),
                priority=priority,
            )
            jobs_created = result.get("jobs_created", 0)
        else:
            jobs_created = 0

        db.close()

        return {
            "success": True,
            "discovered": total_discovered,
            "jobs_created": jobs_created,
            "skipped": total_skipped,
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}


def get_stats():
    """Get job queue statistics."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        stats = queue.get_stats()
        db.close()
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_jobs(status=None, limit=50, offset=0):
    """List jobs with optional filtering and pagination."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        result = queue.list_jobs(status=status, limit=limit, offset=offset)
        db.close()
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def cancel_job(job_id):
    """Cancel a job."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        success = queue.cancel_job(job_id)
        db.close()
        return {"success": success}
    except Exception as e:
        return {"success": False, "error": str(e)}


def retry_failed():
    """Retry all failed jobs."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        count = queue.retry_failed_jobs()
        db.close()
        return {"success": True, "retried": count}
    except Exception as e:
        return {"success": False, "error": str(e)}


def clear_completed():
    """Clear all completed jobs."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        count = queue.clear_completed_jobs()
        db.close()
        return {"success": True, "deleted": count}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_work_requests(include_completed=False):
    """List work requests."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        wrs = queue.get_work_requests(include_completed)
        db.close()
        return {"success": True, "work_requests": wrs}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_work_request_detail(wr_id):
    """Get work request detail with subtasks."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        detail = queue.get_work_request_detail(wr_id)
        db.close()
        if detail is None:
            return {"success": False, "error": "Work request not found"}
        return {"success": True, **detail}
    except Exception as e:
        return {"success": False, "error": str(e)}


def pause_work_request(wr_id):
    """Pause a work request."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        ok = queue.pause_work_request(wr_id)
        db.close()
        return {"success": ok}
    except Exception as e:
        return {"success": False, "error": str(e)}


def resume_work_request(wr_id):
    """Resume a paused work request."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        ok = queue.resume_work_request(wr_id)
        db.close()
        return {"success": ok}
    except Exception as e:
        return {"success": False, "error": str(e)}


def cancel_work_request(wr_id):
    """Cancel a work request and its pending jobs."""
    try:
        db = SQLiteDB()
        queue = JobQueueManager(db)
        result = queue.cancel_work_request(wr_id)
        db.close()
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def backfill_folders():
    """Backfill folder_path/folder_tags from file_path for all files.

    - Files with missing folder info: always update
    - WebDAV files: always recalculate (previous scan may have stored incomplete paths)
    - Local files with existing folder info: skip
    """
    try:
        db = SQLiteDB()
        cursor = db.conn.cursor()
        # All files: recalculate folder_path from file_path
        # (previous logic may have stored incomplete paths)
        cursor.execute("SELECT id, file_path, folder_path FROM files")
        rows = cursor.fetchall()
        updated = 0
        for file_id, fpath, current_folder in rows:
            folder_path, depth, tags = _extract_folder_from_path(fpath)
            if folder_path and folder_path != current_folder:
                tags_json = json.dumps(tags)
                cursor.execute(
                    "UPDATE files SET folder_path=?, folder_depth=?, folder_tags=? WHERE id=?",
                    (folder_path, depth, tags_json, file_id)
                )
                db._refresh_fts_row(cursor, file_id)
                updated += 1
        db.conn.commit()
        db.close()
        return {"success": True, "updated": updated, "total": len(rows)}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No command specified"}))
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "register-paths":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = register_paths(data.get("file_paths", []), data.get("priority", 0))
    elif cmd == "scan-folder":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = scan_folder(data.get("folder_path", ""), data.get("priority", 0), data.get("webdav_configs"))
    elif cmd == "scan-folders":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = scan_folders(data.get("folder_paths", []), int(data.get("priority", 0)), data.get("webdav_configs"))
    elif cmd == "stats":
        result = get_stats()
    elif cmd == "list-jobs":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = list_jobs(
            status=data.get("status"),
            limit=data.get("limit", 50),
            offset=data.get("offset", 0),
        )
    elif cmd == "cancel-job":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = cancel_job(data.get("job_id", 0))
    elif cmd == "retry-failed":
        result = retry_failed()
    elif cmd == "clear-completed":
        result = clear_completed()
    elif cmd == "list-work-requests":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = list_work_requests(data.get("include_completed", False))
    elif cmd == "work-request-detail":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = get_work_request_detail(data.get("wr_id", 0))
    elif cmd == "pause-wr":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = pause_work_request(data.get("wr_id", 0))
    elif cmd == "resume-wr":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = resume_work_request(data.get("wr_id", 0))
    elif cmd == "cancel-wr":
        data = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(sys.stdin.readline())
        result = cancel_work_request(data.get("wr_id", 0))
    elif cmd == "backfill-folders":
        result = backfill_folders()
    else:
        result = {"success": False, "error": f"Unknown command: {cmd}"}

    print(json.dumps(result, ensure_ascii=False))
