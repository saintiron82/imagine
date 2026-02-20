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
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.manager import JobQueueManager


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


def scan_folder(folder_path, priority=0):
    """DFS scan folder and create processing jobs."""
    try:
        from backend.pipeline.ingest_engine import discover_files

        folder = Path(folder_path).resolve()
        if not folder.exists() or not folder.is_dir():
            return {"success": False, "error": "Directory not found"}

        discovered = discover_files(folder)
        if not discovered:
            return {"success": True, "discovered": 0, "jobs_created": 0, "skipped": 0}

        db = SQLiteDB()
        queue = JobQueueManager(db)
        cursor = db.conn.cursor()

        file_ids = []
        file_paths_list = []
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
                file_ids.append(fid)
                file_paths_list.append(fpath_str)
            except Exception:
                pass

        db.conn.commit()
        jobs_created = queue.create_jobs(file_ids, file_paths_list, priority) if file_ids else 0
        db.close()

        return {
            "success": True,
            "discovered": len(discovered),
            "jobs_created": jobs_created,
            "skipped": skipped,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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
        result = scan_folder(data.get("folder_path", ""), data.get("priority", 0))
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
    else:
        result = {"success": False, "error": f"Unknown command: {cmd}"}

    print(json.dumps(result, ensure_ascii=False))
