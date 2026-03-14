"""
API wrapper for archive browsing.
Returns JSON with folder listing or file listing.

Usage:
  python api_archive.py folders
  python api_archive.py files '{"folder_path":"Characters","image_type":null,"limit":50,"offset":0}'
  python api_archive.py image-types
"""
import sys
import json
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB


def get_folders():
    try:
        db = SQLiteDB()
        folders = db.get_archive_folders()
        db.close()
        return {"success": True, "folders": folders}
    except Exception as e:
        return {"success": False, "error": str(e), "folders": []}


def get_files(params):
    try:
        db = SQLiteDB()
        result = db.get_archive_files(
            folder_path=params.get("folder_path"),
            image_type=params.get("image_type"),
            limit=params.get("limit", 50),
            offset=params.get("offset", 0),
        )
        db.close()
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e), "total": 0, "files": []}


def get_image_types():
    try:
        db = SQLiteDB()
        types = db.get_image_type_stats()
        db.close()
        return {"success": True, "types": types}
    except Exception as e:
        return {"success": False, "error": str(e), "types": []}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Missing command"}))
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "folders":
        result = get_folders()
    elif cmd == "files":
        params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = get_files(params)
    elif cmd == "image-types":
        result = get_image_types()
    else:
        result = {"success": False, "error": f"Unknown command: {cmd}"}

    print(json.dumps(result, ensure_ascii=False))
