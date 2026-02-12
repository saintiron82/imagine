"""
API wrapper for folder-level phase completion stats.
Returns JSON with per-folder MC/VV/MV counts.

Usage: python api_folder_stats.py <storage_root>
"""
import sys
import json
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB


def get_folder_stats(storage_root: str):
    try:
        db = SQLiteDB()
        folders = db.get_folder_phase_stats(storage_root)
        db.close()
        return {"success": True, "folders": folders}
    except Exception as e:
        return {"success": False, "error": str(e), "folders": []}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Missing storage_root argument", "folders": []}))
        sys.exit(1)

    storage_root = sys.argv[1]
    result = get_folder_stats(storage_root)
    print(json.dumps(result, ensure_ascii=False))
