"""
API wrapper for incomplete file stats.
Returns JSON with per-folder incomplete file counts for resume dialog.
"""
import sys
import json
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB


def get_incomplete_stats():
    try:
        db = SQLiteDB()
        stats = db.get_incomplete_stats()
        db.close()
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": str(e), "total_files": 0, "total_incomplete": 0, "folders": []}


if __name__ == "__main__":
    result = get_incomplete_stats()
    print(json.dumps(result, ensure_ascii=False))
