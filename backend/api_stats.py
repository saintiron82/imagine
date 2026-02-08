"""
API wrapper for database stats.
Returns JSON with archived image count and format distribution.
"""
import sys
import json
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB


def get_stats():
    try:
        db = SQLiteDB()
        stats = db.get_stats()
        db.close()
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": str(e), "total_files": 0}


if __name__ == "__main__":
    result = get_stats()
    print(json.dumps(result, ensure_ascii=False))
