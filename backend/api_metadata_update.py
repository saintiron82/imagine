#!/usr/bin/env python
"""
API endpoint for updating user metadata.
Reads JSON from stdin and updates the database.
"""

import sys
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB


def update_user_metadata(file_path: str, updates: dict):
    """
    Update user metadata for a file.

    Args:
        file_path: Absolute path to the file
        updates: Dict with keys: user_note, user_tags, user_category, user_rating

    Returns:
        Success status and message
    """
    try:
        db = SQLiteDB()
        cursor = db.conn.cursor()

        # Build UPDATE query dynamically
        set_clauses = []
        params = []

        if 'user_note' in updates:
            set_clauses.append("user_note = ?")
            params.append(updates['user_note'])

        if 'user_tags' in updates:
            # Convert list to JSON string
            if isinstance(updates['user_tags'], list):
                tags_json = json.dumps(updates['user_tags'])
            else:
                tags_json = updates['user_tags']  # Already JSON string
            set_clauses.append("user_tags = ?")
            params.append(tags_json)

        if 'user_category' in updates:
            set_clauses.append("user_category = ?")
            params.append(updates['user_category'])

        if 'user_rating' in updates:
            rating = int(updates['user_rating'])
            if not 0 <= rating <= 5:
                return {"success": False, "error": "Rating must be between 0 and 5"}
            set_clauses.append("user_rating = ?")
            params.append(rating)

        if not set_clauses:
            return {"success": False, "error": "No updates provided"}

        # Execute update
        params.append(file_path)
        query = f"UPDATE files SET {', '.join(set_clauses)} WHERE file_path = ?"
        cursor.execute(query, params)

        if cursor.rowcount == 0:
            db.conn.rollback()
            db.close()
            return {"success": False, "error": "File not found in database"}

        # Refresh FTS index after update (trigger inserts empty values)
        row = cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,)).fetchone()
        if row:
            db._refresh_fts_row(cursor, row[0])

        db.conn.commit()
        db.close()
        return {"success": True, "message": "Metadata updated"}

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == '__main__':
    try:
        # Read JSON from stdin
        input_data = json.loads(sys.stdin.read())
        result = update_user_metadata(input_data['file_path'], input_data['updates'])
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)
