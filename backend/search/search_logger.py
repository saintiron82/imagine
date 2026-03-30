"""
Search logger — shared search request logging for local (Electron) and server modes.

Uses a module-level SQLiteDB singleton for fire-and-forget logging.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_log_db = None


def log_search(
    query: str,
    mode: str,
    result_count: int,
    elapsed_ms: int,
    username: Optional[str] = None,
    ip_address: Optional[str] = None,
    filters: Optional[dict] = None,
    threshold: Optional[float] = None,
):
    """Record a search request to the search_logs table (fire-and-forget).

    Args:
        query: Search query text (truncated to 500 chars).
        mode: Search mode (triaxis, vector, text_vector, fts).
        result_count: Number of results returned.
        elapsed_ms: Total search time in milliseconds.
        username: User who performed the search (default: None).
        ip_address: Client IP address (server mode only).
        filters: Search filters dict (serialized to JSON).
        threshold: Similarity threshold used.
    """
    global _log_db
    try:
        if _log_db is None:
            from backend.db.sqlite_client import SQLiteDB
            _log_db = SQLiteDB()
        _log_db.conn.execute(
            """INSERT INTO search_logs
               (query, mode, result_count, elapsed_ms, username, ip_address, filters, threshold)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                query[:500],
                mode,
                result_count,
                elapsed_ms,
                username,
                ip_address,
                json.dumps(filters) if filters else None,
                threshold,
            ),
        )
        _log_db.conn.commit()
    except Exception as e:
        logger.debug(f"Failed to log search: {e}")
