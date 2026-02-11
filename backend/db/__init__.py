"""
Database module for SQLite + sqlite-vec storage.

This module provides database access for the ImageParser project.
SQLite is the primary database (PostgreSQL is deprecated).
"""

from .sqlite_client import SQLiteDB

try:
    from .pg_client import PostgresDB
except ImportError:
    PostgresDB = None

__all__ = ['SQLiteDB', 'PostgresDB']
