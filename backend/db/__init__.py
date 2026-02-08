"""
Database module for PostgreSQL + pgvector storage.

This module provides database access for the ImageParser project,
replacing the JSON file + ChromaDB dual storage system with a unified
PostgreSQL database using the pgvector extension.
"""

from .pg_client import PostgresDB

__all__ = ['PostgresDB']
