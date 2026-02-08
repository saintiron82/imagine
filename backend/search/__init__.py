"""
Search module for PostgreSQL + pgvector queries.

This module provides search functionality including:
- Vector similarity search (CLIP embeddings)
- Hybrid search (vector + metadata filters)
- JSONB queries (nested structures)
- Full-text search (AI captions, tags)
"""

from .pg_search import PgVectorSearch

__all__ = ['PgVectorSearch']
