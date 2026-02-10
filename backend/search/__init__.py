"""
Search module for SQLite + sqlite-vec Triaxis queries.

This module provides search functionality including:
- V-axis: Visual similarity search (SigLIP2 embeddings)
- S-axis: Semantic text vector search (Qwen3-Embedding)
- M-axis: Metadata keyword search (FTS5 BM25)
- Triaxis RRF merge (V + S + M)

Legacy: PgVectorSearch (PostgreSQL, deprecated)
"""

from .sqlite_search import SqliteVectorSearch

__all__ = ['SqliteVectorSearch']
