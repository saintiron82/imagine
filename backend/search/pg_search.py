"""
PostgreSQL vector search with pgvector.

This module provides search functionality using pgvector for CLIP embeddings,
replacing the ChromaDB-based VectorSearcher while maintaining compatibility.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

from backend.db.pg_client import PostgresDB

logger = logging.getLogger(__name__)


class PgVectorSearch:
    """PostgreSQL vector search with CLIP embeddings."""

    def __init__(self, db: Optional[PostgresDB] = None, model_name: str = "clip-ViT-L-14"):
        """
        Initialize vector search.

        Args:
            db: PostgresDB instance (creates new if None)
            model_name: CLIP model name for text encoding
        """
        self.db = db if db else PostgresDB()
        self.model_name = model_name
        self._clip_model = None  # Lazy loading

        logger.info(f"PgVectorSearch initialized with model: {model_name}")

    @property
    def clip_model(self) -> SentenceTransformer:
        """Lazy load CLIP model."""
        if self._clip_model is None:
            logger.info(f"Loading CLIP model: {self.model_name}...")
            self._clip_model = SentenceTransformer(self.model_name)
            logger.info("✅ CLIP model loaded")
        return self._clip_model

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query to CLIP embedding.

        Args:
            text: Text query

        Returns:
            CLIP embedding vector (768 dimensions)
        """
        return self.clip_model.encode(text)

    def vector_search(
        self,
        query: str,
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using CLIP embeddings.

        Args:
            query: Text query (will be encoded with CLIP)
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of file records with similarity scores
        """
        # Encode query text
        query_embedding = self.encode_text(query)

        cursor = self.db.conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute("""
                SELECT
                    id,
                    file_path,
                    file_name,
                    format,
                    width,
                    height,
                    ai_caption,
                    ai_tags,
                    metadata,
                    thumbnail_url,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM files
                WHERE embedding IS NOT NULL
                    AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                query_embedding.tolist(),
                query_embedding.tolist(),
                threshold,
                query_embedding.tolist(),
                top_k
            ))

            results = [dict(row) for row in cursor.fetchall()]

            logger.info(f"Vector search '{query}' returned {len(results)} results")
            return results

        finally:
            cursor.close()

    def hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: vector similarity + metadata filters.

        Args:
            query: Text query for vector search
            filters: Metadata filters, e.g.:
                     {
                         "format": "PSD",
                         "min_width": 2000,
                         "max_width": 4000,
                         "tags": "cartoon",  # ILIKE search in semantic_tags
                         "ai_caption": "city"  # Full-text search
                     }
            top_k: Number of results
            threshold: Minimum similarity

        Returns:
            Filtered and ranked results
        """
        query_embedding = self.encode_text(query)

        # Build dynamic WHERE clause
        where_clauses = ["embedding IS NOT NULL"]
        params = [query_embedding.tolist(), query_embedding.tolist(), threshold]

        if filters:
            if "format" in filters:
                where_clauses.append("format = %s")
                params.append(filters["format"])

            if "min_width" in filters:
                where_clauses.append("width >= %s")
                params.append(filters["min_width"])

            if "max_width" in filters:
                where_clauses.append("width <= %s")
                params.append(filters["max_width"])

            if "min_height" in filters:
                where_clauses.append("height >= %s")
                params.append(filters["min_height"])

            if "max_height" in filters:
                where_clauses.append("height <= %s")
                params.append(filters["max_height"])

            if "tags" in filters:
                where_clauses.append("metadata->>'semantic_tags' ILIKE %s")
                params.append(f"%{filters['tags']}%")

            if "ai_caption" in filters:
                where_clauses.append("ai_caption ILIKE %s")
                params.append(f"%{filters['ai_caption']}%")

        where_sql = " AND ".join(where_clauses)
        params.extend([query_embedding.tolist(), top_k])

        cursor = self.db.conn.cursor(cursor_factory=RealDictCursor)

        try:
            sql = f"""
                SELECT
                    id,
                    file_path,
                    file_name,
                    format,
                    width,
                    height,
                    ai_caption,
                    ai_tags,
                    metadata,
                    thumbnail_url,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM files
                WHERE {where_sql}
                    AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            cursor.execute(sql, params)
            results = [dict(row) for row in cursor.fetchall()]

            logger.info(f"Hybrid search '{query}' with {len(filters or {})} filters returned {len(results)} results")
            return results

        finally:
            cursor.close()

    def metadata_query(
        self,
        filters: Dict[str, Any],
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Pure metadata query (no vector search).

        Args:
            filters: Same as hybrid_search filters
            top_k: Number of results

        Returns:
            Filtered results ordered by parsed_at DESC
        """
        where_clauses = []
        params = []

        if "format" in filters:
            where_clauses.append("format = %s")
            params.append(filters["format"])

        if "min_width" in filters:
            where_clauses.append("width >= %s")
            params.append(filters["min_width"])

        if "tags" in filters:
            where_clauses.append("metadata->>'semantic_tags' ILIKE %s")
            params.append(f"%{filters['tags']}%")

        if "ai_caption" in filters:
            where_clauses.append("ai_caption ILIKE %s")
            params.append(f"%{filters['ai_caption']}%")

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
        params.append(top_k)

        cursor = self.db.conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(f"""
                SELECT
                    id,
                    file_path,
                    file_name,
                    format,
                    width,
                    height,
                    ai_caption,
                    ai_tags,
                    metadata,
                    thumbnail_url,
                    parsed_at
                FROM files
                WHERE {where_sql}
                ORDER BY parsed_at DESC
                LIMIT %s
            """, params)

            results = [dict(row) for row in cursor.fetchall()]

            logger.info(f"Metadata query with {len(filters)} filters returned {len(results)} results")
            return results

        finally:
            cursor.close()

    def jsonb_query(
        self,
        jsonb_path: str,
        value: Any,
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query nested JSONB metadata.

        Args:
            jsonb_path: JSONB path (e.g., "layer_tree->name")
            value: Value to match
            top_k: Number of results

        Returns:
            Matching files

        Example:
            # Find files with layer_tree.name = "Root"
            results = search.jsonb_query("layer_tree->>'name'", "Root")
        """
        cursor = self.db.conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(f"""
                SELECT
                    id,
                    file_path,
                    file_name,
                    metadata,
                    parsed_at
                FROM files
                WHERE metadata->{jsonb_path} = %s
                ORDER BY parsed_at DESC
                LIMIT %s
            """, (value, top_k))

            results = [dict(row) for row in cursor.fetchall()]

            logger.info(f"JSONB query '{jsonb_path}' = '{value}' returned {len(results)} results")
            return results

        finally:
            cursor.close()

    def search(
        self,
        query: str,
        mode: str = "vector",
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Unified search interface (compatibility with VectorSearcher).

        Args:
            query: Search query
            mode: "vector", "hybrid", or "metadata"
            filters: Optional metadata filters
            top_k: Number of results
            threshold: Similarity threshold (vector modes only)

        Returns:
            Search results
        """
        if mode == "vector":
            return self.vector_search(query, top_k, threshold)
        elif mode == "hybrid":
            return self.hybrid_search(query, filters, top_k, threshold)
        elif mode == "metadata":
            if not filters:
                raise ValueError("Metadata mode requires filters")
            return self.metadata_query(filters, top_k)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'vector', 'hybrid', or 'metadata'")
