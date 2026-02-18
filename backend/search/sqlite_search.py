"""
SQLite vector search with sqlite-vec (Triaxis Architecture).

This module replaces pg_search.py with SQLite-based vector search,
maintaining API compatibility for minimal code changes.

Triaxis Search (V + S + M):
- VV (Visual): SigLIP 2 embedding similarity (image pixels)
- MV (Meaning Vector): Qwen3 text embedding (AI-interpreted captions + context)
- FTS (Metadata): FTS5 metadata-only search (file facts, no AI content)

User Filters: Format, category, rating, tags, folder paths
"""

import logging
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from backend.db.sqlite_client import SQLiteDB
from backend.search.query_decomposer import QueryDecomposer

logger = logging.getLogger(__name__)

# Search diagnostic logging (disable with SEARCH_DIAGNOSTIC=0)
_DIAGNOSTIC_ENABLED = os.getenv("SEARCH_DIAGNOSTIC", "1") != "0"
_DIAGNOSTIC_LOG_DIR = Path(__file__).parent.parent.parent / "logs"


class SqliteVectorSearch:
    """SQLite vector search with SigLIP 2 embeddings."""

    def __init__(self, db: Optional[SQLiteDB] = None):
        """
        Initialize vector search.

        Args:
            db: SQLiteDB instance (creates new if None)
        """
        self.db = db if db else SQLiteDB()
        self._encoder = None  # Lazy loading (VV)
        self._text_provider = None  # Lazy loading (MV)
        self._structure_encoder = None  # Lazy loading (Structure)
        self._text_enabled = None  # Cache for MV availability check

        logger.info("SqliteVectorSearch initialized")

    @property
    def encoder(self):
        """Lazy load VV embedding encoder (SigLIP 2)."""
        if self._encoder is None:
            from backend.vector.siglip2_encoder import SigLIP2Encoder
            self._encoder = SigLIP2Encoder()
            logger.info(f"Embedding encoder loaded: {self._encoder.model_name}")
        return self._encoder

    @property
    def text_provider(self):
        """Lazy load MV provider."""
        if self._text_provider is None:
            from backend.vector.text_embedding import get_text_embedding_provider
            self._text_provider = get_text_embedding_provider()
        return self._text_provider

    @property
    def structure_encoder(self):
        """Lazy load DINOv2 structure encoder."""
        if self._structure_encoder is None:
            from backend.vector.dinov2_encoder import DinoV2Encoder
            self._structure_encoder = DinoV2Encoder()
            logger.info(f"Structure encoder loaded: {self._structure_encoder.model_name}")
        return self._structure_encoder

    @property
    def text_search_enabled(self) -> bool:
        """Check if MV search is available (vec_text table exists with data)."""
        if self._text_enabled is None:
            try:
                count = self.db.conn.execute("SELECT COUNT(*) FROM vec_text").fetchone()[0]
                self._text_enabled = count > 0
            except Exception:
                self._text_enabled = False
        return self._text_enabled

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query to VV embedding vector (SigLIP 2).

        Args:
            text: Text query

        Returns:
            Embedding vector
        """
        return self.encoder.encode_text(text)

    def encode_structure(self, image) -> np.ndarray:
        """
        Encode image to Structure embedding vector (DINOv2).

        Args:
            image: PIL Image or scalar

        Returns:
            768-dim embedding vector
        """
        return self.structure_encoder.encode_image(image)

    def vector_search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using a pre-computed embedding.

        Args:
            query_embedding: Embedding vector (np.ndarray)
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of file records with similarity scores
        """
        embedding_json = json.dumps(query_embedding.astype(np.float32).tolist())

        cursor = self.db.conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    f.id,
                    f.file_path,
                    f.file_name,
                    f.format,
                    f.width,
                    f.height,
                    f.mc_caption,
                    f.ai_tags,
                    f.ocr_text,
                    f.metadata,
                    f.thumbnail_url,
                    f.user_note,
                    f.user_tags,
                    f.user_category,
                    f.user_rating,
                    f.folder_path,
                    f.folder_depth,
                    f.folder_tags,
                    f.storage_root,
                    f.relative_path,
                    f.image_type,
                    f.art_style,
                    f.color_palette,
                    f.scene_type,
                    f.time_of_day,
                    f.weather,
                    f.character_type,
                    f.item_type,
                    f.ui_type,
                    (1.0 - vec_distance_cosine(v.embedding, ?)) AS similarity
                FROM files f
                JOIN vec_files v ON f.id = v.file_id
                WHERE (1.0 - vec_distance_cosine(v.embedding, ?)) >= ?
                ORDER BY vec_distance_cosine(v.embedding, ?) ASC
                LIMIT ?
            """, (embedding_json, embedding_json, threshold, embedding_json, top_k))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                self._parse_json_fields(result)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Vector search by embedding failed: {e}")
            return []
        finally:
            cursor.close()

    def vector_search(
        self,
        query: str,
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using SigLIP2 embeddings.

        Args:
            query: Text query (will be encoded with SigLIP2)
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of file records with similarity scores
        """
        query_embedding = self.encode_text(query)
        results = self.vector_search_by_embedding(query_embedding, top_k, threshold)
        logger.info(f"Vector search '{query}' returned {len(results)} results")
        return results

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
                         "tags": "cartoon",  # LIKE search in semantic_tags
                         "mc_caption": "city"  # Full-text search
                     }
            top_k: Number of results
            threshold: Minimum similarity

        Returns:
            Filtered and ranked results
        """
        query_embedding = self.encode_text(query)
        embedding_json = json.dumps(query_embedding.astype(np.float32).tolist())

        # Build dynamic WHERE clause
        where_clauses = []
        params = [embedding_json, embedding_json, threshold]

        if filters:
            if "format" in filters:
                where_clauses.append("f.format = ?")
                params.append(filters["format"])

            if "min_width" in filters:
                where_clauses.append("f.width >= ?")
                params.append(filters["min_width"])

            if "max_width" in filters:
                where_clauses.append("f.width <= ?")
                params.append(filters["max_width"])

            if "min_height" in filters:
                where_clauses.append("f.height >= ?")
                params.append(filters["min_height"])

            if "max_height" in filters:
                where_clauses.append("f.height <= ?")
                params.append(filters["max_height"])

            if "tags" in filters:
                where_clauses.append("json_extract(f.metadata, '$.semantic_tags') LIKE ?")
                params.append(f"%{filters['tags']}%")

            if "mc_caption" in filters:
                where_clauses.append("f.mc_caption LIKE ?")
                params.append(f"%{filters['mc_caption']}%")

            if "folder_path" in filters:
                where_clauses.append("f.folder_path LIKE ?")
                params.append(f"{filters['folder_path']}%")  # prefix match

            if "folder_tag" in filters:
                where_clauses.append("f.folder_tags LIKE ?")
                params.append(f"%\"{filters['folder_tag']}\"%")

        where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""
        params.extend([embedding_json, top_k])

        cursor = self.db.conn.cursor()

        try:
            sql = f"""
                SELECT
                    f.id,
                    f.file_path,
                    f.file_name,
                    f.format,
                    f.width,
                    f.height,
                    f.mc_caption,
                    f.ai_tags,
                    f.metadata,
                    f.thumbnail_url,
                    f.folder_path,
                    f.folder_depth,
                    f.folder_tags,
                    f.storage_root,
                    f.relative_path,
                    f.image_type,
                    f.art_style,
                    f.color_palette,
                    f.scene_type,
                    f.time_of_day,
                    f.weather,
                    f.character_type,
                    f.item_type,
                    f.ui_type,
                    (1.0 - vec_distance_cosine(v.embedding, ?)) AS similarity
                FROM files f
                JOIN vec_files v ON f.id = v.file_id
                WHERE (1.0 - vec_distance_cosine(v.embedding, ?)) >= ?
                {where_sql}
                ORDER BY vec_distance_cosine(v.embedding, ?) ASC
                LIMIT ?
            """

            cursor.execute(sql, params)

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                self._parse_json_fields(result)
                results.append(result)

            logger.info(f"Hybrid search '{query}' with {len(filters or {})} filters returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
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
            where_clauses.append("format = ?")
            params.append(filters["format"])

        if "min_width" in filters:
            where_clauses.append("width >= ?")
            params.append(filters["min_width"])

        if "tags" in filters:
            where_clauses.append("json_extract(metadata, '$.semantic_tags') LIKE ?")
            params.append(f"%{filters['tags']}%")

        if "mc_caption" in filters:
            where_clauses.append("mc_caption LIKE ?")
            params.append(f"%{filters['mc_caption']}%")

        if "folder_path" in filters:
            where_clauses.append("folder_path LIKE ?")
            params.append(f"{filters['folder_path']}%")

        if "folder_tag" in filters:
            where_clauses.append("folder_tags LIKE ?")
            params.append(f"%\"{filters['folder_tag']}\"%")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        params.append(top_k)

        cursor = self.db.conn.cursor()

        try:
            cursor.execute(f"""
                SELECT
                    id,
                    file_path,
                    file_name,
                    format,
                    width,
                    height,
                    mc_caption,
                    ai_tags,
                    metadata,
                    thumbnail_url,
                    folder_path,
                    folder_depth,
                    folder_tags,
                    parsed_at
                FROM files
                WHERE {where_sql}
                ORDER BY parsed_at DESC
                LIMIT ?
            """, params)

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                self._parse_json_fields(result)
                results.append(result)

            logger.info(f"Metadata query with {len(filters)} filters returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Metadata query failed: {e}")
            return []
        finally:
            cursor.close()

    def json_query(
        self,
        json_path: str,
        value: Any,
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query nested JSON metadata.

        Args:
            json_path: JSON path (e.g., "$.layer_tree.name")
            value: Value to match
            top_k: Number of results

        Returns:
            Matching files

        Example:
            # Find files with layer_tree.name = "Root"
            results = search.json_query("$.layer_tree.name", "Root")
        """
        cursor = self.db.conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    id,
                    file_path,
                    file_name,
                    metadata,
                    parsed_at
                FROM files
                WHERE json_extract(metadata, ?) = ?
                ORDER BY parsed_at DESC
                LIMIT ?
            """, (json_path, value, top_k))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except:
                        result['metadata'] = {}
                results.append(result)

            logger.info(f"JSON query '{json_path}' = '{value}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"JSON query failed: {e}")
            return []
        finally:
            cursor.close()

    def text_vector_search(
        self,
        query: str,
        top_k: int = 20,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        MV: Text vector similarity search using Qwen3-Embedding.

        Searches vec_text table (caption+tags embeddings) for semantic text matching.
        Complements VV (visual similarity) with textual semantic similarity.

        Args:
            query: Text query (encoded with MV model)
            top_k: Number of results
            threshold: Minimum similarity threshold

        Returns:
            List of file records with text_similarity scores
        """
        query_vec = self.text_provider.encode(query, is_query=True)
        embedding_json = json.dumps(query_vec.astype(np.float32).tolist())

        cursor = self.db.conn.cursor()
        try:
            cursor.execute("""
                SELECT
                    f.id,
                    f.file_path,
                    f.file_name,
                    f.format,
                    f.width,
                    f.height,
                    f.mc_caption,
                    f.ai_tags,
                    f.ocr_text,
                    f.metadata,
                    f.thumbnail_url,
                    f.user_note,
                    f.user_tags,
                    f.user_category,
                    f.user_rating,
                    f.folder_path,
                    f.folder_depth,
                    f.folder_tags,
                    f.storage_root,
                    f.relative_path,
                    f.image_type,
                    f.art_style,
                    f.color_palette,
                    f.scene_type,
                    f.time_of_day,
                    f.weather,
                    f.character_type,
                    f.item_type,
                    f.ui_type,
                    (1.0 - vec_distance_cosine(vt.embedding, ?)) AS text_similarity
                FROM files f
                JOIN vec_text vt ON f.id = vt.file_id
                WHERE (1.0 - vec_distance_cosine(vt.embedding, ?)) >= ?
                ORDER BY vec_distance_cosine(vt.embedding, ?) ASC
                LIMIT ?
            """, (embedding_json, embedding_json, threshold, embedding_json, top_k))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                self._parse_json_fields(result)
                results.append(result)

            logger.info(f"MV search '{query[:50]}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"MV text vector search failed: {e}")
            return []
        finally:
            cursor.close()

    def _text_vector_search_by_embedding(
        self,
        query_vec: np.ndarray,
        top_k: int = 20,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        MV search using a pre-computed embedding vector.

        Same as text_vector_search() but accepts a pre-encoded vector,
        allowing callers to cache the embedding for reuse.
        """
        embedding_json = json.dumps(query_vec.astype(np.float32).tolist())

        cursor = self.db.conn.cursor()
        try:
            cursor.execute("""
                SELECT
                    f.id,
                    f.file_path,
                    f.file_name,
                    f.format,
                    f.width,
                    f.height,
                    f.mc_caption,
                    f.ai_tags,
                    f.ocr_text,
                    f.metadata,
                    f.thumbnail_url,
                    f.user_note,
                    f.user_tags,
                    f.user_category,
                    f.user_rating,
                    f.folder_path,
                    f.folder_depth,
                    f.folder_tags,
                    f.storage_root,
                    f.relative_path,
                    f.image_type,
                    f.art_style,
                    f.color_palette,
                    f.scene_type,
                    f.time_of_day,
                    f.weather,
                    f.character_type,
                    f.item_type,
                    f.ui_type,
                    (1.0 - vec_distance_cosine(vt.embedding, ?)) AS text_similarity
                FROM files f
                JOIN vec_text vt ON f.id = vt.file_id
                WHERE (1.0 - vec_distance_cosine(vt.embedding, ?)) >= ?
                ORDER BY vec_distance_cosine(vt.embedding, ?) ASC
                LIMIT ?
            """, (embedding_json, embedding_json, threshold, embedding_json, top_k))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                self._parse_json_fields(result)
                results.append(result)

            logger.info(f"MV search (by embedding) returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"MV text vector search by embedding failed: {e}")
            return []
        finally:
            cursor.close()

    def search_structure(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Structure search (DINOv2) using a pre-computed embedding.

        Args:
            query_embedding: Structure Vector (768-dim)
            top_k: Number of results
            threshold: Minimum similarity

        Returns:
            List of file records with structural_similarity
        """
        embedding_json = json.dumps(query_embedding.astype(np.float32).tolist())

        cursor = self.db.conn.cursor()
        try:
            cursor.execute("""
                SELECT
                    f.id,
                    f.file_path,
                    f.file_name,
                    f.format,
                    f.width,
                    f.height,
                    f.mc_caption,
                    f.ai_tags,
                    f.ocr_text,
                    f.metadata,
                    f.thumbnail_url,
                    f.user_note,
                    f.user_tags,
                    f.user_category,
                    f.user_rating,
                    f.folder_path,
                    f.folder_depth,
                    f.folder_tags,
                    f.storage_root,
                    f.relative_path,
                    f.image_type,
                    f.art_style,
                    f.color_palette,
                    f.scene_type,
                    f.time_of_day,
                    f.weather,
                    f.character_type,
                    f.item_type,
                    f.ui_type,
                    (1.0 - vec_distance_cosine(vs.embedding, ?)) AS structural_similarity
                FROM files f
                JOIN vec_structure vs ON f.id = vs.file_id
                WHERE (1.0 - vec_distance_cosine(vs.embedding, ?)) >= ?
                ORDER BY vec_distance_cosine(vs.embedding, ?) ASC
                LIMIT ?
            """, (embedding_json, embedding_json, threshold, embedding_json, top_k))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                self._parse_json_fields(result)
                results.append(result)

            logger.info(f"Structure search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Structure search failed: {e}")
            return []
        finally:
            cursor.close()

    def find_similar_structure(
        self,
        file_id: int,
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find files with similar structure/texture to the given file_id.

        Args:
            file_id: Database ID of the reference file
            top_k: Number of results
            threshold: Minimum similarity

        Returns:
            List of matching file records
        """
        cursor = self.db.conn.cursor()
        try:
            # Fetch existing structure embedding
            row = cursor.execute(
                "SELECT embedding FROM vec_structure WHERE file_id = ?",
                (file_id,)
            ).fetchone()

            if not row:
                logger.warning(f"No structure embedding found for file_id={file_id}")
                return []

            embedding_json = row[0]
            # Convert JSON string back to list/array if needed, but search_structure needs ndarray
            # SQLite stores it as JSON string '[-0.1, 0.5, ...]'
            vec_list = json.loads(embedding_json)
            query_vec = np.array(vec_list, dtype=np.float32)

            return self.search_structure(query_vec, top_k, threshold)

        except Exception as e:
            logger.error(f"find_similar_structure failed: {e}")
            return []
        finally:
            cursor.close()

    def find_similar_visual(
        self,
        file_id: int,
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find files with similar visual appearance to the given file_id using VV (SigLIP2).

        Args:
            file_id: Database ID of the reference file
            top_k: Number of results
            threshold: Minimum similarity

        Returns:
            List of matching file records with vector_score
        """
        cursor = self.db.conn.cursor()
        try:
            row = cursor.execute(
                "SELECT embedding FROM vec_files WHERE file_id = ?",
                (file_id,)
            ).fetchone()

            if not row:
                logger.warning(f"No VV embedding found for file_id={file_id}")
                return []

            vec_list = json.loads(row[0])
            query_vec = np.array(vec_list, dtype=np.float32)

            results = self.vector_search_by_embedding(query_vec, top_k + 1, threshold)
            # Exclude self from results
            results = [r for r in results if r.get("id") != file_id]
            return results[:top_k]

        except Exception as e:
            logger.error(f"find_similar_visual failed: {e}")
            return []
        finally:
            cursor.close()

    def fts_search(
        self,
        keywords: List[str],
        top_k: int = 20,
        exclude_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search using FTS5 (FTS: Metadata-only).

        Args:
            keywords: List of keywords to search (combined with OR)
            top_k: Number of results to return
            exclude_keywords: Optional list of keywords to exclude via FTS5 NOT operator

        Returns:
            List of file records with FTS rank scores
        """
        if not keywords:
            return []

        # Build FTS5 MATCH query: split multi-word keywords into individual
        # tokens so "crossroads at night" matches documents containing any of
        # those words, not just the exact phrase.
        tokens = set()
        for kw in keywords:
            for word in kw.split():
                word = word.strip().replace('"', '""')
                if word:
                    tokens.add(word)
        if not tokens:
            return []

        match_expr = " OR ".join(f'"{t}"' for t in tokens)

        # Build exclude expression using FTS5 NOT operator
        if exclude_keywords:
            exclude_tokens = set()
            for kw in exclude_keywords:
                for word in kw.split():
                    word = word.strip().replace('"', '""')
                    if word:
                        exclude_tokens.add(word)
            if exclude_tokens:
                exclude_expr = " OR ".join(f'"{t}"' for t in exclude_tokens)
                match_expr = f"({match_expr}) NOT ({exclude_expr})"

        # Triaxis: Load BM25 weights from config (2 columns: meta_strong, meta_weak)
        from backend.utils.config import get_config as _cfg
        cfg = _cfg()
        w_strong = cfg.get("search.fts.bm25_weights.meta_strong", 3.0)
        w_weak = cfg.get("search.fts.bm25_weights.meta_weak", 1.5)

        cursor = self.db.conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    f.id,
                    f.file_path,
                    f.file_name,
                    f.format,
                    f.width,
                    f.height,
                    f.mc_caption,
                    f.ai_tags,
                    f.ocr_text,
                    f.metadata,
                    f.thumbnail_url,
                    f.user_note,
                    f.user_tags,
                    f.user_category,
                    f.user_rating,
                    f.folder_path,
                    f.folder_depth,
                    f.folder_tags,
                    f.storage_root,
                    f.relative_path,
                    f.image_type,
                    f.art_style,
                    f.color_palette,
                    f.scene_type,
                    f.time_of_day,
                    f.weather,
                    f.character_type,
                    f.item_type,
                    f.ui_type,
                    bm25(files_fts, ?, ?) AS fts_rank
                FROM files_fts fts
                JOIN files f ON f.id = fts.rowid
                WHERE files_fts MATCH ?
                ORDER BY fts_rank
                LIMIT ?
            """, (w_strong, w_weak, match_expr, top_k))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                self._parse_json_fields(result)
                results.append(result)

            logger.info(f"FTS search '{match_expr}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            return []
        finally:
            cursor.close()

    def triaxis_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        threshold: float = 0.0,
        return_diagnostic: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        3-axis search: Vector + FTS5 + User Filters with RRF merge.

        1. QueryDecomposer decomposes query (LLM or fallback)
        2. Vector search with decomposed.vector_query
        3. FTS5 search with decomposed.fts_keywords
        4. RRF merge results
        5. Apply user filters

        Args:
            query: Natural language search query
            filters: User-specified metadata filters
            top_k: Number of results
            threshold: Vector similarity threshold
            return_diagnostic: If True, return (results, diagnostic) tuple

        Returns:
            Merged and filtered search results.
            If return_diagnostic=True, returns (results, diagnostic_dict).
        """
        t_start = time.perf_counter()
        diag = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "top_k": top_k,
            "threshold": threshold,
            "user_filters": filters,
        }

        # Step 1: Decompose query
        t0 = time.perf_counter()
        decomposer = QueryDecomposer()
        plan = decomposer.decompose(query)
        diag["decomposition_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        vector_query = plan["vector_query"]
        fts_keywords = plan["fts_keywords"]
        llm_filters = plan.get("filters", {})
        negative_query = plan.get("negative_query", "")
        exclude_keywords = plan.get("exclude_keywords", [])

        query_type = plan.get("query_type", "balanced")

        diag["decomposition"] = {
            "decomposed": plan.get("decomposed", False),
            "vector_query": vector_query,
            "fts_keywords": fts_keywords,
            "llm_filters": llm_filters,
            "negative_query": negative_query,
            "exclude_keywords": exclude_keywords,
            "query_type": query_type,
        }

        # Merge LLM-suggested filters with user filters (user takes precedence)
        user_filters = filters or {}

        # Per-axis thresholds: SigLIP (V) and Qwen3 (Tv) have very different score ranges
        # VV: 0.10-0.17 typical match, MV: 0.65-0.78 typical match
        from backend.utils.config import get_config as _cfg
        _search_cfg = _cfg()
        v_threshold = _search_cfg.get("search.threshold.visual", 0.05)
        tv_threshold = _search_cfg.get("search.threshold.text_vec", threshold)

        # Per-axis candidate pool: larger pool → more cross-axis overlap → better RRF
        candidate_mul = _search_cfg.get("search.rrf.candidate_multiplier", 5)
        candidate_k = top_k * candidate_mul

        # Step 2: VV vector search (cache embedding for post-merge enrichment)
        vector_results = []
        v_query_embedding = None
        t0 = time.perf_counter()
        try:
            v_query_embedding = self.encode_text(vector_query)
            vector_results = self.vector_search_by_embedding(
                v_query_embedding, top_k=candidate_k, threshold=v_threshold
            )
        except Exception as e:
            logger.warning(f"VV search unavailable: {e}")
            diag["vector_error"] = str(e)
        diag["vector_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        diag["vector_results"] = {
            "count": len(vector_results),
            "top5": [
                {
                    "file": r.get("file_name", r.get("file_path", "")),
                    "similarity": round(r.get("similarity", 0), 4),
                    "rank": i + 1,
                }
                for i, r in enumerate(vector_results[:5])
            ],
        }

        # Step 2b: MV text vector search (cache embedding for post-merge enrichment)
        text_vec_results = []
        t_query_embedding = None
        t0 = time.perf_counter()
        if self.text_search_enabled:
            try:
                t_query_embedding = self.text_provider.encode(vector_query, is_query=True)
                text_vec_results = self._text_vector_search_by_embedding(
                    t_query_embedding, top_k=candidate_k, threshold=tv_threshold
                )
            except Exception as e:
                logger.warning(f"MV search unavailable: {e}")
                diag["text_vec_error"] = str(e)
        diag["text_vec_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        diag["text_vec_results"] = {
            "count": len(text_vec_results),
            "top5": [
                {
                    "file": r.get("file_name", r.get("file_path", "")),
                    "text_similarity": round(r.get("text_similarity", 0), 4),
                    "rank": i + 1,
                }
                for i, r in enumerate(text_vec_results[:5])
            ],
        }

        # Step 3: FTS FTS5 search
        fts_results = []
        t0 = time.perf_counter()
        try:
            fts_results = self.fts_search(fts_keywords, top_k=candidate_k, exclude_keywords=exclude_keywords)
        except Exception as e:
            logger.warning(f"FTS search unavailable: {e}")
            diag["fts_error"] = str(e)
        diag["fts_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        diag["fts_results"] = {
            "count": len(fts_results),
            "top5": [
                {
                    "file": r.get("file_name", r.get("file_path", "")),
                    "fts_rank": round(r.get("fts_rank", 0), 4),
                    "rank": i + 1,
                }
                for i, r in enumerate(fts_results[:5])
            ],
        }

        # Step 4: 3-axis RRF merge (V + T + F)
        # Build rank lookup before merge for diagnostic
        vector_rank_map = {
            r["file_path"]: i + 1 for i, r in enumerate(vector_results)
        }
        text_vec_rank_map = {
            r["file_path"]: i + 1 for i, r in enumerate(text_vec_results)
        }
        fts_rank_map = {
            r["file_path"]: i + 1 for i, r in enumerate(fts_results)
        }

        # Collect all non-empty result lists for RRF
        rrf_weights = None
        all_result_lists = []
        if vector_results:
            all_result_lists.append(("visual", vector_results))
        if text_vec_results:
            all_result_lists.append(("text_vec", text_vec_results))
        if fts_results:
            all_result_lists.append(("fts", fts_results))

        if len(all_result_lists) >= 2:
            from backend.search.rrf import get_weights
            from backend.utils.config import get_config as _get_config
            rrf_k = _get_config().get("search.rrf.k", 60)
            active_axes = [name for name, _ in all_result_lists]
            rrf_weights = get_weights(query_type, active_axes)
            merged = self._rrf_merge_multi(all_result_lists, k=rrf_k, weights=rrf_weights)
        elif len(all_result_lists) == 1:
            axis_name, single_results = all_result_lists[0]
            if axis_name == "visual":
                for r in single_results:
                    r["vector_score"] = r.get("similarity", 0)
                    r["text_vec_score"] = None
                    r["text_score"] = None
            elif axis_name == "text_vec":
                for r in single_results:
                    r["vector_score"] = None
                    r["text_vec_score"] = r.get("text_similarity", 0)
                    r["text_score"] = None
            else:  # fts
                fts_ranks = [r.get("fts_rank", 0) for r in single_results]
                best = min(fts_ranks)   # most negative = best match
                worst = max(fts_ranks)  # closest to 0 = worst match
                span = worst - best
                for r in single_results:
                    r["vector_score"] = None
                    r["text_vec_score"] = None
                    raw = r.get("fts_rank", 0)
                    r["text_score"] = (worst - raw) / span if span else 1.0
            merged = single_results
        else:
            merged = []

        # NOTE: Per-axis thresholds are already applied at the SQL level
        # (v_threshold for VV, tv_threshold for MV, MATCH for FTS).
        # No post-merge threshold filter needed — it caused scale mismatch
        # (frontend 0.15 vs SigLIP2 range 0.06-0.17, killing most VV results).

        diag["rrf_merge"] = {
            "axes": len(all_result_lists),
            "query_type": query_type,
            "weights": rrf_weights if len(all_result_lists) >= 2 else None,
            "count": len(merged),
            "top5": [
                {
                    "file": r.get("file_name", r.get("file_path", "")),
                    "rrf_score": round(r.get("rrf_score", 0), 6),
                    "vector_rank": vector_rank_map.get(r.get("file_path")),
                    "text_vec_rank": text_vec_rank_map.get(r.get("file_path")),
                    "fts_rank": fts_rank_map.get(r.get("file_path")),
                    "vector_score": round(r["vector_score"], 4) if r.get("vector_score") is not None else None,
                    "text_vec_score": round(r["text_vec_score"], 4) if r.get("text_vec_score") is not None else None,
                    "text_score": round(r["text_score"], 4) if r.get("text_score") is not None else None,
                }
                for i, r in enumerate(merged[:5])
            ],
        }

        # Step 5a: Apply negative filter (demote results matching exclusion terms)
        # Encode negative_query as VV embedding for visual penalty
        neg_v_embedding = None
        if negative_query:
            try:
                neg_v_embedding = self.encode_text(negative_query)
            except Exception as e:
                logger.debug(f"Negative VV encoding failed: {e}")

        pre_neg_count = len(merged)
        if negative_query:
            merged = self._apply_negative_filter(merged, negative_query, neg_v_embedding)

        # Step 5b: Apply LLM filters (lenient -- don't exclude results missing the field)
        pre_filter_count = len(merged)
        llm_removed = 0
        if llm_filters:
            merged = self._apply_user_filters(merged, llm_filters, strict=False)
            llm_removed = pre_filter_count - len(merged)

        # Step 5c: Apply user filters (strict -- exact match required)
        pre_user_count = len(merged)
        user_removed = 0
        if user_filters:
            merged = self._apply_user_filters(merged, user_filters, strict=True)
            user_removed = pre_user_count - len(merged)

        diag["filter_applied"] = bool(user_filters) or bool(llm_filters)
        diag["filter_removed"] = llm_removed + user_removed
        diag["negative_filter_active"] = bool(negative_query)
        diag["negative_v_axis"] = neg_v_embedding is not None

        # Step 5d: quality rerank on filtered candidate pool
        rerank_enabled = bool(_search_cfg.get("search.rerank.enabled", True))
        rerank_pool = int(_search_cfg.get("search.rerank.pool_size", max(top_k * 3, 80)))
        rerank_pool = max(top_k, rerank_pool)
        rerank_used = False
        if rerank_enabled and len(merged) > 1:
            rerank_n = min(len(merged), rerank_pool)
            # Ensure rerank has dense axis scores in its candidate pool
            self._enrich_axis_scores(
                merged[:rerank_n],
                v_query_embedding,
                t_query_embedding,
                fts_keywords,
            )
            merged = self._quality_rerank(
                merged,
                top_k=top_k,
                query=query,
                llm_filters=llm_filters,
                user_filters=user_filters,
                axis_weights=rrf_weights,
                pool_size=rerank_n,
            )
            rerank_used = True

        diag["rerank"] = {
            "enabled": rerank_enabled,
            "used": rerank_used,
            "pool_size": min(len(merged), rerank_pool),
        }

        # Trim to top_k
        merged = merged[:top_k]

        # Step 6: Enrich missing per-axis scores via direct DB lookup
        # Files in final results may lack V/S scores if they weren't in that axis's
        # candidate pool. Compute their actual similarity for complete badge display.
        v_missing_before = sum(1 for r in merged if r.get("vector_score") is None)
        s_missing_before = sum(1 for r in merged if r.get("text_vec_score") is None)
        self._enrich_axis_scores(merged, v_query_embedding, t_query_embedding, fts_keywords)
        v_missing_after = sum(1 for r in merged if r.get("vector_score") is None)
        s_missing_after = sum(1 for r in merged if r.get("text_vec_score") is None)
        diag["enrichment"] = {
            "v_missing_before": v_missing_before,
            "v_enriched": v_missing_before - v_missing_after,
            "s_missing_before": s_missing_before,
            "s_enriched": s_missing_before - s_missing_after,
        }
        diag["final_top5"] = [
            {
                "file": r.get("file_name", r.get("file_path", "")),
                "vector_score": round(r["vector_score"], 4) if r.get("vector_score") is not None else None,
                "text_vec_score": round(r["text_vec_score"], 4) if r.get("text_vec_score") is not None else None,
                "text_score": round(r["text_score"], 4) if r.get("text_score") is not None else None,
                "structure_score": round(r["structure_score"], 4) if r.get("structure_score") is not None else None,
                "quality_score": round(r["quality_score"], 4) if r.get("quality_score") is not None else None,
            }
            for r in merged[:5]
        ]

        diag["final_results_count"] = len(merged)
        diag["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

        logger.info(
            f"Triaxis search '{query}': vector={len(vector_results)}, "
            f"fts={len(fts_results)}, merged={len(merged)}, "
            f"decomposed={plan.get('decomposed', False)}"
        )

        # Write diagnostic log
        if _DIAGNOSTIC_ENABLED:
            self._write_diagnostic(diag)

        if return_diagnostic:
            return merged, diag
        return merged

    @staticmethod
    def _write_diagnostic(diagnostic: Dict[str, Any]) -> None:
        """Append diagnostic data to logs/search_diagnostic.jsonl."""
        try:
            _DIAGNOSTIC_LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_path = _DIAGNOSTIC_LOG_DIR / "search_diagnostic.jsonl"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write diagnostic log: {e}")

    def _rrf_merge(
        self,
        vector_results: List[Dict],
        fts_results: List[Dict],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) to merge results from multiple sources.
        Preserves per-axis scores: vector_score (cosine similarity) and
        text_score (min-max normalized FTS rank).

        Args:
            vector_results: Results from vector search (ordered by similarity)
            fts_results: Results from FTS5 search (ordered by rank)
            k: RRF constant (default 60)

        Returns:
            Merged results ordered by RRF score (descending),
            each with vector_score and text_score fields.
        """
        scores = {}  # file_path -> rrf_score
        result_map = {}  # file_path -> result dict
        vector_scores = {}  # file_path -> cosine similarity
        fts_raw_ranks = {}  # file_path -> fts_rank (negative)

        # Score vector results
        for rank, result in enumerate(vector_results):
            fp = result["file_path"]
            scores[fp] = scores.get(fp, 0) + 1.0 / (k + rank + 1)
            vector_scores[fp] = result.get("similarity", 0)
            if fp not in result_map:
                result_map[fp] = result

        # Score FTS results
        for rank, result in enumerate(fts_results):
            fp = result["file_path"]
            scores[fp] = scores.get(fp, 0) + 1.0 / (k + rank + 1)
            fts_raw_ranks[fp] = result.get("fts_rank", 0)
            if fp not in result_map:
                result_map[fp] = result

        # Normalize FTS ranks (negative → 0~1, more negative = better match → higher score)
        normalized_text = {}
        if fts_raw_ranks:
            ranks = list(fts_raw_ranks.values())
            best = min(ranks)   # most negative = best match
            worst = max(ranks)  # closest to 0 = worst match
            span = worst - best
            for fp, r in fts_raw_ranks.items():
                normalized_text[fp] = (worst - r) / span if span else 1.0

        # Sort by RRF score descending
        sorted_paths = sorted(scores.keys(), key=lambda fp: scores[fp], reverse=True)

        merged = []
        for fp in sorted_paths:
            result = result_map[fp]
            result["rrf_score"] = scores[fp]
            result["vector_score"] = vector_scores.get(fp)   # None if not in vector results
            result["text_score"] = normalized_text.get(fp)    # None if not in FTS results
            merged.append(result)

        return merged

    def _rrf_merge_multi(
        self,
        result_lists: list[tuple[str, list[dict]]],
        k: int = 60,
        weights: dict[str, float] | None = None,
    ) -> list[dict]:
        """
        Multi-axis RRF merge for 2+ result lists.

        Generalizes _rrf_merge to handle V + T + F (+X) axes.
        Preserves per-axis scores: vector_score, text_vec_score, text_score,
        structure_score.

        Args:
            result_lists: List of (axis_name, results) tuples.
                          axis_name: "visual", "text_vec", "fts", or "structure"
            k: RRF constant (default 60)
            weights: Per-axis weight dict (e.g. {"visual": 0.5, "text_vec": 0.3, "fts": 0.2}).
                     If None, uniform weighting (1.0 per axis).

        Returns:
            Merged results ordered by RRF score (descending).
        """
        scores = {}       # file_path -> cumulative rrf_score
        result_map = {}   # file_path -> result dict
        axis_scores = {}  # file_path -> {axis_name: raw_score}

        for axis_name, results in result_lists:
            w = weights.get(axis_name, 1.0) if weights else 1.0
            for rank, result in enumerate(results):
                fp = result["file_path"]
                scores[fp] = scores.get(fp, 0) + w / (k + rank + 1)

                if fp not in axis_scores:
                    axis_scores[fp] = {}

                # Store raw per-axis score
                if axis_name == "visual":
                    axis_scores[fp]["visual"] = result.get("similarity", 0)
                elif axis_name == "text_vec":
                    axis_scores[fp]["text_vec"] = result.get("text_similarity", 0)
                elif axis_name == "fts":
                    axis_scores[fp]["fts_rank"] = result.get("fts_rank", 0)
                elif axis_name == "structure":
                    axis_scores[fp]["structure"] = result.get("structural_similarity", 0)

                if fp not in result_map:
                    result_map[fp] = result

        # Normalize FTS ranks to 0~1 (more negative = better match → higher score)
        fts_raw = {fp: s.get("fts_rank") for fp, s in axis_scores.items() if "fts_rank" in s}
        normalized_fts = {}
        if fts_raw:
            ranks = list(fts_raw.values())
            best = min(ranks)   # most negative = best match
            worst = max(ranks)  # closest to 0 = worst match
            span = worst - best
            for fp, r in fts_raw.items():
                normalized_fts[fp] = (worst - r) / span if span else 1.0

        # Sort by cumulative RRF score
        sorted_paths = sorted(scores.keys(), key=lambda fp: scores[fp], reverse=True)

        merged = []
        for fp in sorted_paths:
            result = result_map[fp]
            result["rrf_score"] = scores[fp]
            result["vector_score"] = axis_scores.get(fp, {}).get("visual")
            result["text_vec_score"] = axis_scores.get(fp, {}).get("text_vec")
            result["text_score"] = normalized_fts.get(fp)
            result["structure_score"] = axis_scores.get(fp, {}).get("structure")
            merged.append(result)

        return merged

    def _enrich_axis_scores(
        self,
        merged: List[Dict],
        v_embedding: Optional[np.ndarray],
        t_embedding: Optional[np.ndarray],
        fts_keywords: Optional[List[str]] = None,
        s_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """
        Enrich merged results with missing per-axis scores via direct DB lookup.

        Display-only: does NOT affect ranking (runs after RRF + trim).
        Computes actual V/S/M scores so all badges can be displayed in the UI.
        """
        if not merged:
            return

        # Collect file IDs missing VV scores
        v_missing = [r["id"] for r in merged if r.get("vector_score") is None and r.get("id")]
        # Collect file IDs missing MV scores
        s_missing = [r["id"] for r in merged if r.get("text_vec_score") is None and r.get("id")]
        # Collect file IDs missing FTS scores
        m_missing = [r["id"] for r in merged if r.get("text_score") is None and r.get("id")]
        # Collect file IDs missing Structure scores
        st_missing = [r["id"] for r in merged if r.get("structure_score") is None and r.get("id")]

        # Batch-compute VV similarity for missing files
        if v_missing and v_embedding is not None:
            v_scores = self._batch_similarity("vec_files", v_embedding, v_missing)
            for r in merged:
                if r.get("vector_score") is None and r.get("id") in v_scores:
                    r["vector_score"] = v_scores[r["id"]]

        # Batch-compute MV similarity for missing files
        if s_missing and t_embedding is not None:
            s_scores = self._batch_similarity("vec_text", t_embedding, s_missing)
            for r in merged:
                if r.get("text_vec_score") is None and r.get("id") in s_scores:
                    r["text_vec_score"] = s_scores[r["id"]]

        # Batch-compute FTS (FTS5 BM25) scores for missing files
        if m_missing and fts_keywords:
            m_scores = self._batch_fts_score(fts_keywords, m_missing)
            for r in merged:
                if r.get("text_score") is None and r.get("id") in m_scores:
                    r["text_score"] = m_scores[r["id"]]

        # Batch-compute Structure similarity for missing files
        if st_missing and s_embedding is not None:
            st_scores = self._batch_similarity("vec_structure", s_embedding, st_missing)
            for r in merged:
                if r.get("structure_score") is None and r.get("id") in st_scores:
                    r["structure_score"] = st_scores[r["id"]]
                if r.get("structural_similarity") is None and r.get("id") in st_scores:
                    r["structural_similarity"] = st_scores[r["id"]]

    @staticmethod
    def _query_tokens(query: str) -> List[str]:
        """Extract lightweight query tokens for soft intent matching."""
        if not query:
            return []
        tokens = []
        for raw in query.lower().split():
            t = raw.strip(" \t\r\n,.;:!?\"'()[]{}")
            if len(t) >= 2:
                tokens.append(t)
        # Preserve order, remove duplicates
        seen = set()
        uniq = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    @staticmethod
    def _safe_norm(val: Optional[float], low: float, high: float) -> Optional[float]:
        """Min-max normalize a score to [0, 1]."""
        if val is None:
            return None
        span = high - low
        if span <= 1e-12:
            return 1.0
        x = (float(val) - low) / span
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _quality_rerank(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
        query: str,
        llm_filters: Optional[Dict[str, Any]] = None,
        user_filters: Optional[Dict[str, Any]] = None,
        axis_weights: Optional[Dict[str, float]] = None,
        pool_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Quality-focused rerank over top candidate pool.

        Goal:
        - Keep RRF recall benefits
        - Promote results with stronger cross-axis agreement
        - Prefer entries with richer stored metadata (caption/tags/structured fields)
        """
        if not results:
            return results

        llm_filters = llm_filters or {}
        user_filters = user_filters or {}

        pool_n = min(len(results), int(pool_size or max(top_k * 3, 80)))
        if pool_n <= 1:
            return results

        pool = results[:pool_n]
        tail = results[pool_n:]

        # Axis ranges for normalization
        def _axis_range(key: str):
            vals = [float(r[key]) for r in pool if r.get(key) is not None]
            if not vals:
                return 0.0, 1.0
            return min(vals), max(vals)

        v_low, v_high = _axis_range("vector_score")
        x_low, x_high = _axis_range("structure_score")
        s_low, s_high = _axis_range("text_vec_score")
        m_low, m_high = _axis_range("text_score")

        # Axis blend defaults (same names as _rrf_merge_multi axes)
        axis_w = {
            "visual": 0.30,
            "structure": 0.15,
            "text_vec": 0.35,
            "fts": 0.20,
        }
        if axis_weights:
            axis_w.update({k: float(v) for k, v in axis_weights.items() if k in axis_w})

        q_tokens = self._query_tokens(query)
        q_path = (query or "").replace("\\", "/").strip().lower()
        path_hint = ("/" in q_path) or bool(q_path.endswith((".psd", ".png", ".jpg", ".jpeg")))
        soft_filter_keys = {
            "format", "image_type", "scene_type", "art_style",
            "time_of_day", "weather", "folder_path"
        }
        all_filters = {}
        all_filters.update(llm_filters)
        all_filters.update(user_filters)

        rescored = []
        n = len(pool)
        for idx, r in enumerate(pool):
            # Base rank prior from merged order
            rrf_prior = 1.0 if n <= 1 else 1.0 - (idx / (n - 1))

            # Per-axis normalized scores
            v_norm = self._safe_norm(r.get("vector_score"), v_low, v_high)
            x_norm = self._safe_norm(r.get("structure_score"), x_low, x_high)
            s_norm = self._safe_norm(r.get("text_vec_score"), s_low, s_high)
            m_norm = self._safe_norm(r.get("text_score"), m_low, m_high)

            axis_num = 0.0
            axis_den = 0.0
            if v_norm is not None:
                axis_num += axis_w["visual"] * v_norm
                axis_den += axis_w["visual"]
            if x_norm is not None:
                axis_num += axis_w["structure"] * x_norm
                axis_den += axis_w["structure"]
            if s_norm is not None:
                axis_num += axis_w["text_vec"] * s_norm
                axis_den += axis_w["text_vec"]
            if m_norm is not None:
                axis_num += axis_w["fts"] * m_norm
                axis_den += axis_w["fts"]
            axis_blend = (axis_num / axis_den) if axis_den > 0 else rrf_prior

            # Metadata completeness (stored-information utilization)
            has_caption = 1.0 if (r.get("mc_caption") or "").strip() else 0.0
            tags = r.get("ai_tags")
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = []
            has_tags = 1.0 if isinstance(tags, list) and len(tags) > 0 else 0.0
            has_struct = 1.0 if (r.get("image_type") or r.get("scene_type") or r.get("art_style")) else 0.0
            has_user = 1.0 if (r.get("user_note") or r.get("user_tags") or r.get("user_category")) else 0.0
            meta_completeness = (has_caption * 0.35) + (has_tags * 0.25) + (has_struct * 0.30) + (has_user * 0.10)

            # Soft intent match (query token overlap + filter consistency)
            hay_parts = [
                str(r.get("file_name") or ""),
                str(r.get("folder_path") or ""),
                str(r.get("relative_path") or ""),
                str(r.get("file_path") or ""),
                str(r.get("mc_caption") or ""),
                str(r.get("image_type") or ""),
                str(r.get("scene_type") or ""),
                str(r.get("art_style") or ""),
            ]
            if isinstance(tags, list):
                hay_parts.extend(str(t) for t in tags)
            hay = " ".join(hay_parts).lower()
            token_hits = sum(1 for t in q_tokens if t in hay)
            token_score = (token_hits / max(1, len(q_tokens))) if q_tokens else 0.0

            filter_hits = 0
            filter_total = 0
            for fk, fv in all_filters.items():
                if fk not in soft_filter_keys or fv in (None, ""):
                    continue
                filter_total += 1
                rv = str(r.get(fk) or "").lower()
                if rv and rv == str(fv).lower():
                    filter_hits += 1
            filter_score = (filter_hits / filter_total) if filter_total else 0.0

            # Path intent boost: if query itself is path-like, reward direct path hit.
            path_score = 0.0
            if path_hint and q_path:
                cands = [
                    str(r.get("relative_path") or "").replace("\\", "/").lower(),
                    str(r.get("file_path") or "").replace("\\", "/").lower(),
                    str(r.get("folder_path") or "").replace("\\", "/").lower(),
                ]
                for cp in cands:
                    if not cp:
                        continue
                    if cp == q_path or cp.endswith(q_path):
                        path_score = 1.0
                        break
                    if q_path in cp:
                        path_score = 0.8
                        break

            intent_boost = (token_score * 0.55) + (filter_score * 0.25) + (path_score * 0.20)

            # Final quality score
            quality_score = (
                (0.62 * axis_blend) +
                (0.23 * rrf_prior) +
                (0.10 * meta_completeness) +
                (0.05 * intent_boost)
            )

            r["quality_score"] = quality_score
            rescored.append((r, quality_score, idx))

        rescored.sort(key=lambda x: (x[1], x[0].get("rrf_score", 0.0), -x[2]), reverse=True)
        return [x[0] for x in rescored] + tail

    def _batch_similarity(
        self,
        table: str,
        query_embedding: np.ndarray,
        file_ids: List[int],
    ) -> Dict[int, float]:
        """
        Compute cosine similarity for specific files against a query embedding.

        Uses JOIN with files table (vec0 virtual tables don't support
        arbitrary WHERE clauses directly).

        Args:
            table: "vec_files", "vec_text", or "vec_structure"
            query_embedding: Pre-encoded query embedding vector
            file_ids: List of file IDs to compute similarity for

        Returns:
            Dict mapping file_id -> similarity score
        """
        if not file_ids:
            return {}
        if table not in {"vec_files", "vec_text", "vec_structure"}:
            logger.warning(f"Batch similarity lookup rejected unknown table: {table}")
            return {}

        embedding_json = json.dumps(query_embedding.astype(np.float32).tolist())
        placeholders = ",".join("?" * len(file_ids))
        cursor = self.db.conn.cursor()
        try:
            cursor.execute(f"""
                SELECT f.id,
                       (1.0 - vec_distance_cosine(v.embedding, ?)) AS sim
                FROM files f
                JOIN {table} v ON f.id = v.file_id
                WHERE f.id IN ({placeholders})
            """, (embedding_json, *file_ids))
            return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.warning(f"Batch similarity lookup failed ({table}): {e}")
            return {}
        finally:
            cursor.close()

    def _batch_fts_score(
        self,
        fts_keywords: List[str],
        file_ids: List[int],
    ) -> Dict[int, float]:
        """
        Compute normalized FTS5 BM25 scores for specific files.

        Returns scores normalized to 0-1 range. Files not matching
        the keywords at all are omitted from the result.
        """
        if not file_ids or not fts_keywords:
            return {}

        # Build FTS5 MATCH expression (OR of all keywords)
        safe_kw = [kw.replace('"', '""') for kw in fts_keywords if kw.strip()]
        if not safe_kw:
            return {}
        match_expr = " OR ".join(f'"{kw}"' for kw in safe_kw)

        placeholders = ",".join("?" * len(file_ids))
        cursor = self.db.conn.cursor()
        try:
            cursor.execute(f"""
                SELECT rowid, rank
                FROM files_fts
                WHERE files_fts MATCH ? AND rowid IN ({placeholders})
            """, (match_expr, *file_ids))
            raw = {row[0]: row[1] for row in cursor.fetchall()}

            if not raw:
                return {}

            # Normalize BM25 ranks to 0-1 (more negative = better)
            ranks = list(raw.values())
            best = min(ranks)
            worst = max(ranks)
            span = worst - best
            return {
                fid: (worst - r) / span if span else 1.0
                for fid, r in raw.items()
            }
        except Exception as e:
            logger.warning(f"Batch FTS score lookup failed: {e}")
            return {}
        finally:
            cursor.close()

    def _apply_user_filters(
        self,
        results: List[Dict],
        filters: Dict[str, Any],
        strict: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to results in-memory.

        Args:
            results: Search results to filter
            filters: Metadata filter dict
            strict: If True (user filters), exclude results missing the field.
                    If False (LLM filters), pass results where the field is None/empty.

        Supported filters:
            format: File format (PSD, PNG, JPG)
            user_category: User-assigned category
            min_rating: Minimum star rating (1-5)
            user_tags: Tag that must be present
            dominant_color_hint: Boost results matching this color (soft filter)
            image_type: v3 P0 image classification filter
            art_style: v3 P0 art style filter
            scene_type: v3 P0 scene type filter (backgrounds)
            time_of_day: v3 P0 time of day filter (backgrounds)
            weather: v3 P0 weather filter (backgrounds)
        """
        filtered = []

        for result in results:
            # Hard filters (exclude non-matching)
            if "format" in filters and filters["format"]:
                if result.get("format", "").upper() != filters["format"].upper():
                    continue

            if "user_category" in filters and filters["user_category"]:
                if result.get("user_category", "") != filters["user_category"]:
                    continue

            if "min_rating" in filters and filters["min_rating"]:
                if (result.get("user_rating") or 0) < int(filters["min_rating"]):
                    continue

            if "user_tags" in filters and filters["user_tags"]:
                result_tags = result.get("user_tags", [])
                if isinstance(result_tags, str):
                    try:
                        result_tags = json.loads(result_tags)
                    except:
                        result_tags = []
                filter_tag = filters["user_tags"].lower()
                if not any(filter_tag in t.lower() for t in result_tags):
                    continue

            if "folder_path" in filters and filters["folder_path"]:
                result_folder = result.get("folder_path") or ""
                if not result_folder.startswith(filters["folder_path"]):
                    continue

            if "folder_tag" in filters and filters["folder_tag"]:
                result_ftags = result.get("folder_tags", [])
                if isinstance(result_ftags, str):
                    try:
                        result_ftags = json.loads(result_ftags)
                    except:
                        result_ftags = []
                filter_ftag = filters["folder_tag"].lower()
                if not any(filter_ftag in t.lower() for t in result_ftags):
                    continue

            # v3 P0: structured vision filters
            # When strict=False (LLM filters), pass results that lack the field entirely
            if "image_type" in filters and filters["image_type"]:
                result_val = (result.get("image_type") or "")
                if not strict and not result_val:
                    pass  # lenient: skip filter when result lacks the field
                elif result_val.lower() != filters["image_type"].lower():
                    continue

            if "art_style" in filters and filters["art_style"]:
                result_val = (result.get("art_style") or "")
                if not strict and not result_val:
                    pass
                elif result_val.lower() != filters["art_style"].lower():
                    continue

            if "scene_type" in filters and filters["scene_type"]:
                result_val = (result.get("scene_type") or "")
                if not strict and not result_val:
                    pass
                elif result_val.lower() != filters["scene_type"].lower():
                    continue

            if "time_of_day" in filters and filters["time_of_day"]:
                result_val = (result.get("time_of_day") or "")
                if not strict and not result_val:
                    pass
                elif result_val.lower() != filters["time_of_day"].lower():
                    continue

            if "weather" in filters and filters["weather"]:
                result_val = (result.get("weather") or "")
                if not strict and not result_val:
                    pass
                elif result_val.lower() != filters["weather"].lower():
                    continue

            filtered.append(result)

        return filtered

    def _apply_negative_filter(
        self,
        results: List[Dict],
        negative_query: str,
        neg_v_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Post-filter: demote results matching negative concepts via two layers:

        Layer 1 (Text): Check mc_caption + ai_tags for negative term text matches.
        Layer 2 (Visual): If neg_v_embedding is provided, compute VV similarity
                          between each result's stored visual embedding and the
                          negative concept. High visual similarity → demote.

        Results are scored by negative match strength and reordered (not removed).

        Args:
            results: Search results to filter
            negative_query: Space-separated negative terms (e.g. "cold snow winter")
            neg_v_embedding: Optional SigLIP2 embedding of negative_query for VV penalty

        Returns:
            Reordered results with negative-matching items demoted to the end
        """
        if not negative_query or not results:
            return results

        neg_terms = [t.lower().strip() for t in negative_query.split() if t.strip()]
        if not neg_terms:
            return results

        # Layer 2: Compute VV negative similarity scores
        neg_v_scores = {}
        if neg_v_embedding is not None:
            file_ids = [r["id"] for r in results if r.get("id")]
            if file_ids:
                neg_v_scores = self._batch_similarity("vec_files", neg_v_embedding, file_ids)

        # Compute per-result negative match score using two layers:
        # Layer 1 (Text): direct term matching on captions/tags
        # Layer 2 (Visual): statistical outlier detection on neg_v_sim distribution
        #
        # VV scoring uses distribution-based outlier detection because
        # text-to-image and image-to-image similarities live on different scales:
        #   - Text-only search: pos ~0.10-0.17, neg ~0.04-0.08 (same scale)
        #   - Image+text search: pos ~0.70-0.80, neg ~0.04-0.08 (different scales)
        # A simple ratio fails for image search. Instead, we flag results whose
        # neg_v_sim is significantly above the group mean (statistical outlier).

        # Pre-compute visual negative outlier threshold
        v_outlier_flag = {}
        if neg_v_scores:
            all_neg_sims = list(neg_v_scores.values())
            if len(all_neg_sims) >= 3:
                mean_neg = sum(all_neg_sims) / len(all_neg_sims)
                variance = sum((x - mean_neg) ** 2 for x in all_neg_sims) / len(all_neg_sims)
                std_neg = variance ** 0.5
                # Outlier: neg_v_sim > mean + 1.0 * std (top ~16% of distribution)
                outlier_thresh = mean_neg + 1.0 * std_neg
                for fid, sim in neg_v_scores.items():
                    if sim > outlier_thresh:
                        v_outlier_flag[fid] = sim
                logger.debug(
                    f"Negative VV stats: mean={mean_neg:.4f}, std={std_neg:.4f}, "
                    f"outlier_thresh={outlier_thresh:.4f}, outliers={len(v_outlier_flag)}"
                )

        scored = []
        for r in results:
            # Layer 1: Text-based negative match
            caption = (r.get("mc_caption") or "").lower()
            tags = r.get("ai_tags") or []
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except (json.JSONDecodeError, TypeError):
                    tags = []
            if not isinstance(tags, list):
                tags = []
            tags_text = " ".join(str(t).lower() for t in tags)
            combined_text = f"{caption} {tags_text}"

            text_neg = sum(1 for term in neg_terms if term in combined_text)

            # Layer 2: Visual outlier detection
            fid = r.get("id")
            v_is_outlier = fid in v_outlier_flag

            # Combined: text match = 1.0 per hit, visual outlier = 0.6
            neg_score = (text_neg * 1.0) + (0.6 if v_is_outlier else 0.0)

            scored.append((r, neg_score, text_neg, v_is_outlier))

        # Threshold: demote if neg_score > 0.45
        # - Text-only: 1 term hit → 1.0 (always demotes)
        # - VV outlier: 0.6 (demotes alone)
        # - Combined: even weak signals add up
        neg_threshold = 0.45
        filtered = [(r, ns) for r, ns, _, _ in scored if ns <= neg_threshold]
        demoted = [(r, ns) for r, ns, _, _ in scored if ns > neg_threshold]

        # Sort demoted by neg_score descending (worst offenders last)
        demoted.sort(key=lambda x: x[1], reverse=True)

        if demoted:
            demoted_info = [
                (r.get("file_name", "?"), round(ns, 3))
                for r, ns in demoted[:5]
            ]
            logger.info(
                f"Negative filter: demoted {len(demoted)} results "
                f"(threshold={neg_threshold}, terms={neg_terms[:5]}, "
                f"outlier_count={len(v_outlier_flag)}, "
                f"top_demoted={demoted_info})"
            )

        # Demoted results go to the end (not removed)
        return [r for r, _ in filtered] + [r for r, _ in demoted]

    @staticmethod
    def _parse_json_fields(result: Dict) -> None:
        """Parse JSON string fields in a result dict."""
        if result.get("ai_tags"):
            try:
                result["ai_tags"] = json.loads(result["ai_tags"])
            except (json.JSONDecodeError, TypeError):
                result["ai_tags"] = []
        if result.get("metadata"):
            try:
                result["metadata"] = json.loads(result["metadata"])
            except (json.JSONDecodeError, TypeError):
                result["metadata"] = {}
        if result.get("user_tags") and isinstance(result["user_tags"], str):
            try:
                result["user_tags"] = json.loads(result["user_tags"])
            except (json.JSONDecodeError, TypeError):
                result["user_tags"] = []
        if result.get("folder_tags") and isinstance(result["folder_tags"], str):
            try:
                result["folder_tags"] = json.loads(result["folder_tags"])
            except (json.JSONDecodeError, TypeError):
                result["folder_tags"] = []

    def triaxis_image_search(
        self,
        query: str,
        image_embeddings: list,
        structure_embeddings: Optional[list[np.ndarray]] = None,
        image_mode: str = "and",
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Combined text + image search.

        Axes:
        - VV: SigLIP image embeddings
        - X: DINOv2 structure embeddings (optional)
        - MV: Qwen text embeddings
        - FTS: metadata lexical search

        Args:
            query: Text query for MV and FTS
            image_embeddings: Pre-computed SigLIP2 image embeddings
            structure_embeddings: Optional pre-computed DINOv2 embeddings
            image_mode: "and" (average) or "or" (union)
            filters: Optional metadata filters
            top_k: Number of results
            threshold: Similarity threshold
        """
        from backend.search.rrf import get_weights
        from backend.utils.config import get_config as _cfg
        _search_cfg = _cfg()

        candidate_mul = _search_cfg.get("search.rrf.candidate_multiplier", 5)
        candidate_k = top_k * candidate_mul
        v_threshold = _search_cfg.get("search.threshold.visual", 0.05)
        x_threshold = _search_cfg.get("search.threshold.structure", v_threshold)
        tv_threshold = _search_cfg.get("search.threshold.text_vec", threshold)

        # VV: image embeddings
        if image_mode == "and":
            mean_emb = np.mean(image_embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            vector_results = self.vector_search_by_embedding(mean_emb, candidate_k, v_threshold)
        else:  # OR
            all_v = {}
            for emb in image_embeddings:
                for r in self.vector_search_by_embedding(emb, candidate_k, v_threshold):
                    fid = r.get("id")
                    if fid not in all_v or r.get("similarity", 0) > all_v[fid].get("similarity", 0):
                        all_v[fid] = r
            vector_results = sorted(
                all_v.values(), key=lambda x: x.get("similarity", 0), reverse=True
            )[:candidate_k]

        # X: structure embeddings (DINOv2)
        structure_results = []
        if structure_embeddings:
            if image_mode == "and":
                mean_struct = np.mean(structure_embeddings, axis=0).astype(np.float32)
                norm = np.linalg.norm(mean_struct)
                if norm > 0:
                    mean_struct = mean_struct / norm
                structure_results = self.search_structure(mean_struct, candidate_k, x_threshold)
            else:
                all_x = {}
                for emb in structure_embeddings:
                    for r in self.search_structure(emb, candidate_k, x_threshold):
                        fid = r.get("id")
                        sim = r.get("structural_similarity", 0)
                        if fid not in all_x or sim > all_x[fid].get("structural_similarity", 0):
                            all_x[fid] = r
                structure_results = sorted(
                    all_x.values(), key=lambda x: x.get("structural_similarity", 0), reverse=True
                )[:candidate_k]

        # MV: text query (Qwen3-Embedding)
        decomposer = QueryDecomposer()
        plan = decomposer.decompose(query)
        vector_query = plan.get("vector_query", query)
        query_type = plan.get("query_type", "balanced")

        text_vec_results = []
        t_query_embedding = None
        if self.text_search_enabled:
            try:
                t_query_embedding = self.text_provider.encode(vector_query, is_query=True)
                text_vec_results = self._text_vector_search_by_embedding(
                    t_query_embedding, top_k=candidate_k, threshold=tv_threshold
                )
            except Exception as e:
                logger.warning(f"MV search unavailable in triaxis_image: {e}")

        # FTS: FTS5 keywords (text query)
        fts_keywords = plan.get("fts_keywords", [query])
        exclude_kw = plan.get("exclude_keywords", [])
        fts_results = []
        try:
            fts_results = self.fts_search(fts_keywords, top_k=candidate_k, exclude_keywords=exclude_kw)
        except Exception as e:
            logger.warning(f"FTS search unavailable in triaxis_image: {e}")

        # RRF merge (3~4 axes)
        all_result_lists = []
        if vector_results:
            all_result_lists.append(("visual", vector_results))
        if structure_results:
            all_result_lists.append(("structure", structure_results))
        if text_vec_results:
            all_result_lists.append(("text_vec", text_vec_results))
        if fts_results:
            all_result_lists.append(("fts", fts_results))

        rrf_weights = None
        if len(all_result_lists) >= 2:
            rrf_k = _search_cfg.get("search.rrf.k", 60)
            active_axes = [name for name, _ in all_result_lists]
            rrf_weights = get_weights(query_type, active_axes)
            merged = self._rrf_merge_multi(all_result_lists, k=rrf_k, weights=rrf_weights)
        elif len(all_result_lists) == 1:
            _, merged = all_result_lists[0]
            for r in merged:
                r["vector_score"] = r.get("similarity", r.get("vector_score"))
        else:
            merged = []

        # Filters
        user_filters = filters or {}
        llm_filters = plan.get("filters", {})
        negative_query = plan.get("negative_query", "")

        # VV negative embedding for visual penalty
        neg_v_embedding = None
        if negative_query:
            try:
                neg_v_embedding = self.encode_text(negative_query)
            except Exception as e:
                logger.debug(f"Negative VV encoding failed in image search: {e}")
            merged = self._apply_negative_filter(merged, negative_query, neg_v_embedding)

        if llm_filters:
            merged = self._apply_user_filters(merged, llm_filters, strict=False)
        if user_filters:
            merged = self._apply_user_filters(merged, user_filters, strict=True)

        # Enrich missing axis scores (VV/X use image embeddings)
        if image_mode == "and" and len(image_embeddings) > 0:
            v_emb_for_enrich = np.mean(image_embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(v_emb_for_enrich)
            if norm > 0:
                v_emb_for_enrich = v_emb_for_enrich / norm
        elif image_embeddings:
            v_emb_for_enrich = image_embeddings[0]
        else:
            v_emb_for_enrich = None

        if structure_embeddings and image_mode == "and":
            x_emb_for_enrich = np.mean(structure_embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(x_emb_for_enrich)
            if norm > 0:
                x_emb_for_enrich = x_emb_for_enrich / norm
        elif structure_embeddings:
            x_emb_for_enrich = structure_embeddings[0]
        else:
            x_emb_for_enrich = None

        # Quality rerank on filtered candidate pool
        rerank_enabled = bool(_search_cfg.get("search.rerank.enabled", True))
        rerank_pool = int(_search_cfg.get("search.rerank.pool_size", max(top_k * 3, 80)))
        rerank_pool = max(top_k, rerank_pool)
        if rerank_enabled and len(merged) > 1:
            rerank_n = min(len(merged), rerank_pool)
            self._enrich_axis_scores(
                merged[:rerank_n],
                v_emb_for_enrich,
                t_query_embedding,
                fts_keywords,
                s_embedding=x_emb_for_enrich,
            )
            merged = self._quality_rerank(
                merged,
                top_k=top_k,
                query=query,
                llm_filters=llm_filters,
                user_filters=user_filters,
                axis_weights=rrf_weights,
                pool_size=rerank_n,
            )

        merged = merged[:top_k]

        self._enrich_axis_scores(
            merged,
            v_emb_for_enrich,
            t_query_embedding,
            fts_keywords,
            s_embedding=x_emb_for_enrich,
        )

        logger.info(
            f"Triaxis image search '{query}' + {len(image_embeddings)} images: "
            f"V={len(vector_results)}, X={len(structure_results)}, S={len(text_vec_results)}, "
            f"M={len(fts_results)}, merged={len(merged)}"
        )

        # Write diagnostic log
        if _DIAGNOSTIC_ENABLED:
            diag = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "search_type": "triaxis_image",
                "image_count": len(image_embeddings),
                "image_mode": image_mode,
                "top_k": top_k,
                "decomposition": {
                    "vector_query": plan.get("vector_query"),
                    "negative_query": plan.get("negative_query"),
                    "exclude_keywords": plan.get("exclude_keywords"),
                    "query_type": query_type,
                },
                "axis_counts": {
                    "V": len(vector_results),
                    "X": len(structure_results),
                    "S": len(text_vec_results),
                    "M": len(fts_results),
                },
                "negative_v_axis": neg_v_embedding is not None,
                "final_results_count": len(merged),
                "final_top5": [
                    {
                        "file": r.get("file_name", r.get("file_path", "")),
                        "vector_score": round(r["vector_score"], 4) if r.get("vector_score") is not None else None,
                        "structure_score": round(r["structure_score"], 4) if r.get("structure_score") is not None else None,
                        "text_vec_score": round(r["text_vec_score"], 4) if r.get("text_vec_score") is not None else None,
                        "text_score": round(r["text_score"], 4) if r.get("text_score") is not None else None,
                        "quality_score": round(r["quality_score"], 4) if r.get("quality_score") is not None else None,
                    }
                    for r in merged[:5]
                ],
            }
            self._write_diagnostic(diag)

        return merged

    def multi_image_search(
        self,
        query_images: List[str],
        mode: str = "and",
        top_k: int = 20,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Multi-image search with AND/OR modes.

        Args:
            query_images: List of base64-encoded images
            mode: "and" (similar to ALL) or "or" (similar to ANY)
            top_k: Number of results
            threshold: Similarity threshold
        """
        # Encode all images to embeddings
        embeddings = []
        for img_b64 in query_images:
            emb = self.encoder.encode_image_from_base64(img_b64)
            embeddings.append(emb)
        logger.info(f"Multi-image search: {len(embeddings)} images, mode={mode}")

        if mode == "and":
            # Average embeddings → re-normalize → single search
            mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            results = self.vector_search_by_embedding(mean_emb, top_k, threshold)
        else:
            # OR: search per image → union with max score per file
            all_results = {}
            for emb in embeddings:
                hits = self.vector_search_by_embedding(emb, top_k, threshold)
                for r in hits:
                    fid = r.get("id")
                    sim = r.get("similarity", 0)
                    if fid not in all_results or sim > all_results[fid].get("similarity", 0):
                        all_results[fid] = r
            results = sorted(
                all_results.values(),
                key=lambda x: x.get("similarity", 0),
                reverse=True
            )[:top_k]

        for r in results:
            r["vector_score"] = r.get("similarity", 0)
            r["text_vec_score"] = None
            r["text_score"] = None
        logger.info(f"Multi-image search returned {len(results)} results")
        return results

    def search(
        self,
        query: str = "",
        mode: str = "vector",
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        threshold: float = 0.0,
        return_diagnostic: bool = False,
        query_image: Optional[str] = None,
        query_images: Optional[List[str]] = None,
        image_search_mode: str = "and",
        query_file_id: Optional[int] = None,
    ):
        """
        Unified search interface (compatibility with VectorSearcher).

        Args:
            query: Search query (text)
            mode: "vector", "hybrid", "metadata", "fts", "triaxis", or "structure"
            filters: Optional metadata filters
            top_k: Number of results
            threshold: Similarity threshold (vector modes only)
            return_diagnostic: If True and mode=triaxis, return (results, diagnostic)
            query_image: Base64-encoded image for single image-to-image search
            query_images: List of base64-encoded images for multi-image search
            image_search_mode: "and" or "or" (for multi-image search)
            query_file_id: File ID for "find similar" queries (avoids re-encoding)

        Returns:
            Search results. If return_diagnostic=True with triaxis mode,
            returns (results, diagnostic_dict).
        """
        # Structure Search (DINOv2)
        if mode == "structure":
            if query_file_id:
                return self.find_similar_structure(query_file_id, top_k, threshold)
            elif query_image:
                embedding = self.structure_encoder.encode_image_from_base64(query_image)
                return self.search_structure(embedding, top_k, threshold)
            else:
                raise ValueError("Structure search requires 'query_file_id' or 'query_image'")

        # Combined text + image search (triaxis with VV=image, MV+FTS=text)
        has_images = (query_images and len(query_images) > 0) or query_image
        has_text = bool(query and query.strip())

        if has_text and has_images:
            # Encode images
            embeddings = []
            structure_embeddings = []
            if query_images and len(query_images) > 0:
                for img_b64 in query_images:
                    embeddings.append(self.encoder.encode_image_from_base64(img_b64))
                    structure_embeddings.append(self.structure_encoder.encode_image_from_base64(img_b64))
            elif query_image:
                embeddings.append(self.encoder.encode_image_from_base64(query_image))
                structure_embeddings.append(self.structure_encoder.encode_image_from_base64(query_image))

            return self.triaxis_image_search(
                query, embeddings, structure_embeddings, image_search_mode, filters, top_k, threshold
            )

        # Multi-image search (images only, no text)
        if query_images and len(query_images) > 0:
            results = self.multi_image_search(query_images, image_search_mode, top_k, threshold)
            if filters:
                results = self._apply_user_filters(results, filters)
            return results

        # Single image-to-image search (backward compatible, image only)
        # Note: If mode="structure" was intended, it's handled above.
        # This block is for legacy/default visual search (SigLIP).
        if query_image:
            image_embedding = self.encoder.encode_image_from_base64(query_image)
            results = self.vector_search_by_embedding(image_embedding, top_k, threshold)
            if filters:
                results = self._apply_user_filters(results, filters)
            for r in results:
                r["vector_score"] = r.get("similarity", 0)
                r["text_vec_score"] = None
                r["text_score"] = None
            logger.info(f"Image search returned {len(results)} results")
            return results

        # Visual similarity by file ID (e.g. "Find Similar (Visual)" context menu)
        if mode == "vector" and query_file_id:
            return self.find_similar_visual(query_file_id, top_k, threshold)

        if mode == "vector":
            return self.vector_search(query, top_k, threshold)
        elif mode == "hybrid":
            return self.hybrid_search(query, filters, top_k, threshold)
        elif mode == "metadata":
            if not filters:
                raise ValueError("Metadata mode requires filters")
            return self.metadata_query(filters, top_k)
        elif mode == "fts":
            return self.fts_search([query], top_k)
        elif mode == "triaxis":
            return self.triaxis_search(query, filters, top_k, threshold, return_diagnostic=return_diagnostic)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'vector', 'hybrid', 'metadata', 'fts', 'triaxis', or 'structure'")
