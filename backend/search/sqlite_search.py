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
from backend.search.encoders import EncoderManager
from backend.search.scoring import (
    safe_norm,
    query_tokens,
    rrf_merge,
    rrf_merge_multi,
    enrich_axis_scores,
    quality_rerank,
    apply_user_filters,
    apply_negative_filter,
)

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
        self.encoders = EncoderManager(db=self.db)

        logger.info("SqliteVectorSearch initialized")

    @property
    def encoder(self):
        """Lazy load VV embedding encoder (SigLIP 2). Delegates to EncoderManager."""
        return self.encoders.vv_encoder

    @property
    def text_provider(self):
        """Lazy load MV provider. Delegates to EncoderManager."""
        return self.encoders.mv_encoder

    @property
    def structure_encoder(self):
        """Lazy load DINOv2 structure encoder. Delegates to EncoderManager."""
        return self.encoders.structure_encoder

    @property
    def text_search_enabled(self) -> bool:
        """Check if MV search is available (vec_text table exists with data)."""
        return self.encoders.text_search_enabled

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query to VV embedding vector (SigLIP 2).

        Args:
            text: Text query

        Returns:
            Embedding vector
        """
        return self.encoders.encode_text(text)

    def encode_structure(self, image) -> np.ndarray:
        """
        Encode image to Structure embedding vector (DINOv2).

        Args:
            image: PIL Image or scalar

        Returns:
            768-dim embedding vector
        """
        return self.encoders.encode_structure(image)

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
                WHERE f.preview_only = 0
                  AND (1.0 - vec_distance_cosine(v.embedding, ?)) >= ?
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
                WHERE f.preview_only = 0
                  AND (1.0 - vec_distance_cosine(v.embedding, ?)) >= ?
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

        where_clauses.insert(0, "preview_only = 0")
        where_sql = " AND ".join(where_clauses)
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
                WHERE f.preview_only = 0
                  AND (1.0 - vec_distance_cosine(vt.embedding, ?)) >= ?
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
                WHERE f.preview_only = 0
                  AND (1.0 - vec_distance_cosine(vt.embedding, ?)) >= ?
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
                WHERE f.preview_only = 0
                  AND (1.0 - vec_distance_cosine(vs.embedding, ?)) >= ?
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
                  AND f.preview_only = 0
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

    # ------------------------------------------------------------------
    # Plan-based search: Codex generates search plan, engine executes it
    # ------------------------------------------------------------------

    def plan_search(
        self,
        query: str,
        top_k: int = 20,
        threshold: float = 0.0,
        return_diagnostic: bool = False,
        _plan: dict = None,
    ) -> List[Dict[str, Any]]:
        """
        2-stage search: Codex generates a search plan, engine executes it.

        Stage 1: QueryDecomposer (Codex CLI) → search plan with pre_filter + search intent
        Stage 2: pre_filter → SQL WHERE → file_id set → vector search within that set

        Falls back to triaxis_search if Codex fails or no pre_filter.
        """
        t_start = time.perf_counter()

        # Use provided plan or decompose
        plan = _plan
        if not plan:
            decomposer = QueryDecomposer()
            plan = decomposer.decompose(query)

        # Read unified schema
        scope = plan.get("scope", {})
        find = plan.get("find", {})
        exclude = plan.get("exclude", {})
        fallback_kw = find.get("keywords", [])

        # If no scope, fall back to standard triaxis
        if not scope or not any(scope.get(k) for k in ("folder", "image_type", "format")):
            logger.info(f"Plan search: no scope, falling back to triaxis")
            return self.triaxis_search(query, None, top_k, threshold, return_diagnostic=return_diagnostic)

        # Stage 1: Apply scope → get file_id set
        file_ids = self._apply_plan_filter(scope)
        if not file_ids:
            logger.warning(f"Plan search: scope matched 0 files, falling back to triaxis")
            return self.triaxis_search(query, None, top_k, threshold, return_diagnostic=return_diagnostic)

        logger.info(f"Plan search: scope matched {len(file_ids)} files (scope={scope})")

        # Stage 2: VV + MV vector search within file_id set
        search_query = find.get("description", query)

        # MV: semantic similarity within scope
        mv_results = self._mv_search_within(search_query, file_ids, top_k * 3, threshold)

        # VV: visual similarity within scope
        vv_results = self._vv_search_within(search_query, file_ids, top_k * 3, threshold)

        # RRF merge VV + MV (within scope)
        if mv_results and vv_results:
            # Simple RRF: combine ranks from both axes
            path_to_mv_rank = {r["file_path"]: i + 1 for i, r in enumerate(mv_results)}
            path_to_vv_rank = {r["file_path"]: i + 1 for i, r in enumerate(vv_results)}
            all_paths = set(path_to_mv_rank) | set(path_to_vv_rank)

            rrf_k = 60
            scored = []
            for fp in all_paths:
                mv_rank = path_to_mv_rank.get(fp, len(mv_results) + 100)
                vv_rank = path_to_vv_rank.get(fp, len(vv_results) + 100)
                # MV weight 0.6, VV weight 0.4 (semantic > visual for text queries)
                score = 0.6 / (mv_rank + rrf_k) + 0.4 / (vv_rank + rrf_k)
                scored.append((fp, score))
            scored.sort(key=lambda x: x[1], reverse=True)

            # Build result list from top scored paths
            mv_map = {r["file_path"]: r for r in mv_results}
            vv_map = {r["file_path"]: r for r in vv_results}
            results = []
            for fp, rrf_score in scored[:top_k]:
                r = mv_map.get(fp) or vv_map.get(fp)
                if r:
                    # Enrich with scores from both axes
                    if fp in mv_map:
                        r["text_vec_score"] = mv_map[fp].get("text_vec_score")
                    if fp in vv_map:
                        r["vector_score"] = vv_map[fp].get("vector_score")
                    r["rrf_score"] = rrf_score
                    results.append(r)
        elif mv_results:
            results = mv_results[:top_k]
        elif vv_results:
            results = vv_results[:top_k]
        else:
            results = []

        # Fallback: if not enough results, add FTS matches within scope
        if len(results) < top_k and fallback_kw:
            existing_ids = {r["id"] for r in results}
            fts_fill = self.fts_search(fallback_kw, top_k * 2)
            for r in fts_fill:
                if r["id"] in file_ids and r["id"] not in existing_ids:
                    r["vector_score"] = None
                    r["text_vec_score"] = None
                    r["text_score"] = 0.5
                    results.append(r)
                    existing_ids.add(r["id"])
                    if len(results) >= top_k:
                        break

        results = results[:top_k]

        elapsed = time.perf_counter() - t_start
        logger.info(f"Plan search '{query}': {len(results)} results in {elapsed:.1f}s "
                     f"(pre_filter={len(file_ids)} files)")

        if return_diagnostic:
            diag = {
                "mode": "plan",
                "pre_filter": pre_filter,
                "search_query": search_query,
                "pre_filter_count": len(file_ids),
                "result_count": len(results),
                "total_ms": round(elapsed * 1000, 1),
            }
            return results, diag
        return results

    def _apply_plan_filter(self, pre_filter: dict) -> set:
        """Apply pre_filter to get file_id set from DB."""
        cursor = self.db.conn.cursor()
        conditions = ["preview_only = 0"]
        params = []

        folder = pre_filter.get("folder", "")
        if folder:
            conditions.append("(folder_path LIKE ? OR file_path LIKE ?)")
            params.extend([f"%{folder}%", f"%{folder}%"])

        img_type = pre_filter.get("image_type")
        if img_type:
            conditions.append("image_type = ?")
            params.append(img_type)

        fmt = pre_filter.get("format")
        if fmt:
            conditions.append("UPPER(format) = ?")
            params.append(fmt.upper())

        where = " AND ".join(conditions)
        cursor.execute(f"SELECT id FROM files WHERE {where}", params)
        return {row[0] for row in cursor.fetchall()}

    def _mv_search_within(
        self, query: str, file_ids: set, top_k: int = 20, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """MV vector search within a specific file_id set.

        Loads vectors for the file_id set and computes cosine similarity
        in-memory (sqlite-vec doesn't support WHERE + ORDER BY distance).
        """
        if not self.text_search_enabled:
            return []

        try:
            q_vec = self.text_provider.encode(query, is_query=True)
        except Exception as e:
            logger.warning(f"MV encode failed: {e}")
            return []

        # Load vectors for file_ids
        cursor = self.db.conn.cursor()
        id_list = list(file_ids)

        # Batch query (SQLite has a limit of ~999 params, chunk if needed)
        all_scores = []
        for chunk_start in range(0, len(id_list), 500):
            chunk = id_list[chunk_start:chunk_start + 500]
            placeholders = ','.join('?' * len(chunk))
            cursor.execute(f"""
                SELECT vt.file_id, vt.embedding
                FROM vec_text vt
                WHERE vt.file_id IN ({placeholders})
            """, chunk)

            for row in cursor.fetchall():
                fid = row[0]
                raw = row[1]
                if isinstance(raw, bytes):
                    emb = np.frombuffer(raw, dtype=np.float32)
                else:
                    emb = np.array(json.loads(raw), dtype=np.float32)
                # Cosine similarity
                sim = float(np.dot(q_vec, emb) / (np.linalg.norm(q_vec) * np.linalg.norm(emb) + 1e-8))
                if sim >= threshold:
                    all_scores.append((fid, sim))

        # Sort by similarity descending
        all_scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = all_scores[:top_k]

        if not top_ids:
            return []

        # Fetch full file metadata for top results
        results = []
        for fid, sim in top_ids:
            id_placeholder = '?'
            cursor.execute(f"""
                SELECT f.* FROM files f WHERE f.id = {id_placeholder}
            """, (fid,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                self._parse_json_fields(result)
                result["text_similarity"] = sim
                result["text_vec_score"] = sim
                result["vector_score"] = None
                result["text_score"] = None
                results.append(result)

        logger.info(f"MV search within {len(file_ids)} files: {len(results)} results "
                     f"(top sim={all_scores[0][1]:.3f})" if all_scores else "")
        return results

    def _vv_search_within(
        self, query: str, file_ids: set, top_k: int = 20, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """VV visual search within a specific file_id set.

        Encodes query text with SigLIP2, loads VV vectors for file_ids,
        and computes cosine similarity in-memory.
        """
        try:
            q_vec = self.encode_text(query)
        except Exception as e:
            logger.warning(f"VV encode failed: {e}")
            return []

        cursor = self.db.conn.cursor()
        id_list = list(file_ids)

        all_scores = []
        for chunk_start in range(0, len(id_list), 500):
            chunk = id_list[chunk_start:chunk_start + 500]
            placeholders = ','.join('?' * len(chunk))
            cursor.execute(f"""
                SELECT vf.file_id, vf.embedding
                FROM vec_files vf
                WHERE vf.file_id IN ({placeholders})
            """, chunk)

            for row in cursor.fetchall():
                fid = row[0]
                raw = row[1]
                if isinstance(raw, bytes):
                    emb = np.frombuffer(raw, dtype=np.float32)
                else:
                    emb = np.array(json.loads(raw), dtype=np.float32)
                sim = float(np.dot(q_vec, emb) / (np.linalg.norm(q_vec) * np.linalg.norm(emb) + 1e-8))
                if sim >= threshold:
                    all_scores.append((fid, sim))

        all_scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = all_scores[:top_k]

        if not top_ids:
            return []

        results = []
        for fid, sim in top_ids:
            cursor.execute("SELECT f.* FROM files f WHERE f.id = ?", (fid,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                self._parse_json_fields(result)
                result["similarity"] = sim
                result["vector_score"] = sim
                result["text_vec_score"] = None
                result["text_score"] = None
                results.append(result)

        logger.info(f"VV search within {len(file_ids)} files: {len(results)} results "
                     f"(top sim={all_scores[0][1]:.3f})" if all_scores else "")
        return results

    def triaxis_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        threshold: float = 0.0,
        return_diagnostic: bool = False,
        use_codex: bool = True,
        file_ids: Optional[set] = None,
        progress_callback: Optional[callable] = None,
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

        # Progress callback helper
        _progress = progress_callback or (lambda stage: None)

        # Step 1: Decompose query → unified schema {scope, find, exclude}
        _progress("decompose")
        t0 = time.perf_counter()
        decomposer = QueryDecomposer(use_codex=use_codex)
        unified = decomposer.decompose(query)
        diag["decomposition_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        diag["decomp_backend"] = unified.pop("_decomp_backend", "unknown")

        scope = unified.get("scope", {})
        find = unified.get("find", {})
        exclude = unified.get("exclude", {})
        legacy = unified.get("_legacy", {})

        # Scope → file_id filter (search within scope only)
        scope_file_ids = file_ids  # Direct file_ids from refine search
        t0 = time.perf_counter()
        if not scope_file_ids and any(scope.get(k) for k in ("folder", "image_type", "format")):
            scope_file_ids = self._apply_plan_filter(scope)
            if scope_file_ids:
                logger.info(f"Scope filter: {len(scope_file_ids)} files (scope={scope})")
                diag["scope_filter"] = {"scope": scope, "file_count": len(scope_file_ids)}
            else:
                logger.warning(f"Scope filter matched 0 files, searching full DB")
        diag["scope_filter_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Extract fields for triaxis (from unified + legacy fallback)
        vector_query = find.get("description", "") or legacy.get("vector_query", query)
        fts_keywords = find.get("keywords", []) or legacy.get("fts_keywords", [query])
        llm_filters = legacy.get("filters", {})
        negative_query = exclude.get("description", "")
        exclude_keywords = exclude.get("keywords", [])
        folder_filter = scope.get("folder", "")

        # Remove scope keywords from FTS — scope is a filter, not a search term
        if scope_file_ids and folder_filter:
            fts_keywords = [kw for kw in fts_keywords
                           if folder_filter.lower() not in kw.lower()
                           and kw.lower() not in folder_filter.lower()]
            if not fts_keywords:
                fts_keywords = [vector_query] if vector_query else [query]

        query_type = legacy.get("query_type", "balanced")

        diag["decomposition"] = {
            "decomposed": unified.get("decomposed", False),
            "scope": scope,
            "find_description": find.get("description", ""),
            "find_keywords": find.get("keywords", []),
            "exclude": exclude,
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
        _progress("visual")
        # Step 2: VV vector search
        vector_results = []
        v_query_embedding = None
        t0 = time.perf_counter()
        try:
            if scope_file_ids:
                vector_results = self._vv_search_within(vector_query, scope_file_ids, candidate_k, v_threshold)
            else:
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

        _progress("semantic")
        # Step 2b: MV text vector search
        text_vec_results = []
        t_query_embedding = None
        t0 = time.perf_counter()
        if self.text_search_enabled:
            try:
                if scope_file_ids:
                    text_vec_results = self._mv_search_within(vector_query, scope_file_ids, candidate_k, tv_threshold)
                else:
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

        _progress("keyword")
        # Step 3: FTS FTS5 search
        fts_results = []
        t0 = time.perf_counter()
        try:
            fts_results = self.fts_search(fts_keywords, top_k=candidate_k, exclude_keywords=exclude_keywords)
            # Scope filter: keep only files within scope
            if scope_file_ids:
                fts_results = [r for r in fts_results if r.get("id") in scope_file_ids]
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

        # Step 3b: Folder pre-filter (legacy — only when scope_file_ids not used)
        # scope_file_ids already handles this at the search level
        if folder_filter and not scope_file_ids:
            folder_fts = self.fts_search([folder_filter], top_k=500)
            folder_paths = {r["file_path"] for r in folder_fts}
            if folder_paths:
                pre_vv = len(vector_results)
                pre_mv = len(text_vec_results)
                vector_results = [r for r in vector_results if r.get("file_path") in folder_paths]
                text_vec_results = [r for r in text_vec_results if r.get("file_path") in folder_paths]
                logger.info(
                    f"Folder filter '{folder_filter}': {len(folder_paths)} files, "
                    f"VV {pre_vv}→{len(vector_results)}, MV {pre_mv}→{len(text_vec_results)}"
                )
                diag["folder_filter"] = {
                    "keyword": folder_filter,
                    "folder_files": len(folder_paths),
                    "vv_filtered": len(vector_results),
                    "mv_filtered": len(text_vec_results),
                }

        _progress("ranking")
        # Step 4: 3-axis RRF merge (V + T + F)
        t0 = time.perf_counter()
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
            merged = rrf_merge_multi(all_result_lists, k=rrf_k, weights=rrf_weights)
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

        diag["rrf_merge_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Step 5a: Apply negative filter (demote results matching exclusion terms)
        t0 = time.perf_counter()
        # Encode negative_query as VV embedding for visual penalty
        neg_v_embedding = None
        if negative_query:
            try:
                neg_v_embedding = self.encode_text(negative_query)
            except Exception as e:
                logger.debug(f"Negative VV encoding failed: {e}")

        pre_neg_count = len(merged)
        if negative_query:
            merged = apply_negative_filter(merged, negative_query, neg_v_embedding, batch_similarity_fn=self._batch_similarity)
        diag["negative_filter_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Step 5b: Apply LLM filters (lenient -- don't exclude results missing the field)
        t0 = time.perf_counter()
        pre_filter_count = len(merged)
        llm_removed = 0
        if llm_filters:
            merged = apply_user_filters(merged, llm_filters, strict=False)
            llm_removed = pre_filter_count - len(merged)

        # Step 5c: Apply user filters (strict -- exact match required)
        pre_user_count = len(merged)
        user_removed = 0
        if user_filters:
            merged = apply_user_filters(merged, user_filters, strict=True)
            user_removed = pre_user_count - len(merged)
        diag["user_filter_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        diag["filter_applied"] = bool(user_filters) or bool(llm_filters)
        diag["filter_removed"] = llm_removed + user_removed
        diag["negative_filter_active"] = bool(negative_query)
        diag["negative_v_axis"] = neg_v_embedding is not None

        # Step 5d: quality rerank on filtered candidate pool
        t0 = time.perf_counter()
        rerank_enabled = bool(_search_cfg.get("search.rerank.enabled", True))
        rerank_pool = int(_search_cfg.get("search.rerank.pool_size", max(top_k * 3, 80)))
        rerank_pool = max(top_k, rerank_pool)
        rerank_used = False
        if rerank_enabled and len(merged) > 1:
            rerank_n = min(len(merged), rerank_pool)
            # Ensure rerank has dense axis scores in its candidate pool
            enrich_axis_scores(
                merged[:rerank_n],
                v_query_embedding,
                t_query_embedding,
                fts_keywords,
                batch_similarity_fn=self._batch_similarity,
                batch_fts_fn=self._batch_fts_score,
            )
            merged = quality_rerank(
                merged,
                top_k=top_k,
                query=query,
                llm_filters=llm_filters,
                user_filters=user_filters,
                axis_weights=rrf_weights,
                pool_size=rerank_n,
            )
            rerank_used = True

        diag["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        diag["rerank"] = {
            "enabled": rerank_enabled,
            "used": rerank_used,
            "pool_size": min(len(merged), rerank_pool),
        }

        # Step 5e: FTS priority for keyword queries.
        # When query_type is "keyword", FTS results take full priority —
        # fill all slots with FTS matches first, remaining slots get VV/MV.
        if query_type == "keyword" and fts_results:
            fts_path_set = {r["file_path"] for r in fts_results}
            # Split merged into FTS-matching and non-FTS
            fts_in_merged = [r for r in merged if r["file_path"] in fts_path_set]
            non_fts = [r for r in merged if r["file_path"] not in fts_path_set]
            # FTS-only results not already in merged
            merged_path_set = {r["file_path"] for r in merged}
            fts_only = [r for r in fts_results if r["file_path"] not in merged_path_set]
            # Normalize FTS scores for display
            if fts_only:
                all_fts_ranks = [r.get("fts_rank", 0) for r in fts_only]
                best = min(all_fts_ranks)
                worst = max(all_fts_ranks)
                span = worst - best
                for r in fts_only:
                    r["vector_score"] = None
                    r["text_vec_score"] = None
                    raw = r.get("fts_rank", 0)
                    r["text_score"] = (worst - raw) / span if span else 1.0
                    r["rrf_score"] = 0
            # FTS first (in-merged + FTS-only), then non-FTS to fill remaining
            merged = fts_in_merged + fts_only + non_fts
            fts_count = len(fts_in_merged) + len(fts_only)
            logger.info(f"FTS keyword priority: {fts_count} FTS files placed first")

        # Trim to top_k
        merged = merged[:top_k]

        # Step 6: Enrich missing per-axis scores via direct DB lookup
        t0 = time.perf_counter()
        # Files in final results may lack V/S scores if they weren't in that axis's
        # candidate pool. Compute their actual similarity for complete badge display.
        v_missing_before = sum(1 for r in merged if r.get("vector_score") is None)
        s_missing_before = sum(1 for r in merged if r.get("text_vec_score") is None)
        enrich_axis_scores(merged, v_query_embedding, t_query_embedding, fts_keywords,
                          batch_similarity_fn=self._batch_similarity, batch_fts_fn=self._batch_fts_score)
        v_missing_after = sum(1 for r in merged if r.get("vector_score") is None)
        s_missing_after = sum(1 for r in merged if r.get("text_vec_score") is None)
        diag["enrich_ms"] = round((time.perf_counter() - t0) * 1000, 1)
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
            f"decomposed={unified.get('decomposed', False)}"
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
            merged = rrf_merge_multi(all_result_lists, k=rrf_k, weights=rrf_weights)
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
            merged = apply_negative_filter(merged, negative_query, neg_v_embedding, batch_similarity_fn=self._batch_similarity)

        if llm_filters:
            merged = apply_user_filters(merged, llm_filters, strict=False)
        if user_filters:
            merged = apply_user_filters(merged, user_filters, strict=True)

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
            enrich_axis_scores(
                merged[:rerank_n],
                v_emb_for_enrich,
                t_query_embedding,
                fts_keywords,
                s_embedding=x_emb_for_enrich,
                batch_similarity_fn=self._batch_similarity,
                batch_fts_fn=self._batch_fts_score,
            )
            merged = quality_rerank(
                merged,
                top_k=top_k,
                query=query,
                llm_filters=llm_filters,
                user_filters=user_filters,
                axis_weights=rrf_weights,
                pool_size=rerank_n,
            )

        merged = merged[:top_k]

        enrich_axis_scores(
            merged,
            v_emb_for_enrich,
            t_query_embedding,
            fts_keywords,
            s_embedding=x_emb_for_enrich,
            batch_similarity_fn=self._batch_similarity,
            batch_fts_fn=self._batch_fts_score,
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
        use_codex: bool = True,
        file_ids: Optional[set] = None,
        progress_callback: Optional[callable] = None,
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
                results = apply_user_filters(results, filters)
            return results

        # Single image-to-image search (backward compatible, image only)
        # Note: If mode="structure" was intended, it's handled above.
        # This block is for legacy/default visual search (SigLIP).
        if query_image:
            image_embedding = self.encoder.encode_image_from_base64(query_image)
            results = self.vector_search_by_embedding(image_embedding, top_k, threshold)
            if filters:
                results = apply_user_filters(results, filters)
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
            return self.triaxis_search(query, filters, top_k, threshold, return_diagnostic=return_diagnostic, use_codex=use_codex, file_ids=file_ids, progress_callback=progress_callback)
        elif mode == "plan":
            return self.plan_search(query, top_k, threshold, return_diagnostic=return_diagnostic)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'vector', 'hybrid', 'metadata', 'fts', 'triaxis', 'plan', or 'structure'")
