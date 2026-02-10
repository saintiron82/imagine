"""
SQLite vector search with sqlite-vec (Triaxis Architecture).

This module replaces pg_search.py with SQLite-based vector search,
maintaining API compatibility for minimal code changes.

Triaxis Search (V + S + M):
- V-axis (Visual): SigLIP 2 embedding similarity (image pixels)
- S-axis (Semantic): Qwen3 text embedding (AI-interpreted captions + context)
- M-axis (Metadata): FTS5 metadata-only search (file facts, no AI content)

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
        self._encoder = None  # Lazy loading (V-axis)
        self._text_provider = None  # Lazy loading (S-axis)
        self._text_enabled = None  # Cache for S-axis availability check

        logger.info("SqliteVectorSearch initialized")

    @property
    def encoder(self):
        """Lazy load V-axis embedding encoder (SigLIP 2)."""
        if self._encoder is None:
            from backend.vector.siglip2_encoder import SigLIP2Encoder
            self._encoder = SigLIP2Encoder()
            logger.info(f"Embedding encoder loaded: {self._encoder.model_name}")
        return self._encoder

    @property
    def text_provider(self):
        """Lazy load S-axis text embedding provider."""
        if self._text_provider is None:
            from backend.vector.text_embedding import get_text_embedding_provider
            self._text_provider = get_text_embedding_provider()
        return self._text_provider

    @property
    def text_search_enabled(self) -> bool:
        """Check if S-axis text vector search is available (vec_text table exists with data)."""
        if self._text_enabled is None:
            try:
                count = self.db.conn.execute("SELECT COUNT(*) FROM vec_text").fetchone()[0]
                self._text_enabled = count > 0
            except Exception:
                self._text_enabled = False
        return self._text_enabled

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query to V-axis embedding vector (SigLIP 2).

        Args:
            text: Text query

        Returns:
            Embedding vector
        """
        return self.encoder.encode_text(text)

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
        S-axis: Text vector similarity search using Qwen3-Embedding.

        Searches vec_text table (caption+tags embeddings) for semantic text matching.
        Complements V-axis (visual similarity) with textual semantic similarity.

        Args:
            query: Text query (encoded with text embedding model)
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

            logger.info(f"S-axis search '{query[:50]}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"S-axis text vector search failed: {e}")
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
        S-axis search using a pre-computed text embedding vector.

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

            logger.info(f"S-axis search (by embedding) returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"S-axis text vector search by embedding failed: {e}")
            return []
        finally:
            cursor.close()

    def fts_search(
        self,
        keywords: List[str],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search using FTS5 (M-axis: Metadata-only).

        Args:
            keywords: List of keywords to search (combined with OR)
            top_k: Number of results to return

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

        query_type = plan.get("query_type", "balanced")

        diag["decomposition"] = {
            "decomposed": plan.get("decomposed", False),
            "vector_query": vector_query,
            "fts_keywords": fts_keywords,
            "llm_filters": llm_filters,
            "query_type": query_type,
        }

        # Separate LLM-suggested filters (soft boost) from user filters (hard gate)
        user_filters = filters or {}
        soft_filters = llm_filters  # applied as score boost, never removes results

        # Per-axis thresholds: SigLIP (V) and Qwen3 (Tv) have very different score ranges
        # V-axis: 0.10-0.17 typical match, S-axis: 0.65-0.78 typical match
        from backend.utils.config import get_config as _cfg
        _search_cfg = _cfg()
        v_threshold = _search_cfg.get("search.threshold.visual", 0.05)
        tv_threshold = _search_cfg.get("search.threshold.text_vec", threshold)

        # Per-axis candidate pool: larger pool → more cross-axis overlap → better RRF
        candidate_mul = _search_cfg.get("search.rrf.candidate_multiplier", 5)
        candidate_k = top_k * candidate_mul

        # Step 2: V-axis vector search (cache embedding for post-merge enrichment)
        vector_results = []
        v_query_embedding = None
        t0 = time.perf_counter()
        try:
            v_query_embedding = self.encode_text(vector_query)
            vector_results = self.vector_search_by_embedding(
                v_query_embedding, top_k=candidate_k, threshold=v_threshold
            )
        except Exception as e:
            logger.warning(f"V-axis search unavailable: {e}")
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

        # Step 2b: S-axis text vector search (cache embedding for post-merge enrichment)
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
                logger.warning(f"S-axis search unavailable: {e}")
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

        # Step 3: M-axis FTS5 search
        fts_results = []
        t0 = time.perf_counter()
        try:
            fts_results = self.fts_search(fts_keywords, top_k=candidate_k)
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
        # (v_threshold for V-axis, tv_threshold for S-axis, MATCH for M-axis).
        # No post-merge threshold filter needed — it caused scale mismatch
        # (frontend 0.15 vs SigLIP2 range 0.06-0.17, killing most V-axis results).

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

        # Step 5: Apply user filters only (LLM filters no longer used as hard gate)
        pre_filter_count = len(merged)
        if user_filters:
            merged = self._apply_user_filters(merged, user_filters)
        diag["filter_applied"] = bool(user_filters)
        diag["filter_removed"] = pre_filter_count - len(merged)

        # Trim to top_k
        merged = merged[:top_k]

        # Step 6: Enrich missing per-axis scores via direct DB lookup
        # Files in final results may lack V/S scores if they weren't in that axis's
        # candidate pool. Compute their actual similarity for complete badge display.
        self._enrich_axis_scores(merged, v_query_embedding, t_query_embedding)

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

        Generalizes _rrf_merge to handle V + T + F axes.
        Preserves per-axis scores: vector_score, text_vec_score, text_score.

        Args:
            result_lists: List of (axis_name, results) tuples.
                          axis_name: "visual", "text_vec", or "fts"
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
            merged.append(result)

        return merged

    def _enrich_axis_scores(
        self,
        merged: List[Dict],
        v_embedding: Optional[np.ndarray],
        t_embedding: Optional[np.ndarray],
    ) -> None:
        """
        Enrich merged results with missing per-axis scores via direct DB lookup.

        After RRF merge, some results may lack V-axis or S-axis scores because
        they weren't in that axis's candidate pool. This method computes the
        actual similarity for those files so all badges can be displayed in the UI.
        """
        if not merged:
            return

        # Collect file IDs missing V-axis scores
        v_missing = [r["id"] for r in merged if r.get("vector_score") is None and r.get("id")]
        # Collect file IDs missing S-axis scores
        s_missing = [r["id"] for r in merged if r.get("text_vec_score") is None and r.get("id")]

        # Batch-compute V-axis similarity for missing files
        if v_missing and v_embedding is not None:
            v_scores = self._batch_similarity("vec_files", v_embedding, v_missing)
            for r in merged:
                if r.get("vector_score") is None and r.get("id") in v_scores:
                    r["vector_score"] = v_scores[r["id"]]

        # Batch-compute S-axis similarity for missing files
        if s_missing and t_embedding is not None:
            s_scores = self._batch_similarity("vec_text", t_embedding, s_missing)
            for r in merged:
                if r.get("text_vec_score") is None and r.get("id") in s_scores:
                    r["text_vec_score"] = s_scores[r["id"]]

    def _batch_similarity(
        self,
        table: str,
        query_embedding: np.ndarray,
        file_ids: List[int],
    ) -> Dict[int, float]:
        """
        Compute cosine similarity for specific files against a query embedding.

        Args:
            table: "vec_files" or "vec_text"
            query_embedding: Pre-encoded query embedding vector
            file_ids: List of file IDs to compute similarity for

        Returns:
            Dict mapping file_id -> similarity score
        """
        if not file_ids:
            return {}

        embedding_json = json.dumps(query_embedding.astype(np.float32).tolist())
        placeholders = ",".join("?" * len(file_ids))
        cursor = self.db.conn.cursor()
        try:
            cursor.execute(f"""
                SELECT file_id,
                       (1.0 - vec_distance_cosine(embedding, ?)) AS sim
                FROM {table}
                WHERE file_id IN ({placeholders})
            """, (embedding_json, *file_ids))
            return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.warning(f"Batch similarity lookup failed ({table}): {e}")
            return {}
        finally:
            cursor.close()

    def _apply_user_filters(
        self,
        results: List[Dict],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Apply user metadata filters to results in-memory.

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
            if "image_type" in filters and filters["image_type"]:
                if (result.get("image_type") or "").lower() != filters["image_type"].lower():
                    continue

            if "art_style" in filters and filters["art_style"]:
                if (result.get("art_style") or "").lower() != filters["art_style"].lower():
                    continue

            if "scene_type" in filters and filters["scene_type"]:
                if (result.get("scene_type") or "").lower() != filters["scene_type"].lower():
                    continue

            if "time_of_day" in filters and filters["time_of_day"]:
                if (result.get("time_of_day") or "").lower() != filters["time_of_day"].lower():
                    continue

            if "weather" in filters and filters["weather"]:
                if (result.get("weather") or "").lower() != filters["weather"].lower():
                    continue

            filtered.append(result)

        return filtered

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

    def search(
        self,
        query: str = "",
        mode: str = "vector",
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        threshold: float = 0.0,
        return_diagnostic: bool = False,
        query_image: Optional[str] = None,
    ):
        """
        Unified search interface (compatibility with VectorSearcher).

        Args:
            query: Search query (text)
            mode: "vector", "hybrid", "metadata", "fts", or "triaxis"
            filters: Optional metadata filters
            top_k: Number of results
            threshold: Similarity threshold (vector modes only)
            return_diagnostic: If True and mode=triaxis, return (results, diagnostic)
            query_image: Base64-encoded image for image-to-image search

        Returns:
            Search results. If return_diagnostic=True with triaxis mode,
            returns (results, diagnostic_dict).
        """
        # Image-to-image search: encode image and use vector similarity
        if query_image:
            image_embedding = self.encoder.encode_image_from_base64(query_image)
            results = self.vector_search_by_embedding(image_embedding, top_k, threshold)
            if filters:
                results = self._apply_user_filters(results, filters)
            # Add vector_score field for consistency with triaxis results
            for r in results:
                r["vector_score"] = r.get("similarity", 0)
                r["text_vec_score"] = None
                r["text_score"] = None
            logger.info(f"Image search returned {len(results)} results")
            return results

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
            raise ValueError(f"Invalid mode: {mode}. Use 'vector', 'hybrid', 'metadata', 'fts', or 'triaxis'")
