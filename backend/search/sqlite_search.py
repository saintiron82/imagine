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
import re
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
from backend.search.metadata_quality import annotate_metadata_quality

logger = logging.getLogger(__name__)

# Search diagnostic logging (disable with SEARCH_DIAGNOSTIC=0 or config)
_DIAGNOSTIC_LOG_DIR = Path(__file__).parent.parent.parent / "logs"

_KO_SCOPE_HINT_RE = re.compile(
    r"^\s*(?P<scope>.+?)\s*(?:중에서|중에|에서)\b",
    re.IGNORECASE,
)
_KO_SCOPE_SUFFIX_RE = re.compile(
    r"(?:\s*(?:폴더|프로젝트|자료|이미지|사진|파일))+$",
    re.IGNORECASE,
)
_SCOPE_HINT_BLOCKLIST = {
    "이미지",
    "사진",
    "그림",
    "파일",
    "자료",
    "폴더",
    "프로젝트",
    "asset",
    "assets",
    "image",
    "images",
    "file",
    "files",
}

_SPATIAL_KO_LOCATION_ALIASES = {
    "좌상단": "top-left",
    "왼쪽위": "top-left",
    "왼쪽 위": "top-left",
    "상단": "top",
    "위쪽": "top",
    "우상단": "top-right",
    "오른쪽위": "top-right",
    "오른쪽 위": "top-right",
    "왼쪽": "left",
    "좌측": "left",
    "중앙": "center",
    "가운데": "center",
    "오른쪽": "right",
    "우측": "right",
    "좌하단": "bottom-left",
    "왼쪽아래": "bottom-left",
    "왼쪽 아래": "bottom-left",
    "하단": "bottom",
    "아래쪽": "bottom",
    "우하단": "bottom-right",
    "오른쪽아래": "bottom-right",
    "오른쪽 아래": "bottom-right",
}
_SPATIAL_RELATION_ALIASES = {
    "on": "on",
    "over": "on",
    "above": "above",
    "under": "under",
    "below": "below",
    "behind": "behind",
    "inside": "inside",
    "around": "around",
    "near": "near",
    "left-of": "left_of",
    "left_of": "left_of",
    "right-of": "right_of",
    "right_of": "right_of",
    "in-front-of": "in_front_of",
    "in_front_of": "in_front_of",
    "attached-to": "attached_to",
    "attached_to": "attached_to",
    "위": "on",
    "위에": "on",
    "아래": "under",
    "아래에": "under",
    "왼쪽에": "left_of",
    "오른쪽에": "right_of",
    "앞": "in_front_of",
    "앞쪽": "in_front_of",
    "전면": "in_front_of",
    "뒤": "behind",
    "뒤쪽": "behind",
    "후면": "behind",
    "안": "inside",
    "내부": "inside",
    "주변": "around",
    "근처": "near",
    "가까이": "near",
    "붙은": "attached_to",
    "연결": "attached_to",
}
_SPATIAL_DEPTH_ALIASES = {
    "foreground": "foreground",
    "front": "foreground",
    "전경": "foreground",
    "앞쪽": "foreground",
    "midground": "midground",
    "middle": "midground",
    "중경": "midground",
    "중간": "midground",
    "뒤쪽": "background",
}
_SPATIAL_STOPWORDS = {
    "이미지", "사진", "그림", "파일", "자료", "찾기", "찾아줘", "보이는",
    "보이", "있는", "있고", "있음", "있다", "있", "함께", "같이", "모두",
    "있어", "그리고", "와", "과", "가", "이", "을", "를", "에", "의", "은",
    "는", "도", "로", "으로", "함", "with", "and", "together", "the", "a",
    "an", "of", "in", "to", "is", "are",
}
_SPATIAL_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_-]+")
_SPATIAL_KO_PARTICLES_RE = re.compile(
    r"(?:에게서|에게|한테|으로|부터|까지|보다|처럼|만큼|대로|마다|"
    r"이랑|이나|하고|과|와|에|의|은|는|이|가|을|를|도|만|나)$"
)
_SPATIAL_LOCATION_COMPONENTS = {
    "top-left": {"top-left", "top", "left"},
    "top-right": {"top-right", "top", "right"},
    "bottom-left": {"bottom-left", "bottom", "left"},
    "bottom-right": {"bottom-right", "bottom", "right"},
    "top": {"top"},
    "bottom": {"bottom"},
    "left": {"left"},
    "right": {"right"},
    "center": {"center"},
}


def _ordered_unique(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _extract_spatial_locations_from_text(text: str) -> List[str]:
    """Extract location aliases from one text segment without cross-keyword joins."""
    segment = str(text or "").strip()
    lowered = segment.lower()
    compact_ko = re.sub(r"\s+", "", segment)
    locations: List[str] = []
    for raw, canonical in SQLiteDB._SPATIAL_LOCATION_ALIASES.items():
        if raw in lowered:
            locations.append(canonical)
    for canonical in SQLiteDB._SPATIAL_LOCATIONS:
        if canonical in lowered:
            locations.append(canonical)
    for raw, canonical in _SPATIAL_KO_LOCATION_ALIASES.items():
        if raw in segment or raw.replace(" ", "") in compact_ko:
            locations.append(canonical)
    return _ordered_unique(locations)


def _location_conflicts(candidate: str, allowed_locations: List[str]) -> bool:
    candidate_parts = _SPATIAL_LOCATION_COMPONENTS.get(candidate, {candidate})
    allowed_parts: set[str] = set()
    for location in allowed_locations:
        allowed_parts.update(_SPATIAL_LOCATION_COMPONENTS.get(location, {location}))
    if not candidate_parts or not allowed_parts:
        return False
    horizontal = {"left", "right"}
    vertical = {"top", "bottom"}
    return (
        bool(candidate_parts & horizontal)
        and bool(allowed_parts & horizontal)
        and not bool(candidate_parts & allowed_parts & horizontal)
    ) or (
        bool(candidate_parts & vertical)
        and bool(allowed_parts & vertical)
        and not bool(candidate_parts & allowed_parts & vertical)
    )


def _sanitize_spatial_fts_keywords(query: str, keywords: Optional[List[str]]) -> List[str]:
    """Remove decomposer location keywords that contradict explicit query location."""
    if not keywords:
        return []
    query_locations = _extract_spatial_locations_from_text(query)
    if not query_locations:
        return _ordered_unique([str(kw) for kw in keywords if str(kw).strip()])

    sanitized: List[str] = []
    for keyword in keywords:
        text = str(keyword or "").strip()
        if not text:
            continue
        keyword_locations = _extract_spatial_locations_from_text(text)
        if keyword_locations and all(
            _location_conflicts(location, query_locations)
            for location in keyword_locations
        ):
            continue
        sanitized.append(text)
    return _ordered_unique(sanitized)


def _extract_spatial_intent(query: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """Extract rule-based spatial intent for relation/location/depth searches."""
    text = str(query or "").strip()
    clean_keywords = _sanitize_spatial_fts_keywords(text, keywords)
    joined = " ".join([text] + [str(k) for k in clean_keywords if k])
    lowered = joined.lower()

    query_locations = _extract_spatial_locations_from_text(text)
    if query_locations:
        locations = query_locations
    else:
        locations = []
        for segment in [text] + clean_keywords:
            locations.extend(_extract_spatial_locations_from_text(str(segment)))

    relations: List[str] = []
    for raw, canonical in SQLiteDB._SPATIAL_RELATION_ALIASES.items():
        if raw in lowered:
            relations.append(canonical)
    for canonical in SQLiteDB._SPATIAL_RELATIONS:
        if canonical in lowered:
            relations.append(canonical)
    for raw, canonical in _SPATIAL_RELATION_ALIASES.items():
        if raw in lowered or raw in joined:
            relations.append(canonical)

    depth_layers: List[str] = []
    for raw, canonical in _SPATIAL_DEPTH_ALIASES.items():
        if raw in lowered or raw in joined:
            depth_layers.append(canonical)

    marker_terms = {
        *locations,
        *relations,
        *depth_layers,
        *SQLiteDB._SPATIAL_LOCATIONS,
        *SQLiteDB._SPATIAL_RELATIONS,
        *SQLiteDB._DEPTH_LAYERS,
        *_SPATIAL_KO_LOCATION_ALIASES.keys(),
        *_SPATIAL_RELATION_ALIASES.keys(),
        *_SPATIAL_DEPTH_ALIASES.keys(),
    }
    terms: List[str] = []
    for token in _SPATIAL_TOKEN_RE.findall(joined):
        normalized = token.strip("_- ")
        if any("\uac00" <= c <= "\ud7af" for c in normalized):
            normalized = _SPATIAL_KO_PARTICLES_RE.sub("", normalized) or normalized
        cleaned = normalized.lower()
        if not cleaned or cleaned in _SPATIAL_STOPWORDS:
            continue
        if cleaned in {str(t).lower() for t in marker_terms}:
            continue
        terms.append(normalized)

    locations = _ordered_unique(locations)
    relations = _ordered_unique(relations)
    depth_layers = _ordered_unique(depth_layers)
    terms = _ordered_unique(terms)

    return {
        "active": bool(locations or relations or depth_layers),
        "terms": terms,
        "locations": locations,
        "relations": relations,
        "depth_layers": depth_layers,
    }


def _split_scope_segments(value: str) -> List[str]:
    """Split a folder/path string into normalized path segments."""
    if not value:
        return []
    normalized = str(value).replace("\\", "/").strip("/")
    return [part.strip().lower() for part in normalized.split("/") if part.strip()]


def _path_has_scope_segments(path_value: str, scope_value: str) -> bool:
    """Return True when scope_value matches full path segment(s), not substring."""
    path_segments = _split_scope_segments(path_value)
    scope_segments = _split_scope_segments(scope_value)
    if not path_segments or not scope_segments:
        return False

    width = len(scope_segments)
    if width > len(path_segments):
        return False

    for idx in range(0, len(path_segments) - width + 1):
        if path_segments[idx:idx + width] == scope_segments:
            return True
    return False


def _bool_setting(value: Any, default: bool = True) -> bool:
    """Parse bool-like config/env values without treating 'false' as truthy."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _diagnostic_log_enabled(config: Any = None) -> bool:
    """Return whether diagnostic JSONL file logging is enabled."""
    env_value = os.getenv("SEARCH_DIAGNOSTIC")
    if env_value is not None:
        return _bool_setting(env_value, default=True)

    if config is None:
        try:
            from backend.utils.config import get_config
            config = get_config()
        except Exception:
            config = None

    if config is not None:
        return _bool_setting(config.get("search.diagnostic.enabled", True), default=True)
    return True


def _query_explicitly_requests_format(query: str, fmt: str) -> bool:
    """Return True when a format token is a user-requested file type."""
    if not query or not fmt:
        return False

    fmt = str(fmt).lower()
    aliases = {"jpg": {"jpg", "jpeg"}, "jpeg": {"jpg", "jpeg"}}.get(fmt, {fmt})
    alias_expr = "|".join(re.escape(alias) for alias in sorted(aliases))
    q = str(query).lower()

    explicit_patterns = [
        rf"(?<![\w.])(?:{alias_expr})(?![\w.])\s*(?:파일|포맷|형식|확장자|이미지)",
        rf"(?:파일|포맷|형식|확장자)\s*(?:이|가|은|는|:)?\s*(?:{alias_expr})(?![\w.])",
        rf"(?:{alias_expr})(?:만|로만|인\s*파일)(?![\w.])",
    ]
    return any(re.search(pattern, q, re.IGNORECASE) for pattern in explicit_patterns)


def _relax_unmatched_scope(scope: dict, query: str) -> tuple[dict, set[str]]:
    """Relax LLM-only scope filters that commonly cause false-empty results.

    Only called when the combined scope matched 0 files. The folder is never
    relaxed here (it has fuzzy/hint resolution and a deliberate strict-empty
    policy); image_type and non-explicit format are dropped because the LLM
    routinely absorbs *search elements* into them — e.g. "#08에서 캐릭터과 밤"
    became {folder: '#08', image_type: 'character'}, and since that folder has
    no character-classified files the user got an empty page instead of #08's
    night scenes.
    """
    if not scope or not scope.get("folder"):
        return dict(scope or {}), set()

    relaxed = dict(scope)
    relaxed_keys: set[str] = set()
    fmt = relaxed.get("format")
    if fmt and not _query_explicitly_requests_format(query, str(fmt)):
        relaxed["format"] = None
        relaxed_keys.add("format")
    if relaxed.get("image_type"):
        relaxed["image_type"] = None
        relaxed_keys.add("image_type")
    return relaxed, relaxed_keys


def _extract_scope_hint_candidates(query: str) -> List[str]:
    """Return conservative folder-scope candidates from Korean scoped queries."""
    if not query:
        return []

    match = _KO_SCOPE_HINT_RE.search(str(query).strip())
    if not match:
        return []

    scope = match.group("scope").strip(" \t\r\n\"'`“”‘’[](){}")
    scope = _KO_SCOPE_SUFFIX_RE.sub("", scope).strip(" \t\r\n\"'`“”‘’[](){}")
    scope = re.sub(r"\s+", " ", scope)
    if not scope:
        return []

    lowered = scope.lower()
    if lowered in _SCOPE_HINT_BLOCKLIST:
        return []
    if len(scope) < 2 and not re.fullmatch(r"#\d+", scope):
        return []

    candidates = [scope]
    if " " in scope:
        candidates.append("/".join(part for part in scope.split(" ") if part))

    deduped = []
    seen = set()
    for candidate in candidates:
        key = candidate.lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


class SqliteVectorSearch:
    """SQLite vector search with SigLIP 2 embeddings."""

    _FTS_KO_PARTICLES = re.compile(
        r"(?:중에서|중에|에서|에게서|에게|한테|으로|부터|까지|보다|처럼|만큼|대로|마다|"
        r"이랑|이나|하고|과|와|에|의|은|는|이|가|을|를|도|만|나)$"
    )
    _FTS_KO_STOPWORDS = {"이미지", "사진", "파일", "자료", "있는", "있다", "있고", "있음"}

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

            # vec_structure stores embeddings as raw float32 bytes (sqlite-vec vec0 native format)
            query_vec = np.frombuffer(row[0], dtype=np.float32)

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

            # vec_files stores embeddings as raw float32 bytes (sqlite-vec vec0 native format)
            query_vec = np.frombuffer(row[0], dtype=np.float32)

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
        file_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search using FTS5 (FTS: Metadata-only).

        Args:
            keywords: List of keywords to search (combined with OR)
            top_k: Number of results to return
            exclude_keywords: Optional list of keywords to exclude via FTS5 NOT operator
            file_ids: Optional scope filter applied IN-SQL.
                     When provided, only rows whose id is in this set are
                     considered for ranking. This matters: a post-fetch
                     Python filter on the global top-K can drop every
                     scope-relevant match when the scope is small relative
                     to the DB (e.g. 745 of 17k → top-60 likely contains
                     0 in-scope rows). VV/MV axes already do this in-SQL
                     via _vv_search_within / _mv_search_within; this
                     parameter brings FTS to parity.

        Returns:
            List of file records with FTS rank scores
        """
        if not keywords:
            return []

        # Build FTS5 MATCH query: split multi-word keywords into individual
        # tokens so "crossroads at night" matches documents containing any of
        # those words, not just the exact phrase.
        # ASCII alphabetic tokens get a trailing '*' (FTS5 prefix match) so
        # singular/plural/derived forms hit: query "mountain" matches captions
        # like "mountains", "mountainous". CJK / numeric / single-char tokens
        # stay literal (single-char Korean like "산" would over-match into
        # 산책/산들/etc. with prefix).
        wildcard_tokens = set()
        literal_tokens = set()
        for kw in keywords:
            for word in kw.split():
                word = word.strip().replace('"', '""')
                if not word:
                    continue
                if word in self._FTS_KO_STOPWORDS:
                    continue
                if not word.isascii():
                    stripped = self._FTS_KO_PARTICLES.sub("", word)
                    if stripped and stripped not in self._FTS_KO_STOPWORDS:
                        word = stripped.replace('"', '""')
                if word.isascii() and word.isalpha() and len(word) >= 3:
                    wildcard_tokens.add(word.lower())
                else:
                    literal_tokens.add(word)
        if not wildcard_tokens and not literal_tokens:
            return []

        # Wildcards must be unquoted in FTS5; literals stay quoted to escape
        # tokenizer-special chars and to disable accidental prefix matching.
        match_parts = [f'{t}*' for t in wildcard_tokens] + \
                      [f'"{t}"' for t in literal_tokens]
        match_expr = " OR ".join(match_parts)

        # Build exclude expression using FTS5 NOT operator (same wildcard
        # rule as positive tokens for symmetric behavior).
        if exclude_keywords:
            ex_wild = set()
            ex_lit = set()
            for kw in exclude_keywords:
                for word in kw.split():
                    word = word.strip().replace('"', '""')
                    if not word:
                        continue
                    if word.isascii() and word.isalpha() and len(word) >= 3:
                        ex_wild.add(word.lower())
                    else:
                        ex_lit.add(word)
            if ex_wild or ex_lit:
                ex_parts = [f'{t}*' for t in ex_wild] + \
                           [f'"{t}"' for t in ex_lit]
                exclude_expr = " OR ".join(ex_parts)
                match_expr = f"({match_expr}) NOT ({exclude_expr})"

        # v4: Load BM25 weights from config (6 columns)
        from backend.utils.config import get_config as _cfg
        cfg = _cfg()
        w_strong = cfg.get("search.fts.bm25_weights.meta_strong", 3.0)
        w_weak = cfg.get("search.fts.bm25_weights.meta_weak", 1.5)
        w_caption = cfg.get("search.fts.bm25_weights.caption", 2.5)
        w_ai_tags = cfg.get("search.fts.bm25_weights.ai_tags", 2.0)
        w_classification = cfg.get("search.fts.bm25_weights.classification", 1.5)
        w_spatial = cfg.get("search.fts.bm25_weights.spatial", 2.2)

        cursor = self.db.conn.cursor()

        # Optional in-SQL scope filter (see file_ids docstring above)
        scope_clause = ""
        scope_binds: tuple = ()
        if file_ids:
            id_list = list(file_ids)
            placeholders = ",".join("?" * len(id_list))
            scope_clause = f" AND f.id IN ({placeholders})"
            scope_binds = tuple(id_list)

        try:
            cursor.execute(f"""
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
                    bm25(files_fts, ?, ?, ?, ?, ?, ?) AS fts_rank
                FROM files_fts fts
                JOIN files f ON f.id = fts.rowid
                WHERE files_fts MATCH ?
                  AND f.preview_only = 0
                  {scope_clause}
                ORDER BY fts_rank
                LIMIT ?
            """, (w_strong, w_weak, w_caption, w_ai_tags, w_classification, w_spatial,
                  match_expr, *scope_binds, top_k))

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
        file_ids, scope_match_info = self._apply_plan_filter_with_info(scope)
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
                "scope_match": scope_match_info,
                "search_query": search_query,
                "pre_filter_count": len(file_ids),
                "result_count": len(results),
                "total_ms": round(elapsed * 1000, 1),
            }
            return results, diag
        return results

    def _row_matches_folder_scope(self, row: Any, folder: str) -> bool:
        """Check exact folder scope against full path segments."""
        if not folder:
            return False
        try:
            folder_path = row["folder_path"] if "folder_path" in row.keys() else None
            file_path = row["file_path"] if "file_path" in row.keys() else None
        except AttributeError:
            folder_path = row[1] if len(row) > 1 else None
            file_path = row[2] if len(row) > 2 else None
        return (
            _path_has_scope_segments(str(folder_path or ""), folder)
            or _path_has_scope_segments(str(file_path or ""), folder)
        )

    def _folder_exact_scope_exists(self, folder: str) -> bool:
        """Return True if folder exists as exact path segment(s)."""
        if not folder:
            return False
        cursor = self.db.conn.cursor()
        try:
            cursor.execute(
                """
                SELECT id, folder_path, file_path
                FROM files
                WHERE preview_only = 0
                  AND (folder_path LIKE ? OR file_path LIKE ?)
                """,
                (f"%{folder}%", f"%{folder}%"),
            )
            return any(self._row_matches_folder_scope(row, folder) for row in cursor.fetchall())
        finally:
            cursor.close()

    def _filter_exact_folder_scope(self, rows: list[Any], folder: str) -> set:
        """Keep only rows whose path contains folder as exact segment(s)."""
        ids = set()
        for row in rows:
            if self._row_matches_folder_scope(row, folder):
                ids.add(row["id"] if hasattr(row, "keys") else row[0])
        return ids

    @staticmethod
    def _scope_match_info(
        *,
        requested_folder: str | None,
        applied_folder: str | None,
        match_mode: str,
        resolved_folder: str | None = None,
        resolved_match_mode: str | None = None,
    ) -> dict[str, Any]:
        return {
            "requested_folder": requested_folder or None,
            "applied_folder": applied_folder or None,
            "match_mode": match_mode,
            "resolved_folder": resolved_folder or None,
            "resolved_match_mode": resolved_match_mode or None,
        }

    def _apply_plan_filter_with_info(self, pre_filter: dict) -> tuple[set, dict[str, Any]]:
        """Apply pre_filter to get file_id set from DB.

        Folder matching is exact-first. If the requested folder exists as full
        path segment(s), only those exact segments match, so `#3` does not also
        match `#30` or `#33`. If no exact segment exists, the legacy substring
        result is kept first; fuzzy correction is used only when the original
        substring finds nothing. This prevents code-like scopes such as `3DBG`
        from being "corrected" to a broad `bg` folder.
        """
        cursor = self.db.conn.cursor()
        try:
            img_type = pre_filter.get("image_type")
            fmt = pre_filter.get("format")

            def fetch_rows(folder_value: str | None) -> list[Any]:
                conditions = ["preview_only = 0"]
                params = []
                if folder_value:
                    conditions.append("(folder_path LIKE ? OR file_path LIKE ?)")
                    params.extend([f"%{folder_value}%", f"%{folder_value}%"])
                if img_type:
                    conditions.append("image_type = ?")
                    params.append(img_type)
                if fmt:
                    conditions.append("UPPER(format) = ?")
                    params.append(fmt.upper())
                where = " AND ".join(conditions)
                cursor.execute(f"SELECT id, folder_path, file_path FROM files WHERE {where}", params)
                return cursor.fetchall()

            folder = pre_filter.get("folder", "")
            if not folder:
                return (
                    {row[0] for row in fetch_rows(None)},
                    self._scope_match_info(
                        requested_folder=None,
                        applied_folder=None,
                        match_mode="no_folder",
                    ),
                )

            if self._folder_exact_scope_exists(folder):
                return (
                    self._filter_exact_folder_scope(fetch_rows(folder), folder),
                    self._scope_match_info(
                        requested_folder=folder,
                        applied_folder=folder,
                        match_mode="exact_segment",
                    ),
                )

            rows = fetch_rows(folder)
            if rows:
                return (
                    {row[0] for row in rows},
                    self._scope_match_info(
                        requested_folder=folder,
                        applied_folder=folder,
                        match_mode="substring",
                    ),
                )

            resolved = self._resolve_folder_name(folder)
            if resolved and resolved != folder:
                logger.info(f"Folder fuzzy-match: '{folder}' → '{resolved}'")
                rows = fetch_rows(resolved)
                if self._folder_exact_scope_exists(resolved):
                    return (
                        self._filter_exact_folder_scope(rows, resolved),
                        self._scope_match_info(
                            requested_folder=folder,
                            applied_folder=resolved,
                            resolved_folder=resolved,
                            match_mode="fuzzy",
                            resolved_match_mode="exact_segment",
                        ),
                    )
                return (
                    {row[0] for row in rows},
                    self._scope_match_info(
                        requested_folder=folder,
                        applied_folder=resolved,
                        resolved_folder=resolved,
                        match_mode="fuzzy",
                        resolved_match_mode="substring",
                    ),
                )

            return (
                set(),
                self._scope_match_info(
                    requested_folder=folder,
                    applied_folder=None,
                    match_mode="no_match",
                ),
            )
        finally:
            cursor.close()

    def _apply_plan_filter(self, pre_filter: dict) -> set:
        """Apply pre_filter to get file_id set from DB."""
        file_ids, _match_info = self._apply_plan_filter_with_info(pre_filter)
        return file_ids

    def _scope_ids_from_query_hint(
        self,
        query: str,
        base_scope: Optional[dict] = None,
        skip_folder: Optional[str] = None,
        return_match_info: bool = False,
    ) -> tuple[Optional[str], set] | tuple[Optional[str], set, dict[str, Any]]:
        """Resolve Korean `X에서 ...` scope hints only when they hit the DB."""
        skip_key = (skip_folder or "").strip().lower()
        for candidate in _extract_scope_hint_candidates(query):
            if skip_key and candidate.lower() == skip_key:
                continue
            scope = dict(base_scope or {})
            scope["folder"] = candidate
            file_ids, match_info = self._apply_plan_filter_with_info(scope)
            if file_ids:
                if return_match_info:
                    return candidate, file_ids, match_info
                return candidate, file_ids
        if return_match_info:
            return None, set(), {}
        return None, set()

    def _resolve_folder_name(self, requested: str) -> Optional[str]:
        """Map a possibly-misspelled folder query to a real folder substring.

        Strategy:
          1. Literal exact-segment hit on folder_path/file_path → keep as-is.
          2. Pull distinct folder path segments, find best match by
             difflib ratio (≥0.6 threshold). Korean glyph variants like
             ㅏ↔ㅑ score high enough to cross the threshold.
        """
        if not requested:
            return None
        cur = self.db.conn.cursor()
        try:
            if self._folder_exact_scope_exists(requested):
                return requested  # exact segment works, no fuzzing needed

            # Build candidate folder name set: every distinct path segment
            # from folder_path + each parent dir of file_path.
            segments: set[str] = set()
            for (fp,) in cur.execute(
                "SELECT DISTINCT folder_path FROM files "
                "WHERE folder_path IS NOT NULL AND folder_path != ''"
            ):
                for part in str(fp).replace("\\", "/").split("/"):
                    if part:
                        segments.add(part)
            for (fp,) in cur.execute(
                "SELECT DISTINCT file_path FROM files "
                "WHERE file_path IS NOT NULL LIMIT 5000"
            ):
                parts = str(fp).replace("\\", "/").split("/")
                for part in parts[:-1]:  # skip filename
                    if part and part not in (".", ".."):
                        segments.add(part)
            if not segments:
                return None

            import difflib
            matches = difflib.get_close_matches(
                requested, list(segments), n=1, cutoff=0.6
            )
            return matches[0] if matches else None
        except Exception as e:
            logger.warning(f"Folder fuzzy-match failed: {e}")
            return None
        finally:
            cur.close()

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

    @staticmethod
    def _spatial_term_hits(text: str, terms: List[str]) -> int:
        hay = str(text or "").lower()
        return sum(1 for term in terms if str(term or "").lower() in hay)

    @staticmethod
    def _spatial_confidence_bonus(confidence: str) -> float:
        return {
            "high": 0.20,
            "medium": 0.10,
            "low": 0.0,
        }.get(str(confidence or "").lower(), 0.0)

    def _spatial_evidence_search(
        self,
        intent: Dict[str, Any],
        top_k: int = 20,
        file_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Search normalized spatial evidence tables as a first-class axis."""
        if not intent or not intent.get("active"):
            return []

        terms = [str(t).lower() for t in intent.get("terms", []) if str(t).strip()]
        locations = set(intent.get("locations", []) or [])
        relations = set(intent.get("relations", []) or [])
        depth_layers = set(intent.get("depth_layers", []) or [])
        allowed_ids = set(file_ids or []) if file_ids else None
        matches_by_file: Dict[int, Dict[str, Any]] = {}

        def allowed(file_id: Any) -> bool:
            return allowed_ids is None or file_id in allowed_ids

        def add_match(file_id: int, score: float, match: Dict[str, Any]) -> None:
            if not allowed(file_id):
                return
            bucket = matches_by_file.setdefault(
                int(file_id),
                {"score": 0.0, "matches": []},
            )
            bucket["score"] = max(float(bucket["score"]), float(score))
            bucket["matches"].append(match)

        cursor = self.db.conn.cursor()
        try:
            if relations:
                rows = cursor.execute(
                    """SELECT file_id, subject, relation, object,
                              subject_location, object_location, confidence, spatial_text
                       FROM file_spatial_relations"""
                ).fetchall()
                for row in rows:
                    relation = row["relation"]
                    spatial_text = row["spatial_text"] or ""
                    hay = " ".join(
                        str(row[key] or "")
                        for key in (
                            "subject", "relation", "object",
                            "subject_location", "object_location", "spatial_text",
                        )
                    )
                    term_hits = self._spatial_term_hits(hay, terms)
                    min_hits = min(len(terms), 2) if terms else 0
                    if relation not in relations and not any(rel in spatial_text for rel in relations):
                        continue
                    if term_hits < min_hits:
                        continue
                    loc_hits = sum(
                        1 for loc in locations
                        if loc in {row["subject_location"], row["object_location"]}
                        or loc in spatial_text
                    )
                    score = (
                        0.75
                        + 0.18 * term_hits
                        + 0.20 * loc_hits
                        + self._spatial_confidence_bonus(row["confidence"])
                    )
                    add_match(row["file_id"], score, {
                        "table": "file_spatial_relations",
                        "subject": row["subject"],
                        "relation": relation,
                        "object": row["object"],
                        "confidence": row["confidence"],
                    })

            if locations or (terms and not relations and not depth_layers):
                rows = cursor.execute(
                    """SELECT file_id, name, ko_name, primary_location,
                              locations, extent, confidence, spatial_text
                       FROM file_objects"""
                ).fetchall()
                for row in rows:
                    spatial_text = row["spatial_text"] or ""
                    row_locations = set()
                    try:
                        row_locations.update(json.loads(row["locations"] or "[]"))
                    except (json.JSONDecodeError, TypeError):
                        pass
                    primary_location = row["primary_location"] or ""
                    secondary_locations = set(row_locations)
                    if primary_location:
                        row_locations.add(primary_location)
                        secondary_locations.discard(primary_location)
                    hay = " ".join(
                        str(value or "")
                        for value in (
                            row["name"], row["ko_name"], primary_location,
                            row["locations"], row["extent"], spatial_text,
                        )
                    )
                    term_hits = self._spatial_term_hits(hay, terms)
                    primary_hits = sum(1 for loc in locations if loc == primary_location)
                    secondary_hits = sum(1 for loc in locations if loc in secondary_locations)
                    text_only_hits = sum(
                        1 for loc in locations
                        if loc not in row_locations and loc in spatial_text
                    )
                    loc_hits = primary_hits + secondary_hits + text_only_hits
                    if terms and term_hits == 0:
                        continue
                    if locations and loc_hits == 0:
                        continue
                    if primary_hits:
                        loc_bonus = 0.52 * primary_hits
                        match_strength = "primary"
                    elif secondary_hits:
                        loc_bonus = 0.18 * secondary_hits
                        match_strength = "secondary"
                    elif text_only_hits:
                        loc_bonus = 0.07 * text_only_hits
                        match_strength = "text"
                    else:
                        loc_bonus = 0.0
                        match_strength = "term_only"
                    score = (
                        0.45
                        + 0.18 * term_hits
                        + loc_bonus
                        + self._spatial_confidence_bonus(row["confidence"])
                    )
                    add_match(row["file_id"], score, {
                        "table": "file_objects",
                        "name": row["name"],
                        "ko_name": row["ko_name"],
                        "primary_location": primary_location,
                        "match_strength": match_strength,
                        "confidence": row["confidence"],
                    })

            if depth_layers:
                rows = cursor.execute(
                    """SELECT file_id, name, ko_name, layer, confidence, spatial_text
                       FROM file_depth_layers"""
                ).fetchall()
                for row in rows:
                    spatial_text = row["spatial_text"] or ""
                    hay = " ".join(
                        str(value or "")
                        for value in (
                            row["name"], row["ko_name"], row["layer"], spatial_text,
                        )
                    )
                    term_hits = self._spatial_term_hits(hay, terms)
                    if row["layer"] not in depth_layers and not any(layer in spatial_text for layer in depth_layers):
                        continue
                    if terms and term_hits == 0:
                        continue
                    score = (
                        0.65
                        + 0.20 * term_hits
                        + self._spatial_confidence_bonus(row["confidence"])
                    )
                    add_match(row["file_id"], score, {
                        "table": "file_depth_layers",
                        "name": row["name"],
                        "ko_name": row["ko_name"],
                        "layer": row["layer"],
                        "confidence": row["confidence"],
                    })

            if not matches_by_file:
                return []

            ranked = sorted(
                matches_by_file.items(),
                key=lambda item: item[1]["score"],
                reverse=True,
            )[:top_k]
            results: List[Dict[str, Any]] = []
            for file_id, payload in ranked:
                row = cursor.execute("SELECT f.* FROM files f WHERE f.id = ?", (file_id,)).fetchone()
                if not row:
                    continue
                result = dict(row)
                self._parse_json_fields(result)
                result["spatial_score"] = float(payload["score"])
                result["spatial_matches"] = payload["matches"]
                results.append(result)
            return results
        except Exception as e:
            logger.warning(f"Spatial evidence search unavailable: {e}")
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
        from backend.utils.config import get_config as _cfg
        _search_cfg = _cfg()

        # Scope → file_id filter (search within scope only)
        scope_file_ids = file_ids  # Direct file_ids from refine search
        scope_requested = bool(scope.get("folder") or scope.get("image_type") or scope.get("format"))
        scope_unmatched = False
        scope_source = "decomposition"
        relaxed_scope_keys: set[str] = set()
        soft_scope_file_ids: set = set()
        soft_scope_folder: Optional[str] = None
        scope_match_info: dict[str, Any] = {}
        soft_scope_match_info: dict[str, Any] = {}
        t0 = time.perf_counter()
        if not scope_file_ids:
            if scope_requested:
                scope_file_ids, scope_match_info = self._apply_plan_filter_with_info(scope)
                if not scope_file_ids:
                    relaxed_scope, relaxed_keys = _relax_unmatched_scope(scope, query)
                    if relaxed_keys:
                        relaxed_ids, relaxed_match_info = self._apply_plan_filter_with_info(relaxed_scope)
                        if relaxed_ids:
                            scope = relaxed_scope
                            scope_file_ids = relaxed_ids
                            scope_match_info = relaxed_match_info
                            relaxed_scope_keys = relaxed_keys
                            scope_source = "decomposition_relaxed"
                    if not scope_file_ids:
                        hinted_folder, hinted_ids, hinted_match_info = self._scope_ids_from_query_hint(
                            query,
                            base_scope=scope,
                            skip_folder=scope.get("folder"),
                            return_match_info=True,
                        )
                        if hinted_ids:
                            scope = dict(scope)
                            scope["folder"] = hinted_folder
                            scope_file_ids = hinted_ids
                            scope_match_info = hinted_match_info
                            scope_source = "query_hint"
                    elif scope.get("folder"):
                        hinted_folder, hinted_ids, hinted_match_info = self._scope_ids_from_query_hint(
                            query,
                            base_scope=scope,
                            skip_folder=scope.get("folder"),
                            return_match_info=True,
                        )
                        if hinted_ids:
                            # The user's literal scope should win over an LLM
                            # normalization when both hit the DB. Example:
                            # `#02에서 ...` must not become `#2`.
                            scope = dict(scope)
                            scope["folder"] = hinted_folder
                            scope_file_ids = hinted_ids
                            scope_match_info = hinted_match_info
                            scope_source = "query_hint_override"
            else:
                hinted_folder, hinted_ids, hinted_match_info = self._scope_ids_from_query_hint(
                    query,
                    return_match_info=True,
                )
                if hinted_ids:
                    hard_max = int(_search_cfg.get("search.scope_hint.hard_max_files", 1000))
                    if 0 < len(hinted_ids) <= hard_max:
                        scope = dict(scope)
                        scope["folder"] = hinted_folder
                        scope_file_ids = hinted_ids
                        scope_match_info = hinted_match_info
                        scope_source = "query_hint_hard"
                    else:
                        soft_scope_folder = hinted_folder
                        soft_scope_file_ids = hinted_ids
                        soft_scope_match_info = hinted_match_info

            if scope_file_ids:
                logger.info(
                    f"Scope filter: {len(scope_file_ids)} files "
                    f"(scope={scope}, source={scope_source})"
                )
                diag["scope_filter"] = {
                    "scope": scope,
                    "file_count": len(scope_file_ids),
                    "source": scope_source,
                }
                for key, value in scope_match_info.items():
                    if value is not None:
                        diag["scope_filter"][key] = value
                if relaxed_scope_keys:
                    diag["scope_filter"]["relaxed_keys"] = sorted(relaxed_scope_keys)
            elif scope_requested:
                # Strict policy: when the user explicitly asked for a scope
                # (folder / image_type / format) and nothing matches even
                # after fuzzy folder resolution, return empty rather than
                # silently falling back to a full-DB search. Falling back
                # produced confusing UX — the user typed
                # "마카베리즈무에서 산 이미지" expecting that folder only,
                # but every other folder's mountains showed up too.
                logger.warning(f"Scope filter matched 0 files (scope={scope}); returning empty")
                scope_unmatched = True
                diag["scope_filter"] = {"scope": scope, "file_count": 0,
                                         "out_of_scope": True}
                for key, value in scope_match_info.items():
                    if value is not None:
                        diag["scope_filter"][key] = value
            elif soft_scope_file_ids:
                diag["scope_hint"] = {
                    "folder": soft_scope_folder,
                    "file_count": len(soft_scope_file_ids),
                    "source": "query_hint",
                    "mode": "soft_rerank",
                }
                for key, value in soft_scope_match_info.items():
                    if value is not None:
                        diag["scope_hint"][key] = value
        diag["scope_filter_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if scope_unmatched:
            diag["final_results_count"] = 0
            diag["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            if return_diagnostic:
                return [], diag
            return []

        # Extract fields for triaxis (from unified + legacy fallback)
        vector_query = find.get("description", "") or legacy.get("vector_query", query)
        fts_keywords = find.get("keywords", []) or legacy.get("fts_keywords", [query])
        llm_filters = legacy.get("filters", {})
        if relaxed_scope_keys and llm_filters:
            llm_filters = dict(llm_filters)
            for key in relaxed_scope_keys:
                llm_filters.pop(key, None)
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

        fts_keywords = _sanitize_spatial_fts_keywords(query, fts_keywords)
        if not fts_keywords:
            fts_keywords = [vector_query] if vector_query else [query]

        query_type = legacy.get("query_type", "balanced")
        spatial_intent = _extract_spatial_intent(query, fts_keywords)
        if spatial_intent.get("active"):
            query_type = "spatial"

        diag["decomposition"] = {
            "decomposed": unified.get("decomposed", False),
            "scope": scope,
            "find_description": find.get("description", ""),
            "find_keywords": find.get("keywords", []),
            "exclude": exclude,
            "query_type": query_type,
            "spatial_intent": spatial_intent,
        }

        # Merge LLM-suggested filters with user filters (user takes precedence)
        user_filters = filters or {}

        # Per-axis thresholds: SigLIP (V) and Qwen3 (Tv) have very different score ranges
        # VV: 0.10-0.17 typical match, MV: 0.65-0.78 typical match
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
            # Encode unconditionally so v_query_embedding is available to
            # enrich_axis_scores below (the scope path used to skip caching
            # this, leaving every result with axes_present=1 and forcing
            # quality_rerank to rank from single-axis scores only).
            v_query_embedding = self.encode_text(vector_query)
            if scope_file_ids:
                vector_results = self._vv_search_within(vector_query, scope_file_ids, candidate_k, v_threshold)
            else:
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
                # Same parity fix as VV above: cache t_query_embedding for
                # enrich_axis_scores so MV scores get backfilled into
                # results that only matched on FTS or VV.
                t_query_embedding = self.text_provider.encode(vector_query, is_query=True)
                if scope_file_ids:
                    text_vec_results = self._mv_search_within(vector_query, scope_file_ids, candidate_k, tv_threshold)
                else:
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
        # Pass scope_file_ids IN-SQL so ranking happens within scope.
        # The previous post-filter on global top-K silently dropped
        # every in-scope match when scope was small relative to the DB.
        fts_results = []
        t0 = time.perf_counter()
        try:
            fts_results = self.fts_search(
                fts_keywords,
                top_k=candidate_k,
                exclude_keywords=exclude_keywords,
                file_ids=scope_file_ids,
            )
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

        _progress("spatial")
        spatial_results = []
        t0 = time.perf_counter()
        if spatial_intent.get("active"):
            spatial_results = self._spatial_evidence_search(
                spatial_intent,
                top_k=candidate_k,
                file_ids=scope_file_ids,
            )
        diag["spatial_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        diag["spatial_results"] = {
            "active": bool(spatial_intent.get("active")),
            "intent": spatial_intent,
            "count": len(spatial_results),
            "top5": [
                {
                    "file": r.get("file_name", r.get("file_path", "")),
                    "spatial_score": round(r.get("spatial_score", 0), 4),
                    "rank": i + 1,
                    "matches": r.get("spatial_matches", [])[:2],
                }
                for i, r in enumerate(spatial_results[:5])
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
        spatial_rank_map = {
            r["file_path"]: i + 1 for i, r in enumerate(spatial_results)
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
        if spatial_results:
            all_result_lists.append(("spatial", spatial_results))

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
            elif axis_name == "fts":
                fts_ranks = [r.get("fts_rank", 0) for r in single_results]
                best = min(fts_ranks)   # most negative = best match
                worst = max(fts_ranks)  # closest to 0 = worst match
                span = worst - best
                for r in single_results:
                    r["vector_score"] = None
                    r["text_vec_score"] = None
                    raw = r.get("fts_rank", 0)
                    r["text_score"] = (worst - raw) / span if span else 1.0
            else:  # spatial
                for r in single_results:
                    r["vector_score"] = None
                    r["text_vec_score"] = None
                    r["text_score"] = None
                    r["spatial_score"] = r.get("spatial_score", 0)
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
                    "spatial_rank": spatial_rank_map.get(r.get("file_path")),
                    "vector_score": round(r["vector_score"], 4) if r.get("vector_score") is not None else None,
                    "text_vec_score": round(r["text_vec_score"], 4) if r.get("text_vec_score") is not None else None,
                    "text_score": round(r["text_score"], 4) if r.get("text_score") is not None else None,
                    "spatial_score": round(r["spatial_score"], 4) if r.get("spatial_score") is not None else None,
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
        metadata_quality_weight = float(_search_cfg.get("search.rerank.metadata_quality_weight", 0.0))
        rerank_pool = max(top_k, rerank_pool)
        rerank_used = False
        annotate_metadata_quality(merged)
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
                metadata_quality_weight=metadata_quality_weight,
            )
            rerank_used = True

        diag["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        diag["rerank"] = {
            "enabled": rerank_enabled,
            "used": rerank_used,
            "pool_size": min(len(merged), rerank_pool),
            "metadata_quality_weight": metadata_quality_weight,
        }

        # If the decomposer did not emit an explicit scope, treat `X에서 ...`
        # as a soft intent signal only. This preserves recall while nudging
        # matching folders upward for user-scoped Korean queries.
        if soft_scope_file_ids and len(merged) > 1:
            scope_hint_boost = float(_search_cfg.get("search.scope_hint.soft_boost", 0.05))
            if scope_hint_boost > 0:
                rescored = []
                boosted = 0
                for idx, r in enumerate(merged):
                    base_score = float(r.get("quality_score") or r.get("rrf_score") or 0.0)
                    bonus = scope_hint_boost if r.get("id") in soft_scope_file_ids else 0.0
                    if bonus:
                        boosted += 1
                    r["scope_hint_score"] = bonus
                    rescored.append((base_score + bonus, r.get("rrf_score", 0.0), -idx, r))
                rescored.sort(reverse=True)
                merged = [r for _, _, _, r in rescored]
                diag["scope_hint_rerank"] = {
                    "folder": soft_scope_folder,
                    "file_count": len(soft_scope_file_ids),
                    "boost": scope_hint_boost,
                    "boosted_candidates": boosted,
                }

        # Enforce decomposed scene conditions before trimming so full-condition
        # candidates can rise from the wider RRF/rerank pool.
        elements_for_check: list[str] = []
        evidence_boost = 0.0
        try:
            import os as _bench_os2
            _disable_and = _bench_os2.environ.get("IMAGINE_BENCH_DISABLE_AND") == "1"
            if not _disable_and and isinstance(unified, dict):
                source: list[str] = []
                find_block = unified.get("find") or {}
                if isinstance(find_block, dict):
                    src = find_block.get("keywords")
                    if isinstance(src, list):
                        source = [
                            str(x).strip() for x in src
                            if isinstance(x, str) and str(x).strip()
                        ]
                if not source:
                    legacy_block = unified.get("_legacy") or {}
                    if isinstance(legacy_block, dict):
                        src = legacy_block.get("fts_keywords")
                        if isinstance(src, list):
                            source = [
                                str(x).strip() for x in src
                                if isinstance(x, str) and str(x).strip()
                            ]

                elements_for_check = _build_element_verification_groups(source)
            if len(elements_for_check) >= 2 and len(merged) > 1:
                pre_count = len(merged)
                merged = apply_element_verification(
                    merged, elements=elements_for_check, penalty=0.15,
                )
                evidence_boost = float(_search_cfg.get("search.rerank.evidence_matrix_boost", 0.20))
                if evidence_boost > 0:
                    merged = apply_evidence_matrix_rerank(
                        merged,
                        elements=elements_for_check,
                        boost=evidence_boost,
                    )
                    diag["evidence_matrix"] = {
                        "elements": elements_for_check,
                        "pre_count": pre_count,
                        "stage": "pre_trim",
                        "boost": evidence_boost,
                    }
                diag["element_verification"] = {
                    "elements": elements_for_check,
                    "pre_count": pre_count,
                    "stage": "pre_trim",
                }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Element verification skipped: {exc}")

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
                "spatial_score": round(r["spatial_score"], 4) if r.get("spatial_score") is not None else None,
                "quality_score": round(r["quality_score"], 4) if r.get("quality_score") is not None else None,
            }
            for r in merged[:5]
        ]

        # Cross-encoder rerank over top candidates.
        # BGE-reranker-v2-m3 is multilingual and CPU-friendly; load is
        # lazy so this is a no-op when transformers/the model is not
        # available. Env var IMAGINE_BENCH_DISABLE_RERANK=1 lets benches
        # skip it for A/B comparison.
        try:
            import os as _bench_os
            _disable_rerank = _bench_os.environ.get("IMAGINE_BENCH_DISABLE_RERANK") == "1"
            from backend.search.cross_encoder import (
                rerank as _ce_rerank,
                load_default_reranker,
            )
            reranker = None if _disable_rerank else load_default_reranker()
            if reranker is not None and len(merged) > 1:
                pool_size = min(30, len(merged))
                pool = merged[:pool_size]
                reranked = _ce_rerank(query=query, rows=pool, reranker=reranker)
                merged = reranked + merged[pool_size:]
                diag["cross_encoder_rerank"] = {"pool_size": pool_size}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Cross-encoder rerank skipped: {exc}")

        if diag.get("cross_encoder_rerank") and evidence_boost > 0 and len(elements_for_check) >= 2 and len(merged) > 1:
            merged = apply_evidence_matrix_rerank(
                merged,
                elements=elements_for_check,
                boost=evidence_boost,
            )
            diag.setdefault("evidence_matrix", {})["post_cross_encoder"] = True

        # Spatial-intent queries should keep structured spatial matches high.
        # Env IMAGINE_BENCH_DISABLE_SPATIAL=1 skips this for A/B comparisons.
        try:
            import os as _bench_os_sp
            _disable_spatial = _bench_os_sp.environ.get("IMAGINE_BENCH_DISABLE_SPATIAL") == "1"
            if not _disable_spatial and query_type == "spatial":
                pre_count = len(merged)
                merged = apply_spatial_intent_boost(
                    merged, query_type=query_type, boost=0.35,
                )
                boosted = sum(1 for r in merged if (r.get("spatial_score") or 0) > 0)
                diag["spatial_intent_boost"] = {
                    "pre_count": pre_count,
                    "rows_with_spatial_score": boosted,
                }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Spatial intent boost skipped: {exc}")

        # Phase D: soft demotion using accumulated user "irrelevant" feedback.
        # Files repeatedly flagged irrelevant for the same query get a small
        # rrf_score penalty so they drop in subsequent searches.
        try:
            penalty_rows = self.db.conn.execute(
                """SELECT file_id, COUNT(*) AS n FROM search_feedback
                   WHERE query = ? AND label = 'irrelevant'
                   GROUP BY file_id""",
                (query,),
            ).fetchall()
            if penalty_rows:
                penalty_map = {fid: 0.05 * min(n, 5) for fid, n in penalty_rows}
                penalized = 0
                for r in merged:
                    p = penalty_map.get(r.get("id"))
                    if p:
                        r["rrf_score"] = float(r.get("rrf_score") or 0.0) - p
                        r["feedback_penalty"] = p
                        penalized += 1
                if penalized:
                    merged.sort(key=lambda r: r.get("rrf_score", 0.0), reverse=True)
                    logger.info(
                        f"Phase D feedback demotion: {penalized} file(s) penalised "
                        f"for query={query!r}"
                    )
                    diag["feedback_demotion"] = {
                        "penalised_count": penalized,
                        "max_penalty": max(penalty_map.values()),
                    }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Feedback demotion skipped: {exc}")

        # Phase B: hard folder substring filter on folder_path as final safety net.
        # The earlier FTS-based pre-filter (Step 3b) catches folders that match via
        # indexed text; this catches the remaining cases where folder appears in
        # the path but not in indexed content (e.g., hash-prefixed names like #07).
        if folder_filter:
            pre_count = len(merged)
            hard_filtered = apply_folder_filter(merged, folder_filter)
            # Only apply when the hard filter keeps a usable result count.
            if len(hard_filtered) >= 3:
                merged = hard_filtered
                logger.info(
                    f"Phase B folder hard-filter '{folder_filter}': "
                    f"{pre_count} → {len(merged)} results"
                )
                diag["folder_hard_filter"] = {
                    "folder": folder_filter,
                    "before": pre_count,
                    "after": len(merged),
                }

        diag["final_results_count"] = len(merged)
        diag["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

        logger.info(
            f"Triaxis search '{query}': vector={len(vector_results)}, "
            f"fts={len(fts_results)}, spatial={len(spatial_results)}, merged={len(merged)}, "
            f"decomposed={unified.get('decomposed', False)}"
        )

        # Write diagnostic log
        if _diagnostic_log_enabled(_search_cfg):
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

    def _load_spatial_objects(self, file_id: Any) -> list[dict]:
        """Load normalized object-location evidence for a file."""
        if not file_id:
            return []
        try:
            rows = self.db.conn.execute(
                """SELECT name, ko_name, primary_location, locations,
                          extent, confidence, spatial_text
                   FROM file_objects
                   WHERE file_id = ?
                   ORDER BY id""",
                (file_id,),
            ).fetchall()
        except Exception:
            return []

        objects: list[dict] = []
        for row in rows:
            try:
                locations = json.loads(row["locations"] or "[]")
            except (json.JSONDecodeError, TypeError):
                locations = []
            objects.append({
                "name": row["name"],
                "ko_name": row["ko_name"],
                "primary_location": row["primary_location"],
                "locations": locations,
                "extent": row["extent"],
                "confidence": row["confidence"],
                "spatial_text": row["spatial_text"],
            })
        return objects

    def _load_spatial_relations(self, file_id: Any) -> list[dict]:
        """Load normalized object-to-object spatial relations for a file."""
        if not file_id:
            return []
        try:
            rows = self.db.conn.execute(
                """SELECT subject, relation, object, subject_location,
                          object_location, confidence, spatial_text
                   FROM file_spatial_relations
                   WHERE file_id = ?
                   ORDER BY id""",
                (file_id,),
            ).fetchall()
        except Exception:
            return []

        return [
            {
                "subject": row["subject"],
                "relation": row["relation"],
                "object": row["object"],
                "subject_location": row["subject_location"],
                "object_location": row["object_location"],
                "confidence": row["confidence"],
                "spatial_text": row["spatial_text"],
            }
            for row in rows
        ]

    def _load_depth_layers(self, file_id: Any) -> list[dict]:
        """Load normalized foreground/midground/background evidence for a file."""
        if not file_id:
            return []
        try:
            rows = self.db.conn.execute(
                """SELECT name, ko_name, layer, confidence, spatial_text
                   FROM file_depth_layers
                   WHERE file_id = ?
                   ORDER BY id""",
                (file_id,),
            ).fetchall()
        except Exception:
            return []

        return [
            {
                "name": row["name"],
                "ko_name": row["ko_name"],
                "layer": row["layer"],
                "confidence": row["confidence"],
                "spatial_text": row["spatial_text"],
            }
            for row in rows
        ]

    def _parse_json_fields(self, result: Dict) -> None:
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
        if result.get("structured_meta"):
            if isinstance(result["structured_meta"], dict):
                structured_meta = result["structured_meta"]
            else:
                try:
                    structured_meta = json.loads(result["structured_meta"])
                except (json.JSONDecodeError, TypeError):
                    structured_meta = {}
            result["structured_meta"] = structured_meta
            result["spatial_processing_quality"] = structured_meta.get(
                "spatial_processing_quality", {}
            )
        else:
            result["spatial_processing_quality"] = {}
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
        result["spatial_objects"] = self._load_spatial_objects(result.get("id"))
        result["spatial_relations"] = self._load_spatial_relations(result.get("id"))
        result["depth_layers"] = self._load_depth_layers(result.get("id"))

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

        annotate_metadata_quality(merged)

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
        metadata_quality_weight = float(_search_cfg.get("search.rerank.metadata_quality_weight", 0.0))
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
                metadata_quality_weight=metadata_quality_weight,
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
        if _diagnostic_log_enabled():
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
            return self.fts_search([query], top_k, file_ids=file_ids)
        elif mode == "triaxis":
            return self.triaxis_search(query, filters, top_k, threshold, return_diagnostic=return_diagnostic, use_codex=use_codex, file_ids=file_ids, progress_callback=progress_callback)
        elif mode == "plan":
            return self.plan_search(query, top_k, threshold, return_diagnostic=return_diagnostic)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'vector', 'hybrid', 'metadata', 'fts', 'triaxis', 'plan', or 'structure'")


# ── Phase B: hard folder constraint ───────────────────────────────

def apply_folder_filter(rows, folder: str):
    """Drop rows whose folder_path doesn't contain the requested folder.

    Substring match for now — the same string the Decomposer extracts
    ("#07", "studio_bg" etc.) is matched against folder_path. Exact
    boundary handling is upgraded once we have evidence it matters.
    """
    if not folder:
        return rows
    needle = folder.strip()
    if not needle:
        return rows
    return [r for r in rows if needle in (r.get("folder_path") or "")]


def _build_element_verification_groups(
    source: list[str],
    *,
    max_groups: int = 4,
) -> list[str]:
    """Build synonym groups for multi-condition evidence verification."""
    if not source:
        return []

    location_noise = {
        *SQLiteDB._SPATIAL_LOCATIONS,
        *SQLiteDB._SPATIAL_LOCATION_ALIASES.keys(),
        *_SPATIAL_KO_LOCATION_ALIASES.keys(),
    }
    generic = {
        "이미지", "사진", "그림", "파일", "자료",
        "image", "images", "photo", "photos", "picture", "pictures",
    }

    def _classify_lang(token: str) -> str:
        for ch in token:
            if 0xAC00 <= ord(ch) <= 0xD7A3:
                return "ko"
        return "en"

    def _clean(token: str) -> str:
        normalized = str(token or "").strip()
        if any("\uac00" <= c <= "\ud7af" for c in normalized):
            normalized = _SPATIAL_KO_PARTICLES_RE.sub("", normalized) or normalized
        return normalized.strip()

    known_pairs = {
        "벽": {"wall", "brick wall"},
        "커튼": {"curtain", "curtains", "blind"},
        "창문": {"window", "bright window"},
        "문": {"door", "gate"},
        "바닥": {"floor", "ground"},
        "천장": {"ceiling"},
        "등": {"lamp", "light", "sconce"},
        "조명": {"light", "lighting", "light fixture"},
        "배관": {"pipe", "pipes"},
        "레일": {"rail", "railing"},
        "수납장": {"cabinet"},
        "책장": {"bookshelf"},
        "병": {"bottle"},
        "상자": {"box", "boxes"},
        "선반": {"shelf"},
        "화면": {"screen"},
        "버튼": {"button"},
        "식물": {"plant"},
        "의자": {"chair", "bench", "armchair"},
        "테이블": {"table"},
        "초콜릿": {"chocolate", "candy"},
    }

    def _phrase_noise(token: str, all_source: set[str]) -> bool:
        low = token.lower()
        if "," in token or "，" in token:
            return True
        if re.search(r"[가-힣]\s*(?:과|와|하고|및)\s*[가-힣]", token):
            return True
        if any(glue in low.split() for glue in {"and", "with", "together"}):
            return True
        parts = [part for part in re.split(r"\s+", low) if part]
        if len(parts) >= 2 and any(part in all_source for part in parts):
            return True
        return False

    def _keep(token: str) -> bool:
        low = token.lower()
        if not token or low in _SPATIAL_STOPWORDS or low in generic:
            return False
        if token in location_noise or low in {str(v).lower() for v in location_noise}:
            return False
        if len(token) == 1 and _classify_lang(token) != "ko":
            return False
        return True

    raw_cleaned = [_clean(raw) for raw in source]
    source_terms = {token.lower() for token in raw_cleaned if token}
    cleaned = _ordered_unique([
        token for token in raw_cleaned
        if _keep(token) and not _phrase_noise(token, source_terms)
    ])
    ko_tokens = [t for t in cleaned if _classify_lang(t) == "ko"]
    en_tokens = [t for t in cleaned if _classify_lang(t) == "en"]

    groups: list[str] = []
    remaining_en = list(en_tokens)
    for ko_token in ko_tokens[:max_groups]:
        expected = known_pairs.get(ko_token, set())
        match_idx = next(
            (
                idx for idx, en_token in enumerate(remaining_en)
                if en_token.lower() in expected
            ),
            None,
        )
        if match_idx is None:
            continue
        en_token = remaining_en.pop(match_idx)
        groups.append(f"{ko_token}|{en_token}")

    paired_ko = {group.split("|", 1)[0] for group in groups}
    unpaired_ko = [token for token in ko_tokens if token not in paired_ko]
    pair_count = min(len(unpaired_ko), len(remaining_en), max_groups - len(groups))
    for idx in range(pair_count):
        groups.append(f"{unpaired_ko[idx]}|{remaining_en[idx]}")

    remaining = unpaired_ko[pair_count:] + remaining_en[pair_count:]
    for token in remaining:
        if len(groups) >= max_groups:
            break
        groups.append(token)

    return groups


def apply_element_verification(
    rows,
    *,
    elements,
    penalty: float = 0.10,
):
    """Penalise rows missing requested condition elements.

    Counts how many elements appear (case-insensitive substring) in the
    combined text of mc_caption + ai_tags + str(spatial_objects). For
    each missing element, subtract `penalty` from `rrf_score`. Then
    re-sort by rrf_score descending. Rows matching every element are
    unchanged.

    Elements can be in Korean or English; the substring match works on
    whichever language the row's text uses.
    """
    if not elements:
        return rows

    # Each "element" can be either a plain string OR a `|`-separated
    # synonym group ("balcony|발코니"). A group counts as PRESENT when
    # ANY of its synonyms appears in the row text. This is what lets
    # us match Korean intent ("발코니") against English captions
    # ("balcony") without manual translation tables.
    needle_groups: list[tuple[str, list[str]]] = []
    for e in elements:
        if not isinstance(e, str):
            continue
        synonyms = [
            piece.strip().lower()
            for piece in e.split("|")
            if piece.strip()
        ]
        if synonyms:
            needle_groups.append((e, synonyms))
    if not needle_groups:
        return rows

    def _as_text(value) -> str:
        # ai_tags / spatial_objects can arrive as list OR comma-string
        # depending on the enrichment path. Normalise both to plain text
        # so " ".join() never gets a list element.
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return " ".join(str(x) for x in value)
        return str(value)

    uses_cross_encoder = any(
        r.get("cross_encoder_score") is not None for r in rows
    )

    for r in rows:
        haystack = " ".join([
            _as_text(r.get("mc_caption")),
            _as_text(r.get("ai_tags")),
            _as_text(r.get("spatial_objects")),
        ]).lower()

        matched: list[str] = []
        missing_groups: list[str] = []
        evidence: dict[str, list[str]] = {}
        for label, group in needle_groups:
            hits = [n for n in group if n in haystack]
            if hits:
                matched.append(label)
                evidence[label] = hits
            else:
                missing_groups.append(label)
        present = len(matched)
        missing = len(needle_groups) - present
        r["element_match_count"] = present
        r["element_miss_count"] = missing
        r["element_match_ratio"] = round(present / len(needle_groups), 4)
        r["element_missing"] = missing_groups
        r["element_evidence"] = evidence
        if missing:
            r["rrf_score"] = float(r.get("rrf_score") or 0.0) - penalty * missing

    rows = list(rows)
    if any(r.get("quality_score") is not None for r in rows):
        rows.sort(
            key=lambda r: float(r.get("quality_score") or 0.0)
            - penalty * (r.get("element_miss_count") or 0),
            reverse=True,
        )
    elif uses_cross_encoder:
        # Cross-encoder ran upstream; preserve its ordering but layer the
        # AND penalty on top so missing-element rows drop within the
        # rerank's preference order.
        rows.sort(
            key=lambda r: float(r.get("cross_encoder_score") or 0.0)
            - penalty * (r.get("element_miss_count") or 0),
            reverse=True,
        )
    else:
        rows.sort(key=lambda r: r.get("rrf_score", 0.0), reverse=True)
    return rows


def _evidence_value_text(value) -> str:
    """Compact recursive text for already-loaded search result evidence."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(
            _evidence_value_text(v)
            for v in value.values()
            if v is not None
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_evidence_value_text(v) for v in value)
    return str(value)


def _candidate_evidence_text(row: dict) -> str:
    """Build evidence text from existing data/search axes without new parsing."""
    fields = [
        "file_name", "folder_path", "relative_path", "file_path",
        "mc_caption", "ai_tags", "metadata", "structured_meta",
        "image_type", "art_style", "color_palette", "scene_type",
        "time_of_day", "weather", "character_type", "item_type", "ui_type",
        "user_note", "user_tags", "user_category",
        "spatial_objects", "spatial_relations", "spatial_matches",
        "depth_layers",
    ]
    return " ".join(_evidence_value_text(row.get(field)) for field in fields).lower()


def _candidate_structured_object_text(row: dict) -> str:
    """Build text only from structured object rows attached to the result."""
    return _evidence_value_text(row.get("spatial_objects")).lower()


def _axis_evidence(row: dict) -> dict:
    """Summarize which existing axes supplied usable evidence for a row."""
    return {
        "visual": {
            "present": row.get("vector_score") is not None or row.get("similarity") is not None,
            "score": row.get("vector_score", row.get("similarity")),
        },
        "text_vec": {
            "present": row.get("text_vec_score") is not None or row.get("text_similarity") is not None,
            "score": row.get("text_vec_score", row.get("text_similarity")),
        },
        "fts": {
            "present": row.get("text_score") is not None or row.get("fts_rank") is not None,
            "score": row.get("text_score", row.get("fts_rank")),
        },
        "spatial": {
            "present": bool(row.get("spatial_score")) or bool(row.get("spatial_matches")),
            "score": row.get("spatial_score"),
        },
        "metadata": {
            "present": bool(
                row.get("metadata")
                or row.get("structured_meta")
                or row.get("mc_caption")
                or row.get("ai_tags")
                or row.get("image_type")
                or row.get("scene_type")
                or row.get("art_style")
                or row.get("user_tags")
                or row.get("user_note")
                or row.get("user_category")
            ),
            "score": row.get("metadata_reliability_score"),
        },
    }


def apply_evidence_matrix_rerank(
    rows,
    *,
    elements,
    boost: float = 0.20,
):
    """Attach condition/axis evidence and rerank using existing search data.

    This does not create a new search pipeline. It only reads fields already
    loaded by VV/MV/FTS/spatial/metadata paths and makes their support explicit.
    """
    if not rows or not elements:
        return rows

    needle_groups: list[tuple[str, list[str]]] = []
    for element in elements:
        if not isinstance(element, str):
            continue
        synonyms = [piece.strip().lower() for piece in element.split("|") if piece.strip()]
        if synonyms:
            needle_groups.append((element, synonyms))
    if not needle_groups:
        return rows

    rescored = []
    for idx, row in enumerate(rows):
        haystack = _candidate_evidence_text(row)
        object_haystack = _candidate_structured_object_text(row)
        matches: dict[str, list[str]] = {}
        missing: list[str] = []
        object_matches: dict[str, list[str]] = {}
        object_missing: list[str] = []
        for label, synonyms in needle_groups:
            hits = [synonym for synonym in synonyms if synonym in haystack]
            if hits:
                matches[label] = hits
            else:
                missing.append(label)
            object_hits = [synonym for synonym in synonyms if synonym in object_haystack]
            if object_hits:
                object_matches[label] = object_hits
            else:
                object_missing.append(label)

        axis_evidence = _axis_evidence(row)
        axis_count = sum(1 for axis in axis_evidence.values() if axis["present"])
        match_count = len(matches)
        object_match_count = len(object_matches)
        match_ratio = round(match_count / len(needle_groups), 4)
        object_full_bonus = 1.0 if object_match_count == len(needle_groups) and len(needle_groups) >= 2 else 0.0
        evidence_score = round(
            match_count + (0.25 * axis_count) + object_match_count + object_full_bonus,
            4,
        )
        row["evidence_score"] = evidence_score
        row["evidence_matrix"] = {
            "conditions": {
                "total": len(needle_groups),
                "matched": match_count,
                "missing": missing,
                "match_ratio": match_ratio,
                "matches": matches,
                "object_matched": object_match_count,
                "object_missing": object_missing,
                "object_matches": object_matches,
            },
            "axes": axis_evidence,
        }

        base = float(row.get("quality_score") or row.get("rrf_score") or 0.0)
        rescored.append((base + (float(boost) * evidence_score), row.get("rrf_score", 0.0), -idx, row))

    rescored.sort(reverse=True)
    return [row for _, _, _, row in rescored]


def apply_spatial_intent_boost(rows, *, query_type, boost: float = 0.10):
    """Boost rows with spatial evidence for spatial-intent queries."""
    if query_type != "spatial" or not rows:
        return rows

    has_ce = any(r.get("cross_encoder_score") is not None for r in rows)
    boosted_count = 0
    for r in rows:
        s = float(r.get("spatial_score") or 0.0)
        if s <= 0.0:
            continue
        gain = boost * s
        r["rrf_score"] = float(r.get("rrf_score") or 0.0) + gain
        if r.get("quality_score") is not None:
            r["quality_score"] = float(r.get("quality_score") or 0.0) + gain
        if r.get("cross_encoder_score") is not None:
            r["cross_encoder_score"] = (
                float(r.get("cross_encoder_score") or 0.0) + gain
            )
        boosted_count += 1

    rows = list(rows)
    if any(r.get("quality_score") is not None for r in rows):
        rows.sort(
            key=lambda r: float(r.get("quality_score") or 0.0),
            reverse=True,
        )
    elif has_ce:
        rows.sort(
            key=lambda r: float(r.get("cross_encoder_score") or 0.0),
            reverse=True,
        )
    else:
        rows.sort(key=lambda r: r.get("rrf_score", 0.0), reverse=True)
    return rows
