"""
Search router — Triaxis search API.

Wraps existing SqliteVectorSearch with FastAPI endpoints.
Reuses format_result() from api_search.py for response compatibility.
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.db.sqlite_client import SQLiteDB
from backend.search.sqlite_search import SqliteVectorSearch
from backend.server.deps import get_db, get_db_safe, get_current_user
from backend.api_search import format_result
from backend.search.search_logger import log_search as _log_search
from backend.search.confidence import (
    ConfidenceLevel,
    classify_topk,
    thresholds_from_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

# Singleton searcher (lazy init, models loaded once)
_searcher: Optional[SqliteVectorSearch] = None


def _get_searcher() -> SqliteVectorSearch:
    global _searcher
    if _searcher is None:
        _searcher = SqliteVectorSearch()
    return _searcher


# ── Request schemas ──────────────────────────────────────────

class SearchRequest(BaseModel):
    query: Optional[str] = ""
    query_image: Optional[str] = None       # base64 image
    query_images: Optional[List[str]] = None  # multiple base64 images
    image_search_mode: str = "and"           # "and" | "or"
    query_file_id: Optional[int] = None      # search by file ID
    limit: int = Field(20, ge=1, le=200)
    threshold: float = Field(0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None
    diagnostic: bool = False
    use_codex: bool = True
    effort: Optional[str] = None              # accepted for local/HTTP parity
    file_ids: Optional[List[int]] = Field(None, max_length=5000)


# ── Endpoints ────────────────────────────────────────────────

@router.post("/triaxis")
def search_triaxis(
    req: SearchRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
):
    """Triaxis search: VV + MV + FTS combined via RRF."""
    return _do_search(req, mode="triaxis", user=_user, request=request)


@router.post("/visual")
def search_visual(
    req: SearchRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
):
    """VV-only search (SigLIP2 visual similarity)."""
    return _do_search(req, mode="vector", user=_user, request=request)


@router.post("/semantic")
def search_semantic(
    req: SearchRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
):
    """MV-only search (Qwen3-Embedding text similarity)."""
    return _do_search(req, mode="text_vector", user=_user, request=request)


@router.post("/keyword")
def search_keyword(
    req: SearchRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
):
    """FTS-only search (FTS5 BM25 keyword matching)."""
    return _do_search(req, mode="fts", user=_user, request=request)


@router.post("/structure")
def search_structure(
    req: SearchRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
):
    """Structure-only search (DINOv2 structural similarity)."""
    return _do_search(req, mode="structure", user=_user, request=request)


def _do_search(req: SearchRequest, mode: str,
               user: dict = None, request: Request = None) -> dict:
    """Execute search and format results."""
    t0 = time.time()
    try:
        searcher = _get_searcher()
        result_data = searcher.search(
            req.query or "",
            mode=mode,
            filters=req.filters,
            top_k=req.limit,
            threshold=req.threshold,
            return_diagnostic=req.diagnostic,
            query_image=req.query_image,
            query_images=req.query_images,
            image_search_mode=req.image_search_mode,
            query_file_id=req.query_file_id,
            use_codex=req.use_codex,
            file_ids=set(req.file_ids) if req.file_ids else None,
        )

        if req.diagnostic and isinstance(result_data, tuple):
            results, diag = result_data
        else:
            results = result_data
            diag = None

        formatted = [format_result(r, skip_fs=True) for r in results]
        elapsed_ms = int((time.time() - t0) * 1000)

        # Log search request
        query_text = req.query or f"[file_id:{req.query_file_id}]" if req.query_file_id else (req.query or "")
        if req.query_images:
            query_text = query_text or f"[image_search:{len(req.query_images)} images]"
        elif req.query_image:
            query_text = query_text or "[image_search]"
        username = user.get("username") if user else None
        ip = request.client.host if request and request.client else None
        _log_search(
            query=query_text, mode=mode, result_count=len(formatted),
            elapsed_ms=elapsed_ms, username=username, ip_address=ip,
            filters=req.filters, threshold=req.threshold,
        )

        # Phase A: confidence + empty-mode envelope.
        try:
            from backend.utils.config import get_config
            thresholds = thresholds_from_config(get_config())
        except Exception:
            from backend.search.confidence import ConfidenceThresholds
            thresholds = ConfidenceThresholds()

        top1 = formatted[0] if formatted else None
        if top1 is None:
            confidence = ConfidenceLevel.EMPTY
            top1_score = 0.0
        else:
            vec = float(top1.get("vector_score") or 0.0)
            txt_vec = float(top1.get("text_vec_score") or 0.0)
            fts_hit = bool(top1.get("text_score"))
            confidence = classify_topk(
                vector_score=vec,
                text_vec_score=txt_vec,
                fts_hit=fts_hit,
                thresholds=thresholds,
            )
            top1_score = max(vec, txt_vec)

        if confidence is ConfidenceLevel.EMPTY:
            response = {
                "success": True,
                "results": [],
                "count": 0,
                "elapsed_ms": elapsed_ms,
                "confidence": confidence.value,
                "top1_raw_score": top1_score,
                "empty_reason": "no_result_above_confidence_threshold",
            }
        else:
            response = {
                "success": True,
                "results": formatted,
                "count": len(formatted),
                "elapsed_ms": elapsed_ms,
                "confidence": confidence.value,
                "top1_raw_score": top1_score,
            }
        if diag is not None:
            response["diagnostic"] = diag

        return response

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ── Search logs API ─────────────────────────────────────────

@router.get("/logs")
def get_search_logs(
    limit: int = 50,
    offset: int = 0,
    username: Optional[str] = None,
    _admin: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List search logs (admin-visible)."""
    cursor = db.conn.cursor()
    conditions = []
    params = []
    if username:
        conditions.append("username = ?")
        params.append(username)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    cursor.execute(f"SELECT COUNT(*) FROM search_logs {where}", params)
    total = cursor.fetchone()[0]

    cursor.execute(
        f"""SELECT id, query, mode, result_count, elapsed_ms,
                   username, ip_address, filters, threshold, created_at
            FROM search_logs {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?""",
        params + [min(limit, 100), offset]
    )
    logs = [
        {
            "id": r[0], "query": r[1], "mode": r[2],
            "result_count": r[3], "elapsed_ms": r[4],
            "username": r[5], "ip_address": r[6],
            "filters": json.loads(r[7]) if r[7] else None,
            "threshold": r[8], "created_at": r[9],
        }
        for r in cursor.fetchall()
    ]
    return {"success": True, "logs": logs, "total": total}
