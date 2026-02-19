"""
Search router — Triaxis search API.

Wraps existing SqliteVectorSearch with FastAPI endpoints.
Reuses format_result() from api_search.py for response compatibility.
"""

import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.db.sqlite_client import SQLiteDB
from backend.search.sqlite_search import SqliteVectorSearch
from backend.server.deps import get_db, get_current_user
from backend.api_search import format_result

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


# ── Endpoints ────────────────────────────────────────────────

@router.post("/triaxis")
def search_triaxis(
    req: SearchRequest,
    _user: dict = Depends(get_current_user),
):
    """Triaxis search: VV + MV + FTS combined via RRF."""
    return _do_search(req, mode="triaxis")


@router.post("/visual")
def search_visual(
    req: SearchRequest,
    _user: dict = Depends(get_current_user),
):
    """VV-only search (SigLIP2 visual similarity)."""
    return _do_search(req, mode="vector")


@router.post("/semantic")
def search_semantic(
    req: SearchRequest,
    _user: dict = Depends(get_current_user),
):
    """MV-only search (Qwen3-Embedding text similarity)."""
    return _do_search(req, mode="text_vector")


@router.post("/keyword")
def search_keyword(
    req: SearchRequest,
    _user: dict = Depends(get_current_user),
):
    """FTS-only search (FTS5 BM25 keyword matching)."""
    return _do_search(req, mode="fts")


@router.post("/similar/{file_id}")
def search_similar(
    file_id: int,
    limit: int = 20,
    _user: dict = Depends(get_current_user),
):
    """Find similar images to a given file (VV + Structure)."""
    return _do_search(
        SearchRequest(query_file_id=file_id, limit=limit),
        mode="triaxis",
    )


def _do_search(req: SearchRequest, mode: str) -> dict:
    """Execute search and format results."""
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
        )

        if req.diagnostic and isinstance(result_data, tuple):
            results, diag = result_data
        else:
            results = result_data
            diag = None

        formatted = [format_result(r) for r in results]
        response = {
            "success": True,
            "results": formatted,
            "count": len(formatted),
        }
        if diag is not None:
            response["diagnostic"] = diag

        return response

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
