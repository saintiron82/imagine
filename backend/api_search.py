"""
API wrapper for SQLite search.
Returns JSON for Electron frontend.

Supports two input modes:
1. stdin JSON: {"query":"...", "limit":20, "mode":"triaxis", "filters":{...}}
2. Positional args: python api_search.py "query" [limit]  (backward compatible)
"""
import sys
import json
import logging
import io
import os
from pathlib import Path

# Force UTF-8 stdout/stdin for multilingual support (JP, KR, CN, etc.)
# Windows defaults to cp949/cp932 which breaks non-local characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# Suppress tqdm/transformers progress bars that pollute stdout
# (Electron reads stdout for JSON — any non-JSON breaks parsing)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.search.sqlite_search import SqliteVectorSearch

# Suppress noisy logs from libraries during search
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def format_result(result: dict) -> dict:
    """Format a single search result for the frontend."""
    metadata = result.get("metadata", {})

    formatted = {
        "path": result["file_path"],
        "folder_path": result.get("folder_path", ""),
        "relative_path": result.get("relative_path", ""),
        "storage_root": result.get("storage_root", ""),
        "vector_score": result.get("vector_score", result.get("similarity")),     # V-axis: SigLIP visual
        "text_vec_score": result.get("text_vec_score", result.get("text_similarity")),  # S-axis: Qwen3 text vector
        "text_score": result.get("text_score"),          # M-axis: FTS5 keyword
        "combined_score": result.get("rrf_score", result.get("similarity", 0)),
        "metadata": metadata,
        "thumbnail_path": result.get("thumbnail_url", ""),
        "format": result.get("format", ""),
        "width": result.get("width", 0),
        "height": result.get("height", 0),
        "layer_count": metadata.get("layer_count", 0),
        "mc_caption": result.get("mc_caption", ""),
        "ai_tags": result.get("ai_tags", []),
        "semantic_tags": metadata.get("semantic_tags", ""),
        "user_note": result.get("user_note", ""),
        "user_tags": result.get("user_tags", []),
        "user_category": result.get("user_category", ""),
        "user_rating": result.get("user_rating", 0),
        # v3 P0: structured vision fields
        "image_type": result.get("image_type"),
        "art_style": result.get("art_style"),
        "scene_type": result.get("scene_type"),
    }

    return formatted


def search(query: str = "", limit: int = 20, mode: str = "triaxis", filters: dict = None, threshold: float = 0.0, diagnostic: bool = False, query_image: str = None, query_images: list = None, image_search_mode: str = "and"):
    """Search SQLite and return JSON results."""
    try:
        searcher = SqliteVectorSearch()
        result_data = searcher.search(
            query, mode=mode, filters=filters, top_k=limit,
            threshold=threshold, return_diagnostic=diagnostic,
            query_image=query_image,
            query_images=query_images,
            image_search_mode=image_search_mode,
        )

        if diagnostic and isinstance(result_data, tuple):
            results, diag = result_data
        else:
            results = result_data
            diag = None

        formatted = [format_result(r) for r in results]
        response = {"success": True, "results": formatted, "count": len(formatted)}

        if diag is not None:
            response["diagnostic"] = diag

        return response

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"success": False, "error": str(e), "results": []}


if __name__ == "__main__":
    # Try reading stdin JSON first (new protocol)
    stdin_data = None
    if not sys.stdin.isatty():
        try:
            raw = sys.stdin.read().strip()
            if raw:
                stdin_data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass

    if stdin_data and isinstance(stdin_data, dict) and ("query" in stdin_data or "query_image" in stdin_data or "query_images" in stdin_data):
        query = stdin_data.get("query", "")
        query_image = stdin_data.get("query_image", None)
        query_images = stdin_data.get("query_images", None)
        image_search_mode = stdin_data.get("image_search_mode", "and")
        limit = stdin_data.get("limit", 20)
        mode = stdin_data.get("mode", "triaxis")
        threshold = float(stdin_data.get("threshold", 0.0))
        filters = stdin_data.get("filters", None)
        diag_flag = stdin_data.get("diagnostic", False)
    elif len(sys.argv) >= 2:
        # Backward compatible: positional args
        diag_flag = "--diagnostic" in sys.argv
        positional = [a for a in sys.argv[1:] if not a.startswith("--")]
        query = positional[0]
        query_image = None
        query_images = None
        image_search_mode = "and"
        limit = int(positional[1]) if len(positional) > 1 else 20
        mode = "triaxis"
        threshold = 0.0
        filters = None
    else:
        print(json.dumps({"success": False, "error": "No query provided"}))
        sys.exit(1)

    result = search(query, limit, mode, filters, threshold=threshold, diagnostic=diag_flag, query_image=query_image, query_images=query_images, image_search_mode=image_search_mode)
    print(json.dumps(result, ensure_ascii=False))
