"""
API wrapper for SQLite search — persistent daemon mode.

Runs as a long-lived process. Models are loaded once on first search,
then kept in memory for instant subsequent queries.

Protocol (line-delimited JSON over stdin/stdout):
  → stdin:  one JSON object per line (newline-terminated)
  → stdout: one JSON response per line (newline-terminated)
  Special commands:
    {"cmd":"ping"}   → {"status":"ok","pid":...}
    {"cmd":"warmup"} → pre-loads models, returns {"status":"ready"}
    {"cmd":"quit"}   → exits cleanly

Backward compatible: also works as single-shot CLI with positional args.
"""
import sys
import json
import logging
import io
import os
import time
import traceback
from pathlib import Path

# Force UTF-8 stdout/stdin for multilingual support (JP, KR, CN, etc.)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# Suppress tqdm/transformers progress bars that pollute stdout
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Set process title for Activity Monitor visibility
try:
    import setproctitle
    setproctitle.setproctitle("Imagine-Search")
except ImportError:
    pass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.search.sqlite_search import SqliteVectorSearch

# Suppress noisy logs from libraries during search
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Persistent searcher instance (models loaded once, reused across requests)
_searcher: SqliteVectorSearch = None


def _get_searcher() -> SqliteVectorSearch:
    """Get or create the singleton searcher (lazy init)."""
    global _searcher
    if _searcher is None:
        _searcher = SqliteVectorSearch()
    return _searcher


def format_result(result: dict) -> dict:
    """Format a single search result for the frontend."""
    metadata = result.get("metadata", {})

    formatted = {
        "path": result["file_path"],
        "folder_path": result.get("folder_path", ""),
        "relative_path": result.get("relative_path", ""),
        "storage_root": result.get("storage_root", ""),
        "vector_score": result.get("vector_score", result.get("similarity")),     # VV: SigLIP visual
        "text_vec_score": result.get("text_vec_score", result.get("text_similarity")),  # MV: Qwen3 text vector
        "text_score": result.get("text_score"),          # FTS: FTS5 keyword
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
        searcher = _get_searcher()
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


def _handle_request(data: dict) -> dict:
    """Handle a single request (search or command)."""
    cmd = data.get("cmd")

    if cmd == "ping":
        return {"status": "ok", "pid": os.getpid()}

    if cmd == "warmup":
        t0 = time.time()
        searcher = _get_searcher()
        # Trigger model loading via a dummy search
        searcher.search("warmup", mode="triaxis", top_k=1)
        return {"status": "ready", "warmup_ms": int((time.time() - t0) * 1000)}

    if cmd == "quit":
        return {"status": "bye"}

    # Normal search request
    return search(
        query=data.get("query", ""),
        limit=data.get("limit", 20),
        mode=data.get("mode", "triaxis"),
        filters=data.get("filters"),
        threshold=float(data.get("threshold", 0.0)),
        diagnostic=data.get("diagnostic", False),
        query_image=data.get("query_image"),
        query_images=data.get("query_images"),
        image_search_mode=data.get("image_search_mode", "and"),
    )


def _write_response(response: dict):
    """Write a single JSON response line to stdout."""
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_daemon():
    """Run persistent daemon: read line-delimited JSON from stdin, respond on stdout."""
    _write_response({"status": "ok", "pid": os.getpid(), "mode": "daemon"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            _write_response({"success": False, "error": f"JSON parse error: {e}"})
            continue

        try:
            t0 = time.time()
            response = _handle_request(data)
            response["elapsed_ms"] = int((time.time() - t0) * 1000)
            _write_response(response)

            if data.get("cmd") == "quit":
                break
        except Exception as e:
            _write_response({"success": False, "error": str(e), "traceback": traceback.format_exc()})


def run_oneshot():
    """Run single-shot mode (backward compatible)."""
    stdin_data = None
    if not sys.stdin.isatty():
        try:
            raw = sys.stdin.read().strip()
            if raw:
                stdin_data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass

    if stdin_data and isinstance(stdin_data, dict) and ("query" in stdin_data or "query_image" in stdin_data or "query_images" in stdin_data):
        result = search(
            query=stdin_data.get("query", ""),
            limit=stdin_data.get("limit", 20),
            mode=stdin_data.get("mode", "triaxis"),
            filters=stdin_data.get("filters"),
            threshold=float(stdin_data.get("threshold", 0.0)),
            diagnostic=stdin_data.get("diagnostic", False),
            query_image=stdin_data.get("query_image"),
            query_images=stdin_data.get("query_images"),
            image_search_mode=stdin_data.get("image_search_mode", "and"),
        )
    elif len(sys.argv) >= 2 and sys.argv[1] != "--daemon":
        diag_flag = "--diagnostic" in sys.argv
        positional = [a for a in sys.argv[1:] if not a.startswith("--")]
        result = search(
            query=positional[0],
            limit=int(positional[1]) if len(positional) > 1 else 20,
            diagnostic=diag_flag,
        )
    else:
        print(json.dumps({"success": False, "error": "No query provided"}))
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        run_daemon()
    else:
        run_oneshot()
