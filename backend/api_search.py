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
from typing import List

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
from backend.db.sqlite_client import SQLiteDB
from backend.utils.thumbnail_resolver import resolve_thumbnail_str
from backend.search.search_logger import log_search as _log_search

# Suppress noisy logs from libraries during search
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Persistent searcher instance (models loaded once, reused across requests)
_searcher: SqliteVectorSearch = None


def _candidate_roots() -> List[Path]:
    """
    Build candidate local roots used for path resolution on DB handoff.

    Priority:
    1) IMAGINE_PATH_ROOTS env (os.pathsep-separated)
    2) config.yaml -> registered_folders.folders
    """
    roots: List[Path] = []
    seen = set()

    env_roots = os.getenv("IMAGINE_PATH_ROOTS", "")
    if env_roots:
        for raw in env_roots.split(os.pathsep):
            p = raw.strip()
            if not p:
                continue
            path_obj = Path(p).expanduser()
            key = str(path_obj)
            if key not in seen:
                seen.add(key)
                roots.append(path_obj)

    try:
        from backend.utils.config import get_config
        cfg = get_config()
        cfg_roots = cfg.get("registered_folders.folders", [])
        if isinstance(cfg_roots, list):
            for raw in cfg_roots:
                if not raw:
                    continue
                path_obj = Path(str(raw)).expanduser()
                key = str(path_obj)
                if key not in seen:
                    seen.add(key)
                    roots.append(path_obj)
    except Exception:
        pass

    return roots


def _resolve_local_path(result: dict) -> str:
    """
    Resolve DB path to a local existing path using relative_path + candidate roots.
    """
    db_path = str(result.get("file_path") or "")
    if db_path and Path(db_path).exists():
        return db_path

    relative_path = str(result.get("relative_path") or "").strip()
    if relative_path:
        rel = relative_path.replace("\\", "/").lstrip("/")
        rel_parts = [p for p in rel.split("/") if p]
        if rel_parts:
            rel_path = Path(*rel_parts)
            for root in _candidate_roots():
                cand = root / rel_path
                if cand.exists():
                    return str(cand)

    return db_path


def _get_searcher() -> SqliteVectorSearch:
    """Get or create the singleton searcher (lazy init)."""
    global _searcher
    if _searcher is None:
        _searcher = SqliteVectorSearch()
    return _searcher


# Project root for thumbnail inference
_PROJECT_ROOT = Path(__file__).parent.parent


def format_result(result: dict, skip_fs: bool = False) -> dict:
    """Format a single search result for the frontend.

    Args:
        skip_fs: If True, skip filesystem I/O (Path.exists, thumbnail resolve).
                 Used in server mode where clients use API URLs instead of local paths.
    """
    metadata = result.get("metadata", {})
    db_path = result.get("file_path", "")

    if skip_fs:
        # Server mode: no filesystem access — clients use /files/{id}/thumbnail API
        resolved_path = db_path
        path_exists = True  # assume accessible via server
        thumb_path = result.get("thumbnail_url", "")
    else:
        # Electron local mode: resolve paths on disk
        resolved_path = _resolve_local_path(result)
        path_exists = bool(resolved_path and Path(resolved_path).exists())
        thumb_path = resolve_thumbnail_str(result, _PROJECT_ROOT)

    # Lightweight result for search grid — no heavy metadata/layer_tree.
    # Full metadata is loaded on demand via getFileDetail().
    formatted = {
        "id": result.get("id"),
        "path": resolved_path or db_path,
        "db_path": db_path,
        "resolved_path": resolved_path,
        "path_exists": path_exists,
        "path_mapped": bool(not skip_fs and db_path and resolved_path and db_path != resolved_path),
        "folder_path": result.get("folder_path", ""),
        "relative_path": result.get("relative_path", ""),
        "storage_root": result.get("storage_root", ""),
        "vector_score": result.get("vector_score", result.get("similarity")),
        "text_vec_score": result.get("text_vec_score", result.get("text_similarity")),
        "text_score": result.get("text_score"),
        "combined_score": result.get("rrf_score", result.get("similarity", 0)),
        "thumbnail_path": thumb_path,
        "format": result.get("format", ""),
        "width": result.get("width", 0),
        "height": result.get("height", 0),
        "layer_count": metadata.get("layer_count", 0),
        "mc_caption": result.get("mc_caption", ""),
        "ai_tags": result.get("ai_tags", []),
        "spatial_objects": result.get("spatial_objects", []),
        "user_note": result.get("user_note", ""),
        "user_tags": result.get("user_tags", []),
        "user_category": result.get("user_category", ""),
        "user_rating": result.get("user_rating", 0),
        "image_type": result.get("image_type"),
        "art_style": result.get("art_style"),
    }

    return formatted


def _search_progress_callback(stage: str):
    """Emit search progress event via stdout (Electron IPC)."""
    _write_response({"event": "search_progress", "stage": stage})


def search(query: str = "", limit: int = 20, mode: str = "triaxis", filters: dict = None, threshold: float = 0.0, diagnostic: bool = False, query_image: str = None, query_images: list = None, image_search_mode: str = "and", query_file_id: int = None, use_codex: bool = True, file_ids: list = None, emit_progress: bool = False):
    """Search SQLite and return JSON results."""
    t_start = time.time()
    progress_cb = _search_progress_callback if emit_progress else None
    try:
        searcher = _get_searcher()
        # Always request diagnostic to extract scope info for frontend
        result_data = searcher.search(
            query, mode=mode, filters=filters, top_k=limit,
            threshold=threshold, return_diagnostic=True,
            query_image=query_image,
            query_images=query_images,
            image_search_mode=image_search_mode,
            query_file_id=query_file_id,
            use_codex=use_codex,
            file_ids=set(file_ids) if file_ids else None,
            progress_callback=progress_cb,
        )

        if isinstance(result_data, tuple):
            results, diag = result_data
        else:
            results = result_data
            diag = None

        t_fmt = time.time()
        formatted = [format_result(r) for r in results]
        fmt_ms = round((time.time() - t_fmt) * 1000, 1)
        response = {"success": True, "results": formatted, "count": len(formatted), "format_ms": fmt_ms}

        # Always include lightweight scope info for frontend display
        if diag:
            decomp = diag.get("decomposition", {})
            scope_filter = diag.get("scope_filter", {})
            response["scope"] = {
                "folder": decomp.get("scope", {}).get("folder"),
                "image_type": decomp.get("scope", {}).get("image_type"),
                "format": decomp.get("scope", {}).get("format"),
                "file_count": scope_filter.get("file_count"),
                "find_description": decomp.get("find_description"),
                "query_type": decomp.get("query_type"),
                "decomp_backend": diag.get("decomp_backend"),
                "decomposition_ms": diag.get("decomposition_ms"),
                "scope_filter_ms": diag.get("scope_filter_ms"),
                "vector_ms": diag.get("vector_ms"),
                "text_vec_ms": diag.get("text_vec_ms"),
                "fts_ms": diag.get("fts_ms"),
                "rrf_merge_ms": diag.get("rrf_merge_ms"),
                "negative_filter_ms": diag.get("negative_filter_ms"),
                "user_filter_ms": diag.get("user_filter_ms"),
                "rerank_ms": diag.get("rerank_ms"),
                "enrich_ms": diag.get("enrich_ms"),
                "total_ms": diag.get("total_ms"),
            }

        # Full diagnostic only when explicitly requested
        if diagnostic and diag is not None:
            response["diagnostic"] = diag

        # Log search request
        elapsed_ms = int((time.time() - t_start) * 1000)
        query_text = query or (f"[file_id:{query_file_id}]" if query_file_id else "")
        if query_images:
            query_text = query_text or f"[image_search:{len(query_images)} images]"
        elif query_image:
            query_text = query_text or "[image_search]"
        _log_search(query_text, mode, len(formatted), elapsed_ms,
                    username='local', filters=filters, threshold=threshold)

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
        # Load VV (SigLIP2) and MV (Qwen3-Embedding) models without running a full search.
        # Avoids Codex CLI call (~20s) that was wasted on a dummy query.
        try:
            searcher.encode_text("warmup")  # Load SigLIP2 text encoder
        except Exception:
            pass
        try:
            if searcher.text_search_enabled and searcher.text_provider:
                searcher.text_provider.encode("warmup", is_query=True)  # Load Qwen3-Embedding
        except Exception:
            pass
        return {"status": "ready", "warmup_ms": int((time.time() - t0) * 1000)}

    if cmd == "quit":
        return {"status": "bye"}

    if cmd == "phase_status":
        file_paths = data.get("file_paths", [])
        if not file_paths:
            return {"success": True, "status": {}}
        searcher = _get_searcher()
        status = searcher.db.get_files_phase_status(file_paths[:500])
        return {"success": True, "status": status}

    if cmd == "fix_relative_paths":
        searcher = _get_searcher()
        fixed = searcher.db.fix_missing_relative_paths()
        return {"success": True, "fixed": fixed}

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
        query_file_id=data.get("query_file_id"),
        use_codex=data.get("use_codex", True),
        file_ids=data.get("file_ids"),
        emit_progress=True,
    )


def _write_response(response: dict):
    """Write a single JSON response line to stdout."""
    t0 = time.time()
    json_str = json.dumps(response, ensure_ascii=False)
    json_ms = round((time.time() - t0) * 1000, 1)
    json_bytes = len(json_str.encode('utf-8'))
    # Inject serialization timing (after JSON is built, so append manually)
    if json_bytes > 1000:
        logger.warning(f"[TIMING] JSON serialize: {json_ms}ms, {json_bytes} bytes ({json_bytes/1024:.0f}KB)")
    sys.stdout.write(json_str + "\n")
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
