"""
Benchmark engine runner.
Runs a single engine against a query set and records results.
Each run is independent and saved as a timestamped JSON file.
"""

import json
import random
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class EngineAdapter(ABC):
    """Base class for benchmark engine adapters."""

    engine_id: str = ""
    description: str = ""

    @abstractmethod
    def search(self, query_text: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Run search and return ranked results.

        Returns list of dicts with at least:
            - item_id: str (file id or path)
            - score: float
            - rank: int (1-based)
        """

    def warmup(self):
        """Optional warmup (model loading etc)."""

    def cleanup(self):
        """Optional cleanup."""


class TriaxisAdapter(EngineAdapter):
    """Triaxis VV+MV+FTS full system."""

    engine_id = "triaxis"
    description = "Triaxis VV+MV+FTS (full system)"

    def __init__(self, rrf_override: Optional[dict] = None):
        self._searcher = None
        self._rrf_override = rrf_override
        self._original_preset = None

    def _get_searcher(self):
        if self._searcher is None:
            from backend.search.sqlite_search import SqliteVectorSearch
            self._searcher = SqliteVectorSearch()
        return self._searcher

    def warmup(self):
        self._get_searcher()

    def search(self, query_text: str, top_k: int = 50) -> List[Dict[str, Any]]:
        searcher = self._get_searcher()

        # Temporarily override RRF weights if specified
        if self._rrf_override:
            self._apply_rrf_override()

        try:
            result = searcher.triaxis_search(
                query=query_text,
                top_k=top_k,
                threshold=0.0,
                return_diagnostic=True,
            )
            if isinstance(result, tuple):
                results, diagnostic = result
            else:
                results, diagnostic = result, {}
        finally:
            if self._rrf_override:
                self._restore_rrf()

        ranked = []
        for i, r in enumerate(results):
            ranked.append({
                "item_id": str(r.get("id", r.get("file_path", ""))),
                "score": r.get("rrf_score", r.get("similarity", 0.0)),
                "rank": i + 1,
                "file_path": r.get("file_path", ""),
                "vector_score": r.get("similarity"),
                "text_vec_score": r.get("text_similarity"),
                "text_score": r.get("text_score"),
            })
        return ranked

    def _apply_rrf_override(self):
        from backend.utils.config import get_config
        cfg = get_config()
        presets = cfg.get("search.rrf.presets", {})
        self._original_preset = cfg.get("search.rrf.preset", "balanced")
        # Inject temporary preset
        presets["_bench_override"] = self._rrf_override
        cfg.set("search.rrf.presets._bench_override", self._rrf_override)
        cfg.set("search.rrf.preset", "_bench_override")

    def _restore_rrf(self):
        if self._original_preset:
            from backend.utils.config import get_config
            cfg = get_config()
            cfg.set("search.rrf.preset", self._original_preset)


class FTSOnlyAdapter(EngineAdapter):
    """FTS5 BM25 keyword search only."""

    engine_id = "fts_only"
    description = "FTS5 BM25 keyword only"

    def __init__(self):
        self._searcher = None

    def _get_searcher(self):
        if self._searcher is None:
            from backend.search.sqlite_search import SqliteVectorSearch
            self._searcher = SqliteVectorSearch()
        return self._searcher

    def warmup(self):
        self._get_searcher()

    def search(self, query_text: str, top_k: int = 50) -> List[Dict[str, Any]]:
        searcher = self._get_searcher()
        # Split query into keywords
        keywords = query_text.split()
        results = searcher.fts_search(keywords=keywords, top_k=top_k)
        ranked = []
        for i, r in enumerate(results):
            ranked.append({
                "item_id": str(r.get("id", r.get("file_path", ""))),
                "score": r.get("fts_rank", 0.0),
                "rank": i + 1,
                "file_path": r.get("file_path", ""),
            })
        return ranked


class RandomAdapter(EngineAdapter):
    """Random baseline (chance level)."""

    engine_id = "random"
    description = "Random baseline"

    def __init__(self):
        self._db = None

    def _get_db(self):
        if self._db is None:
            from backend.db.sqlite_client import SQLiteDB
            self._db = SQLiteDB()
        return self._db

    def search(self, query_text: str, top_k: int = 50) -> List[Dict[str, Any]]:
        db = self._get_db()
        rows = db.execute(
            "SELECT id, file_path FROM files ORDER BY RANDOM() LIMIT ?",
            (top_k,),
        )
        ranked = []
        for i, row in enumerate(rows):
            ranked.append({
                "item_id": str(row[0]),
                "score": 1.0 / (i + 1),  # Decreasing pseudo-score
                "rank": i + 1,
                "file_path": row[1],
            })
        return ranked


# --- Engine registry ---

def get_engine(engine_id: str, config: Optional[dict] = None) -> EngineAdapter:
    """Create engine adapter by ID."""
    config = config or {}

    if engine_id == "triaxis":
        return TriaxisAdapter(rrf_override=config.get("rrf_override"))
    elif engine_id == "vv_only":
        return TriaxisAdapter(rrf_override={"v": 1.0, "s": 0.0, "m": 0.0})
    elif engine_id == "mv_only":
        return TriaxisAdapter(rrf_override={"v": 0.0, "s": 1.0, "m": 0.0})
    elif engine_id == "fts_only":
        return FTSOnlyAdapter()
    elif engine_id == "random":
        return RandomAdapter()
    else:
        raise ValueError(f"Unknown engine: {engine_id}")


# --- Run execution ---

def load_queries(path: str) -> List[dict]:
    """Load query set from JSONL file."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_labels(path: str) -> Dict[str, Dict[str, int]]:
    """Load labels from JSONL. Returns {query_id: {item_id: relevance}}."""
    labels = {}
    p = Path(path)
    if not p.exists():
        return labels
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["query_id"]
            if qid not in labels:
                labels[qid] = {}
            labels[qid][row["item_id"]] = row["relevance"]
    return labels


def execute_run(
    engine: EngineAdapter,
    queries: List[dict],
    top_k: int = 50,
    tag: str = "",
) -> dict:
    """Execute benchmark run for a single engine.

    Returns a run record dict with metadata and per-query results.
    """
    run_id = f"{engine.engine_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if tag:
        run_id = f"{tag}_{run_id}"

    engine.warmup()

    query_results = []
    errors = 0

    for q in queries:
        qid = q["query_id"]
        query_text = q["query_text"]

        t0 = time.perf_counter()
        try:
            results = engine.search(query_text, top_k=top_k)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            error = None
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            results = []
            error = str(e)
            errors += 1

        query_results.append({
            "query_id": qid,
            "query_text": query_text,
            "ranked_ids": [r["item_id"] for r in results],
            "scores": [r["score"] for r in results],
            "latency_ms": latency_ms,
            "result_count": len(results),
            "error": error,
        })

    engine.cleanup()

    return {
        "run_id": run_id,
        "engine_id": engine.engine_id,
        "engine_desc": engine.description,
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "n_queries": len(queries),
        "n_errors": errors,
        "top_k": top_k,
        "queries": query_results,
    }


def save_run(run: dict, output_dir: str = "benchmarks/runs") -> str:
    """Save run result to JSON file. Returns file path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{run['run_id']}.json"
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    return str(path)


def load_run(path: str) -> dict:
    """Load a saved run from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_runs(output_dir: str = "benchmarks/runs") -> List[dict]:
    """List all saved runs with summary info."""
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return []
    runs = []
    for p in sorted(out_dir.glob("*.json"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            runs.append({
                "file": str(p),
                "run_id": data.get("run_id", ""),
                "engine_id": data.get("engine_id", ""),
                "tag": data.get("tag", ""),
                "timestamp": data.get("timestamp", ""),
                "n_queries": data.get("n_queries", 0),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return runs
