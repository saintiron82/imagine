#!/usr/bin/env python3
"""Run a fixed Search Evaluation V1 QuerySet against Imagine search engines.

This is the integration layer for objective search benchmarking:

QuerySet + optional LabelSet -> RunResult JSONL -> evaluation -> optional gate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.compare_search_evaluation import (  # noqa: E402
    DEFAULT_METRICS,
    build_markdown as build_compare_markdown,
    compare_summaries,
    load_summary,
    parse_csv,
)
from tools.evaluate_search_quality import (  # noqa: E402
    DEFAULT_K,
    build_markdown as build_eval_markdown,
    evaluate_all,
    load_labels,
    load_queries,
    load_run,
    read_jsonl,
    parse_k_values,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmarks" / "runs"
VALID_ENGINES = ("vv", "mv", "fts", "triaxis")
SCORE_FIELDS = (
    "final_score",
    "combined_score",
    "score",
    "similarity",
    "rrf_score",
    "bm25_score",
    "fts_rank",
)

FTS_QUERY_STOPWORDS = {
    "이미지",
    "사진",
    "파일",
    "자료",
    "폴더",
    "프로젝트",
    "배경",
    "장면",
    "조건",
    "제외",
    "있는",
    "있다",
    "있고",
    "있음",
    "보이는",
    "보이고",
    "보이며",
    "보이지만",
    "함께",
    "같이",
    "느낌",
    "분위기",
    "중심",
    "중",
    "중에",
    "중에서",
    "에서",
    "그리고",
    "또는",
    "찾기",
    "찾아줘",
    "image",
    "images",
    "picture",
    "pictures",
    "photo",
    "photos",
    "file",
    "files",
    "with",
    "and",
    "or",
    "in",
    "from",
}
FTS_QUERY_SPLIT_RE = re.compile(r"[\s,;/]+")
FTS_KO_PARTICLE_SUFFIX_RE = re.compile(
    r"(?:중에서|중에|에서|에게서|에게|한테|으로|부터|까지|보다|처럼|만큼|대로|마다|"
    r"이랑|이나|하고|과|와|에|의|은|는|이|가|을|를|도|만|나)$"
)
FTS_TOKEN_STRIP_CHARS = " \t\r\n\"'`“”‘’[](){}"
FTS_PARTICLE_SUFFIX_EXCEPTIONS = {"마을"}
SCOPED_QUERY_PREFIX_RE = re.compile(r"^\s*(?P<scope>.+?)(?:중에서|중에|에서)\s*(?P<body>.+)$")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    status = run_git(["status", "--porcelain"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
    }


def parse_engines(raw: str) -> list[str]:
    engines = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [engine for engine in engines if engine not in VALID_ENGINES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid engine(s): {', '.join(invalid)}; expected one of: {', '.join(VALID_ENGINES)}"
        )
    return engines


def make_searcher():
    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch

    db = SQLiteDB()
    return SqliteVectorSearch(db=db)


def _normalize_fts_keyword(raw: str) -> str:
    token = raw.strip(FTS_TOKEN_STRIP_CHARS)
    if not token:
        return ""
    if token.lower() in FTS_QUERY_STOPWORDS:
        return ""
    if token not in FTS_PARTICLE_SUFFIX_EXCEPTIONS:
        token = FTS_KO_PARTICLE_SUFFIX_RE.sub("", token)
    token = token.strip(FTS_TOKEN_STRIP_CHARS)
    if len(token) <= 1 and token.isascii():
        return ""
    if token.lower() in FTS_QUERY_STOPWORDS:
        return ""
    return token


def fts_keywords(query_text: str) -> list[str]:
    """Extract meaningful benchmark FTS terms without generic UI/search words."""
    keywords: list[str] = []
    seen: set[str] = set()
    for raw in FTS_QUERY_SPLIT_RE.split(query_text):
        keyword = _normalize_fts_keyword(raw)
        if not keyword:
            continue
        key = keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(keyword)
    return keywords


def benchmark_search_text(query_text: str, query_meta: dict[str, Any]) -> str:
    """Return the semantic condition text used by benchmark engines.

    QuerySet `scope` is already applied as a file-id filter. Keeping the scope
    wording inside the search text double-counts folder names and lets FTS rank
    files by path/folder tokens instead of the visual condition.
    """
    must_terms = [
        str(term).strip()
        for term in query_meta.get("must_terms", [])
        if str(term).strip()
    ]
    soft_terms = [
        str(term).strip()
        for term in query_meta.get("soft_terms", [])
        if str(term).strip()
    ]
    exclude_terms = [
        str(term).strip()
        for term in query_meta.get("exclude_terms", [])
        if str(term).strip()
    ]
    if must_terms:
        parts = [", ".join([*must_terms, *soft_terms])]
        if exclude_terms:
            parts.append("제외: " + ", ".join(exclude_terms))
        return ". ".join(parts)

    if not query_scope(query_meta):
        return query_text

    match = SCOPED_QUERY_PREFIX_RE.match(query_text)
    if match:
        return match.group("body").strip() or query_text
    return query_text


def search_engine(searcher: Any, engine_id: str, query_text: str, top_k: int) -> list[dict[str, Any]]:
    if engine_id == "vv":
        return list(searcher.vector_search(query_text, top_k=top_k, threshold=0.0))
    if engine_id == "mv":
        return list(searcher.text_vector_search(query_text, top_k=top_k, threshold=0.0))
    if engine_id == "fts":
        keywords = fts_keywords(query_text)
        return list(searcher.fts_search(keywords[:10], top_k=top_k)) if keywords else []
    if engine_id == "triaxis":
        return list(searcher.triaxis_search(query_text, top_k=top_k, threshold=0.0, use_codex=False))
    raise ValueError(f"unsupported engine_id: {engine_id}")


def query_metadata_by_id(queries_path: Path) -> dict[str, dict[str, Any]]:
    """Load optional QuerySet fields that the core evaluator ignores."""
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(queries_path):
        query_id = str(row.get("query_id") or "")
        if query_id:
            metadata[query_id] = row
    return metadata


def query_scope(query_meta: dict[str, Any]) -> str:
    """Return the explicit folder scope encoded in a QuerySet row."""
    for key in ("scope", "folder_scope", "folder"):
        value = str(query_meta.get(key) or "").strip()
        if value:
            return value
    return ""


def resolve_scope_file_ids(searcher: Any, scope: str) -> tuple[set[Any] | None, dict[str, Any]]:
    """Resolve an explicit QuerySet folder scope to file ids.

    Scoped benchmark queries must not silently fall back to full-DB search.
    If a QuerySet carries `scope`, every engine run uses this file-id set.
    """
    if not scope:
        return None, {}
    if not hasattr(searcher, "_apply_plan_filter_with_info"):
        raise ValueError("searcher does not support explicit scope filtering")

    file_ids, match_info = searcher._apply_plan_filter_with_info({"folder": scope})
    return set(file_ids), dict(match_info or {})


def scoped_search_engine(
    searcher: Any,
    engine_id: str,
    query_text: str,
    top_k: int,
    *,
    file_ids: set[Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one engine, constraining candidate files when scope is provided."""
    if file_ids is None:
        return search_engine(searcher, engine_id, query_text, top_k)
    if not file_ids:
        return []

    if engine_id == "vv":
        return list(searcher._vv_search_within(query_text, file_ids, top_k=top_k, threshold=0.0))
    if engine_id == "mv":
        return list(searcher._mv_search_within(query_text, file_ids, top_k=top_k, threshold=0.0))
    if engine_id == "fts":
        keywords = fts_keywords(query_text)
        return list(searcher.fts_search(keywords[:10], top_k=top_k, file_ids=file_ids)) if keywords else []
    if engine_id == "triaxis":
        return list(searcher.triaxis_search(
            query_text,
            top_k=top_k,
            threshold=0.0,
            use_codex=False,
            file_ids=file_ids,
        ))
    raise ValueError(f"unsupported engine_id: {engine_id}")


def result_item_id(result: Any) -> str:
    if isinstance(result, dict):
        if "id" in result:
            return str(result["id"])
        if "file_id" in result:
            return str(result["file_id"])
    return str(result)


def result_score(result: Any) -> float:
    if not isinstance(result, dict):
        return 0.0
    for field in SCORE_FIELDS:
        value = result.get(field)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def build_run_rows(
    *,
    queries_path: Path,
    engines: list[str],
    top_k: int,
    run_id: str,
    searcher: Any,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries = load_queries(queries_path)
    query_meta = query_metadata_by_id(queries_path)
    scope_cache: dict[str, tuple[set[Any] | None, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for engine_id in engines:
        for index, query in enumerate(queries.values(), start=1):
            if progress:
                progress(f"{engine_id}: {index}/{len(queries)} {query.query_id}")

            t0 = time.perf_counter()
            error = None
            results: list[Any] = []
            meta = query_meta.get(query.query_id, {})
            scope = query_scope(meta)
            search_text = benchmark_search_text(query.query_text, meta)
            scope_file_ids: set[Any] | None = None
            scope_info: dict[str, Any] = {}
            try:
                if scope:
                    if query.query_id not in scope_cache:
                        scope_cache[query.query_id] = resolve_scope_file_ids(searcher, scope)
                    scope_file_ids, scope_info = scope_cache[query.query_id]
                    if not scope_file_ids:
                        results = []
                    else:
                        results = scoped_search_engine(
                            searcher,
                            engine_id,
                            search_text,
                            top_k,
                            file_ids=scope_file_ids,
                        )
                else:
                    results = scoped_search_engine(
                        searcher,
                        engine_id,
                        search_text,
                        top_k,
                        file_ids=None,
                    )
            except Exception as exc:
                error = str(exc)
            latency_ms = round((time.perf_counter() - t0) * 1000)

            event = {
                "run_id": run_id,
                "engine_id": engine_id,
                "query_id": query.query_id,
                "latency_ms": latency_ms,
                "result_count": len(results),
                "error": error,
            }
            if scope:
                event.update({
                    "scope": scope,
                    "search_text": search_text,
                    "scope_file_count": len(scope_file_ids or []),
                    "scope_match_mode": scope_info.get("match_mode"),
                    "scope_applied_folder": scope_info.get("applied_folder"),
                })
            events.append(event)

            if error:
                continue

            seen: set[str] = set()
            rank = 1
            for result in results:
                item_id = result_item_id(result)
                if item_id in seen:
                    continue
                seen.add(item_id)
                rows.append({
                    "run_id": run_id,
                    "engine_id": engine_id,
                    "query_id": query.query_id,
                    "rank": rank,
                    "item_id": item_id,
                    "score": result_score(result),
                    "latency_ms": latency_ms if rank == 1 else None,
                    "error": None,
                    "cost_usd": None,
                    "scope": scope or None,
                    "search_text": search_text if scope else None,
                    "scope_file_count": len(scope_file_ids or []) if scope else None,
                })
                rank += 1
                if rank > top_k:
                    break

    return rows, events


def run_search_benchmark(
    *,
    queries_path: Path,
    labels_path: Path | None,
    output_dir: Path,
    engines: list[str],
    top_k: int,
    k_values: tuple[int, ...],
    run_id: str,
    baseline_path: Path | None = None,
    metrics: list[str] | None = None,
    min_delta: float = -0.01,
    min_query_ratio: float = 1.0,
    max_latency_ratio: float | None = None,
    searcher: Any | None = None,
    quiet: bool = False,
) -> dict[str, Path | None]:
    if searcher is None:
        searcher = make_searcher()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run_results.jsonl"
    events_path = output_dir / "query_events.jsonl"
    manifest_path = output_dir / "manifest.json"
    evaluation_json_path = output_dir / "evaluation.json"
    evaluation_md_path = output_dir / "evaluation.md"
    compare_json_path = output_dir / "compare.json"
    compare_md_path = output_dir / "compare.md"

    def progress(message: str) -> None:
        if not quiet:
            print(f"  {message}")

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rows, events = build_run_rows(
        queries_path=queries_path,
        engines=engines,
        top_k=top_k,
        run_id=run_id,
        searcher=searcher,
        progress=progress,
    )
    completed_at = datetime.now().astimezone().isoformat(timespec="seconds")

    write_jsonl(run_path, rows)
    write_jsonl(events_path, events)

    evaluation_summary = None
    compare_report = None
    if labels_path:
        evaluation_summary = evaluate_all(
            labels=load_labels(labels_path),
            run_groups=load_run(run_path),
            queries=load_queries(queries_path),
            k_values=k_values,
        )
        evaluation_json_path.write_text(
            json.dumps(evaluation_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        evaluation_md_path.write_text(build_eval_markdown(evaluation_summary), encoding="utf-8")

    if baseline_path:
        if evaluation_summary is None:
            raise ValueError("--baseline requires --labels so candidate evaluation can be computed")
        compare_report = compare_summaries(
            baseline=load_summary(baseline_path),
            candidate=evaluation_summary,
            metrics=metrics or list(DEFAULT_METRICS),
            engines=engines,
            min_delta=min_delta,
            min_query_ratio=min_query_ratio,
            max_latency_ratio=max_latency_ratio,
        )
        compare_json_path.write_text(
            json.dumps(compare_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        compare_md_path.write_text(build_compare_markdown(compare_report), encoding="utf-8")

    manifest = {
        "schema_version": "search_benchmark_run_v1",
        "created_at": started_at,
        "completed_at": completed_at,
        "run_id": run_id,
        "git": git_metadata(),
        "inputs": {
            "queries": str(queries_path),
            "labels": str(labels_path) if labels_path else None,
            "baseline": str(baseline_path) if baseline_path else None,
        },
        "config": {
            "engines": engines,
            "top_k": top_k,
            "k_values": list(k_values),
            "metrics": metrics or list(DEFAULT_METRICS),
            "min_delta": min_delta,
            "min_query_ratio": min_query_ratio,
            "max_latency_ratio": max_latency_ratio,
        },
        "counts": {
            "run_result_rows": len(rows),
            "query_events": len(events),
            "errors": sum(1 for event in events if event["error"]),
            "empty_results": sum(1 for event in events if not event["error"] and event["result_count"] == 0),
        },
        "files": {
            "run_results": run_path.name,
            "query_events": events_path.name,
            "evaluation_json": evaluation_json_path.name if evaluation_summary else None,
            "evaluation_md": evaluation_md_path.name if evaluation_summary else None,
            "compare_json": compare_json_path.name if compare_report else None,
            "compare_md": compare_md_path.name if compare_report else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_results": run_path,
        "query_events": events_path,
        "manifest": manifest_path,
        "evaluation_json": evaluation_json_path if evaluation_summary else None,
        "evaluation_md": evaluation_md_path if evaluation_summary else None,
        "compare_json": compare_json_path if compare_report else None,
        "compare_md": compare_md_path if compare_report else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True, help="QuerySet JSONL path")
    parser.add_argument("--labels", type=Path, help="optional LabelSet JSONL path")
    parser.add_argument("--engines", type=parse_engines, default=list(VALID_ENGINES),
                        help="comma-separated engines: vv,mv,fts,triaxis")
    parser.add_argument("--top-k", type=int, default=50, help="results per query per engine")
    parser.add_argument("--k", type=parse_k_values, default=DEFAULT_K, help="evaluation k values")
    parser.add_argument("--run-id", help="stable run id; defaults to timestamp")
    parser.add_argument("--output-dir", type=Path, help="output directory")
    parser.add_argument("--baseline", type=Path, help="optional baseline evaluation JSON")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="gate metrics")
    parser.add_argument("--min-delta", type=float, default=-0.01)
    parser.add_argument("--min-query-ratio", type=float, default=1.0)
    parser.add_argument("--max-latency-ratio", type=float)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.top_k < 1:
        print("ERROR: --top-k must be >= 1", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"search_{timestamp}"
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)

    try:
        paths = run_search_benchmark(
            queries_path=args.queries,
            labels_path=args.labels,
            output_dir=output_dir,
            engines=args.engines,
            top_k=args.top_k,
            k_values=args.k,
            run_id=run_id,
            baseline_path=args.baseline,
            metrics=parse_csv(args.metrics),
            min_delta=args.min_delta,
            min_query_ratio=args.min_query_ratio,
            max_latency_ratio=args.max_latency_ratio,
            quiet=args.quiet,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"  RunResult: {paths['run_results']}")
        if paths["evaluation_md"]:
            print(f"  Evaluation: {paths['evaluation_md']}")
        if paths["compare_md"]:
            print(f"  Compare: {paths['compare_md']}")

    if paths["compare_json"]:
        report = json.loads(Path(paths["compare_json"]).read_text(encoding="utf-8"))
        return 0 if report["status"] == "pass" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
