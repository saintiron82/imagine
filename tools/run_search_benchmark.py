#!/usr/bin/env python3
"""Run a fixed Search Evaluation V1 QuerySet against Imagine search engines.

This is the integration layer for objective search benchmarking:

QuerySet + optional LabelSet -> RunResult JSONL -> evaluation -> optional gate.
"""

from __future__ import annotations

import argparse
import json
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


def fts_keywords(query_text: str) -> list[str]:
    return [word.strip() for word in query_text.replace(",", " ").split() if len(word.strip()) > 1]


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
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for engine_id in engines:
        for index, query in enumerate(queries.values(), start=1):
            if progress:
                progress(f"{engine_id}: {index}/{len(queries)} {query.query_id}")

            t0 = time.perf_counter()
            error = None
            results: list[Any] = []
            try:
                results = search_engine(searcher, engine_id, query.query_text, top_k)
            except Exception as exc:
                error = str(exc)
            latency_ms = round((time.perf_counter() - t0) * 1000)

            events.append({
                "run_id": run_id,
                "engine_id": engine_id,
                "query_id": query.query_id,
                "latency_ms": latency_ms,
                "result_count": len(results),
                "error": error,
            })

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
