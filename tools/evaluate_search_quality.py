#!/usr/bin/env python3
"""Evaluate Imagine search run results against relevance labels.

This tool intentionally has no third-party dependencies. It is the first
standard evaluation layer for QuerySet, LabelSet, and RunResult JSONL files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_K = (5, 10, 50)
VALID_QUERY_TYPES = {"exact", "semantic", "scoped", "complex", "ambiguous"}
VALID_LABEL_SOURCES = {"weak", "human", "adjudicated"}


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query_text: str
    query_type: str
    locale: str
    created_at: str


@dataclass(frozen=True)
class LabelRecord:
    query_id: str
    item_id: str
    relevance: int
    label_source: str
    label_version: str


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    engine_id: str
    query_id: str
    rank: int
    item_id: str
    score: float
    latency_ms: int | None
    error: str | None
    cost_usd: float | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def require_fields(row: dict[str, Any], fields: Iterable[str], path: Path, index: int) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path}:{index}: missing required field(s): {joined}")


def load_queries(path: Path | None) -> dict[str, QueryRecord]:
    if path is None:
        return {}
    queries: dict[str, QueryRecord] = {}
    for idx, row in enumerate(read_jsonl(path), 1):
        require_fields(row, ("query_id", "query_text", "query_type", "locale", "created_at"), path, idx)
        query_type = str(row["query_type"])
        if query_type not in VALID_QUERY_TYPES:
            raise ValueError(f"{path}:{idx}: invalid query_type: {query_type}")
        record = QueryRecord(
            query_id=str(row["query_id"]),
            query_text=str(row["query_text"]),
            query_type=query_type,
            locale=str(row["locale"]),
            created_at=str(row["created_at"]),
        )
        if record.query_id in queries:
            raise ValueError(f"{path}:{idx}: duplicate query_id: {record.query_id}")
        queries[record.query_id] = record
    return queries


def load_labels(path: Path) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    for idx, row in enumerate(read_jsonl(path), 1):
        require_fields(row, ("query_id", "item_id", "relevance", "label_source", "label_version"), path, idx)
        relevance = int(row["relevance"])
        if relevance not in (0, 1, 2):
            raise ValueError(f"{path}:{idx}: relevance must be 0, 1, or 2")
        source = str(row["label_source"])
        if source not in VALID_LABEL_SOURCES:
            raise ValueError(f"{path}:{idx}: invalid label_source: {source}")
        record = LabelRecord(
            query_id=str(row["query_id"]),
            item_id=str(row["item_id"]),
            relevance=relevance,
            label_source=source,
            label_version=str(row["label_version"]),
        )
        labels[record.query_id][record.item_id] = record.relevance
    return dict(labels)


def _nullable_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _nullable_float(value: Any) -> float | None:
    return None if value is None else float(value)


def load_run(path: Path) -> dict[tuple[str, str], dict[str, list[RunRecord]]]:
    groups: dict[tuple[str, str], dict[str, list[RunRecord]]] = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(read_jsonl(path), 1):
        require_fields(
            row,
            ("run_id", "engine_id", "query_id", "rank", "item_id", "score", "latency_ms", "error", "cost_usd"),
            path,
            idx,
        )
        record = RunRecord(
            run_id=str(row["run_id"]),
            engine_id=str(row["engine_id"]),
            query_id=str(row["query_id"]),
            rank=int(row["rank"]),
            item_id=str(row["item_id"]),
            score=float(row["score"]),
            latency_ms=_nullable_int(row["latency_ms"]),
            error=None if row["error"] is None else str(row["error"]),
            cost_usd=_nullable_float(row["cost_usd"]),
        )
        if record.rank < 1:
            raise ValueError(f"{path}:{idx}: rank must be >= 1")
        groups[(record.run_id, record.engine_id)][record.query_id].append(record)

    for query_runs in groups.values():
        for records in query_runs.values():
            records.sort(key=lambda r: (r.rank, -r.score))
    return {key: dict(value) for key, value in groups.items()}


def dcg(relevances: list[int], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        gain = (2**rel) - 1
        total += gain / math.log2(i + 1)
    return total


def evaluate_group(
    query_runs: dict[str, list[RunRecord]],
    labels: dict[str, dict[str, int]],
    k_values: tuple[int, ...],
    query_filter: set[str] | None = None,
) -> dict[str, Any]:
    query_ids = sorted(query_filter if query_filter is not None else labels.keys())
    query_ids = [qid for qid in query_ids if qid in labels]
    if not query_ids:
        return {"query_count": 0, "metrics": {}}

    metric_sums: dict[str, float] = defaultdict(float)
    evaluated = 0

    for query_id in query_ids:
        rels = labels.get(query_id, {})
        relevant_total = sum(1 for rel in rels.values() if rel > 0)
        if relevant_total == 0:
            continue

        ranked = query_runs.get(query_id, [])
        ranked_relevance = [rels.get(record.item_id, 0) for record in ranked]
        ideal_relevance = sorted(rels.values(), reverse=True)
        evaluated += 1

        for k in k_values:
            top_relevance = ranked_relevance[:k]
            relevant_hits = sum(1 for rel in top_relevance if rel > 0)
            metric_sums[f"P@{k}"] += relevant_hits / k
            metric_sums[f"Recall@{k}"] += relevant_hits / relevant_total

            first_rank = next((i for i, rel in enumerate(top_relevance, start=1) if rel > 0), None)
            metric_sums[f"MRR@{k}"] += (1.0 / first_rank) if first_rank else 0.0

            ideal = dcg(ideal_relevance, k)
            actual = dcg(ranked_relevance, k)
            metric_sums[f"nDCG@{k}"] += (actual / ideal) if ideal > 0 else 0.0

    if evaluated == 0:
        return {"query_count": 0, "metrics": {}}

    return {
        "query_count": evaluated,
        "metrics": {
            name: round(value / evaluated, 6)
            for name, value in sorted(metric_sums.items())
        },
    }


def evaluate_all(
    labels: dict[str, dict[str, int]],
    run_groups: dict[tuple[str, str], dict[str, list[RunRecord]]],
    queries: dict[str, QueryRecord],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    runs = []
    for (run_id, engine_id), query_runs in sorted(run_groups.items()):
        summary = evaluate_group(query_runs, labels, k_values)
        by_type = {}
        if queries:
            for query_type in sorted({query.query_type for query in queries.values()}):
                ids = {query_id for query_id, query in queries.items() if query.query_type == query_type}
                typed_summary = evaluate_group(query_runs, labels, k_values, query_filter=ids)
                if typed_summary["query_count"] > 0:
                    by_type[query_type] = typed_summary

        latency_values = [
            record.latency_ms
            for records in query_runs.values()
            for record in records[:1]
            if record.latency_ms is not None
        ]
        avg_latency = round(sum(latency_values) / len(latency_values), 3) if latency_values else None

        runs.append({
            "run_id": run_id,
            "engine_id": engine_id,
            "query_count": summary["query_count"],
            "metrics": summary["metrics"],
            "by_query_type": by_type,
            "avg_latency_ms": avg_latency,
        })

    return {
        "schema_version": "search_evaluation_v1",
        "k_values": list(k_values),
        "label_query_count": len(labels),
        "run_count": len(runs),
        "runs": runs,
    }


def build_markdown(summary: dict[str, Any]) -> str:
    metric_names = summary["runs"][0]["metrics"].keys() if summary["runs"] else []
    ndcg_name = _choose_metric(metric_names, "nDCG", preferred=10)
    precision_name = _choose_metric(metric_names, "P", preferred=10)
    recall_name = _choose_metric(metric_names, "Recall", preferred=50)

    lines = [
        "# Search Evaluation Report",
        "",
        f"**Schema**: `{summary['schema_version']}`",
        f"**Label queries**: {summary['label_query_count']}",
        f"**Runs**: {summary['run_count']}",
        "",
        "## Overall",
        "",
    ]

    for run in summary["runs"]:
        lines.extend([
            f"### {run['engine_id']} / {run['run_id']}",
            "",
            f"- Queries evaluated: {run['query_count']}",
            f"- Avg latency: {run['avg_latency_ms']} ms" if run["avg_latency_ms"] is not None else "- Avg latency: n/a",
            "",
            "| Metric | Score |",
            "|---|---:|",
        ])
        for name, value in run["metrics"].items():
            lines.append(f"| `{name}` | {value:.6f} |")
        lines.append("")

        if run["by_query_type"]:
            lines.extend([
                "Query type breakdown:",
                "",
                f"| Type | Queries | {ndcg_name or 'nDCG'} | {precision_name or 'P'} | {recall_name or 'Recall'} |",
                "|---|---:|---:|---:|---:|",
            ])
            for query_type, typed in sorted(run["by_query_type"].items()):
                metrics = typed["metrics"]
                lines.append(
                    f"| `{query_type}` | {typed['query_count']} | "
                    f"{metrics.get(ndcg_name, 0.0) if ndcg_name else 0.0:.6f} | "
                    f"{metrics.get(precision_name, 0.0) if precision_name else 0.0:.6f} | "
                    f"{metrics.get(recall_name, 0.0) if recall_name else 0.0:.6f} |"
                )
            lines.append("")

    return "\n".join(lines)


def _choose_metric(metric_names: Iterable[str], prefix: str, preferred: int) -> str | None:
    names = sorted(metric_names)
    preferred_name = f"{prefix}@{preferred}"
    if preferred_name in names:
        return preferred_name
    candidates = [name for name in names if name.startswith(f"{prefix}@")]
    return candidates[-1] if candidates else None


def parse_k_values(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values or any(k < 1 for k in values):
        raise argparse.ArgumentTypeError("--k must contain positive integers")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="LabelSet JSONL path")
    parser.add_argument("--run", type=Path, required=True, help="RunResult JSONL path")
    parser.add_argument("--queries", type=Path, help="optional QuerySet JSONL path")
    parser.add_argument("--k", type=parse_k_values, default=DEFAULT_K, help="comma-separated k values")
    parser.add_argument("--output-json", type=Path, help="write summary JSON")
    parser.add_argument("--output-md", type=Path, help="write Markdown report")
    args = parser.parse_args(argv)

    try:
        queries = load_queries(args.queries)
        labels = load_labels(args.labels)
        run_groups = load_run(args.run)
        summary = evaluate_all(labels, run_groups, queries, args.k)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(build_markdown(summary), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
