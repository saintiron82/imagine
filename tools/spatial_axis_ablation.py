#!/usr/bin/env python3
"""Spatial-axis ablation runner for the frozen spatial queryset.

The prior ON/OFF bench only disabled the final spatial intent boost. This
runner compares the actual spatial evidence axis by monkeypatching the searcher
at runtime. It does not change production search behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB  # noqa: E402
from backend.search.sqlite_search import (  # noqa: E402
    SqliteVectorSearch,
    _extract_spatial_intent,
)
from tools.generate_spatial_queryset import load_file_ids  # noqa: E402


class ReadOnlySQLiteDB(SQLiteDB):
    """SQLiteDB wrapper for benchmarks that must not mutate the live DB."""

    def _create_connection(self) -> sqlite3.Connection:
        uri = Path(self.db_path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        self._load_vec_extension(conn)
        return conn

    def _connect_setup(self):
        self._setup_conn = self._create_connection()
        self._local.conn = self._setup_conn


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def _row_locations(raw: str | None) -> set[str]:
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return set()
    if not isinstance(parsed, list):
        return set()
    return {str(v) for v in parsed if v}


def _confidence_bonus(confidence: str | None) -> float:
    return {"high": 0.20, "medium": 0.10, "low": 0.0}.get(
        str(confidence or "").lower(),
        0.0,
    )


def _term_hits(hay: str, terms: list[str]) -> int:
    hay_l = str(hay or "").lower()
    return sum(1 for term in terms if str(term or "").lower() in hay_l)


def strict_primary_spatial_search(
    searcher: SqliteVectorSearch,
    intent: dict[str, Any],
    top_k: int = 20,
    file_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Spatial search variant requiring primary_location exact matches.

    This is intentionally narrower than production `_spatial_evidence_search`.
    It tests whether broad `locations` / `spatial_text` matching is diluting the
    location signal.
    """
    if not intent or not intent.get("active"):
        return []

    terms = [str(t).lower() for t in intent.get("terms", []) if str(t).strip()]
    locations = {str(v) for v in (intent.get("locations") or []) if v}
    allowed_ids = set(file_ids or []) if file_ids else None
    matches_by_file: dict[int, dict[str, Any]] = {}

    def allowed(file_id: int) -> bool:
        return allowed_ids is None or file_id in allowed_ids

    cursor = searcher.db.conn.cursor()
    try:
        rows = cursor.execute(
            """SELECT file_id, name, ko_name, primary_location,
                      locations, extent, confidence, spatial_text
               FROM file_objects
               WHERE primary_location IS NOT NULL AND primary_location != ''"""
        ).fetchall()
        for row in rows:
            file_id = int(row["file_id"])
            if not allowed(file_id):
                continue
            primary_location = row["primary_location"] or ""
            if locations and primary_location not in locations:
                continue
            hay = " ".join(
                str(value or "")
                for value in (
                    row["name"],
                    row["ko_name"],
                    primary_location,
                    row["extent"],
                    row["spatial_text"],
                )
            )
            hits = _term_hits(hay, terms)
            if terms and hits == 0:
                continue
            score = 0.65 + (0.22 * hits) + _confidence_bonus(row["confidence"])
            bucket = matches_by_file.setdefault(file_id, {"score": 0.0, "matches": []})
            bucket["score"] = max(float(bucket["score"]), score)
            bucket["matches"].append(
                {
                    "table": "file_objects",
                    "name": row["name"],
                    "ko_name": row["ko_name"],
                    "primary_location": primary_location,
                    "confidence": row["confidence"],
                    "mode": "strict_primary",
                }
            )

        ranked = sorted(
            matches_by_file.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[:top_k]
        results: list[dict[str, Any]] = []
        for file_id, payload in ranked:
            row = cursor.execute("SELECT f.* FROM files f WHERE f.id = ?", (file_id,)).fetchone()
            if not row:
                continue
            result = dict(row)
            searcher._parse_json_fields(result)
            result["spatial_score"] = float(payload["score"])
            result["spatial_matches"] = payload["matches"]
            results.append(result)
        return results
    finally:
        cursor.close()


def load_queryset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries", data if isinstance(data, list) else [])
    if limit:
        queries = queries[:limit]
    return queries


def p_at_k(rows: list[dict[str, Any]], gt_ids: list[int], k: int) -> float:
    top = {int(r["id"]) for r in rows[:k] if r.get("id") is not None}
    return len(top & set(int(v) for v in gt_ids)) / k


def diagnostic_summary(diag: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(diag, dict):
        return {}
    decomposition = diag.get("decomposition") or {}
    spatial = diag.get("spatial_results") or {}
    rrf = diag.get("rrf_merge") or {}
    element = diag.get("element_verification") or {}
    evidence = diag.get("evidence_matrix") or {}
    return {
        "query_type": decomposition.get("query_type"),
        "find_description": decomposition.get("find_description"),
        "find_keywords": decomposition.get("find_keywords", []),
        "spatial_active": bool(spatial.get("active")),
        "spatial_count": spatial.get("count", 0),
        "spatial_intent": spatial.get("intent", {}),
        "rrf_axes": rrf.get("axes"),
        "rrf_weights": rrf.get("weights"),
        "element_groups": element.get("elements", []),
        "evidence_matrix_elements": evidence.get("elements", []),
    }


def run_variant(
    searcher: SqliteVectorSearch,
    queries: list[dict[str, Any]],
    variant: str,
    top_k: int,
    file_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    original = searcher._spatial_evidence_search
    if variant == "no_spatial_axis":
        searcher._spatial_evidence_search = types.MethodType(
            lambda self, intent, top_k=20, file_ids=None: [],
            searcher,
        )
    elif variant == "strict_primary":
        searcher._spatial_evidence_search = types.MethodType(
            lambda self, intent, top_k=20, file_ids=None: strict_primary_spatial_search(
                self,
                intent,
                top_k=top_k,
                file_ids=file_ids,
            ),
            searcher,
        )

    rows: list[dict[str, Any]] = []
    try:
        for i, q in enumerate(queries, 1):
            search_output = searcher.triaxis_search(
                q["query"],
                top_k=top_k,
                threshold=0.0,
                use_codex=False,
                file_ids=file_ids,
                return_diagnostic=True,
            )
            if isinstance(search_output, tuple):
                results, diag = search_output
            else:
                results, diag = search_output, {}
            row = {
                "query": q["query"],
                "spatial_location": q.get("spatial_location"),
                "elements_en": q.get("elements_en", []),
                "elements_ko": q.get("elements_ko", []),
                "gt_count": len(q.get("gt_ids", [])),
                "ids5": [int(r["id"]) for r in results[:5] if r.get("id") is not None],
                "ids10": [int(r["id"]) for r in results[:10] if r.get("id") is not None],
                "p5": p_at_k(results, q.get("gt_ids", []), 5),
                "p10": p_at_k(results, q.get("gt_ids", []), 10),
                "diagnostic": diagnostic_summary(diag),
            }
            rows.append(row)
            if i % 10 == 0:
                print(f"[{variant}] {i}/{len(queries)}", flush=True)
        return rows
    finally:
        searcher._spatial_evidence_search = original


def compute_mean_metrics(current: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(rows: list[dict[str, Any]], key: str) -> float:
        return round(sum(float(r[key]) for r in rows) / len(rows), 4) if rows else 0.0

    wins = losses = ties = same_top5 = 0
    per_query = []
    for cur, comp in zip(current, comparison):
        delta = round(float(cur["p5"]) - float(comp["p5"]), 4)
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
        if cur.get("ids5") == comp.get("ids5"):
            same_top5 += 1
        per_query.append(
            {
                "query": cur["query"],
                "current_p5": cur["p5"],
                "comparison_p5": comp["p5"],
                "delta_p5": delta,
                "current_ids5": cur.get("ids5", []),
                "comparison_ids5": comp.get("ids5", []),
            }
        )

    return {
        "queries": len(current),
        "current_p5": mean(current, "p5"),
        "comparison_p5": mean(comparison, "p5"),
        "delta_p5": round(mean(current, "p5") - mean(comparison, "p5"), 4),
        "current_p10": mean(current, "p10"),
        "comparison_p10": mean(comparison, "p10"),
        "delta_p10": round(mean(current, "p10") - mean(comparison, "p10"), 4),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "same_top5": same_top5,
        "per_query": per_query,
    }


def _file_filter(file_ids: set[int] | None, column: str = "file_id") -> tuple[str, tuple[int, ...]]:
    if not file_ids:
        return "", ()
    placeholders = ",".join("?" for _ in file_ids)
    return f" WHERE {column} IN ({placeholders})", tuple(sorted(file_ids))


def data_diagnostics(
    db_path: Path,
    queryset: list[dict[str, Any]],
    file_ids: set[int] | None = None,
) -> dict[str, Any]:
    conn = connect_readonly(db_path)
    try:
        object_where, object_params = _file_filter(file_ids)
        object_summary = dict(
            conn.execute(
                f"""
                SELECT
                    COUNT(*) AS rows,
                    COUNT(DISTINCT file_id) AS files,
                    SUM(CASE WHEN json_valid(locations) AND json_array_length(locations) > 1 THEN 1 ELSE 0 END) AS multi_location_rows,
                    AVG(CASE WHEN json_valid(locations) THEN json_array_length(locations) ELSE NULL END) AS avg_location_count
                FROM file_objects
                {object_where}
                """,
                object_params,
            ).fetchone()
        )
        relation_where, relation_params = _file_filter(file_ids)
        relation_summary = dict(
            conn.execute(
                f"""
                SELECT COUNT(*) AS rows, COUNT(DISTINCT file_id) AS files
                FROM file_spatial_relations
                {relation_where}
                """,
                relation_params,
            ).fetchone()
        )
        primary_where, primary_params = _file_filter(file_ids)
        primary_counts = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT primary_location, COUNT(*) AS rows
                FROM file_objects
                {primary_where}
                GROUP BY primary_location
                ORDER BY rows DESC
                """,
                primary_params,
            ).fetchall()
        ]

        query_label_rows = []
        for q in queryset:
            terms = [str(v).lower() for v in q.get("elements_en", []) + q.get("elements_ko", [])]
            loc = q.get("spatial_location")
            gt_ids = q.get("gt_ids", [])
            if not gt_ids:
                continue
            placeholders = ",".join("?" for _ in gt_ids)
            rows = conn.execute(
                f"""
                SELECT file_id, name, ko_name, primary_location, locations
                FROM file_objects
                WHERE file_id IN ({placeholders})
                """,
                tuple(gt_ids),
            ).fetchall()
            exact = 0
            broad = 0
            for row in rows:
                hay = " ".join(str(row[k] or "").lower() for k in ("name", "ko_name"))
                if terms and not any(t and t in hay for t in terms):
                    continue
                if row["primary_location"] == loc:
                    exact += 1
                if loc in _row_locations(row["locations"]):
                    broad += 1
            query_label_rows.append(
                {
                    "query": q["query"],
                    "gt_count": len(gt_ids),
                    "primary_exact_rows": exact,
                    "broad_location_rows": broad,
                }
            )

        return {
            "object_summary": object_summary,
            "relation_summary": relation_summary,
            "primary_location_counts": primary_counts,
            "query_label_rows": query_label_rows,
        }
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="imageparser.db")
    parser.add_argument("--queryset", default="benchmarks/querysets/frozen_spatial_30_v2.json")
    parser.add_argument("--output", default="benchmarks/results/spatial_axis_ablation_20260603.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--variants",
        default="current,no_spatial_axis,strict_primary",
        help="Comma-separated variants: current,no_spatial_axis,strict_primary",
    )
    parser.add_argument(
        "--file-id-json",
        help="Optional manifest containing files[].file_id/id; limits search and diagnostics to those files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("SEARCH_DIAGNOSTIC", "0")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    queries = load_queryset(Path(args.queryset), limit=args.limit)
    benchmark_file_ids = load_file_ids(Path(args.file_id_json)) if args.file_id_json else None
    db_path = Path(args.db)
    db_mtime_before = db_path.stat().st_mtime_ns if db_path.exists() else None
    db = ReadOnlySQLiteDB(args.db)
    searcher = SqliteVectorSearch(db)

    runs = {}
    for variant in variants:
        if variant not in {"current", "no_spatial_axis", "strict_primary"}:
            raise SystemExit(f"unknown variant: {variant}")
        runs[variant] = run_variant(
            searcher,
            queries,
            variant,
            args.top_k,
            file_ids=benchmark_file_ids,
        )

    comparisons = {}
    if "current" in runs:
        for variant, rows in runs.items():
            if variant == "current":
                continue
            comparisons[f"current_vs_{variant}"] = compute_mean_metrics(runs["current"], rows)

    report = {
        "run_id": Path(args.output).stem,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "queryset": args.queryset,
        "db": args.db,
        "top_k": args.top_k,
        "variants": variants,
        "file_id_filter_count": len(benchmark_file_ids) if benchmark_file_ids else None,
        "data_diagnostics": data_diagnostics(Path(args.db), queries, benchmark_file_ids),
        "runs": runs,
        "comparisons": comparisons,
        "read_only": {
            "query_only": True,
            "db_mtime_unchanged": (
                db_mtime_before == db_path.stat().st_mtime_ns
                if db_mtime_before is not None and db_path.exists()
                else None
            ),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
