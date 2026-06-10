#!/usr/bin/env python3
"""Diagnose multicondition recall misses from existing benchmark artifacts.

The tool is read-only. It reads benchmark JSON and existing SQLite evidence,
then explains whether missed GT items already have object evidence and whether
condition-group labels look incorrectly paired.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise ValueError(f"database not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.split(",")]
    if isinstance(parsed, list):
        return [str(v).strip() for v in parsed if str(v).strip()]
    return []


def build_ko_en_map(db_path: Path) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    conn = _connect_readonly(db_path)
    try:
        for row in conn.execute(
            """
            SELECT ko_name, name
            FROM file_objects
            WHERE ko_name IS NOT NULL AND ko_name != ''
              AND name IS NOT NULL AND name != ''
            """
        ):
            mapping[str(row["ko_name"]).strip()].add(str(row["name"]).strip().lower())
    finally:
        conn.close()
    return {key: sorted(values) for key, values in mapping.items()}


def _load_file_objects(db_path: Path, file_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    if not file_ids:
        return {}
    numeric = [int(v) for v in file_ids if str(v).isdigit()]
    if not numeric:
        return {}
    placeholders = ",".join("?" for _ in numeric)
    conn = _connect_readonly(db_path)
    try:
        objects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in conn.execute(
            f"""
            SELECT file_id, name, ko_name, primary_location, locations, extent, confidence, source
            FROM file_objects
            WHERE file_id IN ({placeholders})
            """,
            numeric,
        ):
            objects[str(row["file_id"])].append({
                "name": str(row["name"] or ""),
                "ko_name": str(row["ko_name"] or ""),
                "primary_location": row["primary_location"],
                "locations": _json_list(row["locations"]),
                "extent": row["extent"],
                "confidence": row["confidence"],
                "source": row["source"],
            })
        return dict(objects)
    finally:
        conn.close()


def _split_condition_label(label: str) -> tuple[str, list[str]]:
    parts = [part.strip() for part in str(label).split("|") if part.strip()]
    if not parts:
        return "", []
    return parts[0], [part.lower() for part in parts[1:]]


def validate_condition_labels(
    elements_ko: list[str],
    matches: dict[str, Any],
    ko_en_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    element_set = {str(v).strip() for v in elements_ko if str(v).strip()}
    expected_by_element = {
        element: set(ko_en_map.get(element, []))
        for element in element_set
    }
    issues: list[dict[str, Any]] = []
    for label in matches:
        ko, actual_en = _split_condition_label(label)
        if not ko or ko not in element_set or not actual_en:
            continue
        actual_set = set(actual_en)
        expected_en = sorted(expected_by_element.get(ko, set()))
        other_expected = set()
        for other_ko, other_terms in expected_by_element.items():
            if other_ko != ko:
                other_expected.update(other_terms)
        if actual_set & other_expected and not actual_set.issubset(set(expected_en)):
            issues.append({
                "label": label,
                "ko": ko,
                "expected_en": expected_en,
                "actual_en": actual_en,
                "issue": "condition_group_cross_pairing",
            })
    return issues


def _matched_object_conditions(
    elements_ko: list[str],
    objects: list[dict[str, Any]],
    ko_en_map: dict[str, list[str]],
) -> list[str]:
    matched: list[str] = []
    for element in elements_ko:
        names = {str(element).strip().lower(), *ko_en_map.get(str(element).strip(), [])}
        for obj in objects:
            obj_values = {
                str(obj.get("ko_name") or "").lower(),
                str(obj.get("name") or "").lower(),
            }
            if names & obj_values:
                matched.append(str(element).strip())
                break
    return matched


def _miss_cause(elements_ko: list[str], matched_conditions: list[str]) -> str:
    if elements_ko and len(matched_conditions) == len(elements_ko):
        return "object_evidence_present_but_not_top10"
    if matched_conditions:
        return "partial_object_evidence_not_top10"
    return "missing_object_evidence_or_label_noise"


def analyze_result(
    result_path: Path,
    *,
    db_path: Path,
    variant: str = "current",
    evidence_audit_path: Path | None = None,
) -> dict[str, Any]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    runs = data.get("runs", {})
    rows = runs.get(variant)
    if not isinstance(rows, list):
        raise ValueError(f"{result_path}: runs.{variant} must be a list")

    ko_en_map = build_ko_en_map(db_path)
    all_gt_ids = {
        str(item_id)
        for row in rows
        for item_id in row.get("gt_ids", [])
    }
    object_map = _load_file_objects(db_path, all_gt_ids)

    evidence_rows: dict[str, dict[str, Any]] = {}
    if evidence_audit_path and evidence_audit_path.exists():
        evidence_data = json.loads(evidence_audit_path.read_text(encoding="utf-8"))
        evidence_rows = {
            str(row.get("query")): row
            for row in evidence_data.get("rows", [])
            if isinstance(row, dict)
        }

    output_rows: list[dict[str, Any]] = []
    cause_counts: Counter[str] = Counter()
    condition_issue_count = 0
    found_at10 = 0
    gt_total = 0
    missed_total = 0

    for row in rows:
        query = str(row.get("query") or "")
        elements_ko = [str(v) for v in row.get("elements_ko", [])]
        gt_ids = {str(v) for v in row.get("gt_ids", [])}
        top10_ids = {str(v) for v in row.get("ids10", [])}
        missed_ids = sorted(gt_ids - top10_ids)
        found_at10 += len(gt_ids & top10_ids)
        gt_total += len(gt_ids)
        missed_total += len(missed_ids)

        matrix = row.get("top_evidence_matrix") or {}
        conditions = matrix.get("conditions") if isinstance(matrix, dict) else {}
        matches = conditions.get("matches", {}) if isinstance(conditions, dict) else {}
        condition_issues = validate_condition_labels(elements_ko, matches, ko_en_map)
        condition_issue_count += len(condition_issues)

        misses = []
        for item_id in missed_ids:
            objects = object_map.get(item_id, [])
            matched_conditions = _matched_object_conditions(elements_ko, objects, ko_en_map)
            cause = _miss_cause(elements_ko, matched_conditions)
            cause_counts[cause] += 1
            misses.append({
                "item_id": item_id,
                "cause": cause,
                "matched_object_conditions": matched_conditions,
                "objects": objects,
            })

        if missed_ids or condition_issues:
            evidence_row = evidence_rows.get(query, {})
            output_rows.append({
                "query": query,
                "elements_ko": elements_ko,
                "gt_count": len(gt_ids),
                "found_at10": len(gt_ids & top10_ids),
                "missed_at10": len(missed_ids),
                "misses": misses,
                "condition_group_issues": condition_issues,
                "s17_gt_rank_positions": evidence_row.get("gt_rank_positions"),
                "s17_full_evidence_non_gt_top10": evidence_row.get("full_evidence_non_gt_in_top10"),
            })

    return {
        "schema_version": "multicondition_recall_diagnosis_v1",
        "source_result": str(result_path),
        "variant": variant,
        "summary": {
            "query_count": len(rows),
            "gt_total": gt_total,
            "found_gt_at10": found_at10,
            "missed_gt_total": missed_total,
            "micro_recall_at10": round(found_at10 / gt_total, 6) if gt_total else None,
            "miss_cause_counts": dict(sorted(cause_counts.items())),
            "condition_group_issue_count": condition_issue_count,
        },
        "rows": output_rows,
        "recommendations": _recommendations(cause_counts, condition_issue_count),
    }


def _recommendations(cause_counts: Counter[str], condition_issue_count: int) -> list[str]:
    recs: list[str] = []
    if condition_issue_count:
        recs.append("validate_condition_group_pairing")
    if cause_counts.get("object_evidence_present_but_not_top10"):
        recs.append("add_object_evidence_recall_guard")
        recs.append("run_top50_miss_trace")
    if cause_counts.get("partial_object_evidence_not_top10"):
        recs.append("inspect_partial_object_evidence")
    return recs


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Multicondition Recall Diagnosis",
        "",
        f"- Source: `{report['source_result']}`",
        f"- Variant: `{report['variant']}`",
        f"- Queries: {summary['query_count']}",
        f"- GT total: {summary['gt_total']}",
        f"- Found@10: {summary['found_gt_at10']}",
        f"- Missed@10: {summary['missed_gt_total']}",
        f"- Micro Recall@10: {summary['micro_recall_at10']}",
        f"- Condition group issues: {summary['condition_group_issue_count']}",
        "",
        "## Miss Causes",
        "",
    ]
    for cause, count in summary["miss_cause_counts"].items():
        lines.append(f"- `{cause}`: {count}")
    lines.extend(["", "## Recommendations", ""])
    for rec in report["recommendations"]:
        lines.append(f"- `{rec}`")
    lines.extend(["", "## Rows", ""])
    for row in report["rows"]:
        lines.append(f"### {row['query']}")
        lines.append(f"- found_at10: {row['found_at10']}/{row['gt_count']}")
        if row["condition_group_issues"]:
            lines.append(f"- condition_group_issues: {len(row['condition_group_issues'])}")
        for miss in row["misses"]:
            lines.append(
                f"- miss `{miss['item_id']}`: `{miss['cause']}` "
                f"matched={','.join(miss['matched_object_conditions'])}"
            )
        lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=Path("imageparser.db"))
    parser.add_argument("--variant", default="current")
    parser.add_argument("--evidence-audit", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)

    try:
        report = analyze_result(
            args.result,
            db_path=args.db_path,
            variant=args.variant,
            evidence_audit_path=args.evidence_audit,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        write_json(args.output_json, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_md:
        write_markdown(args.output_md, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
