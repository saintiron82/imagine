import json
from pathlib import Path

from tools.compare_search_evaluation import (
    build_markdown,
    compare_summaries,
    load_summary,
    main,
)


def summary(metrics, query_count=10, latency=100, engine_id="triaxis"):
    return {
        "schema_version": "search_evaluation_v1",
        "k_values": [10, 50],
        "label_query_count": query_count,
        "run_count": 1,
        "runs": [{
            "run_id": "r1",
            "engine_id": engine_id,
            "query_count": query_count,
            "metrics": metrics,
            "by_query_type": {},
            "avg_latency_ms": latency,
        }],
    }


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_compare_summaries_passes_within_tolerance():
    report = compare_summaries(
        baseline=summary({"nDCG@10": 0.7, "P@10": 0.4}),
        candidate=summary({"nDCG@10": 0.695, "P@10": 0.41}),
        metrics=["nDCG@10", "P@10"],
        min_delta=-0.01,
    )

    assert report["status"] == "pass"
    assert report["comparisons"][0]["delta"] == -0.005
    assert "Metric Gate" in build_markdown(report)


def test_compare_summaries_fails_regression_and_query_drop():
    report = compare_summaries(
        baseline=summary({"nDCG@10": 0.7}, query_count=10),
        candidate=summary({"nDCG@10": 0.68}, query_count=9),
        metrics=["nDCG@10"],
        min_delta=-0.01,
        min_query_ratio=1.0,
    )

    assert report["status"] == "fail"
    assert any("nDCG@10" in failure for failure in report["failures"])
    assert any("query_count" in failure for failure in report["failures"])


def test_compare_summaries_checks_latency_when_requested():
    report = compare_summaries(
        baseline=summary({"nDCG@10": 0.7}, latency=100),
        candidate=summary({"nDCG@10": 0.72}, latency=160),
        metrics=["nDCG@10"],
        max_latency_ratio=1.5,
    )

    assert report["status"] == "fail"
    assert any(check["check"] == "avg_latency_ms" and check["status"] == "fail" for check in report["checks"])


def test_compare_cli_exit_codes(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    report_path = tmp_path / "compare.json"
    write_json(baseline_path, summary({"nDCG@10": 0.7}))
    write_json(candidate_path, summary({"nDCG@10": 0.71}))

    code = main([
        "--baseline", str(baseline_path),
        "--candidate", str(candidate_path),
        "--metrics", "nDCG@10",
        "--output-json", str(report_path),
    ])

    assert code == 0
    assert load_summary(baseline_path)["run_count"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "pass"
