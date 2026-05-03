import json
from pathlib import Path

from tools.run_search_benchmark import build_run_rows, run_search_benchmark


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class FakeSearcher:
    def vector_search(self, query, top_k, threshold):
        return [{"id": "a", "similarity": 0.9}, {"id": "b", "similarity": 0.7}]

    def text_vector_search(self, query, top_k, threshold):
        return [{"id": "b", "score": 0.8}]

    def fts_search(self, keywords, top_k):
        return [{"id": "c", "fts_rank": -1.2}]

    def triaxis_search(self, query, top_k, threshold, use_codex):
        return [{"id": "a", "final_score": 1.0}, {"id": "a", "final_score": 0.9}, {"id": "b", "final_score": 0.8}]


def make_inputs(tmp_path: Path):
    queries_path = tmp_path / "queries.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(queries_path, [
        {
            "query_id": "q1",
            "query_text": "창문과 책상이 있는 방",
            "query_type": "semantic",
            "locale": "ko-KR",
            "created_at": "2026-05-01T00:00:00+09:00",
        }
    ])
    write_jsonl(labels_path, [
        {"query_id": "q1", "item_id": "a", "relevance": 2, "label_source": "human", "label_version": "v1"},
        {"query_id": "q1", "item_id": "b", "relevance": 1, "label_source": "human", "label_version": "v1"},
    ])
    return queries_path, labels_path


def test_build_run_rows_deduplicates_ranked_results(tmp_path: Path):
    queries_path, _labels_path = make_inputs(tmp_path)

    rows, events = build_run_rows(
        queries_path=queries_path,
        engines=["triaxis"],
        top_k=5,
        run_id="r1",
        searcher=FakeSearcher(),
    )

    assert [row["item_id"] for row in rows] == ["a", "b"]
    assert [row["rank"] for row in rows] == [1, 2]
    assert rows[0]["latency_ms"] is not None
    assert rows[1]["latency_ms"] is None
    assert events[0]["result_count"] == 3


def test_run_search_benchmark_writes_evaluation_and_compare(tmp_path: Path):
    queries_path, labels_path = make_inputs(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    output_dir = tmp_path / "run"
    write_json(baseline_path, {
        "schema_version": "search_evaluation_v1",
        "k_values": [1, 2],
        "label_query_count": 1,
        "run_count": 1,
        "runs": [{
            "run_id": "baseline",
            "engine_id": "triaxis",
            "query_count": 1,
            "metrics": {"nDCG@2": 0.5},
            "by_query_type": {},
            "avg_latency_ms": 100,
        }],
    })

    paths = run_search_benchmark(
        queries_path=queries_path,
        labels_path=labels_path,
        output_dir=output_dir,
        engines=["triaxis"],
        top_k=2,
        k_values=(1, 2),
        run_id="candidate",
        baseline_path=baseline_path,
        metrics=["nDCG@2"],
        searcher=FakeSearcher(),
        quiet=True,
    )

    assert paths["run_results"].exists()
    assert paths["evaluation_json"].exists()
    assert paths["compare_json"].exists()
    evaluation = json.loads(paths["evaluation_json"].read_text(encoding="utf-8"))
    compare = json.loads(paths["compare_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert evaluation["runs"][0]["engine_id"] == "triaxis"
    assert evaluation["runs"][0]["metrics"]["nDCG@2"] == 1.0
    assert compare["status"] == "pass"
    assert manifest["counts"]["run_result_rows"] == 2
