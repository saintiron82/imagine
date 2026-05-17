import json
from pathlib import Path

from tools.run_search_benchmark import (
    benchmark_search_text,
    build_run_rows,
    fts_keywords,
    run_search_benchmark,
)
from tools.evaluate_search_quality import load_queries


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


class ScopedFakeSearcher:
    def __init__(self):
        self.calls = []

    def _apply_plan_filter_with_info(self, pre_filter):
        self.calls.append(("scope", pre_filter))
        return {"in1", "in2"}, {
            "match_mode": "exact_segment",
            "applied_folder": pre_filter["folder"],
        }

    def _vv_search_within(self, query, file_ids, top_k, threshold):
        self.calls.append(("vv", query, set(file_ids), top_k, threshold))
        return [{"id": "in1", "similarity": 0.9}]

    def _mv_search_within(self, query, file_ids, top_k, threshold):
        self.calls.append(("mv", query, set(file_ids), top_k, threshold))
        return [{"id": "in2", "score": 0.8}]

    def fts_search(self, keywords, top_k, file_ids=None):
        self.calls.append(("fts", keywords, top_k, set(file_ids or [])))
        return [{"id": "in1", "fts_rank": -1.2}]

    def triaxis_search(self, query, top_k, threshold, use_codex, file_ids=None):
        self.calls.append(("triaxis", query, top_k, threshold, use_codex, set(file_ids or [])))
        return [{"id": "in2", "final_score": 1.0}]


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


def test_build_run_rows_applies_queryset_scope_to_all_engines(tmp_path: Path):
    queries_path = tmp_path / "queries.jsonl"
    write_jsonl(queries_path, [
        {
            "query_id": "q1",
            "query_text": "작품/#08/bg에서 달과 밤하늘 있는 배경",
            "query_type": "complex",
            "locale": "ko-KR",
            "created_at": "2026-05-06T00:00:00+09:00",
            "scope": "작품/#08/bg",
        }
    ])
    searcher = ScopedFakeSearcher()

    rows, events = build_run_rows(
        queries_path=queries_path,
        engines=["vv", "mv", "fts", "triaxis"],
        top_k=5,
        run_id="scoped",
        searcher=searcher,
    )

    assert [event["scope"] for event in events] == ["작품/#08/bg"] * 4
    assert [event["scope_file_count"] for event in events] == [2, 2, 2, 2]
    assert {row["item_id"] for row in rows} == {"in1", "in2"}
    assert all(row["scope"] == "작품/#08/bg" for row in rows)
    assert ("vv", "달과 밤하늘 있는 배경", {"in1", "in2"}, 5, 0.0) in searcher.calls
    assert ("mv", "달과 밤하늘 있는 배경", {"in1", "in2"}, 5, 0.0) in searcher.calls
    assert any(call[0] == "fts" and call[1] == ["달", "밤하늘"] and call[3] == {"in1", "in2"} for call in searcher.calls)
    assert any(call[0] == "triaxis" and call[1] == "달과 밤하늘 있는 배경" and call[5] == {"in1", "in2"} for call in searcher.calls)


def test_benchmark_search_text_uses_must_terms_not_scope_text():
    assert benchmark_search_text(
        "기절용사와 암살공주 #08/bg에서 밤하늘과 달이 함께 보이는 배경",
        {
            "scope": "기절용사와 암살공주/#08/bg",
            "must_terms": ["밤하늘", "달", "조명"],
            "soft_terms": ["성"],
        },
    ) == "밤하늘, 달, 조명, 성"


def test_fts_keywords_drop_generic_korean_search_words():
    assert fts_keywords("#08에서 밤과 달 있는 이미지") == ["#08", "밤", "달"]
    assert fts_keywords("밤하늘과 달이 함께 보이고 성이나 마을 조명이 있는 배경") == ["밤하늘", "달", "성", "마을", "조명"]


def test_fts_keywords_normalize_scope_and_condition_particles():
    assert fts_keywords("로네느의집에서 그림과 방 있는 이미지") == ["로네느의집", "그림", "방"]
    assert fts_keywords("#07에서 캐릭터과 방 있는 이미지") == ["#07", "캐릭터", "방"]
    assert fts_keywords("오른쪽에 달이 있다") == ["오른쪽", "달"]


def test_fts_keywords_preserve_english_terms_but_drop_generic_image_words():
    assert fts_keywords("night forest image with fog") == ["night", "forest", "fog"]


def test_load_queries_accepts_spatial_processing_query_types(tmp_path: Path):
    queries_path = tmp_path / "spatial_queries.jsonl"
    write_jsonl(queries_path, [
        {
            "query_id": "spatial_relation_001",
            "query_text": "컵이 테이블 위에 있는 이미지",
            "query_type": "spatial_relation",
            "locale": "ko-KR",
            "created_at": "2026-05-17T00:00:00+09:00",
        },
        {
            "query_id": "spatial_depth_001",
            "query_text": "전경에 테이블이 있는 이미지",
            "query_type": "spatial_depth",
            "locale": "ko-KR",
            "created_at": "2026-05-17T00:00:00+09:00",
        },
    ])

    queries = load_queries(queries_path)

    assert set(queries) == {"spatial_relation_001", "spatial_depth_001"}


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
