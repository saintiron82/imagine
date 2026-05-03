from pathlib import Path
import sqlite3

from tools.evaluate_search_quality import (
    build_markdown,
    evaluate_all,
    load_labels,
    load_queries,
    load_run,
)
from tools.bench_precision import build_standard_eval_rows
from tools.build_search_label_review import (
    build_review_rows,
    enrich_rows_with_item_metadata,
    load_item_metadata,
)


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text(
        "\n".join(__import__("json").dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_evaluate_search_quality_metrics(tmp_path: Path):
    queries_path = tmp_path / "queries.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    run_path = tmp_path / "run.jsonl"

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
        {"query_id": "q1", "item_id": "c", "relevance": 0, "label_source": "human", "label_version": "v1"},
    ])
    write_jsonl(run_path, [
        {
            "run_id": "r1",
            "engine_id": "triaxis",
            "query_id": "q1",
            "rank": 1,
            "item_id": "c",
            "score": 0.9,
            "latency_ms": 10,
            "error": None,
            "cost_usd": None,
        },
        {
            "run_id": "r1",
            "engine_id": "triaxis",
            "query_id": "q1",
            "rank": 2,
            "item_id": "a",
            "score": 0.8,
            "latency_ms": 10,
            "error": None,
            "cost_usd": None,
        },
        {
            "run_id": "r1",
            "engine_id": "triaxis",
            "query_id": "q1",
            "rank": 3,
            "item_id": "b",
            "score": 0.7,
            "latency_ms": 10,
            "error": None,
            "cost_usd": None,
        },
    ])

    summary = evaluate_all(
        load_labels(labels_path),
        load_run(run_path),
        load_queries(queries_path),
        (1, 2, 3),
    )

    metrics = summary["runs"][0]["metrics"]
    assert metrics["P@1"] == 0.0
    assert metrics["P@2"] == 0.5
    assert metrics["Recall@3"] == 1.0
    assert metrics["MRR@3"] == 0.5
    assert 0.65 < metrics["nDCG@3"] < 0.75
    assert summary["runs"][0]["by_query_type"]["semantic"]["query_count"] == 1
    report = build_markdown(summary)
    assert "nDCG@3" in report
    assert "P@3" in report
    assert "Recall@3" in report


def test_invalid_relevance_fails(tmp_path: Path):
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(labels_path, [
        {"query_id": "q1", "item_id": "a", "relevance": 3, "label_source": "human", "label_version": "v1"},
    ])

    try:
        load_labels(labels_path)
    except ValueError as exc:
        assert "relevance must be 0, 1, or 2" in str(exc)
    else:
        raise AssertionError("expected invalid relevance to fail")


def test_build_standard_eval_rows_marks_weak_labels():
    queries = [{
        "query": "창문과 책상이 있는 방",
        "folder": "",
        "gt_ids": {2, 1},
    }]
    all_results = {
        "triaxis": {
            "per_query": [{
                "ranked_ids": [3, 2, 1],
            }],
        },
    }

    query_rows, label_rows, run_rows = build_standard_eval_rows(
        queries=queries,
        all_results=all_results,
        run_id="r1",
        created_at="2026-05-01T00:00:00+09:00",
        label_source="weak",
        label_version="precision_keyword_v1",
    )

    assert query_rows[0]["query_id"] == "precision-q0001"
    assert query_rows[0]["query_type"] == "semantic"
    assert {row["item_id"] for row in label_rows} == {"1", "2"}
    assert all(row["label_source"] == "weak" for row in label_rows)
    assert [row["rank"] for row in run_rows] == [1, 2, 3]
    assert all(row["score"] == 0.0 for row in run_rows)


def test_build_search_label_review_deduplicates_candidates(tmp_path: Path):
    queries_path = tmp_path / "queries.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    run_path = tmp_path / "run.jsonl"

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
        {"query_id": "q1", "item_id": "a", "relevance": 2, "label_source": "weak", "label_version": "v1"},
    ])
    write_jsonl(run_path, [
        {
            "run_id": "r1",
            "engine_id": "triaxis",
            "query_id": "q1",
            "rank": 1,
            "item_id": "a",
            "score": 0.9,
            "latency_ms": None,
            "error": None,
            "cost_usd": None,
        },
        {
            "run_id": "r1",
            "engine_id": "mv",
            "query_id": "q1",
            "rank": 2,
            "item_id": "a",
            "score": 0.8,
            "latency_ms": None,
            "error": None,
            "cost_usd": None,
        },
    ])

    rows = build_review_rows(
        queries_path=queries_path,
        run_path=run_path,
        labels_path=labels_path,
        top_k=10,
    )

    assert len(rows) == 1
    assert rows[0]["item_id"] == "a"
    assert rows[0]["suggested_relevance"] == 2
    assert rows[0]["engines"] == ["mv", "triaxis"]
    assert rows[0]["engine_ranks"] == {"mv": 2, "triaxis": 1}
    assert rows[0]["run_ranks"] == {"r1:mv": 2, "r1:triaxis": 1}


def test_build_search_label_review_filters_queries_and_keeps_run_ranks(tmp_path: Path):
    queries_path = tmp_path / "queries.jsonl"
    run_path = tmp_path / "run.jsonl"

    write_jsonl(queries_path, [
        {
            "query_id": "q1",
            "query_text": "#3에서 창문과 주방 있는 이미지",
            "query_type": "scoped",
            "locale": "ko-KR",
            "created_at": "2026-05-01T00:00:00+09:00",
        },
        {
            "query_id": "q2",
            "query_text": "늪지대에서 숲과 fog 있는 이미지",
            "query_type": "scoped",
            "locale": "ko-KR",
            "created_at": "2026-05-01T00:00:00+09:00",
        },
    ])
    write_jsonl(run_path, [
        {
            "run_id": "baseline",
            "engine_id": "triaxis",
            "query_id": "q1",
            "rank": 8,
            "item_id": "10",
            "score": 0.2,
            "latency_ms": None,
            "error": None,
            "cost_usd": None,
        },
        {
            "run_id": "candidate",
            "engine_id": "triaxis",
            "query_id": "q1",
            "rank": 2,
            "item_id": "10",
            "score": 0.9,
            "latency_ms": None,
            "error": None,
            "cost_usd": None,
        },
        {
            "run_id": "candidate",
            "engine_id": "triaxis",
            "query_id": "q2",
            "rank": 1,
            "item_id": "11",
            "score": 0.8,
            "latency_ms": None,
            "error": None,
            "cost_usd": None,
        },
    ])

    rows = build_review_rows(
        queries_path=queries_path,
        run_path=run_path,
        query_filter={"q1"},
    )

    assert len(rows) == 1
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["best_rank"] == 2
    assert rows[0]["engine_ranks"] == {"triaxis": 2}
    assert rows[0]["run_ranks"] == {
        "baseline:triaxis": 8,
        "candidate:triaxis": 2,
    }


def test_build_search_label_review_loads_item_metadata(tmp_path: Path):
    db_path = tmp_path / "imageparser.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            file_name TEXT,
            folder_path TEXT,
            thumbnail_url TEXT,
            mc_caption TEXT,
            ai_tags TEXT,
            image_type TEXT
        )
    """)
    conn.execute(
        "INSERT INTO files (id, file_path, file_name, folder_path, thumbnail_url, mc_caption, ai_tags, image_type) "
        "VALUES (10, '/asset/a.png', 'a.png', '홍콩사무실', '/thumb/a.jpg', '소파와 창문', '[\"소파\"]', 'background')"
    )
    conn.commit()
    conn.close()

    metadata = load_item_metadata(db_path, {"10", "999"})
    rows = enrich_rows_with_item_metadata(
        [{"query_id": "q1", "item_id": "10"}, {"query_id": "q1", "item_id": "999"}],
        metadata,
    )

    assert rows[0]["file_path"] == "/asset/a.png"
    assert rows[0]["folder_path"] == "홍콩사무실"
    assert rows[0]["mc_caption"] == "소파와 창문"
    assert rows[0]["ai_tags"] == "[\"소파\"]"
    assert rows[1]["file_path"] is None
