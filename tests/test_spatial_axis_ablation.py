import sqlite3

from tools.spatial_axis_ablation import (
    ReadOnlySQLiteDB,
    compute_mean_metrics,
    connect_readonly,
    data_diagnostics,
    diagnostic_summary,
    run_variant,
    strict_primary_spatial_search,
)


class _SearchStub:
    def __init__(self, conn):
        self.db = type("DB", (), {"conn": conn})()

    def _parse_json_fields(self, result):
        return None


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            file_name TEXT,
            ai_tags TEXT,
            metadata TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE file_objects (
            file_id INTEGER,
            name TEXT,
            ko_name TEXT,
            primary_location TEXT,
            locations TEXT,
            extent TEXT,
            confidence TEXT,
            spatial_text TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE file_spatial_relations (
            file_id INTEGER,
            subject TEXT,
            relation TEXT,
            object TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO files (id, file_path, file_name, ai_tags, metadata) VALUES (?, ?, ?, '[]', '{}')",
        [
            (1, "/top-wall.png", "top-wall.png"),
            (2, "/right-wall-broad-top.png", "right-wall-broad-top.png"),
            (3, "/top-window.png", "top-window.png"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO file_objects
            (file_id, name, ko_name, primary_location, locations, extent, confidence, spatial_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, "wall", "벽", "top", '["top"]', "medium", "high", "wall top 벽 위"),
            (2, "wall", "벽", "right", '["right", "top"]', "wide", "high", "wall right top 벽 오른쪽 위"),
            (3, "window", "창문", "top", '["top"]', "small", "high", "window top 창문 위"),
        ],
    )
    conn.executemany(
        "INSERT INTO file_spatial_relations (file_id, subject, relation, object) VALUES (?, ?, ?, ?)",
        [
            (1, "wall", "right_of", "window"),
            (3, "window", "below", "roof"),
        ],
    )
    conn.commit()
    return conn


def test_strict_primary_spatial_search_requires_primary_location_match():
    conn = _make_conn()
    searcher = _SearchStub(conn)

    rows = strict_primary_spatial_search(
        searcher,
        {"active": True, "terms": ["wall", "벽"], "locations": ["top"]},
        top_k=10,
    )

    assert [row["id"] for row in rows] == [1]
    assert rows[0]["spatial_score"] > 0
    assert rows[0]["spatial_matches"][0]["primary_location"] == "top"


def test_compute_mean_metrics_reports_delta_and_win_loss_counts():
    current = [
        {"query": "q1", "p5": 0.6, "p10": 0.5, "ids5": [1, 2, 3, 4, 5]},
        {"query": "q2", "p5": 0.2, "p10": 0.3, "ids5": [6, 7, 8, 9, 10]},
    ]
    no_axis = [
        {"query": "q1", "p5": 0.2, "p10": 0.3, "ids5": [1, 2, 3, 4, 5]},
        {"query": "q2", "p5": 0.4, "p10": 0.3, "ids5": [6, 7, 8, 9, 11]},
    ]

    metrics = compute_mean_metrics(current, no_axis)

    assert metrics["current_p5"] == 0.4
    assert metrics["comparison_p5"] == 0.3
    assert metrics["delta_p5"] == 0.1
    assert metrics["wins"] == 1
    assert metrics["losses"] == 1
    assert metrics["ties"] == 0
    assert metrics["same_top5"] == 1


def test_run_variant_limits_search_to_sample_file_ids():
    class Searcher:
        def __init__(self):
            self.seen_file_ids = []

        def _spatial_evidence_search(self, intent, top_k=20, file_ids=None):
            return []

        def triaxis_search(
            self,
            query,
            top_k,
            threshold,
            use_codex,
            file_ids=None,
            return_diagnostic=False,
        ):
            self.seen_file_ids.append(file_ids)
            rows = [{"id": 1}, {"id": 99}]
            diag = {"decomposition": {"query_type": "spatial"}}
            return (rows, diag) if return_diagnostic else rows

    searcher = Searcher()

    rows = run_variant(
        searcher,
        [{"query": "왼쪽에 벽이 있는 이미지", "gt_ids": [1]}],
        "current",
        top_k=5,
        file_ids={1, 2},
    )

    assert searcher.seen_file_ids == [{1, 2}]
    assert rows[0]["ids5"] == [1, 99]
    assert rows[0]["diagnostic"]["query_type"] == "spatial"


def test_diagnostic_summary_keeps_root_cause_fields():
    summary = diagnostic_summary(
        {
            "decomposition": {
                "query_type": "spatial",
                "find_description": "wall on right",
                "find_keywords": ["wall", "right"],
            },
            "spatial_results": {
                "active": True,
                "count": 3,
                "intent": {"locations": ["right"]},
            },
            "rrf_merge": {
                "axes": 4,
                "weights": {"spatial": 0.5},
            },
            "element_verification": {
                "elements": ["벽|wall", "커튼|curtain"],
            },
            "evidence_matrix": {
                "elements": ["벽|wall", "커튼|curtain"],
            },
        }
    )

    assert summary["query_type"] == "spatial"
    assert summary["spatial_active"] is True
    assert summary["spatial_count"] == 3
    assert summary["spatial_intent"]["locations"] == ["right"]
    assert summary["element_groups"] == ["벽|wall", "커튼|curtain"]


def test_data_diagnostics_can_limit_to_sample_file_ids(tmp_path):
    db_path = tmp_path / "imageparser.db"
    source = _make_conn()
    target = sqlite3.connect(db_path)
    source.backup(target)
    target.close()
    source.close()

    diagnostics = data_diagnostics(db_path, [], file_ids={1, 2})

    assert diagnostics["object_summary"]["rows"] == 2
    assert diagnostics["object_summary"]["files"] == 2
    assert diagnostics["relation_summary"]["rows"] == 1
    assert diagnostics["relation_summary"]["files"] == 1


def test_connect_readonly_blocks_writes(tmp_path):
    db_path = tmp_path / "imageparser.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE files(id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    ro = connect_readonly(db_path)
    try:
        assert ro.execute("PRAGMA query_only").fetchone()[0] == 1
        try:
            ro.execute("CREATE TABLE blocked(id INTEGER)")
        except sqlite3.OperationalError as exc:
            assert "readonly" in str(exc).lower()
        else:
            raise AssertionError("read-only connection allowed a write")
    finally:
        ro.close()


def test_readonly_sqlitedb_skips_migrations_and_blocks_writes(tmp_path):
    db_path = tmp_path / "imageparser.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE files(id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    db = ReadOnlySQLiteDB(str(db_path))
    try:
        assert db.conn.execute("PRAGMA query_only").fetchone()[0] == 1
        try:
            db.conn.execute("CREATE TABLE blocked(id INTEGER)")
        except sqlite3.OperationalError as exc:
            assert "readonly" in str(exc).lower()
        else:
            raise AssertionError("read-only SQLiteDB allowed a write")
    finally:
        db.close()
