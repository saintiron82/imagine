import json
import sqlite3
from pathlib import Path

from tools.diagnose_multicondition_recall import (
    analyze_result,
    build_ko_en_map,
    validate_condition_labels,
)


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def make_db(path: Path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE file_objects (
            file_id INTEGER,
            name TEXT,
            ko_name TEXT,
            primary_location TEXT,
            locations TEXT,
            extent TEXT,
            confidence TEXT,
            source TEXT
        );
        """
    )
    conn.executemany(
        """INSERT INTO file_objects
           (file_id, name, ko_name, primary_location, locations, extent, confidence, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (10, "wall", "벽", "center", '["center"]', "wide", "high", "vlm"),
            (10, "curtain", "커튼", "right", '["right"]', "medium", "medium", "vlm"),
            (20, "wall", "벽", "center", '["center"]', "wide", "high", "vlm"),
            (20, "curtain", "커튼", "left", '["left"]', "large", "high", "vlm"),
        ],
    )
    conn.commit()
    conn.close()


def test_validate_condition_labels_detects_cross_pairing(tmp_path: Path):
    db_path = tmp_path / "imageparser.db"
    make_db(db_path)
    ko_en = build_ko_en_map(db_path)

    issues = validate_condition_labels(
        ["벽", "커튼"],
        {
            "벽|curtain": ["벽", "curtain"],
            "커튼|wall": ["커튼", "wall"],
        },
        ko_en,
    )

    assert issues == [
        {
            "label": "벽|curtain",
            "ko": "벽",
            "expected_en": ["wall"],
            "actual_en": ["curtain"],
            "issue": "condition_group_cross_pairing",
        },
        {
            "label": "커튼|wall",
            "ko": "커튼",
            "expected_en": ["curtain"],
            "actual_en": ["wall"],
            "issue": "condition_group_cross_pairing",
        },
    ]


def test_validate_condition_labels_ignores_unmapped_inflections(tmp_path: Path):
    db_path = tmp_path / "imageparser.db"
    make_db(db_path)
    ko_en = build_ko_en_map(db_path)

    issues = validate_condition_labels(
        ["벽", "커튼"],
        {
            "벽|walls": ["벽", "walls"],
            "커튼|curtain": ["커튼", "curtain"],
        },
        ko_en,
    )

    assert issues == []


def test_analyze_result_classifies_object_evidence_misses(tmp_path: Path):
    db_path = tmp_path / "imageparser.db"
    make_db(db_path)
    result_path = tmp_path / "s19.json"
    write_json(result_path, {
        "run_id": "s19",
        "runs": {
            "current": [
                {
                    "query": "벽, 커튼가 함께 있는 이미지",
                    "elements_ko": ["벽", "커튼"],
                    "gt_ids": [10, 20],
                    "ids10": [10],
                    "top_evidence_matrix": {
                        "conditions": {
                            "matches": {
                                "벽|curtain": ["벽", "curtain"],
                                "커튼|wall": ["커튼", "wall"],
                            }
                        }
                    },
                }
            ]
        },
    })

    report = analyze_result(result_path, db_path=db_path)

    assert report["summary"]["missed_gt_total"] == 1
    assert report["summary"]["miss_cause_counts"] == {
        "object_evidence_present_but_not_top10": 1,
    }
    row = report["rows"][0]
    assert row["misses"][0]["item_id"] == "20"
    assert row["misses"][0]["matched_object_conditions"] == ["벽", "커튼"]
    assert row["condition_group_issues"][0]["issue"] == "condition_group_cross_pairing"
