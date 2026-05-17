import json
import sqlite3
import types

from backend.db.sqlite_client import SQLiteDB


def test_spatial_objects_are_normalized_from_structured_meta():
    structured_meta = {
        "objects": [
            {
                "name": "Moon",
                "ko_name": "달",
                "primary_location": "upper right",
                "locations": ["right", "top right", "unknown", "right"],
                "extent": "small",
                "confidence": "high",
            },
            {
                "name": "tree",
                "locations": ["lower-left"],
                "extent": "huge",
                "confidence": "certain",
            },
            {"name": "cloud"},
            {"primary_location": "left"},
        ]
    }

    objects = SQLiteDB._normalize_spatial_objects_from_meta(json.dumps(structured_meta))

    assert objects == [
        {
            "name": "moon",
            "ko_name": "달",
            "locations": ["top-right", "right"],
            "primary_location": "top-right",
            "extent": "small",
            "confidence": "high",
        },
        {
            "name": "tree",
            "ko_name": "",
            "locations": ["bottom-left"],
            "primary_location": "bottom-left",
            "extent": "",
            "confidence": "low",
        },
    ]


def test_flat_spatial_object_tokens_are_normalized_from_legacy_fallback():
    structured_meta = {
        "objects": [
            "name",
            "forks",
            "ko_name",
            "포크",
            "locations",
            "top-left",
            "top",
            "center",
        ]
    }

    objects = SQLiteDB._normalize_spatial_objects_from_meta(json.dumps(structured_meta))

    assert objects == [
        {
            "name": "forks",
            "ko_name": "포크",
            "locations": ["top-left", "top", "center"],
            "primary_location": "top-left",
            "extent": "",
            "confidence": "low",
        }
    ]


def test_spatial_relations_and_depth_layers_are_normalized_from_structured_meta():
    structured_meta = {
        "relations": [
            {
                "subject": "Cup",
                "relation": "on",
                "object": "Table",
                "subject_location": "center",
                "object_location": "bottom",
                "confidence": "high",
            },
            {
                "subject": "door",
                "relation": "behind",
                "object": "character",
                "confidence": "certain",
            },
        ],
        "depth_layers": [
            {"name": "Table", "ko_name": "테이블", "layer": "foreground", "confidence": "medium"},
            {"object": "window", "layer": "far", "confidence": "high"},
        ],
    }

    relations = SQLiteDB._normalize_spatial_relations_from_meta(json.dumps(structured_meta))
    depth_layers = SQLiteDB._normalize_depth_layers_from_meta(json.dumps(structured_meta))

    assert relations == [
        {
            "subject": "cup",
            "relation": "on",
            "object": "table",
            "subject_location": "center",
            "object_location": "bottom",
            "confidence": "high",
        },
        {
            "subject": "door",
            "relation": "behind",
            "object": "character",
            "subject_location": "",
            "object_location": "",
            "confidence": "low",
        },
    ]
    assert depth_layers == [
        {
            "name": "table",
            "ko_name": "테이블",
            "layer": "foreground",
            "confidence": "medium",
        }
    ]


def test_spatial_fts_text_contains_relations_and_depth_layers():
    objects = [
        {
            "name": "cup",
            "ko_name": "컵",
            "locations": ["center"],
            "primary_location": "center",
            "extent": "small",
            "confidence": "high",
        }
    ]
    relations = [
        {
            "subject": "cup",
            "relation": "on",
            "object": "table",
            "subject_location": "center",
            "object_location": "bottom",
            "confidence": "high",
        }
    ]
    depth_layers = [
        {
            "name": "table",
            "ko_name": "테이블",
            "layer": "foreground",
            "confidence": "medium",
        }
    ]

    spatial_text = SQLiteDB._build_fts_spatial(objects, relations, depth_layers)

    assert "cup on table" in spatial_text
    assert "cup table 위" in spatial_text
    assert "컵 테이블 위" in spatial_text
    assert "table foreground" in spatial_text
    assert "테이블 전경" in spatial_text


def test_spatial_fts_text_contains_object_location_pairs_in_english_and_korean():
    objects = [
        {
            "name": "moon",
            "ko_name": "달",
            "locations": ["top-right", "right"],
            "primary_location": "top-right",
            "extent": "small",
            "confidence": "high",
        }
    ]

    spatial_text = SQLiteDB._build_fts_spatial(objects)

    assert "moon top-right" in spatial_text
    assert "moon right" in spatial_text
    assert "moon 우상단" in spatial_text
    assert "달 우상단" in spatial_text
    assert "달 오른쪽" in spatial_text


def test_replace_file_objects_persists_normalized_evidence():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("CREATE TABLE files(id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO files(id) VALUES (1)")

    db = object.__new__(SQLiteDB)
    db._local = types.SimpleNamespace(conn=conn)
    db._ensure_file_objects_table()

    objects = SQLiteDB._normalize_spatial_objects_from_meta({
        "objects": [
            {
                "name": "moon",
                "ko_name": "달",
                "primary_location": "right",
                "locations": ["right"],
                "confidence": "medium",
            }
        ]
    })
    db._replace_file_objects(conn.cursor(), 1, objects)
    conn.commit()

    row = conn.execute(
        "SELECT file_id, name, ko_name, primary_location, locations, confidence, spatial_text "
        "FROM file_objects"
    ).fetchone()

    assert {
        key: row[key]
        for key in ("file_id", "name", "ko_name", "primary_location", "locations", "confidence")
    } == {
        "file_id": 1,
        "name": "moon",
        "ko_name": "달",
        "primary_location": "right",
        "locations": '["right"]',
        "confidence": "medium",
    }
    assert "moon right" in row["spatial_text"]
    assert "달 오른쪽" in row["spatial_text"]
