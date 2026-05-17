import json
import sqlite3

from tools.verify_ingest_e2e import _print_verification, _verify


def test_verify_reports_spatial_relations_depth_raw_and_spatial_fts():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            file_name TEXT,
            format TEXT,
            structured_meta TEXT,
            perceptual_hash TEXT,
            dominant_color TEXT,
            ai_style TEXT,
            caption_model TEXT,
            processing_status TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE files_fts (
            rowid INTEGER PRIMARY KEY,
            caption TEXT,
            ai_tags TEXT,
            classification TEXT,
            spatial TEXT
        )"""
    )
    conn.execute("CREATE TABLE file_objects(file_id INTEGER)")
    conn.execute("CREATE TABLE file_spatial_relations(file_id INTEGER)")
    conn.execute("CREATE TABLE file_depth_layers(file_id INTEGER)")
    conn.execute("CREATE TABLE vlm_raw_outputs(file_id INTEGER)")
    conn.execute(
        """INSERT INTO files
           (id, file_path, file_name, format, structured_meta, perceptual_hash,
            dominant_color, ai_style, caption_model, processing_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            1,
            "/tmp/a.png",
            "a.png",
            "png",
            json.dumps({"objects": [{"name": "cup"}]}, ensure_ascii=False),
            "hash",
            "#000000",
            "flat",
            "model",
            "vision_done",
        ),
    )
    conn.execute(
        "INSERT INTO files_fts(rowid, caption, ai_tags, classification, spatial) VALUES (1, 'cap', 'cup', 'item', 'cup on table')"
    )
    conn.execute("INSERT INTO file_objects(file_id) VALUES (1)")
    conn.execute("INSERT INTO file_spatial_relations(file_id) VALUES (1)")
    conn.execute("INSERT INTO file_depth_layers(file_id) VALUES (1)")
    conn.execute("INSERT INTO vlm_raw_outputs(file_id) VALUES (1)")

    result = _verify(conn, 1)

    assert result["objects"]["file_objects_count"] == 1
    assert result["relations"]["file_spatial_relations_count"] == 1
    assert result["depth_layers"]["file_depth_layers_count"] == 1
    assert result["raw"]["vlm_raw_outputs_count"] == 1
    assert result["fts"]["spatial_nonempty"] is True
    assert _print_verification(result) == 0
