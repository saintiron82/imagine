import json
import sqlite3

from tools.generate_spatial_queryset import collect_pairs, load_file_ids


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE file_objects (
            file_id INTEGER,
            name TEXT,
            ko_name TEXT,
            primary_location TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO file_objects (file_id, name, ko_name, primary_location)
        VALUES (?, ?, ?, ?)
        """,
        [
            (1, "wall", "벽", "right"),
            (2, "wall", "벽", "right"),
            (3, "wall", "벽", "right"),
            (4, "castle", "성", "top"),
        ],
    )
    conn.commit()
    conn.close()


def test_collect_pairs_can_limit_to_sample_file_ids(tmp_path):
    db_path = tmp_path / "imageparser.db"
    _make_db(db_path)

    pairs = collect_pairs(db_path, min_files_per_pair=2, file_ids={1, 2})

    assert len(pairs) == 1
    assert pairs[0]["ko_name"] == "벽"
    assert pairs[0]["location"] == "right"
    assert pairs[0]["file_ids"] == [1, 2]


def test_load_file_ids_accepts_sample100_manifest(tmp_path):
    manifest = tmp_path / "sample.json"
    manifest.write_text(
        json.dumps({"files": [{"file_id": 11}, {"id": 12}, {"file_id": 11}]}),
        encoding="utf-8",
    )

    assert load_file_ids(manifest) == {11, 12}
