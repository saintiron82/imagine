import sqlite3

from tools.backfill_spatial_processing import select_reprocess_candidates


def test_select_reprocess_candidates_limits_and_filters_by_missing_relations(tmp_path):
    db_path = tmp_path / "backfill.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE files(id INTEGER PRIMARY KEY, file_path TEXT, mc_caption TEXT)")
    conn.execute("CREATE TABLE file_spatial_relations(file_id INTEGER)")
    conn.execute("INSERT INTO files(id, file_path, mc_caption) VALUES (1, '/a.png', 'caption')")
    conn.execute("INSERT INTO files(id, file_path, mc_caption) VALUES (2, '/b.png', 'caption')")
    conn.execute("INSERT INTO file_spatial_relations(file_id) VALUES (2)")
    conn.commit()
    conn.close()

    rows = select_reprocess_candidates(db_path, reason="missing_relations", limit=10)

    assert rows == [{"id": 1, "file_path": "/a.png", "reason": "missing_relations"}]
