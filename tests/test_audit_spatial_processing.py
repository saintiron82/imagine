import sqlite3

from tools.audit_spatial_processing import collect_spatial_processing_stats


def test_collect_spatial_processing_stats_counts_repair_targets(tmp_path):
    db_path = tmp_path / "audit.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE files(id INTEGER PRIMARY KEY, structured_meta TEXT, mc_caption TEXT)")
    conn.execute("CREATE TABLE file_objects(file_id INTEGER)")
    conn.execute("CREATE TABLE file_spatial_relations(file_id INTEGER)")
    conn.execute("CREATE TABLE file_depth_layers(file_id INTEGER)")
    conn.execute("INSERT INTO files(id, structured_meta, mc_caption) VALUES (1, '{}', 'caption')")
    conn.commit()
    conn.close()

    stats = collect_spatial_processing_stats(db_path)

    assert stats["total_files_with_caption"] == 1
    assert stats["missing_objects"] == 1
    assert stats["missing_relations"] == 1
    assert stats["missing_depth_layers"] == 1
