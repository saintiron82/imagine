import sqlite3

from tools.bench_precision import _filter_ground_truth_to_folder


def test_filter_ground_truth_to_folder_intersects_relevance_with_scope():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE files (id INTEGER, preview_only INTEGER, folder_path TEXT, file_path TEXT)"
    )
    cur.executemany(
        "INSERT INTO files VALUES (?, ?, ?, ?)",
        [
            (1, 0, "작품/호텔실내", "작품/호텔실내/a.png"),
            (2, 0, "작품/다른장소", "작품/다른장소/b.png"),
            (3, 1, "작품/호텔실내", "작품/호텔실내/preview.png"),
            (4, 0, "작품/호텔실내", "작품/호텔실내/c.png"),
        ],
    )

    assert _filter_ground_truth_to_folder(cur, {1, 2, 3, 4}, "호텔실내") == {1, 4}
