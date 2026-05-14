import csv
import re
import sqlite3
from pathlib import Path

from tools.build_metadata_review_sample import (
    build_sample,
    load_candidates,
    translate_caption,
    translated_tag_display,
    write_csv,
)


ENGLISH_TOKEN_RE = r"[A-Za-z][A-Za-z0-9_-]*"


def make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            file_name TEXT,
            relative_path TEXT,
            folder_path TEXT,
            thumbnail_url TEXT,
            format TEXT,
            width INTEGER,
            height INTEGER,
            mc_caption TEXT,
            ai_tags TEXT,
            image_type TEXT,
            scene_type TEXT,
            time_of_day TEXT,
            weather TEXT,
            caption_model TEXT,
            processing_status TEXT,
            processing_error TEXT,
            content_hash TEXT,
            perceptual_hash INTEGER,
            dup_group_id INTEGER,
            preview_only INTEGER
        )
        """
    )
    rows = [
        (1, "/a/one.psd", "one.psd", "one.psd", "A", "/tmp/one.png", "psd", 100, 100, "A classroom.", '["classroom"]', "background", "", "", "", "model", "vision_done", "", "same", 11, None, 0),
        (2, "/a/two.psd", "two.psd", "two.psd", "A", "/tmp/two.png", "psd", 100, 100, "A duplicate.", '["classroom"]', "background", "", "", "", "model", "vision_done", "", "same", 12, None, 0),
        (3, "/b/three.psd", "three.psd", "three.psd", "B", "/tmp/three.png", "psd", 100, 100, "A forest.", '["forest"]', "background", "", "", "", "model", "vision_done", "", "other", 13, None, 0),
        (4, "/c/four.psd", "four.psd", "four.psd", "C", "/tmp/four.png", "psd", 100, 100, "unknown", "[]", "background", "", "", "", "model", "vision_done", "", "missing", 14, None, 0),
    ]
    conn.executemany(
        """
        INSERT INTO files VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_load_candidates_dedupes_and_skips_missing_by_default(tmp_path: Path):
    db_path = tmp_path / "test.db"
    make_db(db_path)

    candidates = load_candidates(db_path)

    assert [row["item_id"] for row in candidates] == ["1", "3"]
    assert {row["analysis_status"] for row in candidates} == {"ok"}


def test_build_sample_and_write_csv(tmp_path: Path):
    db_path = tmp_path / "test.db"
    make_db(db_path)
    candidates = load_candidates(db_path, include_missing=True)

    sample = build_sample(candidates, count=3, seed=7, max_per_source=1)
    output = tmp_path / "sample.csv"
    write_csv(output, sample)

    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert rows[0]["sample_id"].startswith("metadata-quality-v1-")
    assert "caption_alignment" in rows[0]
    assert "tag_alignment" in rows[0]
    assert "overall_alignment" in rows[0]
    assert "caption_ko_display" in rows[0]
    assert "tags_ko_display" in rows[0]
    assert rows[0]["caption_ko_display"]
    assert any(row["tags_ko_display"] for row in rows)


def test_translate_caption_handles_review_gallery_common_patterns():
    assert (
        translate_caption("A translucent anime-style man stands in a lush green landscape under a blue sky.")
        == "무성한 초록 풍경 안에 반투명한 애니메이션풍 남성이 서 있고, 푸른 하늘 아래입니다."
    )
    assert (
        translate_caption("Three translucent anime characters in a blue-lit stadium at night.")
        == "밤의 푸른 조명이 비치는 경기장 안에 세 반투명한 애니메이션 캐릭터가 있습니다."
    )
    assert not re.search(
        ENGLISH_TOKEN_RE,
        translate_caption("A colorful shelf display with various jars, bottles, and packaged goods."),
    )
    assert not re.search(
        ENGLISH_TOKEN_RE,
        translate_caption("Anime characters and robot under rainbow with sparkles and swirls."),
    )


def test_translated_tag_display_handles_motion_and_intensity_tags():
    assert translated_tag_display('["motion blur", "purple", "dynamic", "high intensity"]') == "모션 블러 · 보라색 · 역동적인 · 고강도"


def test_translated_tag_display_skips_noise_and_keeps_korean_only():
    display = translated_tag_display(
        '["grb06_256.psd", "stars", "imagine_dl_2dsrwx33", "purple glow", "sci-fi"]'
    )

    assert display == "별 · 보라색 광채 · 공상과학"
    assert not re.search(ENGLISH_TOKEN_RE, display)
