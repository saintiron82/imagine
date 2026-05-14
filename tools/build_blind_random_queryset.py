#!/usr/bin/env python3
"""Build blind random search questions without looking at image captions/tags."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CREATED_AT = "2026-05-07T00:00:00+09:00"
DEFAULT_OUTPUT_DIR = Path("benchmarks/querysets/blind_random_v1_20260507")
QUERYSET_NAME = "blind_random_v1"

GENERIC_FOLDER_SEGMENTS = {
    "",
    "bg",
    "BG",
    "이미지",
    "image",
    "images",
    "file",
    "files",
    "자료",
    "장소",
    "작품",
    "발송",
    "예비",
}

VISUAL_TERMS = (
    "밤하늘",
    "달",
    "별",
    "구름",
    "푸른 하늘",
    "노을",
    "거리 조명",
    "창문",
    "건물 외관",
    "도로",
    "골목",
    "다리",
    "강",
    "바다",
    "숲",
    "나무",
    "안개",
    "비",
    "눈",
    "실내 조명",
    "소파",
    "책장",
    "책상",
    "침대",
    "커튼",
    "선반",
    "벽돌 벽",
    "돌기둥",
    "폐허",
    "동굴 벽",
    "계단",
    "난간",
    "간판",
    "전선",
    "가로등",
)

SCENE_QUALIFIERS = (
    "야외",
    "실내",
    "도시",
    "자연",
    "마을",
    "학교",
    "상점가",
    "주거지",
    "산업 시설",
    "전통 건축",
    "현대 건축",
    "어두운",
    "밝은 낮",
    "비 오는",
    "조용한",
)

EXCLUDE_TERMS = (
    "캐릭터",
    "인물",
    "텍스트 중심",
    "소품 단독",
    "근접 얼굴",
    "전투 장면",
    "추상 텍스처",
)


@dataclass(frozen=True)
class ScopeCandidate:
    scope: str
    file_count: int


def _folder_scope_name(folder_path: str) -> str:
    return str(folder_path or "").replace("webdav://13730b09/", "").replace("webdav:/13730b09/", "").strip("/")


def _is_good_scope(scope: str, count: int, *, min_files: int, max_files: int) -> bool:
    if count < min_files or count > max_files:
        return False
    segments = [part.strip() for part in scope.replace("\\", "/").split("/") if part.strip()]
    if not segments:
        return False
    if all(segment in GENERIC_FOLDER_SEGMENTS for segment in segments):
        return False
    leaf = segments[-1]
    if leaf in GENERIC_FOLDER_SEGMENTS and len(segments) < 2:
        return False
    return True


def load_scope_candidates(
    db_path: Path,
    *,
    min_files: int = 8,
    max_files: int = 300,
) -> list[ScopeCandidate]:
    """Load folder scopes using only path/count metadata, not visual labels."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
        if "folder_path" not in columns:
            return []
        rows = conn.execute(
            """
            SELECT folder_path, COUNT(*) AS file_count
            FROM files
            WHERE COALESCE(preview_only, 0) = 0
              AND COALESCE(folder_path, '') != ''
            GROUP BY folder_path
            ORDER BY folder_path
            """
        ).fetchall()
    finally:
        conn.close()

    candidates = []
    seen = set()
    for row in rows:
        scope = _folder_scope_name(str(row["folder_path"] or ""))
        count = int(row["file_count"] or 0)
        if scope in seen or not _is_good_scope(scope, count, min_files=min_files, max_files=max_files):
            continue
        seen.add(scope)
        candidates.append(ScopeCandidate(scope=scope, file_count=count))
    return candidates


def _sample_terms(rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    must_count = rng.choice((2, 3, 3, 4))
    must_terms = rng.sample(list(VISUAL_TERMS), must_count)
    soft_terms = rng.sample(list(SCENE_QUALIFIERS), rng.choice((0, 1, 1, 2)))
    exclude_terms = rng.sample(list(EXCLUDE_TERMS), rng.choice((0, 1, 1, 2)))
    return must_terms, soft_terms, exclude_terms


def _condition_text(rng: random.Random, must_terms: list[str], soft_terms: list[str], exclude_terms: list[str]) -> str:
    terms = must_terms[:]
    rng.shuffle(terms)
    if len(terms) >= 4:
        base = f"다음 요소가 한 화면에 함께 보이는 배경: {', '.join(terms)}"
    elif len(terms) == 3:
        base = f"다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: {', '.join(terms)}"
    else:
        base = f"다음 요소가 함께 보이는 배경: {', '.join(terms)}"

    if soft_terms:
        base += f" ({', '.join(soft_terms)} 분위기)"
    if exclude_terms:
        base += f" / 제외 조건: {', '.join(exclude_terms)}"
    return base


def build_rows(
    *,
    count: int,
    seed: int,
    created_at: str = DEFAULT_CREATED_AT,
    scopes: list[ScopeCandidate] | None = None,
    scope_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if not 0 <= scope_ratio <= 1:
        raise ValueError("scope_ratio must be between 0 and 1")

    rng = random.Random(seed)
    scopes = list(scopes or [])
    rows = []
    used_intents = set()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise ValueError("failed to generate enough unique blind random queries")
        must_terms, soft_terms, exclude_terms = _sample_terms(rng)
        intent_key = tuple(sorted([*must_terms, *soft_terms, *exclude_terms]))
        if intent_key in used_intents:
            continue
        used_intents.add(intent_key)

        scope = ""
        if scopes and rng.random() < scope_ratio:
            scope = rng.choice(scopes).scope
        condition = _condition_text(rng, must_terms, soft_terms, exclude_terms)
        query_text = f"{scope}에서 {condition}" if scope else condition
        query_id = f"blind-random-q{len(rows) + 1:04d}"
        difficulty = "hard" if exclude_terms and len(must_terms) >= 3 else "medium" if len(must_terms) >= 3 else "easy"
        rows.append({
            "query_id": query_id,
            "query_text": query_text,
            "query_type": "blind_random",
            "locale": "ko-KR",
            "created_at": created_at,
            "intent": "blind_random_search",
            "difficulty": difficulty,
            "scope": scope,
            "must_terms": must_terms,
            "soft_terms": soft_terms,
            "exclude_terms": exclude_terms,
            "generation": "blind_random_v1",
            "random_seed": seed,
        })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Blind Random QuerySet V1",
        "",
        "이미지, 캡션, 태그를 보지 않고 고정 조건 사전과 랜덤 시드로 만든 검색 질문지입니다.",
        "scope가 있는 경우에도 DB 경로명/파일 수만 사용하며 이미지 내용은 보지 않습니다.",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## {row['query_id']} [{row['difficulty']}]",
            "",
            row["query_text"],
            "",
            f"- scope: {row['scope'] or '-'}",
            f"- must: {', '.join(row['must_terms'])}",
        ])
        if row["soft_terms"]:
            lines.append(f"- soft: {', '.join(row['soft_terms'])}")
        if row["exclude_terms"]:
            lines.append(f"- exclude: {', '.join(row['exclude_terms'])}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-path", type=Path, default=Path("imageparser.db"))
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT)
    parser.add_argument("--scope-ratio", type=float, default=0.5)
    parser.add_argument("--min-scope-files", type=int, default=8)
    parser.add_argument("--max-scope-files", type=int, default=300)
    args = parser.parse_args(argv)

    try:
        scopes = load_scope_candidates(
            args.db_path,
            min_files=args.min_scope_files,
            max_files=args.max_scope_files,
        )
        rows = build_rows(
            count=args.count,
            seed=args.seed,
            created_at=args.created_at,
            scopes=scopes,
            scope_ratio=args.scope_ratio,
        )
        difficulty_counts = Counter(str(row["difficulty"]) for row in rows)
        scoped_count = sum(1 for row in rows if row["scope"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output_dir / "queryset.jsonl", rows)
        write_markdown(args.output_dir / "questions.md", rows)
        (args.output_dir / "manifest.json").write_text(
            json.dumps({
                "name": QUERYSET_NAME,
                "created_at": args.created_at,
                "query_count": len(rows),
                "query_type": "blind_random",
                "locale": "ko-KR",
                "seed": args.seed,
                "scope_ratio": args.scope_ratio,
                "scope_candidate_count": len(scopes),
                "scoped_query_count": scoped_count,
                "difficulty_counts": dict(sorted(difficulty_counts.items())),
                "purpose": "blind random search evaluation independent of image/caption/tag content",
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} blind random queries to {args.output_dir}")
    print(f"Scoped queries: {scoped_count}; scope candidates: {len(scopes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
