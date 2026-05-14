#!/usr/bin/env python3
"""Build a fixed complex background-only QuerySet for search evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CREATED_AT = "2026-05-06T00:00:00+09:00"
DEFAULT_OUTPUT_DIR = Path("benchmarks/querysets/background_complex_v2_20260506")
QUERYSET_NAME = "background_complex_v2"
VALID_DIFFICULTIES = {"easy", "medium", "hard", "extreme"}
NON_BACKGROUND_TERMS = {
    "캐릭터",
    "인물",
    "사람",
    "검",
    "갑옷",
    "character",
    "characters",
    "portrait",
    "sword",
    "armor",
}


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query_text: str
    scope: str
    must_terms: tuple[str, ...]
    soft_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()
    intent: str = "background_complex"
    difficulty: str = "medium"


QUERY_SPECS: tuple[QuerySpec, ...] = (
    QuerySpec(
        "bg-complex-q0001",
        "기절용사와 암살공주 #08/bg에서 밤하늘과 달이 함께 보이고 성이나 마을 조명이 있는 배경",
        "기절용사와 암살공주/#08/bg",
        ("밤하늘", "달", "조명"),
        ("성", "마을"),
        difficulty="easy",
    ),
    QuerySpec(
        "bg-complex-q0002",
        "로네느의집 거실에서 창문과 책장이 함께 보이고 생활감 있는 실내 가구가 배치된 배경",
        "로네느의집/거실",
        ("창문", "책장", "실내 가구"),
        ("거실",),
        difficulty="easy",
    ),
    QuerySpec(
        "bg-complex-q0003",
        "홍콩사무실에서 큰 창문 너머 도시 전망이 보이고 소파나 라운지 좌석이 있는 실내 배경",
        "홍콩사무실",
        ("창문", "도시 전망", "소파"),
        ("사무실", "라운지"),
        difficulty="easy",
    ),
    QuerySpec(
        "bg-complex-q0004",
        "종말투어링 #07/bg에서 교량 구조와 넓은 하늘이 함께 보이고 강이나 도로가 이어지는 야외 배경",
        "종말투어링/#07/bg",
        ("교량", "하늘"),
        ("강", "도로"),
    ),
    QuerySpec(
        "bg-complex-q0005",
        "크랑베르무 마을에서 밤하늘과 거리 조명이 함께 보이는 조용한 마을 배경",
        "크랑베르무/장소/마을",
        ("밤하늘", "거리 조명", "마을"),
    ),
    QuerySpec(
        "bg-complex-q0006",
        "후시 늪지대에서 숲과 안개가 함께 보이고 물가나 습지 분위기가 강한 배경",
        "후시/장소/늪지대",
        ("숲", "안개", "습지"),
        ("물가",),
    ),
    QuerySpec(
        "bg-complex-q0007",
        "안나의집에서 커튼과 창문이 함께 보이고 따뜻한 실내 조명이 느껴지는 방 배경",
        "안나의집",
        ("커튼", "창문", "실내 조명"),
    ),
    QuerySpec(
        "bg-complex-q0008",
        "작품 쫑 후시 장소/숲,자연에서 높은 나무와 밤 분위기가 함께 보이는 숲 배경",
        "후시/장소/숲,자연",
        ("나무", "밤", "숲"),
    ),
    QuerySpec(
        "bg-complex-q0009",
        "기절용사와 암살공주 #09/bg에서 푸른 하늘과 바다가 함께 보이는 탁 트인 해변 배경",
        "기절용사와 암살공주/#09/bg",
        ("푸른 하늘", "바다", "해변"),
    ),
    QuerySpec(
        "bg-complex-q0010",
        "크랑베르무 장소/학교교실에서 칠판이나 책상보다 창문과 교실 구조가 뚜렷한 실내 배경",
        "크랑베르무/장소/학교교실",
        ("창문", "교실", "실내 구조"),
        ("칠판", "책상"),
        difficulty="hard",
    ),
    QuerySpec(
        "bg-complex-q0011",
        "뱅가드 24 #01/bg에서 현대적인 사무실 구조와 큰 창문, 도시 풍경이 함께 보이는 배경",
        "뱅가드/24/#01/bg",
        ("사무실", "큰 창문", "도시 풍경"),
    ),
    QuerySpec(
        "bg-complex-q0012",
        "기절용사와 암살공주 장소/호텔실내에서 침대와 창문 또는 조명이 함께 보이는 호텔 방 배경",
        "기절용사와 암살공주/장소/호텔실내",
        ("침대", "창문", "조명"),
        ("호텔",),
    ),
    QuerySpec(
        "bg-complex-q0013",
        "크랑베르무 장소/신사에서 기둥과 지붕 구조가 보이고 숲이나 나무가 둘러싼 야외 배경",
        "크랑베르무/장소/신사",
        ("기둥", "지붕", "나무"),
        ("신사",),
    ),
    QuerySpec(
        "bg-complex-q0014",
        "작품 쫑 후시 장소/마법창고에서 어두운 실내와 선반 또는 상자가 함께 보이는 창고 배경",
        "후시/장소/마법창고",
        ("어두운 실내", "선반", "상자"),
        ("창고",),
    ),
    QuerySpec(
        "bg-complex-q0015",
        "사카모토 bg에서 도로와 건물 외관이 함께 보이고 도시 골목 느낌이 나는 배경",
        "사카모토/bg",
        ("도로", "건물 외관", "골목"),
        ("도시",),
    ),
    QuerySpec(
        "bg-complex-q0016",
        "켄신 자료에서 목조 교량이나 강이 보이고 전통 건축 분위기가 함께 있는 배경",
        "켄신",
        ("목조 교량", "강", "전통 건축"),
    ),
    QuerySpec(
        "bg-complex-q0017",
        "Fate BG 참고에서 어두운 숲과 밤하늘이 함께 보이는 자연 배경",
        "Fate/BG.참고",
        ("어두운 숲", "밤하늘"),
        ("자연",),
    ),
    QuerySpec(
        "bg-complex-q0018",
        "뱅가드 25 #12/bg에서 하늘과 구름 사이에 미래적인 구조물이 떠 있는 장면 배경",
        "뱅가드/25/#12/bg",
        ("하늘", "구름", "미래적 구조물"),
        difficulty="hard",
    ),
    QuerySpec(
        "bg-complex-q0019",
        "크랑베르무 장소/켐핑장에서 나무와 야외 시설이 함께 보이고 캠핑장처럼 보이는 배경",
        "크랑베르무/장소/켐핑장",
        ("나무", "야외 시설", "캠핑장"),
    ),
    QuerySpec(
        "bg-complex-q0020",
        "기절용사와 암살공주 장소/유적지에서 돌기둥과 폐허 느낌이 함께 나는 야외 배경",
        "기절용사와 암살공주/장소/유적지",
        ("돌기둥", "폐허", "야외"),
    ),
    QuerySpec(
        "bg-complex-q0021",
        "AGM_01_001_013_030 3DBG에서 학교 건물 외관과 창문, 지붕 또는 울타리가 함께 보이는 건축 배경",
        "AGM_01_001_013_030 3DBG",
        ("학교 건물", "창문", "지붕"),
        ("울타리",),
    ),
    QuerySpec(
        "bg-complex-q0022",
        "강다외부에서 콘크리트 복도와 큰 창문, 파이프가 함께 보이는 산업적인 실내 배경",
        "강다외부",
        ("콘크리트 복도", "큰 창문", "파이프"),
        ("산업적",),
    ),
    QuerySpec(
        "bg-complex-q0023",
        "실내소품에서 식물과 창문, 작은 가구가 함께 보이는 아늑한 실내 배경",
        "실내소품",
        ("식물", "창문", "작은 가구"),
        ("아늑한",),
    ),
    QuerySpec(
        "bg-complex-q0024",
        "다윈즈게임 도시 낮에서 도로와 건물이 보이고 낮 시간의 도시 분위기가 분명한 배경",
        "다윈즈게임/장소/도시 낮",
        ("도로", "건물", "낮"),
        ("도시",),
    ),
    QuerySpec(
        "bg-complex-q0025",
        "기절용사와 암살공주 #12/bg에서 발코니나 난간과 하늘이 함께 보이는 높은 장소 배경",
        "기절용사와 암살공주/#12/bg",
        ("발코니", "난간", "하늘"),
    ),
    QuerySpec(
        "bg-complex-q0026",
        "크랑베르무 장소/달에서 달 표면이나 크레이터 느낌이 나고 보라색 또는 푸른색 톤이 강한 배경",
        "크랑베르무/장소/달",
        ("달 표면", "크레이터"),
        ("보라색", "푸른색"),
        difficulty="hard",
    ),
    QuerySpec(
        "bg-complex-q0027",
        "뱅가드 26 #05/bg에서 분수와 로비 구조가 함께 보이는 현대적인 실내 배경",
        "뱅가드/26/#05/bg",
        ("분수", "로비", "현대적 실내"),
    ),
    QuerySpec(
        "bg-complex-q0028",
        "기절용사와 암살공주 장소/동굴에서 어두운 바위 벽과 통로가 함께 보이는 지하 배경",
        "기절용사와 암살공주/장소/동굴",
        ("바위 벽", "통로", "어두운 분위기"),
        ("지하",),
    ),
    QuerySpec(
        "bg-complex-q0029",
        "실내소품에서 카페 조명과 의자, 작은 테이블이 함께 보이는 아늑한 실내 배경",
        "실내소품",
        ("카페 조명", "의자", "작은 테이블"),
        ("아늑한 실내",),
    ),
    QuerySpec(
        "bg-complex-q0030",
        "크랑베르무 장소/마을에서 낮 시간의 거리와 건물, 하늘이 함께 보이는 밝은 마을 배경",
        "크랑베르무/장소/마을",
        ("거리", "건물", "하늘"),
        ("낮",),
    ),
    QuerySpec(
        "bg-complex-q0031",
        "기절용사와 암살공주 #08/bg에서 달이 화면 오른쪽 위에 있고 성 지붕과 창문 조명이 보이지만 전경 피사체는 없는 밤 배경",
        "기절용사와 암살공주/#08/bg",
        ("달", "성 지붕", "창문 조명", "오른쪽 위"),
        ("밤하늘",),
        ("캐릭터", "인물"),
        difficulty="extreme",
    ),
    QuerySpec(
        "bg-complex-q0032",
        "홍콩사무실에서 소파가 화면 하단 쪽에 있고 큰 창문 너머 야경이 보이지만 책상 중심 장면은 아닌 실내 배경",
        "홍콩사무실",
        ("소파", "큰 창문", "야경", "하단"),
        ("라운지",),
        ("책상", "인물"),
        difficulty="hard",
    ),
    QuerySpec(
        "bg-complex-q0033",
        "크랑베르무 장소/학교교실에서 창문 빛과 빈 책상, 칠판 구조가 함께 보이지만 인물이 없는 낮 실내 배경",
        "크랑베르무/장소/학교교실",
        ("창문 빛", "빈 책상", "칠판", "낮"),
        ("교실 구조",),
        ("인물", "캐릭터"),
        difficulty="hard",
    ),
    QuerySpec(
        "bg-complex-q0034",
        "후시 늪지대에서 안개와 물가, 비뚤어진 나무가 함께 보이고 밝은 하늘이나 건물은 없는 어두운 자연 배경",
        "후시/장소/늪지대",
        ("안개", "물가", "비뚤어진 나무", "어두운 자연"),
        ("습지",),
        ("밝은 하늘", "건물"),
        difficulty="extreme",
    ),
    QuerySpec(
        "bg-complex-q0035",
        "기절용사와 암살공주 장소/동굴에서 바위 벽과 깊은 통로, 푸른 조명이 보이지만 전투 장면은 아닌 지하 배경",
        "기절용사와 암살공주/장소/동굴",
        ("바위 벽", "깊은 통로", "푸른 조명", "지하"),
        ("원근감",),
        ("전투", "캐릭터"),
        difficulty="hard",
    ),
    QuerySpec(
        "bg-complex-q0036",
        "AGM_01_001_013_030 3DBG에서 건물 정면과 반복되는 창문, 울타리가 보이지만 실내 장면은 아닌 학교 외관 배경",
        "AGM_01_001_013_030 3DBG",
        ("건물 정면", "반복 창문", "울타리", "학교 외관"),
        ("지붕",),
        ("실내", "인물"),
        difficulty="hard",
    ),
    QuerySpec(
        "bg-complex-q0037",
        "실내소품에서 창문 빛과 식물, 작은 의자가 한 화면에 있지만 카페 조명 중심은 아닌 조용한 실내 배경",
        "실내소품",
        ("창문 빛", "식물", "작은 의자", "조용한 실내"),
        ("소품",),
        ("카페 조명", "인물"),
        difficulty="extreme",
    ),
    QuerySpec(
        "bg-complex-q0038",
        "뱅가드 25 #12/bg에서 구름 사이 미래적 구조물이 떠 있고 넓은 하늘이 보이지만 도시 거리나 실내는 아닌 배경",
        "뱅가드/25/#12/bg",
        ("구름", "미래적 구조물", "넓은 하늘", "떠 있는 장면"),
        ("공중",),
        ("도시 거리", "실내"),
        difficulty="extreme",
    ),
    QuerySpec(
        "bg-complex-q0039",
        "켄신 자료에서 목조 교량과 강, 전통 건축이 함께 보이지만 현대식 건물이나 밤 장면은 아닌 야외 배경",
        "켄신",
        ("목조 교량", "강", "전통 건축", "야외"),
        ("고전적 분위기",),
        ("현대식 건물", "밤"),
        difficulty="hard",
    ),
    QuerySpec(
        "bg-complex-q0040",
        "다윈즈게임 도시 낮에서 도로와 건물 외관, 밝은 낮하늘이 함께 보이지만 실내나 야경이나 인물 중심은 아닌 도시 배경",
        "다윈즈게임/장소/도시 낮",
        ("도로", "건물 외관", "밝은 낮하늘", "도시 배경"),
        ("낮",),
        ("실내", "야경", "인물"),
        difficulty="extreme",
    ),
)


def _intent_key(spec: QuerySpec) -> str:
    return "|".join(sorted(term.lower() for term in spec.must_terms))


def validate_specs(specs: tuple[QuerySpec, ...]) -> None:
    ids: set[str] = set()
    intents: set[str] = set()
    for spec in specs:
        if spec.query_id in ids:
            raise ValueError(f"duplicate query_id: {spec.query_id}")
        ids.add(spec.query_id)
        if spec.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"{spec.query_id}: invalid difficulty: {spec.difficulty}")
        if not spec.must_terms:
            raise ValueError(f"{spec.query_id}: must_terms is empty")
        intent = _intent_key(spec)
        if intent in intents:
            raise ValueError(f"{spec.query_id}: duplicate intent: {intent}")
        intents.add(intent)
        positive_terms = {term.lower() for term in (*spec.must_terms, *spec.soft_terms)}
        blocked = sorted(positive_terms & NON_BACKGROUND_TERMS)
        if blocked:
            raise ValueError(f"{spec.query_id}: non-background term(s): {', '.join(blocked)}")
        if len(spec.must_terms) < 2:
            raise ValueError(f"{spec.query_id}: complex queries need at least 2 must_terms")
        if spec.difficulty == "extreme" and (len(spec.must_terms) < 4 or not spec.exclude_terms):
            raise ValueError(f"{spec.query_id}: extreme queries need at least 4 must_terms and exclude_terms")


def build_rows(*, created_at: str = DEFAULT_CREATED_AT) -> list[dict[str, Any]]:
    validate_specs(QUERY_SPECS)
    rows = []
    for spec in QUERY_SPECS:
        rows.append({
            "query_id": spec.query_id,
            "query_text": spec.query_text,
            "query_type": "complex",
            "locale": "ko-KR",
            "created_at": created_at,
            "intent": spec.intent,
            "difficulty": spec.difficulty,
            "scope": spec.scope,
            "must_terms": list(spec.must_terms),
            "soft_terms": list(spec.soft_terms),
            "exclude_terms": list(spec.exclude_terms),
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
        "# Background Complex QuerySet V2",
        "",
        "배경/장소/장면 검색 품질을 보기 위한 난이도 분산 복합 질문지입니다.",
        "각 질문은 범위(scope), 필수 조건, 보조 조건, 제외 조건, 난이도를 포함할 수 있습니다.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['query_id']} [{row['difficulty']}]")
        lines.append("")
        lines.append(row["query_text"])
        lines.append("")
        lines.append(f"- scope: {row['scope']}")
        lines.append(f"- difficulty: {row['difficulty']}")
        lines.append(f"- must: {', '.join(row['must_terms'])}")
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
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT)
    args = parser.parse_args(argv)

    rows = build_rows(created_at=args.created_at)
    difficulty_counts = Counter(str(row["difficulty"]) for row in rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "queryset.jsonl", rows)
    write_markdown(args.output_dir / "questions.md", rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({
            "name": QUERYSET_NAME,
            "created_at": args.created_at,
            "query_count": len(rows),
            "query_type": "complex",
            "locale": "ko-KR",
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
            "purpose": "background/scene search evaluation with easy-to-extreme compound prompts and hard negatives",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} queries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
