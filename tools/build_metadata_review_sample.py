#!/usr/bin/env python3
"""Build a human review sample for validating AI captions and tags.

This sample is image-first. It does not create search questions. The goal is to
validate whether existing ``mc_caption`` and ``ai_tags`` values are trustworthy
enough to become the basis for later answerable search benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CREATED_AT = "2026-05-10T00:00:00+09:00"
DEFAULT_OUTPUT_DIR = Path("benchmarks/reviews/metadata_quality_v1_20260510")
DEFAULT_SAMPLE_ID_PREFIX = "metadata-quality-v1"

REVIEW_FIELDS = [
    "caption_alignment",
    "tag_alignment",
    "overall_alignment",
    "issue_types",
    "reviewer_notes",
    "reviewer_id",
    "reviewed_at",
]

DISPLAY_FIELDS = [
    "caption_ko_display",
    "tags_ko_display",
]

ITEM_FIELDS = [
    "item_id",
    "file_path",
    "file_name",
    "relative_path",
    "folder_path",
    "thumbnail_url",
    "format",
    "width",
    "height",
    "mc_caption",
    "ai_tags",
    "image_type",
    "scene_type",
    "time_of_day",
    "weather",
    "caption_model",
    "processing_status",
    "processing_error",
    "content_hash",
    "perceptual_hash",
    "dup_group_id",
]

CSV_FIELDS = [
    "sample_id",
    "sample_version",
    "sample_created_at",
    "sample_seed",
    "sample_rank",
    "dedupe_key",
    "source_group",
    "analysis_status",
    "analysis_issue",
    *ITEM_FIELDS,
    *DISPLAY_FIELDS,
    *REVIEW_FIELDS,
]

TERM_KO = {
    "abstract": "추상적인",
    "anatomical lines": "해부학적 선",
    "anime": "애니풍",
    "anime character": "애니메이션 캐릭터",
    "anime characters": "애니메이션 캐릭터",
    "anime style": "애니메이션풍",
    "anime-style": "애니메이션풍",
    "aurora-like lights": "오로라 같은 빛",
    "architecture": "건축",
    "background": "배경",
    "blue": "푸른색",
    "blue-lit": "푸른 조명이 비치는",
    "blue lighting": "푸른 조명",
    "blue sky": "푸른 하늘",
    "blueprint": "청사진",
    "book": "책",
    "bookshelf": "책장",
    "building": "건물",
    "building facade": "건물 외관",
    "buildings": "건물",
    "canyon": "협곡",
    "ceiling": "천장",
    "character": "캐릭터",
    "characters": "캐릭터",
    "chalkboard": "칠판",
    "children's drawings": "어린이 그림",
    "cityscape": "도시 풍경",
    "classroom": "교실",
    "closed eyes": "감은 눈",
    "clouds": "구름",
    "cobblestone pavement": "자갈 포장 바닥",
    "color distortion effects": "색상 왜곡 효과",
    "concrete": "콘크리트",
    "cool twilight light": "차가운 황혼빛",
    "courtyard": "안뜰",
    "creature": "생물",
    "decorative lines": "장식적인 선",
    "desk": "책상",
    "dimly lit": "어둡게 조명된",
    "digital": "디지털",
    "digital art": "디지털 아트",
    "digital sketch": "디지털 스케치",
    "dragon": "용",
    "dungeon": "던전",
    "facade": "외관",
    "fantasy": "판타지",
    "flowing hair": "흐르는 머리카락",
    "floor": "바닥",
    "foot": "발",
    "four": "네",
    "forest": "숲",
    "geometric": "기하학적",
    "giant monster": "거대한 괴물",
    "glowing windows": "빛나는 창문",
    "gothic": "고딕풍",
    "green": "초록",
    "green landscape": "초록 풍경",
    "hand sketch": "손그림 스케치",
    "ice cream": "아이스크림",
    "interior scene": "실내 장면",
    "intricate carvings": "정교한 조각",
    "japanese": "일본식",
    "kimono": "기모노",
    "landscape": "풍경",
    "lava": "용암",
    "lines": "선",
    "lush": "무성한",
    "lush green landscape": "무성한 초록 풍경",
    "man": "남성",
    "mecha dragon": "메카 드래곤",
    "mechanical": "기계적인",
    "medieval courtyard": "중세풍 안뜰",
    "misty": "안개 낀",
    "monster": "괴물",
    "moonlit": "달빛이 비치는",
    "moonlit medieval courtyard": "달빛이 비치는 중세풍 안뜰",
    "motion blur effect": "모션 블러 효과",
    "muted tones": "차분한 색조",
    "mysterious": "신비로운",
    "night": "밤",
    "night sky": "밤하늘",
    "nostalgic rural atmosphere": "향수를 불러일으키는 시골 분위기",
    "notice board": "게시판",
    "one": "한",
    "overlays": "오버레이",
    "pavement": "포장 바닥",
    "pastel tones": "파스텔 톤",
    "pedestrian crossing": "횡단보도",
    "purple": "보라색",
    "purple glowing accents": "보라색으로 빛나는 강조 요소",
    "purple wall": "보라색 벽",
    "quiet": "조용한",
    "red": "붉은",
    "red-tinted illustration": "붉은 색조의 일러스트",
    "rocky cliff": "바위 절벽",
    "rural atmosphere": "시골 분위기",
    "portrait": "인물화",
    "school": "학교",
    "serene": "고요한",
    "shelf": "선반",
    "signpost": "표지판",
    "sketch": "스케치",
    "skyscrapers": "고층 건물",
    "sky": "하늘",
    "snowy autumn landscape": "눈 덮인 가을 풍경",
    "soft pastel tones": "부드러운 파스텔 톤",
    "solitary signpost": "홀로 선 표지판",
    "stadium": "경기장",
    "stand": "서 있음",
    "standing": "서 있는",
    "stands": "서 있음",
    "starry": "별이 있는",
    "starry night sky": "별이 있는 밤하늘",
    "starry sky": "별이 있는 하늘",
    "stone buildings": "석조 건물",
    "street scene": "거리 장면",
    "structural": "구조적인",
    "structural lines": "구조선",
    "thatched house": "초가집",
    "three": "세",
    "torso": "몸통",
    "translucent": "반투명한",
    "translucent figures": "반투명한 인물",
    "traditional building": "전통 건물",
    "traditional japanese thatched house": "전통적인 일본식 초가집",
    "tree": "나무",
    "trees": "나무",
    "two": "두",
    "urban setting": "도시 환경",
    "vertical color distortion effects": "세로 색상 왜곡 효과",
    "warm-toned": "따뜻한 색감의",
    "white foliage": "흰 잎",
    "winged": "날개 달린",
    "window": "창문",
    "windows": "창문",
    "wooden notice board": "나무 게시판",
    "wooden panels": "나무 패널",
    "angel": "천사",
    "angular": "각진",
    "architectural": "건축적인",
    "architectural illustration": "건축 일러스트",
    "art": "아트",
    "atmosphere": "분위기",
    "backdrop": "배경",
    "bedroom": "침실",
    "bench": "벤치",
    "beside": "옆",
    "blurred": "흐릿한",
    "brick": "벽돌",
    "brick wall": "벽돌 벽",
    "bright": "밝은",
    "brown tones": "갈색 톤",
    "candle": "양초",
    "cat ears": "고양이 귀",
    "central counter": "중앙 계산대",
    "clear": "맑은",
    "cliffside village": "절벽 마을",
    "cloudy": "구름 낀",
    "colorful": "화려한",
    "color grading": "색보정",
    "colorful fabrics": "화려한 직물",
    "cozy": "아늑한",
    "cozy bedroom": "아늑한 침실",
    "dark": "어두운",
    "dark cave": "어두운 동굴",
    "dark horizon": "어두운 지평선",
    "decorative": "장식적인",
    "deep": "짙은",
    "dense urban buildings": "빽빽한 도시 건물",
    "diagonal light rays": "대각선 빛줄기",
    "diagonal planks": "대각선 판자",
    "document": "문서",
    "door": "문",
    "doors": "문",
    "dynamic": "역동적인",
    "eerie": "섬뜩한",
    "empty street": "빈 거리",
    "energy": "에너지",
    "ethereal lines": "몽환적인 선",
    "facades": "외벽",
    "fabrics": "직물",
    "faint": "희미한",
    "fiery energy": "불타는 에너지",
    "figure": "인물",
    "figures": "인물",
    "flickering lanterns": "깜빡이는 등불",
    "fluffy clouds": "폭신한 구름",
    "futuristic": "미래적인",
    "futuristic arena": "미래적인 경기장",
    "glimpse": "언뜻 보이는 모습",
    "glowing": "빛나는",
    "glowing light": "빛나는 빛",
    "glowing stars": "빛나는 별",
    "golden": "금빛",
    "golden door": "금빛 문",
    "golden light": "금빛",
    "grand": "웅장한",
    "grand hallway": "웅장한 복도",
    "grain texture": "나뭇결 질감",
    "grass": "풀",
    "greenery": "녹음",
    "hanging": "걸려 있는",
    "hanging curtain": "걸린 커튼",
    "handle": "손잡이",
    "high intensity": "고강도",
    "high-intensity": "고강도",
    "horizontal bands": "수평 띠",
    "illustration": "일러스트",
    "illuminated": "비춰진",
    "indoor": "실내",
    "intense": "강렬한",
    "interior": "실내",
    "lavender walls": "라벤더색 벽",
    "large": "큰",
    "large boulders": "큰 바위",
    "layered": "겹쳐진",
    "layered line art": "겹쳐진 선화",
    "line art": "선화",
    "lineart": "선화",
    "lit": "조명된",
    "map": "지도",
    "modern": "현대적인",
    "motion blur": "모션 블러",
    "mountains": "산",
    "multiple spotlights": "여러 개의 스포트라이트",
    "neon lighting effects": "네온 조명 효과",
    "open doorway": "열린 출입구",
    "orb": "구체",
    "ornate": "화려한",
    "ornate doors": "화려한 문",
    "outline": "윤곽",
    "palm": "야자수",
    "pink": "분홍색",
    "plush rabbit toy": "푹신한 토끼 인형",
    "potted plant": "화분",
    "radiant explosion": "빛나는 폭발",
    "reading": "글자가 적힌",
    "river": "강",
    "rocky": "바위가 많은",
    "rocky terrain": "바위 지형",
    "room": "방",
    "scene": "장면",
    "setting": "환경",
    "shelves": "선반",
    "sign": "간판",
    "sketch-style illustration": "스케치풍 일러스트",
    "sketchy": "스케치풍",
    "skull-like face": "해골 같은 얼굴",
    "sliding door": "미닫이문",
    "small": "작은",
    "small angel": "작은 천사",
    "soft": "부드러운",
    "spiky hair": "뾰족한 머리카락",
    "stage": "무대",
    "stage lighting setup": "무대 조명 장치",
    "stone": "돌",
    "stone ground": "돌바닥",
    "structures": "구조물",
    "style": "스타일",
    "stylized": "양식화된",
    "superimposed": "겹쳐진",
    "supernatural": "초자연적인",
    "surface": "표면",
    "table": "테이블",
    "table surface": "테이블 표면",
    "tassel": "술 장식",
    "texture": "질감",
    "textured": "질감 있는",
    "tiled": "타일로 된",
    "tiled floors": "타일 바닥",
    "tiled path": "타일 길",
    "traditional chinese signage": "전통 중국식 간판",
    "tranquil": "고요한",
    "truss structure": "트러스 구조물",
    "twilight": "황혼",
    "urban": "도시의",
    "vibrant": "선명한",
    "visible": "보이는",
    "warm": "따뜻한",
    "warm lights": "따뜻한 조명",
    "warmly lit": "따뜻하게 조명된",
    "white": "흰색",
    "wooden": "나무",
    "wooden bench": "나무 벤치",
    "wooden furniture": "나무 가구",
    "wooden interior": "나무 실내",
    "wooden surface": "나무 표면",
    "wooden table": "나무 테이블",
    "wooden-floored boutique": "나무 바닥의 부티크",
    "wooden-paneled room": "나무 패널 방",
    "woven baskets": "짠 바구니",
    "aerial": "공중에서 본",
    "alleyway": "골목길",
    "ancient": "오래된",
    "appear": "나타남",
    "appears": "나타남",
    "arched": "아치형",
    "arched ceilings": "아치형 천장",
    "aurora": "오로라",
    "aurora-like": "오로라 같은",
    "aurora-like light streaks": "오로라 같은 빛줄기",
    "bed": "침대",
    "behind": "뒤",
    "blending into": "배경에 녹아드는",
    "books": "책",
    "brown": "갈색",
    "bust pose": "상반신 포즈",
    "cabinet": "수납장",
    "catgirl": "고양이귀 소녀",
    "cave": "동굴",
    "ceilings": "천장",
    "cheerful": "밝은 분위기",
    "chinese signage": "중국식 간판",
    "city": "도시",
    "city street": "도시 거리",
    "cliff": "절벽",
    "cloak": "망토",
    "closet": "수납장",
    "color outlines": "색상 윤곽선",
    "color tones": "색조",
    "columns": "기둥",
    "composition": "구도",
    "concept art": "콘셉트 아트",
    "cool color palette": "차가운 색상 팔레트",
    "cool palette": "차가운 색감",
    "corridor": "복도",
    "cosmic": "우주적인",
    "copper base": "구리색 하단부",
    "crystal": "수정",
    "dappled light": "얼룩진 빛",
    "dappled sunlight": "얼룩진 햇빛",
    "daytime": "낮",
    "design": "디자인",
    "design elements": "디자인 요소",
    "distant": "멀리 보이는",
    "dome structure": "돔 구조물",
    "drawing": "그림",
    "dreamscape": "몽환적인 풍경",
    "dreamy": "몽환적인",
    "empty": "비어 있는",
    "epic": "장대한",
    "ethereal": "몽환적인",
    "european": "유럽풍",
    "european village": "유럽풍 마을",
    "explosion": "폭발",
    "expressive eyes": "표정이 풍부한 눈",
    "fan art": "팬아트",
    "fence": "울타리",
    "field": "들판",
    "filled": "가득 찬",
    "flowers": "꽃",
    "floating rock formations": "떠 있는 암석 지형",
    "form": "형태",
    "frame": "프레임",
    "frame guidelines": "프레임 가이드라인",
    "from": "에서",
    "garden": "정원",
    "gate": "문",
    "ghost": "유령",
    "ghostly": "유령 같은",
    "girl": "소녀",
    "girls": "소녀들",
    "gold trim": "금색 장식",
    "green hills": "초록 언덕",
    "group": "무리",
    "guidelines": "가이드라인",
    "hair": "머리카락",
    "hallway": "복도",
    "hand reaching out": "뻗은 손",
    "high speed": "빠른 속도",
    "hooded": "후드를 쓴",
    "hooded cloak": "후드 달린 망토",
    "horror": "공포",
    "horizon": "지평선",
    "hovering": "떠 있는",
    "hues": "색조",
    "human": "사람",
    "ice cream": "아이스크림",
    "industrial interior": "산업 시설 내부",
    "japanese text": "일본어 텍스트",
    "kitchen": "주방",
    "lamp": "램프",
    "lamp post": "가로등",
    "labeled": "표시된",
    "lattice design": "격자 디자인",
    "leads to": "이어지는",
    "leaves": "잎",
    "library": "도서관",
    "light beams": "빛줄기",
    "light coloring": "옅은 채색",
    "light filtering": "빛이 스며드는",
    "light rays": "빛줄기",
    "lighting": "조명",
    "line drawing": "선화",
    "long hair": "긴 머리카락",
    "lone": "홀로 있는",
    "lying": "누워 있는",
    "market": "시장",
    "market scene": "시장 장면",
    "melancholic": "쓸쓸한",
    "metal": "금속",
    "metal armrests": "금속 팔걸이",
    "metal gate": "금속 문",
    "metal wall": "금속 벽",
    "metalwork": "금속 장식",
    "mood": "분위기",
    "moody": "어두운 분위기의",
    "mountain": "산",
    "mountain peaks": "산봉우리",
    "mouse": "쥐",
    "mystery": "미스터리",
    "natural": "자연스러운",
    "natural environment": "자연 환경",
    "nature": "자연",
    "nebula": "성운",
    "neon": "네온",
    "nighttime": "밤",
    "orange": "주황색",
    "outdoor": "야외",
    "outdoor scene": "야외 장면",
    "outside": "바깥",
    "overlay": "오버레이",
    "paneling": "패널 마감",
    "panels": "패널",
    "paper screen": "종이문",
    "pastel palette": "파스텔 팔레트",
    "path": "길",
    "patterned": "무늬가 있는",
    "person": "사람",
    "poles": "기둥",
    "portrait view": "세로 구도",
    "pots": "냄비",
    "power": "전선",
    "power lines": "전선",
    "profile": "옆모습",
    "radiant light": "빛나는 빛",
    "realistic": "사실적인",
    "residential": "주거지",
    "rifle": "소총",
    "rock": "바위",
    "rock formations": "암석 지형",
    "roof": "지붕",
    "roofs": "지붕",
    "rope": "밧줄",
    "roses": "장미",
    "rough": "거친",
    "rural": "시골",
    "rust stains": "녹 자국",
    "sand": "모래",
    "seasonal colors": "계절감 있는 색상",
    "sheet": "시트",
    "shadows": "그림자",
    "side profile": "옆모습",
    "sci-fi": "SF",
    "sketch lines": "스케치 선",
    "sketch overlay": "스케치 오버레이",
    "softly lit room": "부드럽게 조명된 방",
    "spring": "봄",
    "stack": "더미",
    "starry_sky": "별이 있는 하늘",
    "still life": "정물",
    "still_life": "정물",
    "storyboard": "스토리보드",
    "street": "거리",
    "street market": "거리 시장",
    "street view": "거리 풍경",
    "street_view": "거리 풍경",
    "study": "서재",
    "subtle": "은은한",
    "subway station": "지하철역",
    "subway_station": "지하철역",
    "summer": "여름",
    "sunlit": "햇빛이 드는",
    "sunny outdoor setting": "햇빛 드는 야외 환경",
    "sunset": "일몰",
    "surreal": "초현실적인",
    "tall": "높은",
    "tears": "눈물",
    "thatched roof": "초가지붕",
    "thatched_roof": "초가지붕",
    "toned": "색조의",
    "towering": "높이 솟은",
    "tree trunk": "나무줄기",
    "twinkling stars": "반짝이는 별",
    "utensils": "조리도구",
    "vertical": "세로",
    "vertical distortion": "세로 왜곡",
    "vertical seam": "세로 이음새",
    "viewed": "보이는",
    "vintage aesthetic": "빈티지한 미감",
    "voids": "빈 공간",
    "water": "물",
    "weathered": "낡은",
    "white robe": "흰 로브",
    "wireframe": "와이어프레임",
    "wood": "나무",
    "wooden beams": "나무 들보",
    "wooden posts": "나무 기둥",
    "wooden slats": "나무 판자",
}

TERM_KO.update({
    "2d": "평면",
    "3d": "입체",
    "academic": "학교 분위기",
    "actress": "여배우",
    "adorned": "장식된",
    "amidst": "사이에",
    "anatomy": "해부학",
    "animated": "애니메이션풍",
    "animation": "애니메이션",
    "arena": "경기장",
    "arena details": "경기장 세부 요소",
    "arms": "팔",
    "arms outstretched": "팔을 뻗은",
    "archway": "아치형 통로",
    "atmospheric": "분위기 있는",
    "autumn": "가을",
    "balcony": "발코니",
    "balconies": "발코니",
    "baskets": "바구니",
    "beige": "베이지색",
    "below": "아래",
    "benches": "벤치",
    "bioluminescent": "생물 발광",
    "botanical": "식물",
    "bottles": "병",
    "boulder": "바위",
    "boutique": "부티크",
    "carvings": "조각",
    "casual": "일상적인",
    "central": "중앙의",
    "children": "어린이",
    "children drawings": "어린이 그림",
    "cliffside": "절벽가",
    "close up": "클로즈업",
    "close-up": "클로즈업",
    "coliseum": "콜로세움",
    "concert": "콘서트",
    "contrast": "대비",
    "counter": "계산대",
    "creating": "만드는",
    "cracked": "갈라진",
    "crystalline": "수정 같은",
    "crystals": "수정",
    "curtain": "커튼",
    "curtains": "커튼",
    "details": "세부 요소",
    "diagonal": "대각선",
    "display": "진열",
    "doorway": "출입구",
    "drama": "연극",
    "dramatic": "극적인",
    "drawings": "그림",
    "dress": "의상",
    "ear": "귀",
    "emerald": "에메랄드빛",
    "elements": "요소",
    "eyes": "눈",
    "face": "얼굴",
    "fabric": "직물",
    "fantasy landscape": "판타지 풍경",
    "female": "여성",
    "featuring": "포함하는",
    "fiery": "불타는",
    "floats": "떠 있음",
    "fog": "안개",
    "foliage": "잎",
    "food stalls": "음식 노점",
    "foreground": "전경",
    "framed certificate": "액자에 담긴 증서",
    "frozen": "얼어붙은",
    "furniture": "가구",
    "gazing": "바라보는",
    "ghosts": "유령",
    "glow": "광채",
    "gold": "금색",
    "gradient": "그라데이션",
    "grain": "나뭇결",
    "gray": "회색",
    "ground": "지면",
    "hand": "손",
    "hat": "모자",
    "head": "머리",
    "headwear": "머리 장식",
    "holding": "들고 있는",
    "horizontal": "수평",
    "hotel": "호텔",
    "icy": "얼음 같은",
    "illuminating": "비추는",
    "industrial": "산업적인",
    "jagged": "뾰족한",
    "jars": "항아리",
    "lake": "호수",
    "lamps": "램프",
    "lantern": "등불",
    "lanterns": "등불",
    "light": "빛",
    "lights": "조명",
    "logo": "로고",
    "looking": "바라보는",
    "luxury": "고급스러운",
    "mansion": "저택",
    "mecha": "메카",
    "medieval": "중세풍",
    "minimalist": "미니멀한",
    "mist": "안개",
    "monochrome": "단색",
    "moon": "달",
    "mouth": "입",
    "mystical": "신비로운",
    "otherworldly": "초현실적인",
    "outstretched": "뻗은",
    "packaged goods": "포장 상품",
    "painterly": "회화풍",
    "painting": "그림",
    "pastel": "파스텔",
    "patterns": "무늬",
    "peaceful": "평온한",
    "performance": "공연",
    "plants": "식물",
    "planks": "판자",
    "plush": "푹신한",
    "plush rabbit": "푹신한 토끼",
    "purple glow": "보라색 광채",
    "purple lighting": "보라색 조명",
    "rabbit": "토끼",
    "rainbow": "무지개",
    "reflection": "반사",
    "render": "렌더",
    "retail display": "매장 진열",
    "rise": "솟아오름",
    "robot": "로봇",
    "rocky ground": "바위 지면",
    "ruins": "폐허",
    "sad": "슬픈",
    "screen": "스크린",
    "seating": "좌석",
    "sense": "느낌",
    "serene expression": "고요한 표정",
    "shadow": "그림자",
    "shoji": "쇼지",
    "shoji screen": "쇼지 문",
    "silhouette": "실루엣",
    "silhouettes": "실루엣",
    "sitting": "앉아 있는",
    "skull": "해골",
    "skull face": "해골 얼굴",
    "snow": "눈",
    "sofa": "소파",
    "solitude": "고독",
    "sparkles": "반짝임",
    "spikes": "가시",
    "spotlight": "스포트라이트",
    "spotlights": "스포트라이트",
    "stairs": "계단",
    "stars": "별",
    "statue": "조각상",
    "stones": "돌",
    "storm": "폭풍",
    "stormy": "폭풍우 치는",
    "striped": "줄무늬의",
    "swirling": "소용돌이치는",
    "swirls": "소용돌이",
    "teal": "청록색",
    "text": "텍스트",
    "theater": "극장",
    "tileable": "타일링 가능한",
    "tones": "색조",
    "town": "마을",
    "town square": "광장",
    "traditional": "전통적인",
    "traditional japanese": "일본 전통",
    "traditional japanese house": "일본 전통 가옥",
    "transparent": "투명한",
    "transparency": "투명감",
    "truss": "트러스",
    "underground": "지하",
    "umbrellas": "우산",
    "various": "여러",
    "vast": "광활한",
    "vines": "덩굴",
    "wall": "벽",
    "walls": "벽",
    "whimsical": "기발한",
    "wings": "날개",
    "yellow": "노란색",
    "young": "젊은",
    "young female": "젊은 여성",
    "abandoned": "버려진",
    "alley": "골목",
    "along": "따라",
    "annotations": "주석",
    "area": "구역",
    "arrangement": "배치",
    "asphalt": "아스팔트",
    "attic": "다락방",
    "bamboo": "대나무",
    "banners": "현수막",
    "bar": "바",
    "barrel": "통",
    "barrels": "통",
    "barren": "황량한",
    "beach": "해변",
    "beams": "빛줄기",
    "black": "검은색",
    "blinds": "블라인드",
    "blocks": "블록",
    "body": "몸",
    "bookshelves": "책장",
    "box": "상자",
    "bridge": "다리",
    "bubbles": "거품",
    "cable": "케이블",
    "calm": "차분한",
    "candles": "양초",
    "candlelit": "촛불이 켜진",
    "car": "자동차",
    "cards": "카드",
    "carousel": "회전목마",
    "carrying": "들고 있는",
    "cathedral": "성당",
    "cavern": "동굴",
    "celestial": "천체의",
    "chair": "의자",
    "chairs": "의자",
    "circle": "원",
    "circular": "원형",
    "cliffs": "절벽",
    "clothes": "옷",
    "cobblestone": "자갈길",
    "coastline": "해안선",
    "cold": "차가운",
    "color": "색상",
    "colors": "색상",
    "computer": "컴퓨터",
    "concept": "콘셉트",
    "confrontation": "대치",
    "connecting": "연결된",
    "contemplative": "사색적인",
    "convenience": "편의점",
    "convenience store": "편의점",
    "cool": "차가운",
    "copper": "구리색",
    "coffee": "커피",
    "coffee table": "커피 테이블",
    "corner": "구석",
    "crack": "균열",
    "crates": "상자",
    "creatures": "생물",
    "crowd": "군중",
    "cups": "컵",
    "debris": "잔해",
    "decay": "부식",
    "depth": "깊이",
    "desert": "사막",
    "desks": "책상",
    "desolate": "황량한",
    "detailed": "세밀한",
    "dirt": "흙",
    "dim": "어두운",
    "distance": "거리",
    "distressed": "낡고 거친",
    "distortion": "왜곡",
    "dome": "돔",
    "doorways": "출입구",
    "draft": "초안",
    "dreamlike": "몽환적인",
    "dusk": "해질녘",
    "edge": "가장자리",
    "effect": "효과",
    "exterior": "외부",
    "expression": "표정",
    "faces": "얼굴",
    "fading": "희미해지는",
    "faintly": "희미하게",
    "feet": "발",
    "fences": "울타리",
    "flags": "깃발",
    "flat": "평평한",
    "flat depth": "평평한 깊이감",
    "float": "떠 있음",
    "flower": "꽃",
    "flower arrangement": "꽃꽂이",
    "floral": "꽃무늬",
    "floral arrangement": "꽃꽂이",
    "folded": "접힌",
    "formal": "격식 있는",
    "fountain": "분수",
    "framed": "액자에 담긴",
    "garage": "차고",
    "gears": "기어",
    "giant": "거대한",
    "glass": "유리",
    "graffiti": "그래피티",
    "gradients": "그라데이션",
    "grayscale": "회색조",
    "grey": "회색",
    "grid": "격자",
    "gun": "총",
    "gymnasium": "체육관",
    "haired": "머리카락의",
    "hall": "홀",
    "hands": "손",
    "handwritten": "손글씨",
    "haunting": "음산한",
    "herbs": "허브",
    "hoodie": "후드티",
    "horns": "뿔",
    "house": "집",
    "houses": "집",
    "hued": "색조의",
    "ice": "얼음",
    "image": "이미지",
    "impact": "충격",
    "islands": "섬",
    "items": "물건",
    "laboratory": "실험실",
    "lattice": "격자",
    "leather": "가죽",
    "left": "왼쪽",
    "lightning": "번개",
    "lined": "선이 있는",
    "living room": "거실",
    "lock": "자물쇠",
    "locker": "사물함",
    "lockers": "사물함",
    "long": "긴",
    "luminescence": "발광",
    "magenta": "자홍색",
    "magical": "마법 같은",
    "magic": "마법",
    "maid": "메이드",
    "manga": "만화",
    "marble": "대리석",
    "massive": "거대한",
    "material": "재질",
    "mats": "매트",
    "menu": "메뉴",
    "motion": "움직임",
    "muscle": "근육",
    "muscular": "근육질의",
    "muted": "차분한",
    "nebulae": "성운",
    "noon": "정오",
    "note": "메모",
    "notes": "메모",
    "numbers": "숫자",
    "object": "물체",
    "objects": "물체",
    "obscured": "가려진",
    "ocean": "바다",
    "office": "사무실",
    "old": "오래된",
    "outlined": "윤곽선이 있는",
    "outlines": "윤곽선",
    "overcast": "흐린",
    "overgrown": "무성하게 자란",
    "overlayed": "겹쳐진",
    "pale": "옅은",
    "panel": "패널",
    "paper": "종이",
    "partially": "부분적으로",
    "particle": "입자",
    "particles": "입자",
    "pattern": "무늬",
    "peeking": "살짝 보이는",
    "pencil": "연필",
    "phone": "전화기",
    "pillar": "기둥",
    "pillars": "기둥",
    "pipes": "파이프",
    "plain": "단순한",
    "planet": "행성",
    "pole": "기둥",
    "pointing": "가리키는",
    "reaching": "뻗은",
    "reception": "접수대",
    "reflected": "비친",
    "resembling": "닮은",
    "restaurant": "식당",
    "right": "오른쪽",
    "rises": "솟아오르는",
    "rising": "솟아오르는",
    "road": "도로",
    "robes": "로브",
    "rocks": "바위",
    "rolled": "말린",
    "rooms": "방",
    "roots": "뿌리",
    "row": "줄",
    "rust": "녹",
    "rustic": "소박한",
    "scattered": "흩어진",
    "scenic": "풍경이 좋은",
    "screen": "화면",
    "search": "검색",
    "security": "보안",
    "semi-transparent": "반투명한",
    "shading": "음영",
    "ship": "배",
    "shipping": "운송",
    "shipping containers": "컨테이너",
    "short": "짧은",
    "shimmering": "반짝이는",
    "shrine": "신사",
    "side": "측면",
    "sidewalk": "보도",
    "signs": "표지판",
    "silhouetted": "실루엣으로 보이는",
    "six": "여섯",
    "skeletal": "해골 같은",
    "skeleton": "해골",
    "sketched": "스케치된",
    "skyline": "스카이라인",
    "smile": "미소",
    "smiling": "미소 짓는",
    "sombre": "침울한",
    "somber": "침울한",
    "space": "우주",
    "spirit": "영혼",
    "split-screen": "분할 화면",
    "stall": "가판대",
    "staircase": "계단",
    "star-filled": "별이 가득한",
    "steps": "계단",
    "streaks": "줄기",
    "structure": "구조물",
    "student": "학생",
    "sunlight": "햇빛",
    "sunny": "화창한",
    "surprised": "놀란",
    "sword": "검",
    "swords": "검",
    "tavern": "선술집",
    "teen": "십대",
    "tension": "긴장감",
    "tinted": "색이 입혀진",
    "torches": "횃불",
    "torii": "도리이",
    "totem": "토템",
    "tower": "탑",
    "tray": "쟁반",
    "trophies": "트로피",
    "tv": "TV",
    "underwater": "수중",
    "uniform": "유니폼",
    "up": "위쪽",
    "valley": "계곡",
    "veins": "정맥",
    "village": "마을",
    "vintage": "빈티지",
    "waiting": "기다리는",
    "walks": "걷는",
    "walkway": "보행로",
    "wallpaper": "벽지",
    "warehouse": "창고",
    "wearing": "입고 있는",
    "wavy": "물결 모양의",
    "winter": "겨울",
    "sci-fi": "공상과학",
    "sci fi": "공상과학",
    "sf": "공상과학",
    "sadono": "사도노",
    "tv": "텔레비전",
})

TERM_KO.update({
    "accents": "강조 요소",
    "action": "동작감",
    "afternoon": "오후",
    "arcane": "비전의",
    "arch": "아치",
    "arches": "아치",
    "are": "",
    "ascending": "오름차순의",
    "auroras": "오로라",
    "backview": "뒷모습",
    "baluster": "난간 기둥",
    "banner": "현수막",
    "barrier": "장벽",
    "base": "기반",
    "basin": "대야",
    "battle": "전투",
    "beam": "빛줄기",
    "beauty": "아름다움",
    "borealis": "북극광",
    "branding": "브랜딩",
    "brightly": "밝게",
    "broken": "부서진",
    "bush": "덤불",
    "bushes": "덤불",
    "canopy": "수관",
    "cartoon": "만화풍",
    "cemetery": "묘지",
    "chains": "쇠사슬",
    "chamber": "방",
    "chandelier": "샹들리에",
    "chef": "요리사",
    "claws": "발톱",
    "classic": "고전적인",
    "clothing": "의류",
    "colored": "채색된",
    "conceptual": "개념적인",
    "construction": "구조 작업",
    "containers": "컨테이너",
    "contrasting": "대비되는",
    "confront": "마주하는",
    "cosmos": "우주",
    "covered": "덮인",
    "covering": "덮고 있는",
    "crackled": "갈라진",
    "crater": "분화구",
    "crossed": "교차된",
    "darkness": "어둠",
    "demon": "악마",
    "dense": "빽빽한",
    "depiction": "묘사",
    "decorations": "장식",
    "destruction": "파괴",
    "diary": "일기",
    "driveway": "진입로",
    "dust": "먼지",
    "each": "서로",
    "education": "교육",
    "eight": "여덟",
    "entry": "입구",
    "expressive": "표정이 풍부한",
    "eye": "눈",
    "feather": "깃털",
    "fingers": "손가락",
    "flowerbeds": "화단",
    "flowing": "흐르는",
    "food": "음식",
    "forgotten": "잊힌",
    "formations": "지형",
    "full": "가득한",
    "gas": "가스",
    "gaming": "게임",
    "glides": "미끄러지듯 이동하는",
    "glows": "빛나는",
    "heart": "하트",
    "heart-shaped": "하트 모양의",
    "heat": "열기",
    "high-angle": "하이앵글",
    "hillside": "언덕 비탈",
    "hospital": "병원",
    "hover": "떠 있는",
    "inset": "끼워 넣은",
    "into": "안으로",
    "it": "",
    "its": "그것의",
    "ivy": "담쟁이",
    "ivy-covered": "담쟁이로 덮인",
    "juxtaposes": "대비시키는",
    "ladder": "사다리",
    "ladders": "사다리",
    "label": "라벨",
    "lakes": "호수",
    "layering": "레이어링",
    "leaf": "잎",
    "lies": "놓인",
    "line": "선",
    "lining": "늘어선",
    "liquor": "술",
    "lifts": "들어 올리는",
    "looks": "바라보는",
    "markings": "표식",
    "mask": "가면",
    "matching": "어울리는",
    "meeting": "회의",
    "memory": "기억",
    "minimalism": "미니멀리즘",
    "monitors": "모니터",
    "mounted": "걸린",
    "multiple": "여러 개의",
    "mythical": "신화적인",
    "mythological": "신화적인",
    "nostalgic": "향수를 주는",
    "nightstands": "침대 협탁",
    "open": "열린",
    "orderly": "정돈된",
    "organization": "정리",
    "ornament": "장식",
    "ornaments": "장식",
    "other": "다른 쪽",
    "others": "다른 인물",
    "out": "밖",
    "overlooking": "내려다보는",
    "pair": "한 쌍",
    "pans": "팬",
    "parchment": "양피지",
    "pathway": "길",
    "perspective": "원근감",
    "pillows": "베개",
    "plant": "식물",
    "playful": "장난스러운",
    "plaza": "광장",
    "possibly": "아마도",
    "promotional": "홍보용",
    "projection": "투사",
    "radiant": "빛나는",
    "radiating": "방사되는",
    "railing": "난간",
    "railings": "난간",
    "realm": "영역",
    "remote": "리모컨",
    "rendering": "렌더링",
    "rest": "휴식",
    "revealing": "드러내는",
    "rose-covered": "장미로 덮인",
    "round": "둥근",
    "rugs": "러그",
    "ruined": "폐허가 된",
    "running": "달리는",
    "sadness": "슬픔",
    "screens": "화면",
    "script": "문자",
    "scrolls": "두루마리",
    "seabed": "해저",
    "seamless": "매끄러운",
    "seated": "앉아 있는",
    "shelving": "선반",
    "shoes": "신발",
    "shot": "장면",
    "silver": "은색",
    "sits": "앉아 있는",
    "sleeping": "잠자는",
    "slytherin": "슬리데린",
    "sneakers": "운동화",
    "soaring": "날아오르는",
    "soars": "날아오르는",
    "solo": "단독",
    "sparse": "드문드문한",
    "speakers": "스피커",
    "speed": "속도",
    "sphere": "구체",
    "spines": "책등",
    "spiritual": "영적인",
    "statues": "조각상",
    "streetlamp": "가로등",
    "surrounded": "둘러싸인",
    "symbolic": "상징적인",
    "terrain": "지형",
    "tidy": "정돈된",
    "tombstones": "묘비",
    "tools": "도구",
    "toward": "향해",
    "travel": "여행",
    "tufted": "버튼 장식된",
    "tunnel": "터널",
    "twisted": "뒤틀린",
    "vehicle": "차량",
    "volcano": "화산",
    "volcanic": "화산의",
    "watercolor": "수채화",
    "where": "있는 곳",
    "winding": "구불구불한",
    "wolf": "늑대",
    "worlds": "세계",
    "youthful": "젊은",
})

TERM_KO.update({
    "awe-inspiring": "장엄한",
    "aged": "오래된",
    "animal": "동물",
    "annotated": "주석이 달린",
    "arcane": "신비한",
    "armchair": "안락의자",
    "armored": "갑옷을 입은",
    "arranged": "배치된",
    "artifact": "유물",
    "attire": "복장",
    "audience": "관객",
    "ball": "공",
    "balloons": "풍선",
    "bare": "비어 있는",
    "bleachers": "관람석",
    "blooms": "꽃송이",
    "blur": "흐림",
    "blush": "홍조",
    "boat": "배",
    "bordered": "테두리가 있는",
    "boulders": "바위",
    "bow": "활",
    "brushed": "붓질된",
    "can": "통",
    "campsite": "야영지",
    "candlelight": "촛불",
    "cart": "수레",
    "castle": "성",
    "center": "중앙",
    "chain-link": "체인 링크",
    "cloud formation": "구름 형성",
    "coloring": "채색",
    "contemporary": "현대적인",
    "crabs": "게",
    "decorated": "장식된",
    "delicate": "섬세한",
    "deluxe": "고급",
    "demonic": "악마 같은",
    "displaying": "보여주는",
    "distress": "손상감",
    "division": "분할",
    "dividers": "칸막이",
    "dried": "마른",
    "earthy": "흙빛의",
    "embedded": "박힌",
    "emblem": "문장",
    "emergency": "비상",
    "emerging": "나타나는",
    "emerges": "나타나는",
    "end": "끝",
    "entrance": "입구",
    "entity": "존재",
    "evening": "저녁",
    "exit": "출구",
    "exposed": "노출된",
    "festive": "축제 분위기의",
    "fish": "물고기",
    "fixture": "조명 기구",
    "flanked": "양쪽에 배치된",
    "flies": "날아가는",
    "flying": "날아가는",
    "flooring": "바닥재",
    "floors": "바닥",
    "for": "위한",
    "fragmented": "조각난",
    "gesture": "몸짓",
    "gloom": "어둠",
    "guard": "경비원",
    "guardrail": "가드레일",
    "grandeur": "장엄함",
    "grainy": "거친 입자의",
    "grunge": "거친 질감",
    "hangers": "옷걸이",
    "haze": "안개",
    "hazy": "흐릿한",
    "headband": "머리띠",
    "headboard": "침대 머리판",
    "hedges": "생울타리",
    "hollowed": "속이 빈",
    "hong kong": "홍콩",
    "hour": "시간",
    "illumination": "조명",
    "infrastructure": "기반 시설",
    "intimate": "친밀한",
    "isolation": "고립",
    "japan": "일본",
    "joy": "기쁨",
    "joyful": "즐거운",
    "junk boat": "정크선",
    "layers": "레이어",
    "leaps": "뛰어오르는",
    "limbs": "팔다리",
    "live": "라이브",
    "loom": "어렴풋이 보이는",
    "loose": "느슨한",
    "magical": "마법 같은",
    "majestic": "장엄한",
    "messy": "헝클어진",
    "microphone": "마이크",
    "military": "군사적인",
    "molten": "녹아내린",
    "muscles": "근육",
    "narrative": "서사",
    "neutral": "중립적인",
    "noise": "노이즈",
    "number": "숫자",
    "organized": "정리된",
    "overlaid": "겹쳐진",
    "park": "공원",
    "parked": "주차된",
    "persistence": "지속성",
    "pine": "소나무",
    "platform": "플랫폼",
    "poster": "포스터",
    "presence": "존재감",
    "prison": "감옥",
    "pumpkin": "호박",
    "rainy": "비 오는",
    "reaches": "뻗는",
    "rear": "뒤쪽",
    "reclaiming": "되찾는",
    "reflections": "반사",
    "robed": "로브를 입은",
    "romance": "로맨스",
    "rooftop": "옥상",
    "schedule": "일정표",
    "scissors": "가위",
    "seafood": "해산물",
    "seeds": "씨앗",
    "separated": "분리된",
    "sepia": "세피아",
    "serious": "진지한",
    "shattered": "산산조각난",
    "shapes": "형태",
    "shoes": "신발",
    "shop": "상점",
    "shrimp": "새우",
    "signage": "간판",
    "single": "하나의",
    "singer": "가수",
    "sit": "앉아 있는",
    "skies": "하늘",
    "skylights": "천창",
    "smaller": "작은",
    "sconce": "벽등",
    "soil": "흙",
    "solid": "단단한",
    "solitary": "홀로 있는",
    "spaceship": "우주선",
    "spear": "창",
    "spectators": "관중",
    "spherical": "구형의",
    "spheres": "구체",
    "stained-glass": "스테인드글라스",
    "stark": "강렬한",
    "stem": "줄기",
    "stretching": "뻗은",
    "stripes": "줄무늬",
    "studio": "스튜디오",
    "suburban": "교외의",
    "suit": "정장",
    "tables": "테이블",
    "tatami": "다다미",
    "temple": "사원",
    "tents": "텐트",
    "tense": "긴박한",
    "thematic": "주제적인",
    "themes": "주제",
    "their": "그들의",
    "themed": "테마가 있는",
    "thatched-roof": "초가지붕",
    "tile": "타일",
    "top-down": "위에서 본",
    "torch": "횃불",
    "track": "선로",
    "train": "기차",
    "trash": "쓰레기",
    "trunks": "나무줄기",
    "towels": "수건",
    "twisting": "뒤틀린",
    "uneven": "고르지 않은",
    "vase": "꽃병",
    "vases": "꽃병",
    "volumetric": "입체적인",
    "warriors": "전사",
    "waters": "물",
    "weapon": "무기",
    "worker": "작업자",
    "yard": "마당",
})

TERM_KO.update({
    "a": "",
    "b": "",
    "among": "사이에",
    "arrow": "화살표",
    "artist": "작가",
    "asian": "아시아풍",
    "beneath": "아래",
    "bird": "새",
    "blade": "칼날",
    "boy": "소년",
    "breathtaking": "장엄한",
    "bulbs": "전구",
    "business": "업무",
    "cabin": "오두막",
    "carved": "조각된",
    "charming": "매력적인",
    "chest": "상자",
    "clean": "깨끗한",
    "clock": "시계",
    "collage": "콜라주",
    "contemplation": "사색",
    "creeping": "자라난",
    "cruise": "유람선",
    "curves": "곡선",
    "cuts": "절단선",
    "damage": "손상",
    "dances": "춤추는",
    "dancing": "춤추는",
    "day": "낮",
    "defeat": "패배",
    "depicted": "묘사된",
    "destructive": "파괴적인",
    "displays": "보여주는",
    "down": "아래",
    "drawn": "그려진",
    "east": "동아시아",
    "east asian": "동아시아풍",
    "effects": "효과",
    "emanating": "뿜어져 나오는",
    "expanse": "넓은 공간",
    "fades": "희미해지는",
    "flank": "측면",
    "fly": "날아가는",
    "gentle": "부드러운",
    "glasses": "잔",
    "guides": "가이드",
    "heater": "난방기",
    "hills": "언덕",
    "holes": "구멍",
    "holds": "들고 있는",
    "inside": "안쪽",
    "inscription": "글귀",
    "leading": "이어지는",
    "leg": "다리",
    "lobby": "로비",
    "lunar": "달의",
    "middle": "가운데",
    "moment": "순간",
    "mottled": "얼룩덜룩한",
    "nearby": "근처",
    "nestled": "자리한",
    "numerous": "많은",
    "observe": "바라보는",
    "orbs": "구체",
    "overlapping": "겹쳐진",
    "overlaying": "겹쳐진",
    "palette": "팔레트",
    "past": "지나",
    "plates": "접시",
    "pottery": "도자기",
    "rendered": "표현된",
    "root-covered": "뿌리로 덮인",
    "root-filled": "뿌리로 가득한",
    "rug": "러그",
    "rugged": "거친",
    "ruin": "폐허",
    "rusty": "녹슨",
    "shed": "헛간",
    "shoe": "신발",
    "sings": "노래하는",
    "sink": "싱크대",
    "sparks": "불꽃",
    "spaceships": "우주선",
    "stacked": "쌓인",
    "streetlights": "가로등",
    "streaming": "흘러내리는",
    "students": "학생들",
    "to": "로",
    "umbrella": "우산",
    "vendor": "상인",
    "vein": "정맥",
    "waves": "파도",
    "windowpane": "창유리",
    "year": "연도",
})

TERM_PAIRS = sorted(TERM_KO.items(), key=lambda item: len(item[0]), reverse=True)
TERM_PATTERN = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(needle) for needle, _ in TERM_PAIRS)
    + r")(?![a-z0-9])",
    re.I,
)

NOISE_TAG_PATTERNS = [
    re.compile(r"\.(psd|psb|jpg|jpeg|png|webp|tif|tiff)\b", re.I),
    re.compile(r"^imagine(?:\s|_|-|$)", re.I),
    re.compile(r"^(?:vs|grb|kya|nfb|dwg|fn|tm)\d+", re.I),
    re.compile(r"^(?:genzu)$", re.I),
]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def parse_tags(raw: Any) -> list[str]:
    text = _clean(raw)
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [_clean(item) for item in value if _clean(item)]
    except Exception:
        pass
    return [
        item.strip().strip("\"'")
        for item in text.strip("[]").split(",")
        if item.strip().strip("\"'")
    ]


def translate_loose_term(value: Any) -> str:
    source = _clean(value).replace("_", " ")
    lower = source.lower()
    if not source:
        return ""
    if any(pattern.search(lower) for pattern in NOISE_TAG_PATTERNS):
        return ""
    if lower in TERM_KO:
        return TERM_KO[lower]
    match = TERM_PATTERN.search(lower)
    if match:
        return TERM_KO.get(match.group(1).lower(), match.group(1))
    translated = translate_caption_segment(source)
    if translated and not re.search(r"[A-Za-z]", translated):
        return translated
    return source


def translate_caption_segment(segment: Any) -> str:
    out = _clean(segment)
    if not out:
        return ""
    if re.search(r"[가-힣]", out):
        return out
    out = re.sub(r"[.。]+$", "", out).lower()
    out = re.sub(r"\b([a-z][a-z0-9_-]*)'s\b", r"\1의", out)
    out = re.sub(r"\b([a-z][a-z0-9_-]*)-like\b", r"\1 같은", out)
    out = re.sub(r"\b([a-z][a-z0-9_-]*)-toned\b", r"\1 색조의", out)
    out = re.sub(r"\b([a-z][a-z0-9_-]*)-striped\b", r"\1 줄무늬의", out)
    out = re.sub(r"\b(a|an|the)\b", " ", out)
    out = re.sub(r"\bin front of\b", "앞에", out)
    out = re.sub(r"\bin the foreground\b", "전경에", out)
    out = re.sub(r"\bin the background\b", "배경에", out)
    out = re.sub(r"\bin background\b", "배경에", out)
    out = re.sub(r"\bappears as\b", "처럼 보이는", out)
    out = re.sub(r"\bappears\b", "나타나는", out)
    out = re.sub(r"\bis superimposed on\b", "위에 겹쳐진", out)
    out = re.sub(r"\bsuperimposed on\b", "위에 겹쳐진", out)
    out = re.sub(r"\bblending into\b", "배경에 녹아드는", out)
    out = re.sub(r"\bset against\b", "배경으로 한", out)
    out = re.sub(r"\bset in\b", "안에 있는", out)
    out = re.sub(r"\berupts from\b", "에서 솟아오르는", out)
    out = re.sub(r"\billuminated by\b", "빛을 받는", out)
    out = re.sub(r"\bviewed through\b", "통해 보이는", out)
    out = re.sub(r"\bfilled with\b", "가득한", out)
    out = re.sub(r"\brunning at\b", "로 달리는", out)
    out = re.sub(r"\breaching out\b", "뻗은", out)
    out = re.sub(r"\bcasting shadows\b", "그림자를 드리운", out)
    out = re.sub(r"\bfiltering through\b", "통해 스며드는", out)
    out = re.sub(r"\bsuggesting\b", "암시하는", out)
    out = re.sub(r"\bat night\b", "밤에", out)
    out = re.sub(r"\boverlaid on\b", "위에 겹쳐진", out)
    out = re.sub(r"\bis superimposed over\b", "위에 겹쳐진", out)
    out = re.sub(r"\bsuperimposed over\b", "위에 겹쳐진", out)
    out = re.sub(r"\bsurrounded by\b", "둘러싸인", out)
    out = re.sub(r"\bframed by\b", "둘러싸인", out)
    out = re.sub(r"\brendered in\b", "표현된", out)
    out = re.sub(r"\bbathed in\b", "물든", out)
    out = re.sub(r"\bindicating\b", "나타내는", out)
    out = re.sub(r"\bevoking\b", "느낌을 주는", out)
    out = re.sub(r"\bnestled in\b", "자리한", out)
    out = re.sub(r"\bperched on\b", "앉아 있는", out)
    out = re.sub(r"\bappearing\b", "나타나는", out)
    out = re.sub(r"\bfloating\b", "떠 있는", out)
    out = re.sub(r"\bresting on\b", "위에 놓인", out)
    out = re.sub(r"\brests on\b", "위에 놓인", out)
    out = TERM_PATTERN.sub(lambda match: TERM_KO.get(match.group(1).lower(), match.group(1)), out)
    replacements = [
        (r"\bwith\b", "함께"),
        (r"\band\b", "및"),
        (r"\bor\b", "또는"),
        (r"\bof\b", "의"),
        (r"\bin\b", "안에"),
        (r"\bat\b", "에서"),
        (r"\bon\b", "위에"),
        (r"\bunder\b", "아래"),
        (r"\bover\b", "위에"),
        (r"\babove\b", "위에"),
        (r"\bagainst\b", "배경으로"),
        (r"\bbefore\b", "앞에"),
        (r"\bby\b", "의해"),
        (r"\bthrough\b", "통해"),
        (r"\bbetween\b", "사이에"),
        (r"\bnear\b", "근처"),
        (r"\bshowing\b", "보여주는"),
        (r"\bview\b", "풍경"),
    ]
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out)
    out = re.sub(r"\s*,\s*", ", ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out or _clean(segment)


def with_subject_particle(text: str) -> str:
    if not text:
        return text
    last = text[-1]
    if "가" <= last <= "힣":
        return f"{text}{'이' if (ord(last) - ord('가')) % 28 else '가'}"
    return f"{text}이/가"


def translate_caption(caption: Any) -> str:
    source = _clean(caption)
    if not source or source.lower() == "unknown":
        return "캡션 없음"
    if re.search(r"[가-힣]", source):
        return source
    text = re.sub(r"[.。]+$", "", source).strip()
    if re.match(r"^Digital sketch of a building facade with windows and structural lines$", text, re.I):
        return "창문과 구조선이 보이는 건물 외관의 디지털 스케치입니다."
    match = re.match(r"^(.+?) stands? in (.+?) under (.+?) with (.+)$", text, re.I)
    if match:
        return (
            f"{translate_caption_segment(match.group(2))} 안에 "
            f"{with_subject_particle(translate_caption_segment(match.group(1)))} 서 있고, "
            f"{translate_caption_segment(match.group(3))} 아래에 "
            f"{translate_caption_segment(match.group(4))}이/가 보입니다."
        )
    match = re.match(r"^(.+?) stands? in (.+?) with (.+)$", text, re.I)
    if match:
        return (
            f"{translate_caption_segment(match.group(2))} 안에 "
            f"{with_subject_particle(translate_caption_segment(match.group(1)))} 서 있고, "
            f"{translate_caption_segment(match.group(3))}이/가 보입니다."
        )
    match = re.match(r"^(.+?) stands? in (.+?) under (.+)$", text, re.I)
    if match:
        return (
            f"{translate_caption_segment(match.group(2))} 안에 "
            f"{with_subject_particle(translate_caption_segment(match.group(1)))} 서 있고, "
            f"{translate_caption_segment(match.group(3))} 아래입니다."
        )
    match = re.match(r"^(.+?) stands? before (.+?) under (.+)$", text, re.I)
    if match:
        return (
            f"{with_subject_particle(translate_caption_segment(match.group(1)))} "
            f"{translate_caption_segment(match.group(2))} 앞에 서 있고, "
            f"{translate_caption_segment(match.group(3))} 아래입니다."
        )
    match = re.match(r"^(.+?) in (.+?) at night$", text, re.I)
    if match:
        return (
            f"밤의 {translate_caption_segment(match.group(2))} 안에 "
            f"{with_subject_particle(translate_caption_segment(match.group(1)))} 있습니다."
        )
    match = re.match(r"^(.+?) overlaid on (.+)$", text, re.I)
    if match:
        return (
            f"{with_subject_particle(translate_caption_segment(match.group(1)))} "
            f"{translate_caption_segment(match.group(2))} 위에 겹쳐져 있습니다."
        )
    match = re.match(r"^(.+?) perched on (.+?) against (.+)$", text, re.I)
    if match:
        return (
            f"{with_subject_particle(translate_caption_segment(match.group(1)))} "
            f"{translate_caption_segment(match.group(2))} 위에 앉아 있고, "
            f"{translate_caption_segment(match.group(3))}을/를 배경으로 합니다."
        )
    match = re.match(r"^(.+?) framed by (.+)$", text, re.I)
    if match:
        return (
            f"{translate_caption_segment(match.group(2))}에 둘러싸인 "
            f"{translate_caption_segment(match.group(1))}입니다."
        )
    match = re.match(r"^(.+?) with (.+?) in front of (.+)$", text, re.I)
    if match:
        return (
            f"{translate_caption_segment(match.group(3))} 앞에 "
            f"{translate_caption_segment(match.group(2))}이/가 있는 "
            f"{translate_caption_segment(match.group(1))}입니다."
        )
    match = re.match(r"^(.+?) of (.+?) with (.+)$", text, re.I)
    if match:
        return (
            f"{translate_caption_segment(match.group(3))}이/가 보이는 "
            f"{translate_caption_segment(match.group(2))}의 "
            f"{translate_caption_segment(match.group(1))}입니다."
        )
    match = re.match(r"^(.+?) with (.+?) under (.+)$", text, re.I)
    if match:
        return (
            f"{translate_caption_segment(match.group(3))} 아래의 "
            f"{translate_caption_segment(match.group(1))}이며, "
            f"{translate_caption_segment(match.group(2))}이/가 보입니다."
        )
    match = re.match(r"^(.+?) with (.+)$", text, re.I)
    if match:
        return f"{translate_caption_segment(match.group(1))}이며, {translate_caption_segment(match.group(2))}이/가 보입니다."
    match = re.match(r"^(.+?) suggesting (.+)$", text, re.I)
    if match:
        return f"{translate_caption_segment(match.group(1))}이며, {translate_caption_segment(match.group(2))}을/를 암시합니다."
    translated = translate_caption_segment(text)
    return f"{translated}입니다." if re.search(r"[가-힣]", translated) else f"자동 번역 부족: {source}"


def translated_tag_display(raw: Any) -> str:
    tags = parse_tags(raw)
    translated: list[str] = []
    seen = set()
    for tag in tags:
        ko = translate_loose_term(tag)
        if not ko or ko in seen:
            continue
        translated.append(ko)
        seen.add(ko)
    return " · ".join(translated)


def has_usable_caption(value: Any) -> bool:
    caption = _clean(value)
    return bool(caption) and caption.lower() != "unknown"


def has_usable_tags(value: Any) -> bool:
    raw = _clean(value)
    return bool(raw) and raw != "[]"


def analysis_status(row: dict[str, Any]) -> tuple[str, str]:
    """Classify whether the existing AI metadata can be reviewed."""
    caption_ok = has_usable_caption(row.get("mc_caption"))
    tags_ok = has_usable_tags(row.get("ai_tags"))
    processing_status = _clean(row.get("processing_status"))
    processing_error = _clean(row.get("processing_error"))
    caption_model = _clean(row.get("caption_model"))

    if not caption_ok and not tags_ok:
        return "missing", "missing_caption_and_tags"
    if not caption_ok:
        return "partial", "missing_caption"
    if not tags_ok:
        return "partial", "missing_tags"
    if processing_status == "parse_fallback_legacy" or caption_model == "unknown_legacy":
        return "legacy_warning", "legacy_parse_or_caption_model"
    if "psd-tools failed" in processing_error:
        return "thumbnail_fallback_warning", "caption_from_thumbnail_fallback"
    return "ok", ""


def source_group(folder_path: Any) -> str:
    normalized = _clean(folder_path).replace("\\", "/").strip("/")
    if not normalized:
        return "unknown"
    return normalized


def dedupe_key(row: dict[str, Any]) -> str:
    content_hash = _clean(row.get("content_hash"))
    if content_hash:
        return f"content:{content_hash}"
    perceptual_hash = _clean(row.get("perceptual_hash"))
    if perceptual_hash:
        return f"phash:{perceptual_hash}"
    thumbnail = _clean(row.get("thumbnail_url"))
    if thumbnail:
        return f"thumb:{thumbnail}"
    return f"path:{_clean(row.get('file_path')) or _clean(row.get('item_id'))}"


def load_candidates(
    db_path: Path,
    *,
    include_missing: bool = False,
    include_partial: bool = False,
) -> list[dict[str, Any]]:
    if not db_path.exists():
        raise ValueError(f"database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
        missing = {"id", "thumbnail_url", "file_name", "file_path"} - columns
        if missing:
            raise ValueError(f"files table missing required columns: {', '.join(sorted(missing))}")
        selected = [
            "id AS item_id",
            *[field for field in ITEM_FIELDS if field != "item_id" and field in columns],
        ]
        rows = conn.execute(
            f"""
            SELECT {", ".join(selected)}
            FROM files
            WHERE COALESCE(preview_only, 0) = 0
              AND COALESCE(thumbnail_url, '') != ''
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()

    candidates: list[dict[str, Any]] = []
    seen = set()
    for raw in rows:
        row = dict(raw)
        status, issue = analysis_status(row)
        if not include_missing and status == "missing":
            continue
        if not include_missing and not include_partial and status == "partial":
            continue
        key = dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        row["item_id"] = str(row.get("item_id"))
        row["analysis_status"] = status
        row["analysis_issue"] = issue
        row["dedupe_key"] = key
        row["source_group"] = source_group(row.get("folder_path"))
        candidates.append(row)
    return candidates


def build_sample(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    max_per_source: int,
    created_at: str = DEFAULT_CREATED_AT,
    sample_id_prefix: str = DEFAULT_SAMPLE_ID_PREFIX,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if max_per_source < 1:
        raise ValueError("max_per_source must be >= 1")
    if not candidates:
        raise ValueError("no review candidates found")

    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    selected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    used_items = set()

    def try_pick(row: dict[str, Any], *, enforce_source: bool) -> bool:
        if row["item_id"] in used_items:
            return False
        group = row["source_group"]
        if enforce_source and source_counts[group] >= max_per_source:
            return False
        selected.append(row)
        used_items.add(row["item_id"])
        source_counts[group] += 1
        return True

    for row in shuffled:
        if len(selected) >= count:
            break
        try_pick(row, enforce_source=True)

    if len(selected) < count:
        for row in shuffled:
            if len(selected) >= count:
                break
            try_pick(row, enforce_source=False)

    if len(selected) < count:
        raise ValueError(f"only selected {len(selected)} rows from {len(candidates)} candidates")

    rows = []
    for index, row in enumerate(selected[:count], start=1):
        review_row = {field: "" for field in CSV_FIELDS}
        review_row.update(row)
        review_row.update({
            "sample_id": f"{sample_id_prefix}-{index:04d}",
            "sample_version": sample_id_prefix,
            "sample_created_at": created_at,
            "sample_seed": str(seed),
            "sample_rank": str(index),
            "caption_ko_display": translate_caption(row.get("mc_caption")),
            "tags_ko_display": translated_tag_display(row.get("ai_tags")),
            "caption_alignment": "",
            "tag_alignment": "",
            "overall_alignment": "",
            "issue_types": "",
            "reviewer_notes": "",
            "reviewer_id": "",
            "reviewed_at": "",
        })
        rows.append(review_row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_manifest(path: Path, *, rows: list[dict[str, Any]], candidates: list[dict[str, Any]], args: argparse.Namespace) -> None:
    status_counts = Counter(row.get("analysis_status", "") for row in rows)
    source_counts = Counter(row.get("source_group", "") for row in rows)
    manifest = {
        "schema_version": "metadata_review_sample_v1",
        "created_at": args.created_at,
        "sample_count": len(rows),
        "candidate_count": len(candidates),
        "seed": args.seed,
        "max_per_source": args.max_per_source,
        "include_missing": args.include_missing,
        "include_partial": args.include_partial,
        "analysis_status_counts": dict(sorted(status_counts.items())),
        "source_group_count": len(source_counts),
        "top_source_groups": source_counts.most_common(20),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, *, rows: list[dict[str, Any]], candidates: list[dict[str, Any]], args: argparse.Namespace) -> None:
    status_counts = Counter(row.get("analysis_status", "") for row in rows)
    lines = [
        "# Metadata Quality Review Sample V1",
        "",
        "이미지 자체와 기존 AI 캡션/태그의 정합성을 사람이 평가하기 위한 샘플입니다.",
        "이 결과는 이후 answerable 검색 질문셋을 만드는 기반으로 사용합니다.",
        "",
        f"- sample_count: {len(rows)}",
        f"- candidate_count: {len(candidates)}",
        f"- seed: {args.seed}",
        f"- max_per_source: {args.max_per_source}",
        f"- include_missing: {args.include_missing}",
        f"- include_partial: {args.include_partial}",
        f"- analysis_status_counts: {dict(sorted(status_counts.items()))}",
        "",
        "## First Rows",
        "",
    ]
    for row in rows[:20]:
        lines.extend([
            f"### {row['sample_id']} / item {row['item_id']}",
            "",
            f"- file: {row.get('file_name', '')}",
            f"- source: {row.get('source_group', '')}",
            f"- status: {row.get('analysis_status', '')} {row.get('analysis_issue', '')}",
            f"- caption: {row.get('mc_caption', '')}",
            f"- caption_ko_display: {row.get('caption_ko_display', '')}",
            f"- tags: {row.get('ai_tags', '')}",
            f"- tags_ko_display: {row.get('tags_ko_display', '')}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=Path("imageparser.db"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--max-per-source", type=int, default=8)
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = load_candidates(
        args.db_path,
        include_missing=args.include_missing,
        include_partial=args.include_partial,
    )
    rows = build_sample(
        candidates,
        count=args.count,
        seed=args.seed,
        max_per_source=args.max_per_source,
        created_at=args.created_at,
    )
    output_dir = args.output_dir
    write_csv(output_dir / "metadata_review_sample.csv", rows)
    write_jsonl(output_dir / "metadata_review_sample.jsonl", rows)
    write_manifest(output_dir / "manifest.json", rows=rows, candidates=candidates, args=args)
    write_markdown(output_dir / "README.md", rows=rows, candidates=candidates, args=args)
    print(f"Wrote {len(rows)} review rows to {output_dir / 'metadata_review_sample.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
