# 3-Axis Search Architecture (Triaxis)

ImageParser v3의 검색 시스템 해설.
이미지 한 장을 **3가지 서로 다른 방법**으로 데이터화해서, 검색할 때 3개를 합산하여 순위를 매긴다.

---

## 한눈에 보기

```
사용자 쿼리: "건널목과 상점가가 같이 보이는?"
                │
                ▼
      ┌─── QueryDecomposer ───┐
      │  vector_query:         │
      │   "crosswalk and       │
      │    shopping street"    │
      │  fts_keywords:         │
      │   ["건널목","crosswalk",│
      │    "상점가","shopping"] │
      │  query_type: "keyword" │
      └────────┬───────────────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
  V-axis    S-axis     M-axis
  (시각)    (의미)    (키워드)
     │         │         │
     ▼         ▼         ▼
  ┌──────────────────────────┐
  │   RRF Merge (가중 합산)   │
  │   → 최종 순위 결정        │
  └──────────────────────────┘
```

---

## 축 1: V-axis (Visual) — "이 이미지가 눈에 어떻게 보이는가"

### 개념

이미지 **자체의 시각적 특성**을 벡터로 변환한다.
검색할 때 텍스트 쿼리도 같은 공간의 벡터로 변환해서 **이미지 벡터와 직접 비교**한다.

사람이 이미지를 "눈으로 보고" 판단하는 것의 기계적 근사치.

### 모델

| 항목 | standard | pro | ultra |
|------|----------|-----|-------|
| 모델 | `siglip2-base-patch16-224` | `siglip2-so400m-patch14-384` | `siglip2-giant-opt-patch16-256` |
| 벡터 차원 | **768** | **1152** | **1664** |

> Tier별 모델은 `config.yaml` > `ai_mode.tiers`에서 설정. 현재 `override: standard`.

### 저장하는 데이터

```
원본 이미지 (PSD/PNG/JPG)
  → 썸네일 생성 (output/thumbnails/)
  → SigLIP2 이미지 인코더 통과
  → N차원 벡터 (L2 정규화, tier별: 768/1152/1664)
  → vec_files 테이블에 저장
```

**입력**: 썸네일 이미지 (픽셀 데이터 그 자체)
**출력**: N개의 숫자 배열 = 이미지의 "시각적 지문"

### 검색 원리

```
검색 쿼리: "crosswalk and shopping street"
  → SigLIP2 텍스트 인코더 통과
  → N차원 쿼리 벡터 (tier별)
  → DB의 모든 이미지 벡터와 코사인 유사도 계산
  → 유사도 높은 순으로 정렬
```

### 점수 특성

| 구간 | 의미 |
|------|------|
| 0.15~0.17 | 강한 매치 (이미지가 쿼리와 시각적으로 잘 맞음) |
| 0.10~0.15 | 보통 매치 |
| 0.05~0.10 | 약한 매치 |
| < 0.05 | 무관 (노이즈) |
| **임계값** | **0.05** (config.yaml) |

SigLIP 코사인 유사도의 절대값이 낮은 이유:
1152차원 공간에서 이미지↔텍스트 크로스모달 매칭이므로 0.5+ 같은 높은 값이 나오지 않는다.
17%가 "매우 좋은 매치"인 것은 정상이다.

### 강점과 약점

- **강점**: 텍스트로 설명하기 어려운 시각적 유사성 포착 (색감, 구도, 분위기)
- **약점**: "건널목"처럼 구체적 물체명에는 텍스트 축보다 약함

### 코드 위치

- 인코더: `backend/vector/siglip2_encoder.py`
- 저장: `backend/db/sqlite_client.py` → `vec_files` 테이블
- 검색: `backend/search/sqlite_search.py` → `vector_search()`

---

## 축 2: S-axis (Semantic) — "AI가 쓴 설명이 쿼리와 의미적으로 통하는가"

### 개념

AI 비전 모델이 이미지를 보고 쓴 **캡션과 태그를 텍스트 벡터로 변환**한다.
검색할 때 쿼리도 텍스트 벡터로 변환해서 **텍스트↔텍스트 의미 비교**를 한다.

V-axis가 "눈으로 보는" 것이라면, S-axis는 "설명을 읽고 이해하는" 것.

### 모델

| 항목 | standard/pro | ultra |
|------|-------------|-------|
| 임베딩 모델 | `qwen3-embedding:0.6b` (Ollama) | `qwen3-embedding:8b` (Ollama) |
| 벡터 차원 | **256** (standard) / **1024** (pro) | **4096** |
| 캡션 생성 | Qwen3-VL (transformers) | `qwen3-vl:8b` (Ollama) |
| 서버 | Ollama (`http://localhost:11434`) | Ollama |

### 저장하는 데이터

```
원본 이미지
  → qwen3-vl:8b 비전 모델이 분석
  → ai_caption: "A night scene with a crosswalk and shops"
  → ai_tags: ["crosswalk", "shops", "night", "urban"]
  → 두 개를 합침:
     "A night scene with a crosswalk and shops. Tags: crosswalk, shops, night, urban"
  → qwen3-embedding 텍스트 인코더 통과
  → N차원 벡터 (L2 정규화, tier별: 256/1024/4096)
  → vec_text 테이블에 저장
```

**입력**: AI가 생성한 캡션 + 태그 (텍스트)
**출력**: N개의 숫자 배열 = 이미지 설명의 "의미적 지문"

### 검색 원리

```
검색 쿼리: "crosswalk and shopping street"
  → qwen3-embedding 텍스트 인코더 통과
  → N차원 쿼리 벡터 (tier별)
  → DB의 모든 텍스트 벡터와 코사인 유사도 계산
  → 유사도 높은 순으로 정렬
```

### 점수 특성

| 구간 | 의미 |
|------|------|
| 0.70~0.78 | 강한 매치 (설명이 쿼리와 의미적으로 잘 통함) |
| 0.55~0.70 | 보통 매치 |
| 0.30~0.55 | 약한 매치 |
| < 0.15 | 무관 |
| **임계값** | **0.15** (config.yaml) |

텍스트↔텍스트 비교이므로 V-axis보다 절대값이 훨씬 높다.

### 강점과 약점

- **강점**: 의미적 유사성 (동의어, 유사 개념 매칭). "거리 풍경" 검색 시 "urban street scene"도 매칭
- **약점**: AI 캡션 품질에 의존. 캡션이 부정확하면 검색도 부정확
- **약점**: Ollama 서버가 꺼져 있으면 이 축 비활성화

### 코드 위치

- 임베딩: `backend/vector/text_embedding.py` → `OllamaEmbeddingProvider`
- 저장: `backend/db/sqlite_client.py` → `vec_text` 테이블
- 검색: `backend/search/sqlite_search.py` → `text_vector_search()`

---

## 축 3: M-axis (Metadata) — "검색어가 글자 그대로 데이터에 있는가"

### 개념

이미지에 연결된 **모든 텍스트 데이터**를 FTS5 전문 검색 인덱스에 넣는다.
검색할 때 키워드가 **문자열로 그대로 존재하는지** 찾는다.

V-axis가 "눈으로 보기", S-axis가 "뜻을 이해하기"라면, M-axis는 **"Ctrl+F 찾기"**.

### 기술

| 항목 | 값 |
|------|------|
| 엔진 | SQLite FTS5 (Full-Text Search 5) |
| 인덱스 컬럼 | 16개 |
| 별도 모델 | 없음 (순수 텍스트 매칭) |

### 저장하는 데이터

FTS5 인덱스에 들어가는 **16개 컬럼**:

| 카테고리 | 컬럼 | 출처 | 예시 |
|----------|------|------|------|
| **파일 정보** | `file_path` | 파일 경로 | `Characters/Hero/warrior.psd` |
| | `file_name` | 파일명 | `warrior.psd` |
| **AI 생성** | `ai_caption` | 비전 모델 캡션 | `A warrior in dark armor standing in rain` |
| | `ai_tags` | 비전 모델 태그 | `warrior, armor, dark, rain, fantasy` |
| | `ai_style` | 스타일 분석 | `anime illustration` |
| | `ocr_text` | 이미지 내 글자 (OCR) | `GAME OVER` |
| | `dominant_color` | 주요 색상 | `dark blue` |
| **PSD 구조** | `layer_names` | 레이어명 (원본+번역) | `Background 배경 그림자 Shadow` |
| | `text_content` | 텍스트 레이어 내용 | `Title: The Last Knight` |
| | `used_fonts` | 사용된 폰트 | `NotoSansKR Arial` |
| **사용자 입력** | `user_note` | 메모 | `보스전 배경으로 사용` |
| | `user_tags` | 커스텀 태그 | `boss battle reference` |
| | `folder_tags` | 폴더 경로 태그 | `Characters Hero` |
| **v3 분류** | `image_type` | 이미지 종류 | `character` |
| | `scene_type` | 장면 종류 | `dungeon` |
| | `art_style` | 화풍 | `anime` |

### 검색 원리

```
검색 키워드: ["건널목", "crosswalk", "상점가", "shopping"]
  → FTS5 MATCH 쿼리:
     '"건널목" OR "crosswalk" OR "상점가" OR "shopping"'
  → 16개 컬럼 전체에서 글자 매칭
  → FTS5 내장 랭킹 (BM25 유사)
```

### 점수 특성

| 구간 | 의미 |
|------|------|
| FTS rank | 음수 값. 더 음수 = 더 좋은 매치 |
| 정규화 후 | 0~1로 변환. 1.0 = 최고 매치, 0.0 = 최하 매치 |
| **임계값** | 없음 (매치되면 포함) |

### 강점과 약점

- **강점**: 정확한 키워드 매칭. "warrior"를 검색하면 "warrior"가 들어간 것만 나옴
- **강점**: 16개 컬럼을 한 번에 검색 (파일명, 레이어명, 캡션, 태그, 메모, 폰트 등)
- **강점**: 다국어 지원 (원본 + 한국어 + 영어 모두 인덱싱)
- **약점**: 동의어 불가. "전사"를 검색해도 "warrior"는 안 나옴 (그건 S-axis의 역할)
- **약점**: 오타에 취약

### 코드 위치

- 인덱싱: `backend/db/sqlite_client.py` → `files_fts` 테이블
- 검색: `backend/search/sqlite_search.py` → `fts_search()`

---

## 3축 합산: RRF (Reciprocal Rank Fusion)

### 개념

3개 축이 각자 순위를 매긴 후, **순위를 가중 합산**하여 최종 순위를 결정한다.
점수 스케일이 축마다 다르기 때문에 점수가 아니라 **순위(rank)**를 기반으로 합산한다.

### 공식

```
RRF_score(파일) = Σ  weight[축] / (k + rank[축] + 1)

k = 60 (고정 상수, config.yaml)
```

### 예시

```
"건널목과 상점가" 검색 시:

107.psd:
  V-axis 1등 (rank=0):  0.20 / (60 + 0 + 1) = 0.00328
  S-axis 5등 (rank=4): 0.30 / (60 + 4 + 1) = 0.00462
  M-axis 3등 (rank=2):  0.50 / (60 + 2 + 1) = 0.00794
  ───────────────────────────────────────────
  RRF 합산 = 0.01584
```

### 가중치 (query_type별 자동 선택)

QueryDecomposer가 쿼리를 분석해서 유형을 판단하고, 유형에 맞는 가중치를 적용한다.

| query_type | V (Visual) | S (Semantic) | M (Metadata) | 적용 상황 |
|------------|--------|---------|--------|-----------|
| **visual** | **0.50** | 0.30 | 0.20 | 색감, 분위기, 시각적 묘사 |
| **keyword** | 0.20 | 0.30 | **0.50** | 구체적 물체명, 장면 |
| **semantic** | 0.20 | **0.50** | 0.30 | 용도, 맥락, 추상적 설명 |
| **balanced** | 0.34 | 0.33 | 0.33 | 혼합 또는 분류 불확실 |

예: "파란 톤의 야경" → visual → V에 50% 가중치
예: "건널목 상점가" → keyword → M에 50% 가중치

---

## 데이터 흐름 요약

### 인제스트 (이미지 등록 시)

```
이미지 파일
  │
  ├─→ 썸네일 생성 → SigLIP2 인코더 → N차원 벡터 → vec_files (V-axis)
  │
  ├─→ qwen3-vl 비전 분석 → ai_caption + ai_tags 생성
  │     │
  │     ├─→ qwen3-embedding → N차원 벡터 → vec_text (S-axis)
  │     │
  │     └─→ 캡션+태그+레이어명+메모+... → files_fts 인덱스 (M-axis)
  │
  └─→ 메타데이터 → files 테이블 (format, resolution, layer_tree, ...)
```

### 검색 시

```
사용자 쿼리
  │
  ├─→ QueryDecomposer → vector_query (영문) + fts_keywords (다국어)
  │
  ├─→ V-axis:  SigLIP2 텍스트 인코더 → vec_files 코사인 유사도 → 상위 40개
  ├─→ S-axis: qwen3-embedding 인코더 → vec_text 코사인 유사도 → 상위 40개
  ├─→ M-axis:  FTS5 MATCH 키워드 검색 → 상위 40개
  │
  └─→ RRF Merge (가중 순위 합산) → 최종 top 20 반환
```

---

## DB 스키마 요약

```sql
-- 메인 테이블: 이미지 메타데이터 + AI 분석 결과
files (
    id, file_path, file_name,
    format, width, height,
    ai_caption, ai_tags, ai_style, ocr_text, dominant_color,
    metadata,  -- JSON: layer_tree, semantic_tags, text_content, used_fonts
    user_note, user_tags, user_category, user_rating,
    folder_path, folder_tags,
    image_type, art_style, scene_type, ...
)

-- V-axis 벡터 저장 (SigLIP2, tier별 차원: standard=768, pro=1152, ultra=1664)
vec_files (file_id, embedding FLOAT[dim])

-- S-axis 벡터 저장 (Qwen3-Embedding, tier별 차원: standard=256, pro=1024, ultra=4096)
vec_text (file_id, embedding FLOAT[dim])

-- M-axis 전문 검색 인덱스 (16컬럼)
files_fts (
    file_path, file_name,
    ai_caption, ai_tags, ai_style, ocr_text, dominant_color,
    layer_names, text_content, used_fonts,
    user_note, user_tags, folder_tags,
    image_type, scene_type, art_style
)
```

---

## GUI 뱃지 대응

| 뱃지 | 색상 | 축 | 의미 (한줄 요약) |
|-------|------|-----|-----------------|
| **VV 12** | 파랑 | V-axis | 이미지 자체가 쿼리와 시각적으로 12% 유사 |
| **MV 67** | 보라 | S-axis | AI 설명이 쿼리와 의미적으로 67% 유사 |
| **MC 85** | 초록 | M-axis | 키워드가 데이터에 85% 수준으로 매칭 |
| **★ 98%** | 노랑 | RRF 종합 | 3축 가중 합산 순위 기준 상위 98% |
