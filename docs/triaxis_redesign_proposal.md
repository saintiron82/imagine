# 3-Axis Search 재설계 제안서

## 현재 문제

```
현재 3축:
  V-axis:  이미지 → SigLIP 벡터         (시각)
  S-axis: AI캡션 → Qwen3 벡터          (AI 해석의 벡터 형태)
  M-axis:  AI캡션+메타데이터 → FTS 키워드 (전부 다 Ctrl+F)
           ~~~~~~~ 겹침 ~~~~~~~
```

| 문제 | 설명 |
|------|------|
| S-axis ↔ M-axis 중복 | 둘 다 같은 AI 캡션+태그를 다른 방법으로 검색 |
| S-axis가 메타데이터 무시 | 캡션+태그 2개 필드만 벡터화. 레이어명, 폴더, 사용자 메모 등 미활용 |
| AI가 이미지만 봄 | 폴더 구조, 레이어명 등 맥락 정보 없이 캡션 생성 |
| M-axis 무차별 검색 | AI 해석과 메타데이터를 구분하지 않고 16컬럼 한꺼번에 검색 |

---

## 제안: 새 3축 구조

```
축 1 (V):  이미지 벡터       "이 이미지가 시각적으로 어떻게 생겼는가"
축 2 (M):  메타데이터 검색    "이 이미지에 대해 알려진 구조적/맥락적 정보"
축 3 (S):  통합 AI 해석      "AI가 이미지+메타를 함께 보고 내린 해석"
```

### 왜 이 구조인가

- **V**: 이미지 픽셀에서 직접 추출. 텍스트로 설명 불가능한 시각적 특징 (색감, 구도, 분위기)
- **M**: 이미지 자체가 아닌 "이미지에 대해 알려진 정보". 파일명, 레이어명, 폴더, 사용자 메모 등
- **S**: AI가 V(이미지)와 M(메타데이터)를 **통합 해석**한 결과. 가장 풍부한 정보를 담음

3축이 **서로 다른 정보원**에서 출발하므로 중복이 없다:
- V = 픽셀 (기계가 봄)
- M = 구조적 사실 (파싱/사용자 입력)
- S = AI의 종합 판단 (이미지 + 메타데이터를 보고 해석)

---

## 축별 상세

### 축 1: V-axis (Visual) — 변경 없음

| 항목 | 값 |
|------|------|
| 모델 | SigLIP 2 NaFlex (1152차원) |
| 입력 | 썸네일 이미지 (픽셀) |
| 저장 | `vec_files` 테이블 |
| 검색 | 쿼리 → SigLIP 텍스트 인코더 → 코사인 유사도 |
| 점수 범위 | 0.10~0.17 (좋은 매치) |

현재 잘 동작하므로 변경 불필요.

---

### 축 2: M-axis (Metadata) — FTS5 키워드 검색

**역할**: 이미지의 구조적/맥락적 메타데이터를 **키워드로** 검색

**검색 대상 (메타데이터 컬럼만)**:

| 컬럼 | 출처 | 예시 |
|------|------|------|
| `file_path` | 파일 시스템 | `Characters/Boss/warrior.psd` |
| `file_name` | 파일 시스템 | `warrior_boss.psd` |
| `layer_names` | PSD 파싱 | `Background Shadow_Effect 그림자 Boss_Arena` |
| `text_content` | PSD 텍스트 레이어 | `GAME OVER 最終ボス` |
| `used_fonts` | PSD 파싱 | `NotoSansKR Arial Pretendard` |
| `ocr_text` | 이미지 OCR | `EXIT SAVE` |
| `user_note` | 사용자 입력 | `보스전 배경으로 사용` |
| `user_tags` | 사용자 입력 | `boss reference dark` |
| `folder_tags` | 폴더 경로 | `Characters Boss` |

**왜 FTS(키워드)인가**: 메타데이터는 고유명사가 많다.
"NotoSansKR", "Shadow_Effect", "warrior_boss.psd" 같은 건 의미 매칭보다 **글자 매칭이 정확**하다.

**현재 M-axis와 차이**: AI가 생성한 컬럼(ai_caption, ai_tags, ai_style 등)을 **제외**하고 메타데이터 컬럼만 검색.

---

### 축 3: S-axis (Semantic) — AI 통합 해석 벡터

**역할**: AI가 **이미지 + 메타데이터를 동시에 보고** 종합 해석을 생성, 이를 벡터화

#### 현재 vs 변경 후

```
현재:
  qwen3-vl이 이미지만 봄
  → "A dark alley at night"

변경 후:
  qwen3-vl이 이미지 + 메타데이터 맥락을 함께 받음
  → "A dark alley scene designed as a boss battle arena.
     Contains shadow effect layers and Japanese text '最終ボス' (Final Boss).
     Located in the Characters/Boss asset folder, suggesting this is
     an enemy encounter background with NotoSansKR typography."
```

#### 비전 프롬프트 변경

```
현재 Stage 2 프롬프트:
  "Describe this image in detail. Output JSON: {caption, tags, ...}"

변경 후 Stage 2 프롬프트:
  "Describe this image in detail.
   Consider the following file context:
   - File: warrior_boss.psd
   - Folder: Characters/Boss
   - Layers: Background, Shadow_Effect, Boss_Arena, Sword_Glow
   - Text in file: GAME OVER, 最終ボス
   - Fonts: NotoSansKR, Arial

   Generate a caption that integrates what you SEE in the image
   with the contextual metadata above.
   Output JSON: {caption, tags, ...}"
```

#### 벡터화

통합 캡션을 `qwen3-embedding:0.6b`로 벡터화 → `vec_text` 테이블 (기존 재활용).
메타데이터가 이미 캡션에 **해석되어** 녹아 있으므로 `build_document_text()`는 캡션+태그만 사용.

#### S-axis의 가치

| 메타데이터 원본 | AI가 해석한 결과 |
|----------------|-----------------|
| `layer: Shadow_Effect` | "그림자 효과가 적용된" |
| `folder: Characters/Boss` | "보스전 관련 에셋" |
| `text: 最終ボス` | "일본어 텍스트 '최종 보스'" |
| `font: NotoSansKR` | "한국어 타이포그래피 사용" |

AI가 **기계적 메타데이터를 인간 언어로 해석**해주므로, "보스전 배경" 검색 시 벡터 의미 매칭이 가능해진다.

---

## 기존 대비 변경 요약

| 구성 요소 | 현재 | 변경 후 |
|-----------|------|---------|
| `vec_files` | SigLIP 이미지 벡터 | 그대로 (V-axis) |
| `vec_text` | 캡션+태그만 벡터화 | **통합 AI 해석** 벡터화 (S-axis) |
| `files_fts` | 16컬럼 무차별 검색 | **메타데이터 컬럼만** (M-axis) |
| 비전 프롬프트 | 이미지만 입력 | **이미지 + 메타데이터 맥락** 입력 |
| AI 캡션 품질 | 이미지만 보고 생성 | 이미지 + 맥락 정보 통합 해석 |

---

## RRF 가중치 재설계

```python
# 현재
WEIGHT_PRESETS = {
    "visual":   {"visual": 0.50, "text_vec": 0.30, "fts": 0.20},
    "keyword":  {"visual": 0.20, "text_vec": 0.30, "fts": 0.50},
    "semantic": {"visual": 0.20, "text_vec": 0.50, "fts": 0.30},
    "balanced": {"visual": 0.34, "text_vec": 0.33, "fts": 0.33},
}

# 변경 후
WEIGHT_PRESETS = {
    "visual":   {"visual": 0.50, "semantic": 0.35, "meta": 0.15},
    "keyword":  {"visual": 0.15, "semantic": 0.25, "meta": 0.60},
    "semantic": {"visual": 0.20, "semantic": 0.55, "meta": 0.25},
    "balanced": {"visual": 0.30, "semantic": 0.40, "meta": 0.30},
}
```

| query_type | 의도 | 주력 축 |
|------------|------|---------|
| visual | 색감, 분위기, 시각적 묘사 | V-axis (50%) |
| keyword | 파일명, 레이어명, 폰트명 등 구체적 이름 | M-axis (60%) |
| semantic | 용도, 맥락, 추상적 설명 | S-axis (55%) |
| balanced | 혼합 또는 분류 불확실 | S-axis 약간 우세 (40%) |

---

## GUI 뱃지 변경

| 현재 | 변경 후 | 의미 |
|------|---------|------|
| **V** (파랑) | **V** (파랑) | 시각적 유사도 |
| **Tv** (보라) | **S** (보라) | AI 통합 해석 유사도 |
| **T** (초록) | **M** (초록) | 메타데이터 키워드 매칭 |

---

## 검색 시나리오 비교

| 쿼리 | 현재 동작 | 변경 후 동작 |
|------|----------|-------------|
| "보스전 배경" | user_note에 "보스전" 있어도 S-axis 무반응. M-axis 키워드만 매칭 | M-axis: user_note "보스전" 키워드 매칭. S-axis: 캡션에 "boss battle arena" 의미 매칭 |
| "Shadow 레이어 있는 파일" | M-axis: 16컬럼에서 "Shadow" 키워드 찾음 (AI캡션 노이즈 포함) | M-axis: layer_names에서만 정확 매칭 |
| "NotoSans 폰트 사용" | M-axis에서 찾지만 Tv와 혼합 순위 | M-axis: used_fonts에서 정확 매칭. 깔끔한 결과 |
| "파란 톤 야경" | V-axis만 유효 | V-axis + S-axis(메타 맥락 포함 캡션) 이중 매칭 |
| "Characters 폴더의 검 든 캐릭터" | M-axis 키워드 + Tv 부분 매칭 | M-axis(folder 키워드) + S-axis("sword character" 의미) 정확 분리 |
| "그림자 효과 적용된 일러스트" | M-axis: "그림자" 키워드만 | S-axis: AI가 "Shadow_Effect → 그림자 효과" 해석한 캡션에서 의미 매칭 |

---

## 향후 확장 가능성

현재 제안은 최소 구현이며, 검색 품질 평가 후 다음 확장이 가능하다:

1. **S-axis FTS 보조**: AI 캡션에 대한 키워드 검색을 S-axis에 추가 (벡터 + FTS 하이브리드)
2. **M-axis 벡터화**: 메타데이터도 벡터화해서 의미 검색 추가 (예: "그림자 효과" → "Shadow_Effect")
3. **사용자 메모 실시간 반영**: 사용자가 메모/태그 수정 시 S-axis 캡션 재생성 없이 M-axis FTS만 즉시 업데이트

---

## 사용 중인 AI 모델 (변경 후)

| # | 모델 | 용도 | 변경 |
|---|------|------|------|
| 1 | qwen3-vl:8b | 이미지+메타 통합 해석 (캡션 생성) | **프롬프트에 메타데이터 맥락 추가** |
| 2 | SigLIP 2 NaFlex | 이미지 벡터 인코딩 (V-axis) | 변경 없음 |
| 3 | qwen3-embedding:0.6b | 통합 캡션 벡터 인코딩 (S-axis) | 변경 없음 (입력 캡션이 더 풍부해짐) |
