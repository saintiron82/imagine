# Phase 4: Vision Analysis System (Descriptive Axis) - 상세 명세서

## 📋 목차
1. [목적 및 범위](#목적-및-범위)
2. [기술 스택 선택](#기술-스택-선택)
3. [시스템 아키텍처](#시스템-아키텍처)
4. [유닛 단위 작업 계획](#유닛-단위-작업-계획)
5. [통합 계획](#통합-계획)
6. [성능 및 제약 사항](#성능-및-제약-사항)

---

## 목적 및 범위

### 핵심 목표
**3-Axis 아키텍처의 마지막 축인 "Descriptive Axis (서술적 축)"를 구축**하여, 이미지를 인간이 이해할 수 있는 자연어로 설명하고 정밀한 키워드 검색을 가능하게 합니다.

### 현재 상태 (Phase 1-3)
| Axis | 데이터 | 용도 | 상태 |
|------|--------|------|------|
| **Structural** | 레이어 계층, 텍스트, 좌표 | 구조 분석, 편집 | ✅ 완료 |
| **Latent** | CLIP 768차원 벡터 | 시각적 유사도 검색 | ✅ 완료 |
| **Descriptive** | AI 캡션, 태그 | 키워드 검색, 문맥 이해 | ❌ 미구현 |

### Phase 4가 추가하는 것
```
입력: 이미지 (PSD Composite / PNG / JPG)
    ↓
Vision Language Model (VLM)
    ↓
출력:
  1. 캡션 (Caption): "A fantasy knight character with glowing sword and cape"
  2. 태그 (Tags): ["knight", "sword", "armor", "fantasy", "character"]
  3. 스타일 (Style): "Dark Fantasy", "Cinematic Lighting"
  4. 색감 (Color): "Dark tones with blue highlights"
```

### 사용 사례
1. **정밀 검색**: "Find all images with swords and armor"
2. **컨텍스트 이해**: "Show me fantasy character designs"
3. **자동 태깅**: 수동 태깅 없이 자동 분류
4. **다국어 검색**: 캡션을 번역하여 한글 검색 지원

---

## 기술 스택 선택

### 환경 제약 사항
- **GPU**: NVIDIA RTX 3060 Ti (8GB VRAM, CUDA 12.6)
- **Python**: 3.11.9
- **사용 가능 VRAM**: ~5.5GB (현재 2.5GB 사용 중)
- **요구사항**: 로컬 실행, 빠른 추론 (<1초/image), 다국어 지원

### Vision Language Model 후보 비교

| 모델 | VRAM | 속도 | 정확도 | 다국어 | 추천도 |
|------|------|------|--------|--------|--------|
| **Florence-2** | ~2GB | ⚡ 빠름 | 🟢 우수 | 영어 주력 | ⭐⭐⭐⭐⭐ |
| **Moondream** | ~2GB | ⚡ 매우 빠름 | 🟡 준수 | 영어 전용 | ⭐⭐⭐⭐ |
| **Qwen-VL** | ~6GB | 🐌 보통 | 🟢 매우 우수 | 중국어/영어 | ⭐⭐⭐ |
| **LLaVA-1.5** | ~8GB+ | 🐌 느림 | 🟢 우수 | 영어 주력 | ⭐⭐ |

### ✅ 최종 선택: **Florence-2-large**

**선택 이유:**
1. ✅ **경량**: 2GB VRAM으로 RTX 3060 Ti에 여유롭게 실행
2. ✅ **다목적**: Captioning, Object Detection, OCR 통합 지원
3. ✅ **Microsoft 공식**: 잘 관리되는 오픈소스 (MIT 라이센스)
4. ✅ **HuggingFace 통합**: `transformers` 라이브러리로 간단히 사용
5. ✅ **속도**: ~0.5초/image (CLIP과 병렬 실행 가능)

**모델 정보:**
- **HuggingFace ID**: `microsoft/Florence-2-large`
- **크기**: ~771MB (모델 파일)
- **입력**: 이미지 (RGB, 임의 크기)
- **출력**: JSON 구조화된 결과

---

## 시스템 아키텍처

### 데이터 흐름

```
[기존 파이프라인]
이미지 → PSDParser/ImageParser → AssetMeta → VectorIndexer (CLIP)
                                      ↓
                                    JSON 저장

[Phase 4 추가]
이미지 → VisionAnalyzer (NEW)
           ├─ Florence-2 모델 로드 (lazy)
           ├─ 캡션 생성 (detailed_caption)
           ├─ 객체 탐지 (object_detection)
           └─ OCR (dense_region_caption)
           ↓
       AssetMeta 확장
           ├─ ai_caption: str
           ├─ ai_tags: List[str]
           ├─ ai_objects: List[Dict]
           └─ ai_style: str
           ↓
       ChromaDB 메타데이터 확장
```

### 디렉토리 구조

```
backend/
├── vision/
│   ├── __init__.py
│   ├── analyzer.py           # VisionAnalyzer 클래스
│   ├── prompt_templates.py   # 프롬프트 엔지니어링
│   └── post_processor.py     # 결과 후처리 (태그 추출 등)
├── parser/
│   └── schema.py             # AssetMeta 확장
├── pipeline/
│   └── ingest_engine.py      # Vision 단계 통합
└── vector/
    └── indexer.py            # 메타데이터에 ai_caption 추가
```

### 스키마 확장

**기존 `AssetMeta` 확장:**
```python
class AssetMeta(BaseModel):
    # [기존 필드들...]

    # === Phase 4: Descriptive Axis ===
    ai_caption: Optional[str] = Field(
        None,
        description="AI-generated detailed caption"
    )
    ai_tags: List[str] = Field(
        default_factory=list,
        description="AI-extracted tags (objects, styles, etc.)"
    )
    ai_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected objects with bounding boxes"
    )
    ai_style_description: Optional[str] = Field(
        None,
        description="Style, lighting, mood description"
    )
```

---

## 유닛 단위 작업 계획

### U-015: Vision 모듈 기반 구축 (Foundation)

**목표:**
- Florence-2 모델을 로컬에서 실행할 수 있는 기반 인프라 구축
- VisionAnalyzer 클래스 구현 및 기본 캡션 생성 검증

**정의:**
```
입력: PIL Image 객체
출력: Dict 형태의 분석 결과
  - caption: str
  - objects: List[Dict]
  - tags: List[str]
```

**구현 파일:**
- `backend/vision/__init__.py`
- `backend/vision/analyzer.py`

**의존성 추가:**
```python
# requirements.txt에 추가
transformers>=4.40.0
timm>=0.9.0
einops>=0.7.0
```

**테스트 기준:**
```python
# 테스트 코드
from backend.vision.analyzer import VisionAnalyzer
from PIL import Image

analyzer = VisionAnalyzer()
image = Image.open("test_assets/sample.png")
result = analyzer.analyze(image)

assert "caption" in result
assert len(result["caption"]) > 10
assert "tags" in result
assert len(result["tags"]) > 0
```

**예상 소요 시간:** 2-3시간

---

### U-016: 프롬프트 엔지니어링 (Prompt Engineering)

**목표:**
- Florence-2의 다양한 태스크를 활용하여 최적의 결과 도출
- 프롬프트 템플릿 설계 및 후처리 로직 구현

**정의:**
```
Task Types (Florence-2):
1. <DETAILED_CAPTION>: 상세 설명 (우선)
2. <OD>: Object Detection
3. <DENSE_REGION_CAPTION>: OCR 및 영역별 설명
4. <MORE_DETAILED_CAPTION>: 초상세 모드

Output:
  - 통합 캡션 (200자 내외)
  - 정제된 태그 리스트 (중복 제거, 소문자 통일)
  - 스타일 추론 (휴리스틱)
```

**구현 파일:**
- `backend/vision/prompt_templates.py`
- `backend/vision/post_processor.py`

**프롬프트 예시:**
```python
PROMPTS = {
    "detailed_caption": "<DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "ocr": "<OCR_WITH_REGION>",
}

def build_caption_prompt(image_context: str) -> str:
    """
    Args:
        image_context: "character design", "background art", "UI element"
    Returns:
        Customized prompt string
    """
    return f"<DETAILED_CAPTION> Focus on {image_context}"
```

**테스트 기준:**
```python
result = analyzer.analyze(image, context="character design")
assert "character" in result["caption"].lower()
assert len(result["tags"]) >= 5
assert "style" in result  # 스타일 자동 추론
```

**예상 소요 시간:** 2-3시간

---

### U-017: 파이프라인 통합 (Pipeline Integration)

**목표:**
- `ingest_engine.py`에 Vision 분석 단계 추가
- AssetMeta 스키마에 AI 필드 자동 저장
- ChromaDB 메타데이터에 ai_caption 인덱싱

**정의:**
```
기존 파이프라인:
  파싱 → 번역 → 벡터화 → 저장

Phase 4 파이프라인:
  파싱 → 번역 → 벡터화 → [Vision 분석] → 저장
```

**수정 파일:**
- `backend/pipeline/ingest_engine.py` (Vision 단계 추가)
- `backend/parser/schema.py` (AssetMeta 필드 확장)
- `backend/vector/indexer.py` (메타데이터에 ai_caption 추가)

**코드 예시:**
```python
# ingest_engine.py 수정
def process_file(file_path: Path):
    # ... 기존 파싱 로직 ...

    # === Phase 4: Vision Analysis ===
    try:
        from backend.vision.analyzer import VisionAnalyzer

        if '_global_vision_analyzer' not in globals():
            global _global_vision_analyzer
            _global_vision_analyzer = VisionAnalyzer()

        # 썸네일 이미지로 분석 (속도 최적화)
        if meta.thumbnail_url:
            thumb_path = Path(meta.thumbnail_url.replace("file:///", ""))
            vision_result = _global_vision_analyzer.analyze_file(thumb_path)

            # AssetMeta에 결과 추가
            meta.ai_caption = vision_result["caption"]
            meta.ai_tags = vision_result["tags"]
            meta.ai_objects = vision_result.get("objects", [])
            meta.ai_style_description = vision_result.get("style", "")

    except Exception as e:
        logger.warning(f"Vision analysis failed: {e}")

    # 저장 (JSON + Vector DB)
    parser._save_json(meta, file_path)
    indexer.index_image(file_path, meta.model_dump(), thumb_path)
```

**테스트 기준:**
```powershell
# E2E 테스트
python backend/pipeline/ingest_engine.py --file "test_assets/character.psd"

# 검증
python -c "
import json
data = json.load(open('output/json/character.json'))
assert 'ai_caption' in data
assert 'ai_tags' in data
print('Vision integration OK')
"
```

**예상 소요 시간:** 3-4시간

---

### U-018: 검색 시스템 확장 (Search Enhancement)

**목표:**
- 텍스트 검색 시 ai_caption도 함께 검색
- 하이브리드 검색: Vector (CLIP) + Keyword (Caption)
- 검색 결과에 AI 태그 표시

**정의:**
```
검색 로직:
1. CLIP 벡터 검색 (기존)
2. ai_caption에서 키워드 필터링 (NEW)
3. 두 결과를 스코어 기반으로 병합

예:
  Query: "fantasy sword character"
  - CLIP: 시각적 유사도 스코어
  - Caption: "sword" 포함 여부로 가중치 부여
  - 최종 스코어 = 0.7 * CLIP + 0.3 * Caption Match
```

**수정 파일:**
- `backend/vector/searcher.py` (하이브리드 검색)
- `backend/cli_search.py` (결과 포맷팅)
- `frontend/src/components/FileGrid.jsx` (AI 태그 표시)

**코드 예시:**
```python
# searcher.py 확장
def search_hybrid(self, query: str, top_k: int = 20) -> List[Dict]:
    """
    하이브리드 검색: Vector + Keyword
    """
    # 1. CLIP 벡터 검색
    vector_results = self.search(query, top_k=top_k*2)

    # 2. 키워드 필터링
    query_words = set(query.lower().split())

    # 3. 스코어 재계산
    for result in vector_results:
        caption = result["metadata"].get("ai_caption", "").lower()
        caption_words = set(caption.split())

        # Jaccard 유사도
        keyword_match = len(query_words & caption_words) / len(query_words | caption_words)

        # 통합 스코어
        result["score"] = 0.7 * result["score"] + 0.3 * keyword_match

    # 정렬 및 반환
    return sorted(vector_results, key=lambda x: x["score"], reverse=True)[:top_k]
```

**테스트 기준:**
```powershell
# 검색 테스트
python backend/cli_search.py "fantasy character with sword"

# 예상 결과:
# [1] character_knight.psd (score: 0.89)
#     Caption: "A fantasy knight character wielding a glowing sword..."
#     Tags: knight, sword, armor, fantasy
```

**예상 소요 시간:** 2-3시간

---

### U-019: GUI 통합 및 시각화 (UI Integration)

**목표:**
- Electron GUI에서 AI 분석 결과 표시
- 메타데이터 모달에 "AI 분석" 탭 추가
- 태그 클릭 시 동일 태그 이미지 필터링

**정의:**
```
FileGrid.jsx 확장:
1. 썸네일 위에 AI 태그 배지 표시
2. 메타데이터 모달에 "AI Analysis" 탭 추가
   - Caption 표시
   - Tags (클릭 가능)
   - Objects (바운딩 박스 정보)
3. 태그 클릭 → 해당 태그로 검색 필터링
```

**수정 파일:**
- `frontend/src/components/FileGrid.jsx`
- `frontend/src/components/MetadataModal.jsx` (새 컴포넌트)

**UI 목업:**
```
┌─────────────────────────────────────┐
│ [Image Thumbnail]                   │
│ ┌─────┐ ┌────────┐ ┌──────┐        │
│ │knight│ │fantasy │ │sword │ [+3]  │
│ └─────┘ └────────┘ └──────┘        │
│                                     │
│ [메타데이터 보기] [처리하기]        │
└─────────────────────────────────────┘

[메타데이터 모달 - AI Analysis 탭]
┌─────────────────────────────────────┐
│ 📝 AI Caption                       │
│ A fantasy knight character with     │
│ glowing blue sword and red cape,    │
│ standing in dramatic pose...        │
│                                     │
│ 🏷️ AI Tags                          │
│ #knight #sword #armor #fantasy      │
│ #character #blue-glow #cape         │
│                                     │
│ 🎨 Style                            │
│ Dark Fantasy, Cinematic Lighting    │
└─────────────────────────────────────┘
```

**테스트 기준:**
- ✅ GUI에서 AI 태그 표시 확인
- ✅ 메타데이터 모달에서 Caption 확인
- ✅ 태그 클릭 시 필터링 작동

**예상 소요 시간:** 3-4시간

---

## 통합 계획

### Phase 4 완료 기준

**필수 (Must Have):**
- ✅ Florence-2 모델 로컬 실행
- ✅ 캡션 자동 생성 (ingest_engine 통합)
- ✅ AssetMeta에 AI 필드 저장
- ✅ ChromaDB 메타데이터 확장
- ✅ GUI에서 AI 태그 표시

**선택 (Nice to Have):**
- 🔲 하이브리드 검색 (Vector + Keyword)
- 🔲 스타일 자동 분류
- 🔲 Object Detection 바운딩 박스 표시
- 🔲 다국어 캡션 (번역 통합)

### 작업 순서

```
Week 1: Foundation
├─ Day 1-2: U-015 (Vision 모듈 기반)
├─ Day 3-4: U-016 (프롬프트 엔지니어링)
└─ Day 5:   통합 테스트

Week 2: Integration
├─ Day 1-2: U-017 (파이프라인 통합)
├─ Day 3:   U-018 (검색 확장)
├─ Day 4-5: U-019 (GUI 통합)
└─ Day 6:   E2E 테스트 및 문서화
```

---

## 성능 및 제약 사항

### 성능 목표

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| **모델 로드 시간** | <10초 | 첫 실행 시 측정 |
| **분석 속도** | <1초/image | 100개 이미지 배치 평균 |
| **VRAM 사용량** | <4GB | nvidia-smi 모니터링 |
| **정확도** | >80% | 수동 샘플 검증 (50개) |

### 제약 사항

**하드웨어:**
- RTX 3060 Ti 8GB VRAM 기준으로 최적화
- CPU 전용 환경에서는 속도 10배 저하 예상

**소프트웨어:**
- Florence-2는 영어 캡션 생성 (한글 번역 필요)
- Object Detection은 COCO 데이터셋 기준 (80 classes)
- PSD 레이어별 분석은 Phase 5로 연기

**데이터:**
- 썸네일 이미지로 분석 (224x224)
- 원본 고해상도 분석은 선택적 제공

---

## 다음 단계

1. **의존성 설치 검증**
   ```powershell
   python backend/setup/installer.py --check
   # torch, chromadb, sentence-transformers 설치 확인
   ```

2. **Phase 4 시작 준비**
   ```powershell
   # Vision 디렉토리 생성
   mkdir backend/vision

   # 유닛 개발 시작
   /unit-start  # U-015: Vision 모듈 기반 구축
   ```

3. **테스트 자산 준비**
   - `test_assets/` 에 다양한 스타일의 이미지 10개 준비
   - 수동 레이블링 (정답 캡션/태그) 작성

---

**작성일:** 2026-02-06
**작성자:** Claude (ImageParser Phase 4 기획)
**다음 문서:** `U-015_vision_module_specification.md` (작업 시작 시 생성)
