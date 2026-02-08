# Phase 4: Vision Analysis System - 준비 완료 요약

## ✅ 완료된 작업

### 1. 명세서 작성
- **파일**: `docs/phase4_vision_specification.md` (5,000+ 단어)
- **내용**:
  - 3-Axis 아키텍처의 마지막 축 정의 (Descriptive Axis)
  - 기술 스택 선택 및 비교 (Florence-2 최종 선택)
  - 시스템 아키텍처 및 데이터 흐름
  - 5개 유닛으로 작업 분해 (U-015 ~ U-019)
  - 각 유닛별 목표, 정의, 테스트 기준, 예상 시간

### 2. 작업 체크리스트 업데이트
- **파일**: `docs/project_checklist.md`
- **추가 내용**:
  - U-015: Vision 모듈 기반 구축 (2-3시간)
  - U-016: 프롬프트 엔지니어링 (2-3시간)
  - U-017: 파이프라인 통합 (3-4시간)
  - U-018: 검색 시스템 확장 (2-3시간)
  - U-019: GUI 통합 및 시각화 (3-4시간)
  - **총 예상 시간**: 12-17시간

### 3. 시작 가이드 작성
- **파일**: `docs/phase4_getting_started.md`
- **내용**:
  - 환경 확인 체크리스트
  - 의존성 설치 가이드
  - 기존 시스템 검증 절차
  - Florence-2 예제 코드
  - 타임라인 및 문제 해결 가이드

---

## 🎯 Phase 4 핵심 목표

### What (무엇을)
**Descriptive Axis (서술적 축)** 구축
- AI가 이미지를 자연어로 설명
- 자동 태그 추출
- 키워드 기반 정밀 검색

### Why (왜)
현재는 **시각적 유사도만** 검색 가능:
```
Query: "fantasy sword character"
→ CLIP 벡터 검색 → 시각적으로 비슷한 이미지

문제: "sword"라는 단어를 직접 찾을 수 없음
```

Phase 4 완료 후:
```
Query: "fantasy sword character"
→ Vector 검색 (70%) + Caption 키워드 검색 (30%)
→ "sword" 포함 이미지 우선 표시
```

### How (어떻게)
**Florence-2 모델** 사용:
- Microsoft 공식 오픈소스 (MIT 라이센스)
- 2GB VRAM (RTX 3060 Ti 여유 있음)
- 캡션, 객체 탐지, OCR 통합 지원
- ~0.5초/image (빠른 추론)

---

## 📊 작업 분해 (5개 유닛)

| 유닛 | 제목 | 시간 | 우선순위 | 상태 |
|------|------|------|----------|------|
| **U-015** | Vision 모듈 기반 구축 | 2-3h | 🚨 CRITICAL | 📋 준비됨 |
| **U-016** | 프롬프트 엔지니어링 | 2-3h | 🔴 HIGH | 📋 준비됨 |
| **U-017** | 파이프라인 통합 | 3-4h | 🚨 CRITICAL | 📋 준비됨 |
| **U-018** | 검색 시스템 확장 | 2-3h | 🟡 MEDIUM | 📋 준비됨 |
| **U-019** | GUI 통합 및 시각화 | 3-4h | 🟢 LOW | 📋 준비됨 |

### U-015: Vision 모듈 기반 구축 (첫 번째 작업)
```python
# 목표
- Florence-2 모델 로컬 실행
- VisionAnalyzer 클래스 구현
- 기본 캡션 생성 검증

# 테스트
from backend.vision.analyzer import VisionAnalyzer
analyzer = VisionAnalyzer()
result = analyzer.analyze(image)
assert "caption" in result
```

### U-016: 프롬프트 엔지니어링
```python
# 목표
- 다양한 프롬프트 템플릿 테스트
- 태그 추출 및 후처리
- 스타일 자동 추론

# 프롬프트 예시
PROMPTS = {
    "detailed_caption": "<DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "ocr": "<OCR_WITH_REGION>"
}
```

### U-017: 파이프라인 통합 (핵심)
```python
# ingest_engine.py 수정
def process_file(file_path):
    # ... 기존 파싱 ...

    # Phase 4 추가
    vision_analyzer = VisionAnalyzer()
    result = vision_analyzer.analyze(image)

    meta.ai_caption = result["caption"]
    meta.ai_tags = result["tags"]

    # ChromaDB 저장
    indexer.index_image(file_path, meta.model_dump())
```

### U-018: 검색 시스템 확장
```python
# 하이브리드 검색
final_score = 0.7 * clip_score + 0.3 * keyword_match_score
```

### U-019: GUI 통합
```jsx
// FileGrid.jsx
<div className="thumbnail">
  <img src={image} />
  <div className="ai-tags">
    {aiTags.map(tag => <Badge>{tag}</Badge>)}
  </div>
</div>
```

---

## ⚠️ 작업 시작 전 필수 조건

### 1. 핵심 의존성 설치 (BLOCKING)
```json
{
  "torch": false,              // ❌ 설치 필요
  "chromadb": false,           // ❌ 설치 필요
  "sentence-transformers": false, // ❌ 설치 필요
  "pillow": false              // ❌ 설치 필요
}
```

**해결 방법:**
```powershell
python backend/setup/installer.py --install
python backend/setup/installer.py --download-model
pip install transformers timm einops
```

**예상 시간:** 5-10분
**디스크 공간:** ~5GB

### 2. Phase 1-3 검증
```powershell
# 파서 테스트
python test_image_parser.py

# 파이프라인 테스트
python backend/pipeline/ingest_engine.py --file "test.png"

# 벡터 검색 테스트
python backend/cli_search.py "test"
```

---

## 📅 작업 타임라인

### Week 1: 핵심 구현
```
Day 1 (오늘):
  [x] Phase 4 명세서 작성
  [x] 체크리스트 업데이트
  [ ] 의존성 설치 ⚠️ BLOCKING
  [ ] U-015 시작

Day 2-3:
  [ ] U-015 완료
  [ ] U-016 완료

Day 4-5:
  [ ] U-017 완료 (파이프라인 통합)
  [ ] E2E 테스트
```

### Week 2: 고급 기능 & UI
```
Day 1-2:
  [ ] U-018 완료 (검색 확장)

Day 3-4:
  [ ] U-019 완료 (GUI 통합)

Day 5:
  [ ] Phase 4 완료 검증
  [ ] 회고 작성
```

---

## 🎯 성공 기준 (Definition of Done)

### Must Have (필수)
- ✅ Florence-2 모델 로컬 실행
- ✅ 파이프라인에서 자동 캡션 생성
- ✅ AssetMeta에 `ai_caption`, `ai_tags` 필드 저장
- ✅ ChromaDB 메타데이터 확장
- ✅ GUI에서 AI 태그 표시

### Nice to Have (선택)
- 🔲 하이브리드 검색 (Vector + Keyword)
- 🔲 스타일 자동 분류
- 🔲 Object Detection 바운딩 박스
- 🔲 다국어 캡션 (번역 통합)

### 성능 목표
| 지표 | 목표 |
|------|------|
| 모델 로드 | <10초 |
| 분석 속도 | <1초/image |
| VRAM 사용 | <4GB |
| 캡션 정확도 | >80% (수동 검증) |

---

## 🚀 다음 단계 (지금 바로 실행)

### 1. 의존성 설치
```powershell
python backend/setup/installer.py --install
pip install transformers>=4.40.0 timm>=0.9.0 einops>=0.7.0
```

### 2. 검증
```powershell
python backend/setup/installer.py --check
```

### 3. 작업 시작
```powershell
/unit-start
```

---

## 📚 참고 문서

1. **Phase 4 상세 명세**: `docs/phase4_vision_specification.md`
2. **시작 가이드**: `docs/phase4_getting_started.md`
3. **작업 체크리스트**: `docs/project_checklist.md`
4. **Florence-2 공식 문서**: https://huggingface.co/microsoft/Florence-2-large
5. **트러블슈팅**: `docs/troubleshooting.md`

---

## 💡 핵심 포인트

1. **명확한 목표**: 3-Axis의 마지막 축 완성 (Descriptive)
2. **검증된 기술**: Florence-2 (Microsoft 공식, 경량, 빠름)
3. **구체적 계획**: 5개 유닛, 각 2-4시간, 총 2주
4. **점진적 통합**: U-015→U-016→U-017 순서로 기반 구축
5. **성능 최적화**: RTX 3060 Ti 8GB 기준 최적화

---

**상태**: ✅ 준비 완료
**다음 단계**: 의존성 설치 → `/unit-start`
**예상 완료**: 2주 후 (2026-02-20)
