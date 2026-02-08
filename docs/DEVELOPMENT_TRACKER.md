# ImageParser 종합 개발 계획서

**최종 갱신**: 2026-02-08
**현재 단계**: P0~P2 완료, P2+ 대기
**검색 적중률**: 30~40% → **90~97%** (달성)

---

## 1. 프로젝트 개요

**ImageParser**는 PSD/PNG/JPG 파일을 AI 검색 가능한 데이터로 변환하는 멀티모달 이미지 벡터화 시스템이다.
3-Axis 아키텍처(구조적 + 시각적 + 서술적)로 게임/디자인 에셋 도메인에 최적화된 검색을 제공한다.
모든 AI 모델은 Apache 2.0 라이선스로 상업 전환에 제약이 없다.

---

## 2. 기술 스택

| 영역 | 기술 | 비고 |
|------|------|------|
| **Backend** | Python 3.x | psd-tools, Pillow, watchdog |
| **Database** | SQLite + sqlite-vec + FTS5 | 벡터 + 전문검색 + 메타데이터 통합 |
| **Frontend** | React 19 + Electron 40 + Vite 7 + Tailwind CSS 4 | 데스크톱 앱 |
| **Vision LLM** | Qwen3-VL-8B (Ollama) | 2-Stage 분류/분석, ~6GB VRAM |
| **V축 임베딩** | SigLIP 2 So400m | 1152차원, 109언어, ~0.8GB |
| **T축 임베딩** | Qwen3-Embedding-0.6B (Ollama) | 1024차원, 100+언어, ~0.4GB |
| **i18n** | 커스텀 LocaleContext | ko-KR, en-US |

---

## 3. 아키텍처

```
[ 인제스트 파이프라인 ]

이미지 (PSD/PNG/JPG)
  │
  ├─ [S축] PSD 파싱 → 레이어 트리, 텍스트, 메타데이터
  │
  ├─ [Vision] Qwen3-VL-8B (2-Stage)
  │     Stage 1: image_type 분류 (11종)
  │     Stage 2: 타입별 전용 구조화 분석
  │     → ai_caption, ai_tags, image_type, art_style, scene_type ...
  │
  ├─ [V축] SigLIP 2 → 1152차원 시각 벡터 → vec_files
  │
  ├─ [T축] caption+tags → Qwen3-Embedding → 1024차원 텍스트 벡터 → vec_text
  │
  └─ [DB] SQLite: files + vec_files + vec_text + files_fts(FTS5)


[ 검색 파이프라인 ]

쿼리 → QueryDecomposer (vector_query + fts_keywords + query_type)
  │
  ├─ [V축] SigLIP 2 인코딩 → vec_files 코사인 유사도
  ├─ [T축] Qwen3-Embedding 인코딩 → vec_text 코사인 유사도
  ├─ [F축] FTS5 키워드 매칭 (16컬럼)
  │
  └─ Auto-Weighted RRF 병합 (query_type별 가중치, k=60) → 최종 결과
     ※ 구조화 필터(scene_type 등)는 하드 게이트로 사용하지 않음
```

---

## 4. 로드맵 진행 상태

| 단계 | 작업 | 상태 | 적중률 효과 | 비고 |
|------|------|------|-----------|------|
| **P0** | 2-Stage Vision + DB 스키마 | **완료** | +45~55%p | 99파일 처리, 11개 image_type |
| **P1** | V축 SigLIP 2 교체 | **완료** | +2~5%p | 768→1152차원, 109언어 |
| **P2** | T축 Qwen3-Embedding 추가 | **완료** | +3~7%p | 1024차원, 3축 RRF 완성 |
| **P2+** | Auto-Weighted RRF | **완료** | +2~5%p | 쿼리 유형별 동적 가중치, LLM 하드 필터 폐기 |
| **P3** | 인프라 개선 | 미구현 | 성능/안정성 | FastAPI 서버, 큐 영속성 |
| **P4** | 서버형 전환 | 미구현 | 확장성 | PostgreSQL, 멀티테넌시 |

**누적 적중률:** 30~40% → 85~95% (P0) → 87~97% (P1) → **90~97%** (P2)

---

## 5. 완료된 작업

### P0: 2-Stage Vision Pipeline

- Qwen3-VL-8B로 이미지를 **11개 타입** 자동 분류 (character, background, ui_element, item 등)
- 타입별 전용 프롬프트로 구조화 분석 (scene_type, time_of_day, character_type 등)
- 3단계 JSON Repair 로직으로 LLM 출력 안정성 확보
- DB 15개 컬럼 추가 + FTS5 16컬럼 확장
- 구현: `backend/vision/` (prompts.py, schemas.py, ollama_adapter.py, repair.py)
- 검증: 99파일 100% 처리 — illustration(52), photo(41), background(5), texture(1)

### P1: V축 SigLIP 2 교체

- CLIP ViT-L-14 (768차원) → **SigLIP 2 So400m** (1152차원) 교체
- ImageNet 75% → 85%, VRAM 1.5GB → 0.8GB, 109언어 지원
- 구현: `backend/vector/siglip2_encoder.py`
- 마이그레이션: `backend/db/migrations/v3_p1_vec.py`
- 검증: 한국어 "야경" → yakei.psd 정확 매칭

### P2: T축 텍스트 임베딩

- Qwen3-Embedding-0.6B (1024차원) 통합, Ollama embed API 사용
- ai_caption + ai_tags → 텍스트 벡터 생성 → `vec_text` 테이블 저장
- 3축 RRF 병합 완성 (`_rrf_merge_multi`)
- 구현: `backend/vector/text_embedding.py`, `backend/db/migrations/v3_p2_text_vec.py`
- 검증: 99/99 성공, avg 2.23s/file, T축 유사도 0.73 (V축 0.05 대비 14배 정확)

---

## 6. 미완료 작업

### P2+: Auto-Weighted RRF — ✅ 완료

**목표:** 쿼리 유형에 따라 V/T/F 축 가중치를 동적 조절
**상태:** 완료 (2026-02-08)
**구현 내용:**
- `backend/search/rrf.py` (신규) — 가중치 프리셋 4종 + 비활성 축 재분배
- `backend/search/query_decomposer.py` (수정) — Ollama Chat API + thinking 억제, query_type 반환
- `backend/search/sqlite_search.py` (수정) — weighted RRF 호출 + LLM 하드 필터 폐기
- `config.yaml` — `search.rrf.auto_weight: true` 활성화

#### ⚠️ 구조화 필터 폐기 결정 (Critical Finding)

**문제:** QueryDecomposer가 생성하는 LLM 필터(scene_type, time_of_day, art_style 등)를
AND 조건 하드 게이트로 적용하면 **검색 결과가 전부 0건**으로 삭제됨.

**원인 분석:**
1. **DB 희소성**: 구조화 필드(scene_type, time_of_day 등)의 실제 채워진 비율이 ~5%.
   99파일 중 대부분은 해당 필드가 비어 있어 AND 필터를 통과하지 못함.
2. **근본적 중복**: 구조화 필드가 하는 일을 3축이 이미 더 잘 수행함.
   - `scene_type = "alley"` → FTS가 caption/tags의 "alley" 키워드를 이미 검색
   - `art_style = "anime"` → V축(SigLIP 1152차원)이 시각적 스타일을 이미 인코딩
   - `time_of_day = "night"` → T축이 caption "night scene" 의미를 이미 벡터화
3. **정보 손실**: Qwen3-VL이 생산하는 자유형 caption/tags는 풍부한 서술 정보를 포함하지만,
   구조화 필드는 동일 AI가 같은 이미지를 보고 ~10개 카테고리로 강제 분류한 축소 정보.
   1152차원 벡터가 인코딩하는 정보를 ~10개 체크박스로 대체하려는 시도.

**결론:**
- LLM 필터는 검색 시 하드 게이트로 사용하지 않음 (코드에서 제거)
- 구조화 필드(image_type, scene_type 등)는 DB에 저장은 유지 (프론트엔드 수동 필터 용도)
- 검색 품질은 V축+T축+FTS 3축 RRF가 전담
- 2-Stage Vision 파이프라인은 유지 (타입별 전용 프롬프트가 caption/tags 품질 향상에 기여)

#### 가중치 튜닝 결과

4종 프리셋(A~D)으로 6개 쿼리 비교 실험 수행:
- **A (0.50/0.30/0.20)** 채택 — keyword/visual 양쪽에서 가장 안정적
- keyword 쿼리는 가중치에 둔감 (FTS 0.40~0.60 범위에서 top 1~2 불변)
- visual 쿼리는 가중치에 민감 (V축 0.45 vs 0.50 차이로 1위 변동)
- 균등 분배(C)는 keyword에서 손해, 극단(D)는 visual에서 V축 과의존

### P3: 인프라 개선

**목표:** Python subprocess → FastAPI 장기 실행 서버
- 검색 지연 200ms → ~10ms (모델 상주)
- 처리 큐 영속성 (SQLite `processing_queue` 테이블)
- MetadataModal 컴포넌트 통합

### P4: 서버형 전환

**목표:** 멀티유저/클라우드 배포 준비
- PostgreSQL + pgvector 마이그레이션
- 멀티테넌시 (`user_id`), 오브젝트 스토리지 (S3/MinIO)
- 인증/권한, 태스크 큐 (Celery/Redis)
- (스키마 선제 적용 완료: `storage_root`, `relative_path`, `embedding_model`)

---

## 7. 핵심 디렉토리 구조

```
backend/
  vision/
    prompts.py             # Stage1/Stage2 프롬프트
    schemas.py             # 11개 image_type JSON 스키마
    ollama_adapter.py      # Ollama 호출 + 2-Stage 실행
    repair.py              # 3단계 JSON Repair
  vector/
    siglip2_encoder.py     # [P1] V축 인코더 (1152차원)
    text_embedding.py      # [P2] T축 Provider (1024차원)
  search/
    sqlite_search.py       # 3축 RRF 검색 엔진
    query_decomposer.py    # 쿼리 분석
  db/
    sqlite_client.py       # SQLite 클라이언트 (880줄)
    sqlite_schema.sql      # 스키마 정의
    migrations/            # v3_p0, v3_p1, v3_p2
  pipeline/
    ingest_engine.py       # [메인 진입점] 4단계 처리

frontend/
  src/components/
    SearchPanel.jsx        # 검색 UI + 필터
    FileGrid.jsx           # 결과 그리드
  src/i18n/locales/        # ko-KR.json, en-US.json
  electron/main.cjs        # Electron IPC

config.yaml                # 전체 설정 (vision, embedding, search)
tools/reindex_v3.py        # 재인덱싱 (--vision-only, --embedding-only, --text-embedding, --all)
```

---

## 8. 성능 수치

### VRAM 버짓

| 시나리오 | 모델 | VRAM |
|---------|------|------|
| **검색 시** | SigLIP 2 + Qwen3-Embedding | **~1.2GB** |
| **인제스트 시** | Qwen3-VL + SigLIP 2 + Qwen3-Embedding | **~7.2GB** |

### 적중률 추이

| 단계 | 평균 | 복합 쿼리 | 개선폭 |
|------|------|----------|--------|
| P0 이전 | 30~40% | 10~20% | — |
| P0 후 | 85~95% | 85~95% | +45~55%p |
| P0+P1 | 87~97% | 87~95% | +2~5%p |
| **P0+P1+P2** | **90~97%** | **90~97%** | +3~7%p |

### 처리 속도

| 작업 | 시간 |
|------|------|
| 인제스트 (1파일) | 8~19초 |
| V축 인코딩 | ~200ms |
| T축 인코딩 | ~2.2초 |
| 3축 검색 (99파일) | ~25ms |

---

## 9. 실행 명령어

```bash
# 인제스트
python backend/pipeline/ingest_engine.py --file "image.psd"
python backend/pipeline/ingest_engine.py --discover "C:\assets"
python backend/pipeline/ingest_engine.py --watch "C:\assets"

# 재인덱싱
python tools/reindex_v3.py --vision-only          # Vision 재처리
python tools/reindex_v3.py --embedding-only        # V축 재생성
python tools/reindex_v3.py --text-embedding        # T축 재생성
python tools/reindex_v3.py --all                   # 전체

# 검색 (CLI)
python backend/cli_search_pg.py "fantasy character"

# 프론트엔드
cd frontend && npm install && npm run electron:dev

# 마이그레이션
python -m backend.db.migrations.v3_p2_text_vec     # T축 테이블 생성
```

---

## 10. 문서 체계

| 문서 | 역할 | 위치 |
|------|------|------|
| **이 문서** | 전체 조망, 진행 상태 추적, 퀵 레퍼런스 | `docs/DEVELOPMENT_TRACKER.md` |
| **v3.md** | 상세 기술 명세 (스키마, 모델 비교, 검증 수치) | `docs/v3.md` |
| **CLAUDE.md** | AI 에이전트 개발 가이드, 프로젝트 규칙 | `CLAUDE.md` |
| **troubleshooting.md** | 문제/해결 로그 | `docs/troubleshooting.md` |
| **phase_roadmap.md** | 3-Axis 로드맵 개요 | `docs/phase_roadmap.md` |
| **ollama_setup.md** | Ollama 설치/설정 가이드 | `docs/ollama_setup.md` |

> 상세 스펙이 필요하면 `docs/v3.md` 참조. 이 문서는 **현황 파악용**이다.
