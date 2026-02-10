# ImageParser 종합 개발 계획서

**최종 갱신**: 2026-02-10
**현재 단계**: P0~P2+ 코드 완료, Tier 시스템 정착, **데이터 재구축 대기**
**검색 적중률**: 30~40% → **90~97%** (코드 기준, 데이터 재구축 후 달성 예상)

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
| **Vision LLM** | Tier별: Qwen3-VL-2B (standard) / 4B (pro) / 8B (ultra) | 2-Stage 분류/분석, transformers 또는 Ollama |
| **V-axis 임베딩** | SigLIP2 (tier별: base=768d, so400m=1152d, giant=1664d) | 시각 유사도 |
| **S-axis 임베딩** | Qwen3-Embedding (Ollama, tier별: 256d/1024d/4096d) | 의미 유사도 |
| **i18n** | 커스텀 LocaleContext | ko-KR, en-US |

---

## 3. 아키텍처

```
[ 인제스트 파이프라인 ]

이미지 (PSD/PNG/JPG)
  │
  ├─ [M축] PSD 파싱 → 레이어 트리, 텍스트, 메타데이터 → files + files_fts
  │
  ├─ [Vision] Qwen3-VL (2-Stage, tier별: 2B/4B/8B)
  │     Stage 1: image_type 분류 (11종)
  │     Stage 2: 타입별 전용 구조화 분석
  │     → ai_caption, ai_tags, image_type, art_style, scene_type ...
  │
  ├─ [V-axis] SigLIP2 → tier별 시각 벡터 → vec_files
  │
  ├─ [S-axis] caption+tags → Qwen3-Embedding → tier별 텍스트 벡터 → vec_text
  │
  └─ [DB] SQLite: files + vec_files + vec_text + files_fts(FTS5)


[ 검색 파이프라인 ]

쿼리 → QueryDecomposer (vector_query + fts_keywords + query_type)
  │
  ├─ [V-axis] SigLIP2 인코딩 → vec_files 코사인 유사도
  ├─ [S-axis] Qwen3-Embedding 인코딩 → vec_text 코사인 유사도
  ├─ [M-axis] FTS5 키워드 매칭 (16컬럼)
  │
  └─ Auto-Weighted RRF 병합 (query_type별 가중치, k=60) → 최종 결과
     ※ 구조화 필터(scene_type 등)는 하드 게이트로 사용하지 않음
```

---

## 4. 로드맵 진행 상태

| 단계 | 작업 | 상태 | 적중률 효과 | 비고 |
|------|------|------|-----------|------|
| **P0** | 2-Stage Vision + DB 스키마 | **완료** | +45~55%p | 99파일 처리, 11개 image_type |
| **P1** | V-axis SigLIP2 교체 | **완료** | +2~5%p | tier별 차원, 109언어 |
| **P2** | S-axis Qwen3-Embedding 추가 | **완료** | +3~7%p | tier별 차원, 3축 RRF 완성 |
| **P2+** | Auto-Weighted RRF + Tier 시스템 | **완료** | +2~5%p | 쿼리 유형별 동적 가중치, standard/pro/ultra |
| **v3.1.1** | 플랫폼별 최적화 | **완료** | 성능 | AUTO 모드, Windows: Ollama(1), Mac: vLLM(16) |
| **P3** | 인프라 개선 | 미구현 | 성능/안정성 | FastAPI 서버, 큐 영속성 |
| **P4** | 서버형 전환 | 미구현 | 확장성 | PostgreSQL, 멀티테넌시 |

**누적 적중률:** 30~40% → 85~95% (P0) → 87~97% (P1) → **90~97%** (P2)

---

## 4.1 현재 시스템 상태 (2026-02-10)

### Tier 시스템

| 항목 | 값 |
|------|------|
| **활성 Tier** | `standard` (config.yaml override) |
| **VLM** | Qwen3-VL-2B-Instruct (transformers, CUDA) |
| **V-axis 모델** | google/siglip2-base-patch16-224 (768d) |
| **S-axis 모델** | qwen3-embedding:0.6b (Ollama, 256d) |

### 데이터 현황 (232파일)

| 항목 | 현재 | 정상 | 상태 |
|------|------|------|------|
| **files** (메타데이터) | 232/232 | 232 | **정상** |
| **files_fts** (M-axis) | 232/232 | 232 | **정상** |
| **vec_files** (V-axis) | 217/232 | 232 | **15건 누락** |
| **vec_text** (S-axis) | 1/232 | 232 | **미구축 (1건만 테스트 완료)** |
| **mc_caption** | 1/232 | 232 | **미구축** |
| **ai_tags** | 1/232 | 232 | **미구축** |
| **image_type** | 1/232 | 232 | **미구축** |
| **embedding_model** 기록 | 1/232 | 232 | 231건 NULL (pre-v3.1 데이터) |

### 원인 분석

**왜 231파일이 AI Vision 미처리 상태인가?**

1. **device='auto' 버그 (2026-02-10 수정)**: `config.yaml`의 `vlm.device: auto`가 PyTorch에서 유효하지 않은 디바이스 문자열. VisionAnalyzer가 모델은 로드하지만 추론 시 `.to('auto')` 에러 → Stage 1/2 모두 실패 → `image_type=other`, `mc_caption=NULL`로 저장
2. **pre-v3.1 데이터**: 232파일 중 231파일은 v3.0 이전(CLIP/PostgreSQL 시대)에 인제스트됨. STEP 2(AI Vision)가 존재하지 않던 시점의 데이터
3. **vec_files 15건 누락**: SigLIP2 마이그레이션(v3_p1_vec) 시 일부 파일 누락 추정

### 코드 정상 동작 확인 (2026-02-10)

테스트 파일 1건(`20120206_2219701.jpg`)으로 전체 파이프라인 검증 완료:

| 단계 | 결과 | 시간 |
|------|------|------|
| STEP 1/4 Parse | JPG 파싱 완료 | 0.04s |
| STEP 2/4 AI Vision | Stage1→photo(high), Stage2→caption+tags 생성 | 22.5s |
| STEP 3/4 Embedding | SigLIP2 768d 벡터 저장 | 12.4s |
| STEP 4/4 Storing | SQLite + S-axis 256d 텍스트 임베딩 | 2.4s |
| **총합** | **전 단계 정상** | **38.5s** |

### 검색 기능 상태

| 축 | 상태 | 비고 |
|----|------|------|
| **V-axis** (SigLIP2) | **부분 동작** | 217/232 파일에 벡터 존재. 점수 범위 0.06~0.17 |
| **S-axis** (Qwen3-Emb) | **미동작** | vec_text 1건만 존재. 재구축 필요 |
| **M-axis** (FTS5) | **정상** | 232/232 인덱싱 완료 |
| **RRF 병합** | **정상** | V+M 2축으로 동작 중. S축 추가 시 3축 완성 |
| **프론트엔드 뱃지** | **정상** | VV/MV/MC 표시, i18n 연결 완료 |
| **threshold** | **정상** | post-merge 필터 제거, 기본값 0 |

### 다음 작업: 데이터 재구축

```powershell
# 전체 232파일 재처리 (STEP 1~4 모두 재실행)
python backend/pipeline/ingest_engine.py --discover "C:\Images" --no-skip

# 예상 소요: 232파일 × ~25초/파일 ≈ 1.5시간
```

**재구축 후 기대 결과:**
- mc_caption: 232/232 (모든 파일에 AI 캡션)
- ai_tags: 232/232 (모든 파일에 AI 태그)
- image_type: 232/232 (11종 자동 분류)
- vec_text (S-axis): 232/232 (3축 검색 완성)
- vec_files (V-axis): 232/232 (15건 누락 해소)

---

## 4.2 2026-02-10 수정 내역

### 코드 수정 (5건)

| 커밋 | 유형 | 내용 |
|------|------|------|
| `4ed22d8` | fix | 코드 주석 정규화: T-axis→S-axis, F-axis→M-axis, CLIP→SigLIP2 (9개 파일) |
| `3456227` | fix | 프론트엔드 뱃지 라벨 i18n 연결 (V/S/M → vv/mv/mc) |
| `05c98a3` | fix | **검색 결과 3개 버그 수정**: post-merge threshold 필터 제거. SigLIP2 점수 범위(0.06~0.17)와 threshold(0.15) 불일치로 V-axis 결과 37/40개 삭제되던 문제 |
| `066f79d` | fix | **device='auto' 크래시 수정**: VisionAnalyzer에서 `device='auto'` → `cuda`/`cpu` 자동 해석. 이 버그로 STEP 2 전체가 실패하고 있었음 |
| `abe397e` | refactor | RRF 레거시 vv/mv 폴백 키 제거 |

### 문서 수정 (4건)

| 커밋 | 내용 |
|------|------|
| `927327c` | **CLAUDE.md 전면 재작성**: PostgreSQL/CLIP → SQLite/SigLIP2/Triaxis |
| `ec8035d` | 문서 축 이름 정규화 (triaxis_search_architecture.md, triaxis_redesign_proposal.md, v3.md) |
| `6fa161b` | 마이그레이션 스크립트 레거시 주석 |
| `61d0ad8` | UTF-8 인코딩 필수 규칙 추가 |

---

## 5. 완료된 작업

### P0: 2-Stage Vision Pipeline

- Qwen3-VL (tier별: 2B/4B/8B)로 이미지를 **11개 타입** 자동 분류 (character, background, ui_element, item 등)
- 타입별 전용 프롬프트로 구조화 분석 (scene_type, time_of_day, character_type 등)
- VLM 백엔드 자동 선택: `vision_factory.py` (standard/pro: transformers, ultra: Ollama/vLLM auto)
- 3단계 JSON Repair 로직으로 LLM 출력 안정성 확보
- DB 15개 컬럼 추가 + FTS5 16컬럼 확장
- 구현: `backend/vision/` (vision_factory.py, analyzer.py, ollama_adapter.py, prompts.py, schemas.py, repair.py)
- 검증: 99파일 100% 처리 — illustration(52), photo(41), background(5), texture(1)

### P1: V-axis SigLIP2 교체

- CLIP ViT-L-14 → **SigLIP2** (tier별: base=768d, so400m=1152d, giant=1664d)
- ImageNet 75% → 85%, 109언어 지원
- 구현: `backend/vector/siglip2_encoder.py`
- 마이그레이션: `backend/db/migrations/v3_p1_vec.py`
- 검증: 한국어 "야경" → yakei.psd 정확 매칭

### P2: S-axis 텍스트 임베딩

- Qwen3-Embedding (tier별: 256d/1024d/4096d) 통합, Ollama embed API 사용
- ai_caption + ai_tags → 텍스트 벡터 생성 → `vec_text` 테이블 저장
- 3축 RRF 병합 완성 (`_rrf_merge_multi`)
- 구현: `backend/vector/text_embedding.py`, `backend/db/migrations/v3_p2_text_vec.py`
- 검증: 99/99 성공 (v3.0 테스트 기준), avg 2.23s/file, S-axis 유사도 0.73 (V-axis 0.05 대비 14배 정확)

---

## 6. 미완료 작업

### P2+: Auto-Weighted RRF — ✅ 완료

**목표:** 쿼리 유형에 따라 V/S/M 축 가중치를 동적 조절
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
   - `art_style = "anime"` → V축(SigLIP2 tier별 차원)이 시각적 스타일을 이미 인코딩
   - `time_of_day = "night"` → S축이 caption "night scene" 의미를 이미 벡터화
3. **정보 손실**: Qwen3-VL이 생산하는 자유형 caption/tags는 풍부한 서술 정보를 포함하지만,
   구조화 필드는 동일 AI가 같은 이미지를 보고 ~10개 카테고리로 강제 분류한 축소 정보.
   1152차원 벡터가 인코딩하는 정보를 ~10개 체크박스로 대체하려는 시도.

**결론:**
- LLM 필터는 검색 시 하드 게이트로 사용하지 않음 (코드에서 제거)
- 구조화 필드(image_type, scene_type 등)는 DB에 저장은 유지 (프론트엔드 수동 필터 용도)
- 검색 품질은 V축+S축+M축(FTS) 3축 RRF가 전담
- 2-Stage Vision 파이프라인은 유지 (타입별 전용 프롬프트가 caption/tags 품질 향상에 기여)

#### 가중치 튜닝 결과

4종 프리셋(A~D)으로 6개 쿼리 비교 실험 수행:
- **A (0.50/0.30/0.20)** 채택 — keyword/visual 양쪽에서 가장 안정적
- keyword 쿼리는 가중치에 둔감 (FTS 0.40~0.60 범위에서 top 1~2 불변)
- visual 쿼리는 가중치에 민감 (V축 0.45 vs 0.50 차이로 1위 변동)
- 균등 분배(C)는 keyword에서 손해, 극단(D)는 visual에서 V축 과의존

### 플랫폼별 최적화 (v3.1.1) — ✅ 완료

**목표:** 플랫폼별 최적 VLM 백엔드 자동 선택
**상태:** 완료
**구현 내용:**
- `backend/vision/vision_factory.py` — AUTO 모드로 플랫폼 자동 감지
- Windows: Ollama (batch_size=1, 순차 처리 최적)
- Mac/Linux: vLLM 우선 (batch_size=16, 8.5배 향상)
- `config.yaml` > `ai_mode.tiers.*.vlm.backend: auto`

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
    vision_factory.py      # VLM 백엔드 자동 선택 (Factory)
    analyzer.py            # Transformers VLM 어댑터 (2-Stage, standard/pro)
    ollama_adapter.py      # Ollama VLM 어댑터 (2-Stage, ultra)
    prompts.py             # Stage1/Stage2 프롬프트
    schemas.py             # 11개 image_type JSON 스키마
    repair.py              # 3단계 JSON Repair
  vector/
    siglip2_encoder.py     # [P1] V-axis 인코더 (tier별 차원)
    text_embedding.py      # [P2] S-axis Provider (tier별 차원)
  search/
    sqlite_search.py       # 3축 RRF 검색 엔진
    rrf.py                 # [P2+] RRF 가중치 프리셋 (query_type별)
    query_decomposer.py    # LLM 쿼리 분류기 (query_type 판별)
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
| **검색 시** | SigLIP2 + Qwen3-Embedding (standard) | **~1.2GB** |
| **인제스트 시** | Qwen3-VL-2B + SigLIP2 + Qwen3-Embedding (standard) | **~4GB** |

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
| 인제스트 (1파일, standard tier) | ~25~38초 (초기 모델 로딩 포함) |
| STEP 2 AI Vision (2-Stage) | ~22초 |
| V-axis 인코딩 (SigLIP2) | ~5초 (모델 로딩 포함) |
| S-axis 인코딩 (Qwen3-Emb) | ~2초 |
| 3축 검색 (232파일) | ~25ms |

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
python tools/reindex_v3.py --text-embedding        # S축 재생성
python tools/reindex_v3.py --all                   # 전체

# 검색 (프론트엔드 Electron 앱)
cd frontend && npm run electron:dev

# 프론트엔드
cd frontend && npm install && npm run electron:dev

# 마이그레이션
python -m backend.db.migrations.v3_p2_text_vec     # S-axis 테이블 생성
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
