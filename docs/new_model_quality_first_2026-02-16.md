# 품질 최우선 신규 모델 제안서 (ECM: Evidence-Centric Model)

작성일: 2026-02-16
목표: 로컬 시스템에서 "토큰/연산 절약"보다 "최고의 해(정답 품질)"를 우선하도록 저장 구조/검색 구조를 재설계

## 0. 핵심 원칙
- 목적함수는 비용이 아니라 품질이다.
- 저장은 "최종값 덮어쓰기"가 아니라 "증거(evidence) 누적" 구조여야 한다.
- 검색은 단일 점수 비교가 아니라 "다축 증거 결합 + 재랭킹"이어야 한다.

---

## 1. 현재 구조의 품질 한계 (코드 근거)

### 1.1 파일 경로 중심 upsert로 이력/증거 손실 가능
- `files.file_path UNIQUE` + `ON CONFLICT(file_path) DO UPDATE`로 최신값 덮어쓰기 중심.
- 근거:
  - `backend/db/sqlite_schema.sql` (`file_path TEXT UNIQUE NOT NULL`)
  - `backend/db/sqlite_client.py` (`ON CONFLICT(file_path) DO UPDATE`)
- 문제:
  - 동일 콘텐츠의 분석 버전/모델 버전 비교가 어렵다.
  - "왜 이 결과가 나왔는지" 추적 근거가 약하다.

### 1.2 벡터/캡션/태그의 provenance(생성 이력) 부족
- 현재는 파일별 최종 상태 저장이 중심이고, run 단위 생성 근거가 약함.
- 문제:
  - 모델 교체 시 품질 회귀 원인 추적이 어려움.
  - 신뢰도(confidence) 기반 결합 불가.

### 1.3 FTS 축이 메타 중심으로 제한
- FTS는 `meta_strong`, `meta_weak` 2컬럼 중심이고 캡션/태그 어휘축이 약함.
- 근거:
  - `backend/db/sqlite_client.py` `_FTS_COLUMNS = ['meta_strong', 'meta_weak']`
  - "caption column will be added..." 주석
- 문제:
  - lexical exact-match 회수율이 낮아질 수 있음.

### 1.4 저장 커밋 단위가 세분화되어 대량 구축시 비효율
- `upsert_metadata`/`update_vision_fields`/`upsert_vectors`가 호출 단위 commit.
- 근거:
  - `backend/db/sqlite_client.py` 각 메서드 `self.conn.commit()`
- 문제:
  - 대량 인덱싱에서 write amplification 증가.

### 1.5 스킵 로직이 mtime 문자열 비교에 의존
- `stored_mtime.replace('T',' ') != current_mtime.replace('T',' ')`
- 근거:
  - `backend/pipeline/ingest_engine.py`
- 문제:
  - 타임스탬프 정밀도/포맷 차이로 불필요 재처리 또는 누락 가능성.

---

## 2. 신규 모델: ECM (Evidence-Centric Model)

## 2.1 저장 모델 (최종값 -> 증거 누적)

### A. 엔티티 분리
1. `assets` (콘텐츠 정체성)
- `asset_id`, `content_hash`, `created_at`

2. `asset_versions` (파일 상태/경로 버전)
- `version_id`, `asset_id`, `file_path`, `size`, `mtime`, `fs_snapshot_id`

3. `analysis_runs` (분석 실행 단위 provenance)
- `run_id`, `asset_version_id`, `phase`, `model_name`, `model_ver`, `config_hash`, `started_at`, `ended_at`, `status`

4. `evidence_text` (캡션/태그/OCR/정형 분류)
- `evidence_id`, `run_id`, `kind(caption|tag|ocr|scene|style|...)`, `value`, `confidence`

5. `evidence_vectors` (VV/X/MV 벡터)
- `vector_id`, `run_id`, `space(vv|x|mv)`, `dim`, `embedding`, `norm`, `quant`, `checksum`

6. `search_materialized` (검색용 최신 스냅샷 뷰)
- `asset_id`, `best_vv_vector_id`, `best_x_vector_id`, `best_mv_vector_id`, `fts_doc_id`, `quality_score`

### B. 이 구조의 장점
- 동일 자산에 대해 모델 버전별 증거를 공존 저장 가능.
- 회귀 분석, A/B, 롤백, 재현성 확보.
- confidence 기반 가중 결합 가능.

## 2.2 검색 모델 (후보생성 + 증거기반 재랭킹)

### Stage 1: 다축 후보 생성 (Recall 극대화)
- VV (SigLIP)
- X (DINOv2 Structure)
- MV (Text embedding)
- FTS (BM25)
- 필요시 OCR exact-match 채널

### Stage 2: 후보 통합
- 가중 RRF로 1차 결합
- 쿼리 의도별 초기 가중: visual/structure/semantic/keyword

### Stage 3: 증거 피처화
각 후보에 대해 아래 피처를 계산:
- `sim_vv`, `sim_x`, `sim_mv`, `bm25_meta`, `bm25_caption`, `ocr_hit`
- `tag_overlap`, `style_match`, `scene_match`, `time_weather_match`
- `run_freshness`, `model_confidence`, `evidence_consistency`

### Stage 4: 재랭킹 (Best-solution 지향)
- 로컬 경량 reranker 또는 규칙+학습 혼합 점수:

`final = w1*sim_vv + w2*sim_x + w3*sim_mv + w4*bm25 + w5*structured_match + w6*confidence - penalties`

- penalties: 상충 증거, low-confidence 과다, 스팸 태그

---

## 3. 자원 소비 최적화 (품질 우선, 합리적 소비)

- "모든 쿼리에서 모든 고비용 연산"은 금지.
- 대신 단계적 소비:
  1) 후보 생성은 고정 예산
  2) 재랭킹은 top-N에만 실행
  3) N은 질의 난이도/모호성에 따라 동적 조정

권장 기본값:
- 후보 풀: 축당 200~500
- 재랭킹 대상: 80~200
- 최종 반환: 20~50

---

## 4. 즉시 전환 가능한 개선안 (현 코드에 맞춤)

P0 (1~2일)
1. health gate: sqlite-vec 미로딩 시 벡터 모드 차단 + 명시적 오류
2. ingestion write batching(500~1000) + writer queue 도입
3. mtime 문자열 비교 -> `content_hash + size + mtime_ns` 조합 비교

P1 (3~7일)
1. `analysis_runs`/`evidence_text`/`evidence_vectors` 테이블 추가
2. 기존 files를 materialized current view로 전환
3. FTS에 caption/tag optional 채널 추가(가중치 분리)

P2 (1~2주)
1. feature 로그 축적
2. 쿼리 타입별 동적 가중 튜닝
3. reranker 도입 및 AB 검증

---

## 5. 성과 측정

Offline:
- nDCG@10, Recall@20, MRR@10, Precision@5

Online:
- P95 latency
- query success rate
- top-5 클릭/선택률
- 재검색률(requery rate)

승격 기준:
- nDCG@10 +5% 또는 Recall@20 +8% 이상
- P95는 목표 범위 유지(예: 2초 내)

---

## 6. 결론
현재 구조는 "작동"은 하지만 "최고의 해"를 안정적으로 뽑아내기에는 증거 관리와 재랭킹 체계가 약하다.
따라서 다음 단계는 단순 튜닝이 아니라, ECM처럼 **증거 누적 저장 + 다축 후보 + 재랭킹**으로 전환하는 것이 맞다.
