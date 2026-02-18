# H100 체크리스트 비교 분석 보고서 (2026-02-16)

## 분석 범위
- 코드 정적 분석 기준: `backend/`, `config.yaml`, `requirements.txt`
- 실행 확인: `imageparser.db`의 SQLite pragma 일부 점검
- 기준 문서: 사용자 제공 4개 영역(총 12개 항목) 체크리스트

## 종합 결론
- 충족: 3/12
- 부분충족: 4/12
- 미충족: 4/12
- 미확인/해당없음(현재 구조상): 1/12

핵심 병목은 **(1) 동기식 DB 커밋 빈도**, **(2) PIL 기반 디코딩/로딩 경로**, **(3) vLLM 배치 경로 미연결**입니다.

---

## 1) I/O 및 데이터 파이프라인 체크

### 1-1. Prefetching 병목 확인 (CPU 전처리 vs GPU 추론 독립 동작)
- 판정: **부분충족**
- 근거:
  - Phase1 파싱은 CPU 스레드 병렬 처리(`ThreadPoolExecutor`) 사용: `backend/pipeline/ingest_engine.py:694`, `backend/pipeline/ingest_engine.py:701`
  - 그러나 Vision 단계에서는 서브배치마다 썸네일 로딩 후 즉시 추론하는 동기 루프 구조: `backend/pipeline/ingest_engine.py:771`, `backend/pipeline/ingest_engine.py:776`, `backend/pipeline/ingest_engine.py:803`
  - 별도 producer/consumer 프로세스(예: 미리 128~256장 메모리 선적재) 구조는 없음

### 1-2. 이미지 디코딩 속도 (PIL vs TurboJPEG/OpenCV)
- 판정: **미충족**
- 근거:
  - PIL 직접 사용: `backend/parser/image_parser.py:13`, `backend/parser/image_parser.py:53`, `backend/pipeline/ingest_engine.py:509`
  - PSD도 PIL/psd-tools 경로: `backend/parser/psd_parser.py:13`, `backend/parser/psd_parser.py:60`
  - 의존성에 TurboJPEG/OpenCV 없음: `requirements.txt:7`

### 1-3. 네트워크 대역폭 (S3/GCS ↔ H100 전송)
- 판정: **미확인/현재 구조상 해당 없음**
- 근거:
  - 현재 ingest는 로컬 파일 DFS/폴더 스캔 중심: `backend/pipeline/ingest_engine.py:1573`, `backend/pipeline/ingest_engine.py:1689`
  - `backend/` 내 S3/GCS 클라이언트 코드 확인되지 않음 (`rg` 결과 없음)

---

## 2) vLLM 기반 배치 추론 최적화

### 2-1. Continuous Batching 적용
- 판정: **부분충족**
- 근거:
  - vLLM 어댑터에 배치 처리 메서드 존재: `backend/vision/vllm_adapter.py:313`
  - 다만 실제 ingest는 `classify_and_analyze_sequence` 보유 여부로 분기: `backend/pipeline/ingest_engine.py:803`
  - `vllm_adapter.py`에는 `classify_and_analyze_sequence`가 없어 현재 파이프라인에서 per-image fallback 경로 사용 가능성이 큼
  - 참고: transformers 백엔드는 배치 경로 구현되어 있음: `backend/vision/analyzer.py:917`

### 2-2. KV Cache 관리 (VRAM 80GB 기준 최적화)
- 판정: **부분충족**
- 근거:
  - vLLM 초기화 시 `gpu_memory_utilization=0.9` 사용: `backend/vision/vllm_adapter.py:53`, `backend/vision/vllm_adapter.py:98`
  - 그러나 Factory에서 해당 파라미터를 외부 설정으로 세밀 조정하지 않음: `backend/vision/vision_factory.py:115`
  - KV cache 비율/상한/동적 정책에 대한 별도 런타임 튜닝 로직은 확인되지 않음

### 2-3. FP8/INT4 Quantization
- 판정: **미충족**
- 근거:
  - 코드/설정에서 FP8/INT4/quantization 관련 설정 확인되지 않음 (`rg` 결과 없음)

---

## 3) SQLite 동시성 및 기록 성능

### 3-1. WAL 모드 활성화
- 판정: **충족**
- 근거:
  - 연결 시 `PRAGMA journal_mode = WAL`: `backend/db/sqlite_client.py:65`
  - 런타임 확인 결과도 `wal` 반환

### 3-2. Bulk Insert 주기 (500~1000건 트랜잭션 묶음)
- 판정: **미충족**
- 근거:
  - `upsert_metadata`/`update_vision_fields`/`upsert_vectors`에서 각 호출 단위 `commit`: `backend/db/sqlite_client.py:626`, `backend/db/sqlite_client.py:688`, `backend/db/sqlite_client.py:734`
  - ingest 루프에서 파일별 반복 저장: `backend/pipeline/ingest_engine.py:827`, `backend/pipeline/ingest_engine.py:850`
  - `BEGIN TRANSACTION` 기반 대량 묶음 처리 코드 확인되지 않음

### 3-3. Async DB Writer 분리
- 판정: **미충족**
- 근거:
  - 추론 루프 내부에서 DB write를 직접 호출하는 동기 구조: `backend/pipeline/ingest_engine.py:827`, `backend/pipeline/ingest_engine.py:960`, `backend/pipeline/ingest_engine.py:1087`
  - 전용 DB writer worker/queue 소비자 구조 확인되지 않음

---

## 4) 로컬 이관(Handoff) 및 압축 정책

### 4-1. VACUUM + ANALYZE
- 판정: **부분충족**
- 근거:
  - export 시 VACUUM 수행: `backend/api_export.py:53`, `backend/api_export.py:58`
  - ANALYZE SQL 실행 코드는 확인되지 않음 (`backend/`, `tools/` 검색 결과 없음)
  - 현재 DB `sqlite_stat1` 테이블도 미생성(ANALYZE 미실행 상태로 해석 가능)

### 4-2. Vector Normalization (클라우드/로컬 일관성)
- 판정: **충족**
- 근거:
  - SigLIP2 벡터 L2 정규화: `backend/vector/siglip2_encoder.py:226`, `backend/vector/siglip2_encoder.py:244`
  - 텍스트 임베딩 정규화(및 MRL re-normalize): `backend/vector/text_embedding.py:125`, `backend/vector/text_embedding.py:133`
  - 티어별 normalize 플래그 반영: `backend/vector/text_embedding.py:482`, `config.yaml:31`

### 4-3. Model Parity (클라우드 vs 로컬 체크포인트 100% 일치)
- 판정: **부분충족**
- 근거:
  - import/relink 시 manifest의 `tier`만 비교: `backend/api_relink.py:193`, `backend/api_relink.py:205`
  - tier mismatch면 VV/MV 벡터 삭제 후 재생성 유도: `backend/api_relink.py:347`, `backend/api_relink.py:350`
  - 체크포인트 해시/정확한 모델 revision(커밋 SHA) 검증은 없음

---

## 우선 개선 권고 (병목 중심)

1. **DB Writer 분리 + 배치 트랜잭션**
- 추론 스레드와 DB 쓰기 분리(Queue + 단일 writer)
- 500~1000건 단위 `BEGIN IMMEDIATE ... COMMIT` 묶음

2. **이미지 디코딩 경로 가속**
- JPEG는 TurboJPEG, PNG는 OpenCV/Pillow-SIMD 비교 적용
- 최소한 Phase2/3 JIT 로딩 경로(`_load_and_composite_thumbnail`) 교체 벤치마크

3. **vLLM 배치 경로 실제 파이프라인 연결**
- `VLLMAdapter`에 `classify_and_analyze_sequence` 구현 또는 ingest에서 `process_batch` 직접 호출
- 현재 fallback per-image 경로 제거

4. **이관 후 최적화 마무리**
- export/relink 후 `ANALYZE` 추가
- manifest에 `vlm_model`, `visual_model`, `text_model`, `model_revision/hash` 추가하여 parity 강제

