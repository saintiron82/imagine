# 통합 마스터 플랜 (로컬/클라우드 공통) - 2026-02-16

## 1. 목표
- 현재 단일 프로세스 파이프라인을 유지 가능한 형태로 점진 전환한다.
- 로컬 온프레미스와 클라우드 환경에서 동일한 작업 개념/상태 모델/API 계약을 사용한다.
- 처리량 확장 시 서버는 작업 지시와 수신 저장에 집중하고, 클라이언트 워커는 분석 실행에 집중한다.

## 2. 현재 상태 요약 (기준선)
- 파이프라인: `backend/pipeline/ingest_engine.py` 중심의 phase 기반 처리.
- 배치 처리: adaptive sub-batch 적용.
- 저장소: SQLite(`backend/db/sqlite_client.py`) + 벡터 테이블.
- 비전 백엔드: tier 기반 `transformers`/`vllm`/`ollama` 선택.
- 이관: `backend/api_export.py`, `backend/api_relink.py`.

## 3. 설계 원칙
- 공통 계약 우선: 배포 형태가 달라도 `Task`, `Resource`, `Result`, `State` 스키마는 동일.
- idempotency 기본: 같은 작업 재전송/재시도 시 중복 저장 없이 같은 결과를 보장.
- 전달 보장 명시: 기본은 `at-least-once` 전달 + idempotent persist를 표준으로 채택.
- late binding: 리소스 위치(로컬/서버/오브젝트 스토리지)는 실행 직전에 해석.
- stateless worker: 워커는 상태를 로컬에 고정하지 않고 서버 상태를 기준으로 동작.
- 저장 분리: 추론 경로와 DB commit 경로를 분리해 backpressure를 제어.
- 멀티테넌시 우선: 모든 제어/데이터 계약은 `tenant_id`를 필수 포함한다.

## 4. 목표 아키텍처 (공통 개념)

### 4.1 Control Plane
- Coordinator API
- Scheduler/Dispatcher
- Lease Manager
- State Store

### 4.2 Data Plane
- Resource Sync Service
- Worker Agent (분석/임베딩 실행)
- Result Ingest Writer (비동기 저장)
- Artifact Store (thumbnail/log/manifest)

### 4.3 핵심 흐름
1. 서버가 작업 생성(`TaskSpec`) 후 큐에 등록.
2. 워커가 capability를 제출하고 lease 기반으로 작업 pull.
3. 워커가 리소스 확정(로컬 캐시 우선, 없으면 서버/오브젝트 스토리지 fetch).
4. 워커가 분석 수행 후 `ResultEnvelope` 전송.
5. 서버는 수신 전용 writer가 결과를 검증/저장하고 ACK.
6. 워커는 다음 작업 요청.

## 5. 공통 데이터 계약

### 5.1 TaskSpec
- `schema_version`: 계약 버전
- `tenant_id`: 테넌트 식별자
- `task_id`: 전역 유일 ID
- `job_id`: 배치 단위 ID
- `resource_ref`: `local_path | object_uri | content_hash`
- `task_type`: `vision | vv_embed | mv_embed | structure_embed`
- `model_spec`: `backend`, `model_id`, `model_revision`, `dtype`, `quantization`
- `dedupe_key`: 중복 방지 키 (`tenant_id + content_hash + model_revision + task_type`)
- `ordering_key`: 순서 보장이 필요한 경우의 그룹 키(옵션)
- `priority`: 정수 우선순위
- `deadline_at`: 만료 시각
- `retry_policy`: 최대 재시도/백오프

### 5.2 ResultEnvelope
- `schema_version`, `tenant_id`
- `task_id`, `worker_id`, `attempt`
- `resource_fingerprint`: `content_hash`, `mtime`, `size`
- `outputs`: caption/tags/ocr/vector/metrics
- `telemetry`: latency, batch_size, gpu_mem, io_wait
- `status`: `success | partial | failed`
- `error`: 오류 코드/메시지/재시도 가능 여부
- `result_checksum`: payload 무결성 확인용 해시

### 5.3 상태 전이
- `queued -> leased -> running -> reported -> persisted -> done`
- 실패 경로: `running -> failed -> retry_queued | dead_letter`
- 타임아웃 경로: `leased/running -> lease_expired -> retry_queued`

### 5.4 전달 보장/중복 처리 표준
- 큐 전달 보장: `at-least-once` (기본)
- 저장 보장: idempotent upsert (중복 결과는 overwrite 또는 no-op)
- dedupe 보장 범위: `tenant_id + dedupe_key` 기준으로 최소 7일
- 순서 보장: 전역 순서 미보장, `ordering_key` 단위로만 선택적 보장

### 5.5 멀티테넌시 경계
- API/큐/DB/스토리지 모든 계층에서 `tenant_id`를 필수 키로 사용
- 테넌트 단위 rate limit/쿼터/우선순위 정책 분리
- 감사 로그는 테넌트 경계로 조회 가능해야 함
- 벡터/메타데이터 저장은 row-level 분리를 기본, 고보안 테넌트는 스키마 분리 옵션 제공

## 6. 배포 모드 (같은 코드, 다른 토폴로지)

### 6.1 Mode L1: 단일 머신
- 서버와 워커를 한 프로세스 그룹으로 실행.
- 큐는 로컬(메모리 또는 SQLite queue 테이블) 사용.
- 오프라인 환경 기본값.

### 6.2 Mode L2: 로컬 네트워크 분산
- 서버 1대 + 워커 N대.
- 리소스는 NAS/공유 폴더 + 선택적 오브젝트 스토리지.
- 클라이언트별 capability 기반 스케줄링.

### 6.3 Mode C1: 클라우드/하이브리드
- 서버: API + 중앙 큐 + 중앙 DB(PostgreSQL/pgvector 권장).
- 워커: GPU 노드 오토스케일.
- 리소스: S3/GCS/MinIO + edge cache.

## 7. 현재 코드 재사용/교체 맵

### 7.1 재사용
- phase 처리 로직: `backend/pipeline/ingest_engine.py`
- 모델 팩토리/어댑터: `backend/vision/*`, `backend/vector/*`
- 티어/모델 호환 체크: `backend/utils/tier_config.py`, `backend/utils/tier_compatibility.py`
- 이관/동기화 기반 코드: `backend/api_export.py`, `backend/api_relink.py`

### 7.2 래핑 후 재사용
- `process_batch_phased()`를 Worker 실행 엔진으로 캡슐화.
- SQLite write 함수는 Result Writer 계층 뒤로 이동.

### 7.3 교체/신규
- `server/coordinator` (작업 생성/상태/lease)
- `server/result_writer` (비동기 저장, bulk transaction)
- `worker/agent` (heartbeat, pull loop, local cache)
- `resource/sync` (manifest 기반 증분 동기화)

## 8. 저장소 전략

### 8.1 로컬 기본
- metadata/vector: SQLite 유지.
- queue/state: SQLite 별도 테이블(`tasks`, `task_attempts`, `leases`) 추가.

### 8.2 클라우드 확장
- metadata/state: PostgreSQL
- vector: pgvector 또는 전용 벡터 스토어
- artifact: S3/GCS

### 8.3 공통 규칙
- 결과 저장은 batch commit(`500~1000` 권장) + writer 단일화.
- idempotent upsert 키: `(tenant_id, task_id, model_revision, content_hash)`.

## 9. 모델/버전 일치(Parity) 정책
- 서버가 `model_spec`을 작업 단위로 강제한다.
- 워커는 실행 전 local manifest와 `model_revision/hash`를 검증한다.
- 불일치 시 실행 금지 + `model_mismatch` 상태 보고.
- export/relink manifest에 최소 필드 추가:
- `vlm_model`, `vlm_revision`
- `visual_model`, `visual_revision`
- `text_model`, `text_revision`

### 9.1 모델 롤아웃/롤백 정책
- 롤아웃 순서: `canary(5%) -> ramp(25%) -> ramp(50%) -> full(100%)`
- 각 단계 승격 조건: p95 latency, error rate, quality KPI 임계치 충족
- 실패 시 즉시 이전 `model_revision`으로 rollback
- rollback 이후 `failed_revision`은 자동 재배포 금지(수동 승인 필요)

## 10. 신뢰성/운영 정책
- Lease timeout + heartbeat로 유실 작업 자동 복구.
- 재시도는 오류 코드 기반 분리(네트워크/리소스 부족/모델 오류).
- dead-letter 큐 운영.
- 관측성:
- 큐 길이, lease age, task latency p50/p95/p99
- GPU util, IO wait, DB commit latency
- worker별 실패율

### 10.1 백프레셔 운영 규칙
- `queue_depth > 50,000` 또는 `p95 queue_wait > 120s` 시 ingest throttling 시작
- `queue_depth > 100,000` 또는 `p95 queue_wait > 300s` 시 신규 저우선순위 작업 수락 중지
- writer backlog 임계치 초과 시 worker lease 신규 발급률 제한
- 테넌트별 쿼터 초과 시 해당 테넌트만 제한(전체 정지 금지)

## 11. 보안/권한 (클라우드 포함)
- 서버-워커 mTLS 또는 서명 토큰.
- 작업/결과 API는 최소 권한 토큰 사용.
- 민감 리소스 경로는 서버 측 별칭으로 전달.
- 감사 로그(작업 할당/결과 저장/재시도 사유) 보존.

## 12. 단계별 실행 로드맵

### Phase A (0~2주): 단일 머신 안정화
- 목표: 현행 구조를 서버-워커 전환 가능한 형태로 분리.
- 작업:
- `TaskSpec/ResultEnvelope` 스키마 도입
- ingest 실행부를 Worker 엔진 함수로 캡슐화
- 결과 저장 큐 + 비동기 writer(로컬 SQLite)
- 완료 조건:
- 기존 기능 회귀 없음
- 동일 데이터셋 기준 처리량 1.2x 이상

### Phase B (2~6주): 서버-클라이언트 분리 (L2)
- 목표: 서버 1 + 워커 N 기본 운영.
- 작업:
- Coordinator API, lease, heartbeat 구현
- Worker Agent pull 루프 구현
- 리소스 동기화(manifest + content_hash) 구현
- 완료 조건:
- 워커 장애 시 5분 내 작업 복구
- 단일 머신 대비 처리량 선형 근접 확장

### Phase C (6~12주): 클라우드 공통화 (C1)
- 목표: 로컬/클라우드 공통 운영 모델 완성.
- 작업:
- 중앙 DB(PostgreSQL) 및 중앙 큐 도입
- 오브젝트 스토리지 연계 + edge cache
- model parity 강제 및 감사 로그
- 완료 조건:
- 다중 테넌트 job 분리
- p95 end-to-end latency/SLO 달성

## 13. KPI/SLO
- 처리량: images/min per GPU
- 안정성: task 성공률, 재시도율, dead-letter율
- 성능: p95 task latency, p95 DB persist latency
- 일관성: model mismatch 0건, duplicate persist 0건
- 운영성: 장애 복구 시간(MTTR), 워커 추가 시 확장 계수

## 14. DR/BCP (재해복구/연속성)
- RPO 목표: 5분 이내
- RTO 목표: 30분 이내
- 중앙 DB 스냅샷 + WAL 아카이브 기반 복구
- 아티팩트 스토리지는 다중 AZ 복제
- 분기별 복구 리허설(restore drill)과 결과 기록 의무화

## 15. 즉시 실행 백로그 (우선순위)
1. 작업/결과 공통 스키마 정의 및 코드 반영
2. 비동기 Result Writer + bulk transaction 도입
3. Coordinator 최소 API(`enqueue`, `lease`, `heartbeat`, `report`) 구현
4. Worker Agent 초안 구현(단일 머신 모드부터)
5. export/relink manifest에 model revision/hash 확장
6. 전달 보장/중복 제거 정책을 큐 계층에 반영
7. 테넌트 경계(`tenant_id`)를 API/DB/로그 전 계층에 강제

## 16. 의사결정 항목
- 큐 기술 선택: SQLite queue(초기) vs Redis/RabbitMQ(확장)
- 중앙 벡터 저장소: pgvector vs 분리 벡터 DB
- 리소스 원천 전략: always-sync vs on-demand fetch + cache
- 멀티테넌시 경계: DB 스키마 분리 vs row-level 분리
- 전달 semantics: at-least-once 유지 vs exactly-once(비용 증가) 채택 여부

---

이 플랜은 현재 코드베이스를 폐기하지 않고 확장하는 것을 전제로 한다.  
핵심은 “배치 처리”를 유지하면서도, 작업 계약/상태/저장을 분리해 로컬과 클라우드에서 같은 동작 원리를 확보하는 것이다.
