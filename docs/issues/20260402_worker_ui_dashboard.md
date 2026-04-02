# Bug Report: Worker UI / Pipeline Dashboard

| 항목 | 내용 |
|------|------|
| **보고일** | 2026-04-02 |
| **보고자** | saintiron |
| **발견 경로** | Qwen3.5 VLM 전환 테스트 중 발견 |
| **환경** | Mac M5 32GB + Windows RTX 3060Ti, Electron 앱, 멀티워커(2대) |
| **심각도** | 중간 — 기능 동작에는 영향 없으나 사용자에게 잘못된 정보 표시 |
| **상태** | Open |

---

## BUG-001: stderr 정규식으로 로그 레벨 추측 — 오탐 에러 발생

| 항목 | 내용 |
|------|------|
| **심각도** | 중간 |
| **재현** | 워커가 VV/MV 배치 실행 시 `0 pre-failed` 포함 INFO 로그 발생 → UI에 에러로 표시 |
| **현상** | `[INFO] ImagineWorker: [VV] Validation: 10 active, 0 pre-failed / 10 total`이 빨간색(에러)으로 표시되고 에러 카운터 증가 |
| **원인** | `main.cjs:3002`에서 `/FAIL/i` 정규식이 `pre-failed`의 `fail`에 매칭 → `type: 'error'`로 프론트엔드 전달 |
| **임시 수정** | 커밋 `390027f` — 네거티브 패턴 추가. 근본 해결 아님 |
| **근본 해결** | Python 로거 → JSON 구조화 출력 `{level, worker_id, message}`. main.cjs는 파싱만 수행 |
| **관련 파일** | `frontend/electron/main.cjs:2995-3021`, `frontend/src/App.jsx:236-249`, `frontend/src/components/StatusBar.jsx:100-106` |

### 재현 스크린샷
INFO 레벨 로그가 에러로 카운팅되어 StatusBar에 "오류 15개" 표시

---

## BUG-002: 워커 재시작 시 Phase 카운터 불일치

| 항목 | 내용 |
|------|------|
| **심각도** | 중간 |
| **재현** | 임베디드 워커 재시작 (로그인, 에러 복구 등) 후 Admin Workers 패널 확인 |
| **현상** | MC=`-`(0)인데 VV=`101`, MV=`93` — MC만 리셋되고 VV/MV는 이전 값 유지 |
| **원인** | `WorkerDaemon.__init__`에서 `_phase_counts = {mc:0, vv:0, mv:0}` 전부 초기화되지만, 프론트엔드가 DB(`phase_job_count`)와 메모리(`_phase_counts`) 두 소스를 혼합 표시 |
| **근본 해결** | 카운터 소스 단일화. DB 기반(영속, 재시작 후에도 누적) 또는 메모리 기반(휘발, 재시작 시 전부 0) 중 하나만 사용 |
| **관련 파일** | `backend/server/embedded_worker.py:41`, `backend/worker/worker_daemon.py:132`, `backend/server/routers/workers.py:664` |

---

## BUG-003: Throughput 비현실적 수치 표시

| 항목 | 내용 |
|------|------|
| **심각도** | **높음** |
| **재현** | 멀티워커 동시 운영 중 Pipeline 대시보드 확인 |
| **현상** | 전체 `115.0 files/min, 1s/file` (실제 MC 2.2/m + VV 38.2/m). Saint25PC-worker MC `101.0/m` (Windows 4B Ollama에서 물리적 불가능) |
| **원인** | 1분/5분 슬라이딩 윈도우 계산에서 Phase 중복 카운팅 또는 윈도우 경계 오류 추정. 전체 throughput이 개별 Phase 합산을 초과 |
| **근본 해결** | throughput 계산 로직 재검증. Phase별 독립 카운팅 확인. `throughput = max(r1, r5/5)` 공식 검증. 비현실적 값(>50/m for MC) 클램핑 |
| **관련 파일** | `backend/server/routers/workers.py:578-648`, `frontend/src/components/PipelineBlackboard.jsx` |

### 재현 스크린샷
115.0 files/min으로 표시되나 실제 MC bottleneck은 2.2/m

---

## BUG-004: 남은시간 추정 부정확

| 항목 | 내용 |
|------|------|
| **심각도** | **높음** |
| **재현** | Parse 대기 74개, bottleneck 0.6/m 상태에서 대시보드 확인 |
| **현상** | 남은시간 `1m` 표시. 실제: 74 / 0.6 = ~123분 |
| **원인** | 남은시간 계산이 bottleneck Phase throughput이 아닌 전체 throughput(115/m)을 사용 |
| **근본 해결** | `남은시간 = max(각 Phase별 (대기수 / 해당 Phase throughput))`. bottleneck Phase 기준 |
| **관련 파일** | `frontend/src/components/PipelineBlackboard.jsx` 또는 `StatusBar.jsx` (남은시간 계산 위치 확인 필요) |

---

## BUG-005: 워커 활성 Phase 하이라이트 누락/번갈아 표시

| 항목 | 내용 |
|------|------|
| **심각도** | 중간 |
| **재현** | 두 워커가 동시에 MC 처리 중일 때 Admin Workers 패널 확인 |
| **현상** | Saint25PC-worker의 MC에만 보라색 박스(활성 표시). Mac 임베디드 워커는 현재 Phase 하이라이트 없음. 두 워커가 번갈아 하나만 표시 |
| **원인** | `admin/workers` API가 DB(`worker_sessions.current_phase`)만 조회. 임베디드 워커는 DB에 current_phase를 실시간 기록하지 않고 메모리(`_worker_daemon._current_phase`)에만 보유 |
| **근본 해결** | 방안 A: `admin/workers` 응답에서 임베디드 워커는 메모리 상태를 오버라이드. 방안 B: 임베디드 워커도 하트비트처럼 DB에 current_phase 주기 기록 |
| **관련 파일** | `backend/server/routers/workers.py:554-665`, `backend/server/embedded_worker.py:222-224`, `frontend/src/components/admin/WorkersPanel.jsx:584-611` |

---

## BUG-006: 로그에서 워커 식별 불가

| 항목 | 내용 |
|------|------|
| **심각도** | 낮음 |
| **재현** | 멀티워커 운영 중 로그 패널 확인 |
| **현상** | `ImagineWorker: [MC] Validation: 1 active` — Mac인지 Windows인지 구분 불가 |
| **원인** | 로거 이름이 `ImagineWorker`로 고정. 워커 이름/세션ID 미포함 |
| **근본 해결** | 로거에 워커 식별자 포함: `ImagineWorker[__builtin__]`, `ImagineWorker[Saint25PC]` |
| **관련 파일** | `backend/worker/worker_daemon.py` (logger 초기화), `backend/server/embedded_worker.py` |

---

## BUG-007: Pipeline 대시보드 Phase 카운터 합계 불일치

| 항목 | 내용 |
|------|------|
| **심각도** | **높음** |
| **재현** | 파이프라인 진행 중 Pipeline 대시보드 확인 |
| **현상** | `Download(864) + Parse(121) + MC(52) + VV(64) + MV(26) + 진행중(21) + Done(6747) = 7895` — 전체 큐 `7848`보다 47개 많음. 각 Phase 카운터 합이 전체 큐와 일치하지 않음 |
| **원인** | 각 Phase 대기 카운터가 동일 시점의 스냅샷이 아닌 독립 쿼리로 집계되어, 파일이 Phase 간 이동하는 중에 중복 카운팅되거나, Download 카운터가 별도 소스(WebDAV download-ahead pool)에서 읽혀서 큐 외부 상태를 포함 |
| **근본 해결** | Phase별 카운터를 **단일 트랜잭션 스냅샷**으로 조회. `SELECT status, phase_completed, COUNT(*) FROM job_queue GROUP BY ...`를 한 번에 실행하여 모든 카운터가 동일 시점 기준이 되도록. Download 대기 수는 큐 외부(WebDAV pool)가 아닌 `job_queue`의 `file_ready=0` 기준으로 계산 |
| **관련 파일** | `backend/server/routers/pipeline.py` (queue stats API), `frontend/src/components/PipelineBlackboard.jsx` (대시보드 표시) |

---

## 우선순위

| 순위 | 버그 | 사용자 영향 | 구현 난이도 |
|:----:|------|:---------:|:---------:|
| 1 | BUG-007 Phase 카운터 합계 불일치 | 높음 | 중간 |
| 2 | BUG-003 Throughput 계산 오류 | 높음 | 중간 |
| 3 | BUG-004 남은시간 부정확 | 높음 | 중간 |
| 4 | BUG-002 카운터 불일치 | 중간 | 낮음 |
| 5 | BUG-005 활성 Phase 표시 | 중간 | 중간 |
| 6 | BUG-001 구조화 로그 전환 | 중간 | 높음 |
| 7 | BUG-006 워커 식별 | 낮음 | 낮음 |
