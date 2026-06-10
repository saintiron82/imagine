# 레거시 큐 코드 제거 계획

> **상태: 완료 (2026-06-10)** — 아래 10단계 전부 종료.
> - `manager.py`, `parse_ahead.py`, `api_queue.py`, `routers/pipeline.py`, 프론트 레거시 컴포넌트: 삭제됨
> - 레거시 테이블 DROP: `migrate_drop_legacy_queue_tables` 마이그레이션으로 등록됨
> - 워커의 죽은 `/api/v1/jobs/*` 보고 경로 제거 + 실패 보고를 `/api/v1/tasks/complete`로 일원화 (refactor/legacy-queue-final)
> - 회귀 방지: `tests/test_worker_legacy_cleanup.py`

## 현재 상태 (작성 당시)

새 시스템 (`analysis_jobs` + `file_tasks` + `AnalysisJobManager`) 구현 완료.
기존 시스템 (`job_queue` + `work_requests` + `JobQueueManager`)이 20곳에서 import 중.

## 제거 대상 파일

| 파일 | 줄 수 | 역할 | 상태 |
|------|:-----:|------|------|
| `backend/server/queue/manager.py` | 2931 | 레거시 큐 매니저 | 20곳 import → 점진 제거 |
| `backend/server/queue/parse_ahead.py` | 593 | Parse 사전 처리 | manager import 의존 |
| `backend/server/queue/download_ahead.py` | 712 | WebDAV 다운로드 | 새 시스템에서도 필요 (유지/분리) |

## 제거 대상 테이블

| 테이블 | 역할 | 대체 |
|--------|------|------|
| `job_queue` | 파일별 작업 추적 | `file_tasks` |
| `work_requests` | 큐 단위 관리 | `analysis_jobs` |
| `work_subtasks` | 폴더별 분할 | 폐기 (플랫 큐) |
| `job_completions` | throughput 계산 | `file_tasks` 타임스탬프 |

## 의존성 맵 (manager.py import 20곳)

| 파일 | 사용 기능 | 전환 필요 |
|------|----------|----------|
| `embedded_worker.py` | `_decide_worker_mode`, `get_phase_batch_size` | **완료** (file_tasks 기반) |
| `api_queue.py` | `create_work_request` | **완료** (analysis_manager) |
| `pipeline.py` | `create_work_request`, `get_stats`, claim | 부분 완료 |
| `workers.py` | `_utcnow_sql`, throughput, mode 관리 | 미전환 |
| `admin.py` | embedded worker start | 미전환 |
| `upload.py` | job 완료 보고 | 미전환 |
| `parse_ahead.py` | `_utcnow_sql`, download pool | 미전환 |
| `app.py` | startup audit, pool init | 미전환 |
| `worker_daemon.py` | download pool | 미전환 |

## 제거 순서

1. ✅ 큐 생성 경로 전환 (api_queue.py, pipeline.py)
2. 임베디드 워커 완전 전환 (embedded_worker.py — fallback 제거)
3. 워커 claim 완전 전환 (legacy fallback 제거)
4. `_utcnow_sql` 등 유틸 함수 분리
5. parse_ahead → file_tasks 연결
6. download_ahead → file_tasks 연결
7. workers.py throughput → analysis metrics
8. app.py startup audit → analysis_manager
9. manager.py 삭제
10. 레거시 테이블 DROP
