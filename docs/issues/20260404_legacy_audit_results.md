# 레거시 코드 감사 결과 (2026-04-04)

## 백엔드 레거시 (총 4808줄)

| 파일 | 줄 | 상태 | 조치 |
|------|:---:|------|------|
| `backend/server/queue/manager.py` | 2931 | 16곳 import | 삭제 대상 |
| `backend/server/queue/parse_ahead.py` | 593 | startup에서 제거됨 | 삭제 대상 |
| `backend/server/queue/download_ahead.py` | 712 | startup에서 제거됨 | 삭제 대상 |
| `backend/api_queue.py` | 572 | 일부 전환됨 | 정리 필요 |

## manager.py import 16곳

| 파일:라인 | 사용 기능 | 조치 |
|-----------|---------|------|
| `routers/upload.py:17` | JobQueueManager | 삭제 또는 analysis 전환 |
| `routers/admin.py:482` | JobQueueManager | 삭제 |
| `routers/pipeline.py:17` | JobQueueManager, _utcnow_sql | 정리 (일부 전환됨) |
| `routers/workers.py:17,74,117,290,431,816` | 6곳 | 대부분 삭제 가능 |
| `queue/parse_ahead.py:18,279` | _utcnow_sql, _get_download_pool | 파일 자체 삭제 |
| `app.py:306,403,450` | JobQueueManager | 레거시 함수에서만 사용 |
| `api_queue.py:24` | JobQueueManager | 전환됨, 잔여 import |
| `worker_daemon.py:714` | _get_download_pool | 삭제 |

## pipeline.py 레거시 엔드포인트 (20+개)

```
/api/v1/jobs/claim              ← 새 /api/v1/tasks/claim으로 대체
/api/v1/jobs/stats              ← 새 /api/v1/analysis-jobs으로 대체  
/api/v1/jobs/{id}/complete      ← 새 /api/v1/tasks/complete로 대체
/api/v1/jobs/{id}/complete_mc   ← 새 /api/v1/files/{id}/vision으로 대체
/api/v1/jobs/{id}/complete_vv   ← 새 /api/v1/files/{id}/vv로 대체
/api/v1/jobs/{id}/complete_mv   ← 새 /api/v1/files/{id}/mv로 대체
/api/v1/jobs/{id}/fail          ← 새 /api/v1/tasks/complete (success=false)
... 등 20+개
```

## 프론트엔드 레거시

| 파일 | 상태 | 조치 |
|------|------|------|
| `PipelineBlackboard.jsx` | 탭 제거됨, 미사용 | 삭제 |
| `QueueManagerPanel.jsx` | 레거시 큐 관리 | 삭제 |
| `QueuePanel.jsx` | 레거시 큐 패널 | 삭제 |
| `WRCards.jsx` | 레거시 WR 카드 | 삭제 |

## DB 테이블 레거시

| 테이블 | 참조 수 | 조치 |
|--------|:-------:|------|
| `job_queue` | 154 | 미사용 처리 (DROP은 나중) |
| `work_requests` | 111 | 미사용 처리 |
| `work_subtasks` | 25 | 미사용 처리 |
| `job_completions` | 12 | 미사용 처리 |

## 삭제 순서 (안전)

1. 프론트엔드 레거시 컴포넌트 삭제 (PipelineBlackboard, QueueManagerPanel, QueuePanel, WRCards)
2. `parse_ahead.py` 삭제
3. `download_ahead.py` 삭제  
4. `pipeline.py`에서 레거시 엔드포인트 삭제
5. `workers.py`에서 manager.py import 제거
6. `app.py`에서 레거시 함수 삭제
7. `api_queue.py` 정리 (manager.py import 제거)
8. `worker_daemon.py`에서 레거시 참조 제거
9. `manager.py` 삭제
10. DB 테이블 DROP (최종)
