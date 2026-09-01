# NAS(WebDAV) 다운로드 → 분석 처리 흐름

이 문서는 NAS/WebDAV 원격 파일이 Imagine에서 다운로드되고 분석되는 경로의
**단일 기준 문서**다. 코드를 따라가기 전에 이 문서를 읽으면 전체 구조가 잡힌다.

최종 갱신: 2026-06-10 (refactor/nas-flow-clarity)

## 한눈에 보기

```text
[Archive UI에서 WebDAV 폴더 분석 요청]
        │
        ▼
AnalysisJobManager.create (analysis_manager.py)
  - analysis_jobs 1행 생성 (status=active)
  - 파일마다 file_tasks 1행 생성
      file_path = webdav://{source-id}/{remote/path}
      download_status = 'pending'   ← WebDAV일 때만. 로컬 파일은 'n/a'
        │
        ▼
DownloadAheadPool (download_ahead.py)          [서버 백그라운드 스레드]
  - file_tasks에서 download_status='pending' AND file_path LIKE 'webdav://%' 스캔
  - 등록된 WebDAV 소스 설정(register_webdav_source)으로 원본 다운로드
  - 임시 폴더 {tmp}/imagine_dl_*/{file_id}_{filename} 에 저장
  - bounded buffer: 동시에 max_files개까지만 디스크 점유 (semaphore)
  - 성공: download_status='done' / 실패: 'failed' (5분 후 자동 'pending' 복구)
  - 네트워크 3연속 실패 → 전체 일시정지, health check 후 재개
        │
        ▼
FileTaskParsePool (file_task_parse_pool.py)    [서버 백그라운드 스레드]
  - download_status IN ('done','n/a') AND parse_status='pending' 스캔
  - parse_status: pending → assigned(원자적 클레임) → done | failed
  - 파싱: ParserFactory → PSD 레이어/텍스트/폰트 추출 + 썸네일 생성
      실패 시 PIL thumbnail-only fallback (processing_status='parse_fallback')
  - files 테이블 upsert + 썸네일을 서버 thumbnail_dir로 복사
  - CAS 캐시 적용(apply_cache_hits): 동일 content_hash의 기존 MC/VV/MV가
      있으면 워커를 거치지 않고 즉시 done 처리.
      단, 잡 프로필의 force_reanalyze('전체 다시 분석')면 캐시를 건너뛰고
      전 단계를 재계산한다(게이트 판단 불가 시에도 재계산 쪽).
  - 완료 후 DownloadAheadPool.release_slot(file_id)
      → 원본 임시 파일 삭제 + 버퍼 슬롯 반환 (원본은 더 이상 불필요)
        │
        ▼
AI phases: MC → VV → MV              [워커: 로컬 또는 외부 — 둘 다 독립 프로세스 + HTTP]
  - Scheduler(scheduler.py)가 워커별로 phase(mc|vv|mv)와 batch size 배정
  - 워커 claim:   POST /api/v1/tasks/claim   (file_tasks 기반)
  - 시작 보고:    POST /api/v1/tasks/start
  - 결과 저장:    PATCH /api/v1/files/{id}/vision | /vv | /mv
  - 완료/실패:    POST /api/v1/tasks/complete (success=true|false)
  - 워커는 원본이 아니라 **썸네일만** 받는다 (GET /api/v1/files/{id}/thumbnail)
```

## 핵심 원칙

1. **원본 이동 최소화** — 원본 다운로드는 Parse 단계까지만 필요하다.
   Parse가 끝나면 임시 원본은 즉시 삭제되고, 이후 모든 AI 단계(MC/VV/MV)는
   서버가 보관한 썸네일만 사용한다.
2. **상태는 file_tasks가 단일 소스** — download/parse/mc/vv/mv 각 단계의
   `*_status` 칼럼이 파일 하나의 처리 상태 기계다. 풀과 워커는 모두
   이 테이블을 폴링/갱신할 뿐, 별도의 인메모리 큐가 없다.
3. **다운로드와 파싱은 분리된 단계** — DownloadAheadPool은 다운로드만,
   FileTaskParsePool은 파싱만 한다. 두 풀은 file_tasks의 상태 전이로만
   연결되며 직접 호출 관계는 release_slot(버퍼 반환) 하나뿐이다.

## file_tasks 상태 칼럼

| 칼럼 | 값 | 전이 주체 |
|------|----|----------|
| `download_status` | `n/a`(로컬) / `pending` → `done` / `failed`(5분 후 재시도) | DownloadAheadPool |
| `parse_status` | `pending` → `assigned` → `done` / `failed`(max_retries까지 재시도) | FileTaskParsePool |
| `mc_status` `vv_status` `mv_status` | `pending` → `assigned` → `done` / `failed` | Scheduler + Worker |

## 관련 파일 맵

| 역할 | 파일 |
|------|------|
| Job/Task 생성·상태 관리 | `backend/server/queue/analysis_manager.py` |
| 원본 다운로드 풀 | `backend/server/queue/download_ahead.py` |
| 파싱 풀 | `backend/server/queue/file_task_parse_pool.py` |
| 풀 공통 베이스 (스레드 생명주기) | `backend/server/queue/base_ahead_pool.py` |
| 워커 phase 배정 | `backend/server/queue/scheduler.py` |
| 워커 데몬 (MC/VV/MV 실행) | `backend/worker/worker_daemon.py` |
| 워커 결과 업로드 (files API) | `backend/worker/result_uploader.py` |
| WebDAV 프로토콜 클라이언트 | `backend/remote/webdav_client.py` |
| 태스크 API 라우터 | `backend/server/routers/analysis.py` |
| 파생물 캐시(CAS) 기록·조회·물질화 | `backend/server/queue/derivations.py` |
| 모델 버전 파도(재처리 잡 생성) | `backend/server/queue/waves.py` |
| content_hash 백필 | `backend/server/queue/hash_backfill.py` |

## 설정 키

| 키 | 의미 |
|----|------|
| `server.parse_ahead.poll_interval_s` | 두 풀의 폴링 주기 (이름은 레거시 시절 그대로지만 download/parse 풀이 공유) |
| `server.parse_ahead.parse_workers` | 파싱 병렬 스레드 수 (기본 3, in-flight 2×workers) |
| `server.auto_processing.enabled` | 서버 머신의 로컬 워커 기동 여부 (끄면 활성화 시에도 안 뜬다) |
| temp buffer (`get_temp_buffer_config`) | `max_files`(버퍼 크기), `download_workers`(동시 다운로드 수) |

## 자주 헷갈리는 점

- **로컬 파일도 같은 경로를 탄다.** 차이는 `download_status='n/a'`로 시작해
  DownloadAheadPool을 건너뛴다는 것뿐이다.
- **서버 재시작 시** 임시 폴더가 사라지므로, DownloadAheadPool.start()가
  `download done + parse pending`인 태스크를 `pending`으로 되돌려 재다운로드한다.
- **레거시 `/api/v1/jobs/*` 경로는 존재하지 않는다.** 워커의 모든 보고는
  `/api/v1/tasks/*`와 `/api/v1/files/{id}/*`로 일원화됐다
  (가드 테스트: `tests/test_worker_legacy_cleanup.py`).
