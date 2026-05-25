# Worker Runtime Contract Audit

작성일: 2026-05-25

기준 문서: `docs/worker_runtime_contract_ko.md`

## 결론

현재 구현은 worker core와 서버 API 계약을 기준으로 세 가지 실행 유형을 구분한다.

현재 상태를 한 줄로 정리하면:

**server-local, client-launched, headless worker가 모두 같은 중앙 서버 계약으로 등록될 수 있다.**

## 현재 충족하는 항목

### 1. 서버가 worker session을 이해한다

구현 위치:

- `backend/server/routers/workers.py`
- `backend/db/sqlite_schema_auth.sql`
- `backend/db/sqlite_migrations.py`

현재 가능한 것:

- `/workers/connect`로 worker session 등록
- `/workers/heartbeat`로 상태 갱신
- `/workers/disconnect`로 offline 전환
- `worker_sessions`에 hostname, origin, launcher, status, batch capacity, jobs completed, current phase, current file, resources JSON 저장

판정: **충족**

### 2. 서버가 task claim/start/complete 계약을 제공한다

구현 위치:

- `backend/server/routers/analysis.py`
- `backend/server/queue/analysis_manager.py`
- `backend/server/queue/scheduler.py`

현재 가능한 것:

- `/api/v1/tasks/claim`
- `/api/v1/tasks/start`
- `/api/v1/tasks/complete`
- worker session ownership 검증
- phase assignment 기반 결과 업로드 검증

판정: **충족**

### 3. 서버 관리자가 worker를 제어할 수 있다

구현 위치:

- `backend/server/routers/workers.py`

현재 가능한 것:

- `/admin/workers`
- `/admin/workers/{session_id}/stop`
- `/admin/workers/{session_id}/block`
- batch capacity override
- processing mode override
- origin/launcher 포함 worker 목록 조회

판정: **충족**

### 4. 일반 클라이언트가 자기 worker를 stop할 수 있다

구현 위치:

- `backend/server/routers/workers.py`

현재 가능한 것:

- `/workers/{session_id}/stop`은 `user_id`가 일치하는 online session만 stop pending으로 바꾼다.

판정: **충족**

주의할 점:

- headless worker credential의 owner 정책은 token 발급 방식에 따라 운영 문서에서 더 구체화해야 한다.

### 5. 서버 머신 내부 worker가 있다

구현 위치:

- `backend/server/embedded_worker.py`
- `backend/worker/transport.py`
- `backend/server/routers/admin.py`

현재 가능한 것:

- admin API로 embedded worker start/stop
- `LocalTransport`를 통해 같은 `WorkerDaemon` 사용
- `origin=server-local`, `launcher=server`로 등록

판정: **충족**

### 6. 클라이언트 앱에서 worker를 켜고 끄는 경로가 있다

구현 위치:

- `frontend/electron/main.cjs`
- `backend/worker/worker_ipc.py`
- `backend/worker/worker_daemon.py`

현재 가능한 것:

- Electron main process가 `backend/worker/worker_ipc.py`를 child process로 spawn한다.
- stdin/stdout JSON IPC로 `start`, `stop`, `status`, `update_tokens` 명령을 주고받는다.
- Electron session의 access token/refresh token을 worker에 주입한다.
- worker는 주입된 token으로 서버에 session 등록한다.
- `origin=client-launched`, `launcher=electron`로 등록

판정: **충족**

### 7. Headless worker가 있다

구현 위치:

- `backend/worker/cli.py`
- `backend/server/routers/workers.py`
- `scripts/cloud_worker_boot.sh`

현재 가능한 것:

- `python -m backend.worker.cli`로 Electron 없이 worker 실행
- `IMAGINE_SERVER_URL`, `IMAGINE_WORKER_ACCESS_TOKEN`, `IMAGINE_WORKER_REFRESH_TOKEN` 환경변수 사용
- `origin=headless`, `launcher=cli|service|cloud`로 등록
- cloud boot script가 `python -m backend.worker.cli --launcher cloud`를 호출
- 서버가 `GET /api/v1/workers/bootstrap/linux.sh`로 Linux bootstrap script를 제공
- 서버가 `POST /api/v1/admin/workers/headless-command`로 전용 worker user token과 실행 명령을 발급

판정: **충족**

## 세 유형별 현재 판정

| 유형 | 현재 상태 | 판정 |
| --- | --- | --- |
| Server-local worker | `embedded_worker.py` registers `origin=server-local` | 충족 |
| Client-launched worker | Electron `worker_ipc.py` registers `origin=client-launched` | 충족 |
| Headless worker | `backend.worker.cli` runs without Electron | 충족 |

## 남은 운영 보강

1. Headless token 발급 UX
   - API는 추가됐다.
   - 서버 UI에서 이 API를 호출해 명령을 보여주는 화면은 별도 작업으로 남아 있다.

2. Live cloud E2E
   - 현재 검증은 local unit/static check 중심이다.
   - 실제 Linux CUDA cloud machine에서 connect/heartbeat/claim/batch/disconnect를 확인해야 한다.

3. 설치 문서
   - macOS headless
   - Windows headless
   - Linux CUDA/cloud
   - Electron-launched worker

## 구현 후 검증

```bash
.venv/bin/python -m pytest tests/test_worker_runtime_origin.py tests/test_worker_headless_cli.py -q
.venv/bin/python -m pytest tests/test_worker_bootstrap.py -q
.venv/bin/python tools/check_worker_runtime_contract.py
.venv/bin/python -m py_compile backend/server/routers/workers.py backend/worker/transport.py backend/server/embedded_worker.py backend/worker/worker_daemon.py backend/worker/worker_ipc.py backend/worker/cli.py
bash -n scripts/cloud_worker_boot.sh
```

## 최종 판정

현재 구현은 사용자의 기준 중 **중앙 서버가 모든 워커를 이해하고 제어해야 한다**는 핵심 구조를 만족한다.

새 워커 시스템을 따로 만든 것이 아니라:

**기존 WorkerDaemon을 세 실행 유형에서 같은 서버 계약으로 실행 가능하게 정렬했다.**
