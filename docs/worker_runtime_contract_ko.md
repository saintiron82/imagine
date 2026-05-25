# Imagine Worker Runtime Contract

작성일: 2026-05-25

이 문서는 Imagine의 서버, 클라이언트 앱, 워커 역할을 명확히 나누는 기준이다. 핵심 원칙은 하나다.

**워커 코어는 하나이고, 실행 위치만 셋이다.**

## 1. 용어

### 중앙 서버

중앙 서버는 작업과 상태의 원본이다.

- DB를 소유한다.
- 분석 작업과 `file_tasks` 큐를 소유한다.
- 워커 세션을 등록하고 관리한다.
- 워커에게 작업을 배정한다.
- 워커의 heartbeat, 성능, 현재 작업, 오류 상태를 받는다.
- 워커에게 `stop`, `pause`, `block` 같은 제어 명령을 내린다.

서버가 Electron 안에서 실행되든, 별도 서버 머신에서 실행되든, 워커 입장에서는 항상 같은 Server API로 보인다.

### 클라이언트 앱

클라이언트 앱은 중앙 서버를 사용하는 접속 앱이다.

- 서버에 로그인한다.
- 검색, 분석 요청, 작업 상태 확인 UI를 제공한다.
- 자기 PC에서 도는 워커를 시작하거나 중지할 수 있다.
- 다른 사용자의 워커를 직접 종료하면 안 된다.
- 관리자 권한이 있는 경우에만 전체 워커 제어 UI를 사용할 수 있다.

Electron 앱은 클라이언트 앱이면서, 필요하면 서버를 품고 실행할 수도 있다. 하지만 이 둘은 역할이 다르다.

### 워커

워커는 서버에서 작업을 받아 AI 처리를 수행하고 결과를 돌려주는 실행자다.

- 서버 URL을 알고 있어야 한다.
- 서버 인증을 통과해야 한다.
- 서버에 worker session을 등록해야 한다.
- 주기적으로 heartbeat를 보내야 한다.
- 서버에서 task를 claim해야 한다.
- MC/VV/MV 처리 결과를 서버에 업로드해야 한다.
- 서버 명령을 수신하고 따라야 한다.

워커는 서버 없이 독립적으로 유의미한 작업을 처리하지 않는다. `headless`라는 말은 UI가 없다는 뜻이지, 서버가 필요 없다는 뜻이 아니다.

## 2. 워커 실행 유형

모든 워커는 같은 worker contract를 사용한다. 차이는 실행 위치와 실행 주체뿐이다.

| 유형 | 실행 위치 | 실행 주체 | 서버에서 보여야 하는 origin |
| --- | --- | --- | --- |
| Server-local worker | 중앙 서버 머신 | 서버 관리자 또는 서버 프로세스 | `server-local` |
| Client-launched worker | 클라이언트 앱이 설치된 사용자 PC | 클라이언트 앱 사용자 | `client-launched` |
| Headless worker | UI 없는 별도 머신, 고성능 PC, 클라우드 | CLI, service, boot script | `headless` |

이 셋은 별도 워커 시스템이 아니다. 같은 `WorkerDaemon` 또는 같은 worker core를 서로 다른 runner로 실행하는 방식이어야 한다.

## 3. 권한과 제어 범위

### 클라이언트 사용자가 할 수 있는 것

- 자기 PC의 client-launched worker 시작
- 자기 PC의 client-launched worker 중지
- 자기 worker 상태 확인
- 자기 worker의 batch size, schedule, 자원 제한 설정 요청
- 자기 worker 세션에 대한 stop 요청

클라이언트는 로컬 프로세스를 직접 제어할 수 있다. 하지만 서버에 등록된 전체 worker pool의 최종 제어자는 아니다.

### 중앙 서버 관리자가 할 수 있는 것

- 모든 worker session 조회
- 모든 worker에 stop 명령 전송
- 모든 worker에 block 명령 전송
- batch size override
- phase override
- stale heartbeat 감지
- 장애 worker의 task reclaim
- global processing mode 변경

### 워커가 직접 해야 하는 것

- 서버 연결 실패 시 재시도 또는 명확한 종료
- heartbeat 실패 누적 시 안전 종료
- 서버가 `stop`을 반환하면 현재 batch 경계에서 중지
- 서버가 `block`을 반환하면 즉시 작업을 멈추고 disconnect
- 종료 시 가능한 경우 `/workers/disconnect` 호출

## 4. 서버 API 계약

모든 워커 유형은 같은 서버 API를 사용해야 한다.

### Session

- `POST /api/v1/workers/connect`
- `POST /api/v1/workers/heartbeat`
- `POST /api/v1/workers/disconnect`

`connect` 요청은 최소한 다음 정보를 포함해야 한다.

```json
{
  "worker_name": "string",
  "hostname": "string",
  "batch_capacity": 5,
  "origin": "server-local | client-launched | headless",
  "launcher": "server | electron | cli | service | cloud",
  "resources": {
    "os": "Darwin | Windows | Linux",
    "arch": "string",
    "gpu_type": "cuda | mps | cpu",
    "gpu_name": "string",
    "gpu_memory_total_gb": 0
  }
}
```

현재 schema에 없는 `origin`과 `launcher`는 추가되어야 한다. 그전까지는 `resources_json` 안에 보존해도 되지만, 최종적으로는 서버가 일관되게 필터링할 수 있어야 한다.

### Task

- `POST /api/v1/tasks/claim`
- `POST /api/v1/tasks/start`
- `POST /api/v1/tasks/complete`

워커는 phase를 임의로 소유하지 않는다. 서버 scheduler가 queue 상태와 worker capability를 보고 phase와 count를 결정한다.

### Result Upload

- `PATCH /api/v1/files/{file_id}/vision`
- `PATCH /api/v1/files/{file_id}/vv`
- `PATCH /api/v1/files/{file_id}/mv`
- `GET /api/v1/files/{file_id}/thumbnail`
- `GET /api/v1/files/{file_id}/mc`

서버는 task assignment를 검증한 뒤에만 결과 업로드를 받아야 한다.

## 5. Worker Core와 Runner 분리

### Worker Core

Worker Core는 OS와 launcher에 무관해야 한다.

- 인증된 HTTP session 또는 LocalTransport를 받는다.
- worker session 등록과 heartbeat를 수행한다.
- task claim/start/complete를 수행한다.
- MC/VV/MV 처리를 수행한다.
- 결과를 업로드한다.

### Runner

Runner는 Worker Core를 어떻게 시작할지 담당한다.

| Runner | 용도 |
| --- | --- |
| Embedded server runner | 서버 프로세스 안에서 server-local worker 실행 |
| Electron IPC runner | 클라이언트 앱에서 client-launched worker 실행 |
| Headless CLI runner | 터미널, service, cloud에서 headless worker 실행 |

Runner는 다를 수 있지만, 서버에 등록된 뒤에는 모두 동일한 worker session으로 관리되어야 한다.

## 6. Platform Contract

워커는 macOS, Windows, Linux에서 공통으로 실행 가능해야 한다.

| OS | 우선 backend | 비고 |
| --- | --- | --- |
| macOS | MLX 또는 MPS | Electron 내장 워커와 headless CLI 모두 가능해야 한다. |
| Windows | CUDA, Ollama, CPU fallback | stdin/IPC와 service 실행을 분리 검증해야 한다. |
| Linux | CUDA, vLLM, CPU fallback | cloud worker의 주 검증 대상이다. |

Linux cloud worker는 별도 제품이 아니라, 공통 headless runner의 Linux/CUDA profile이다.

## 7. 정상 동작 판정 기준

다음 항목이 모두 가능해야 worker runtime contract를 만족한다.

- 서버 머신에서 server-local worker를 시작/중지할 수 있다.
- 클라이언트 앱에서 자기 PC의 client-launched worker를 시작/중지할 수 있다.
- Electron 없이 headless worker를 시작할 수 있다.
- 세 유형 모두 중앙 서버의 worker list에 보인다.
- 세 유형 모두 `origin`, hostname, OS, GPU, VRAM, current phase, current file, heartbeat time을 보고한다.
- 서버 관리자는 세 유형 모두에게 stop/block 명령을 보낼 수 있다.
- 일반 클라이언트는 자기 worker만 stop할 수 있다.
- worker가 중지되면 session이 offline으로 전환된다.
- worker가 죽으면 heartbeat timeout 이후 서버가 작업을 reclaim한다.
- worker는 서버의 task assignment 없이 파일 결과를 업로드할 수 없다.

## 8. 현재 구현에서 필요한 보강

이 계약을 완성하려면 최소한 다음이 필요하다.

1. Headless CLI runner 추가
   - 예: `python -m backend.worker.cli`
   - 또는 packaging entrypoint: `imagine-worker`

2. Headless 인증 방식 명확화
   - Electron token injection과 별개로 CLI/service/cloud에서 쓸 인증 입력 필요
   - 예: access/refresh token, worker token, 또는 server-issued worker credential

3. Worker origin 저장
   - `server-local`, `client-launched`, `headless` 구분
   - launcher도 `server`, `electron`, `cli`, `service`, `cloud`로 구분

4. 공통 smoke test 추가
   - server health
   - auth
   - worker connect
   - heartbeat
   - claim
   - disconnect

5. 운영 문서 추가
   - macOS headless
   - Windows headless
   - Linux CUDA/cloud
   - Electron-launched worker

## 9. Headless Worker 설치/연결 계약

Headless worker는 서버가 원격 머신에 직접 접속해서 설치하지 않는다. 서버는 다음 두 가지를 제공하고, 워커 머신에서 사용자가 명령을 실행한다.

- Linux bootstrap script
  - `GET /api/v1/workers/bootstrap/linux.sh`
  - public endpoint
  - credential은 script URL에 포함하지 않고 환경변수로 전달한다.

- Headless worker command 발급
  - `POST /api/v1/admin/workers/headless-command`
  - admin 전용
  - 전용 worker user를 만들거나 재활성화한다.
  - access token, refresh token, bootstrap URL, 실행 명령을 반환한다.

예시:

```bash
IMAGINE_SERVER_URL='https://imagine.example.com' \
IMAGINE_WORKER_ACCESS_TOKEN='...' \
IMAGINE_WORKER_REFRESH_TOKEN='...' \
IMAGINE_WORKER_NAME='cloud-a100-1' \
IMAGINE_WORKER_LAUNCHER='cloud' \
bash -c "$(curl -fsSL 'https://imagine.example.com/api/v1/workers/bootstrap/linux.sh')"
```

이 구조에서 같은 worker package는 실행 시점의 server URL과 credential에 따라 어느 중앙 서버에 붙을지 결정한다.
