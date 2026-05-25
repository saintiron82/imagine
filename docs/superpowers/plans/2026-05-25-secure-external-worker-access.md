# Secure External Worker Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자가 각자 개설한 Imagine 서버에 외부 클라이언트와 클라우드 GPU 워커가 안전하게 붙을 수 있는 on-demand 외부 접속 구조를 만든다.

**Architecture:** 기존 direct 접속은 localhost/LAN/고급 사용자용으로 유지한다. 제품 기본 외부 접속은 사용자 PC 포트를 인터넷에 직접 열지 않고, 서버와 클라/워커가 중앙 relay의 443 endpoint로 outbound 연결하는 세션형 control relay로 만든다. AWS는 control plane만 담당하고, 원본 이미지/대용량 결과는 relay를 통과시키지 않는다.

**Tech Stack:** FastAPI, SQLite, Electron, React, Firebase Firestore/Auth, Python worker runtime, AWS API Gateway WebSocket, Lambda, DynamoDB, AWS Budgets/CloudWatch.

---

## 핵심 결론

- 현재 구조는 `Firebase discovery + 직접 HTTP 접속 + 서버 인증`이다.
- 외부망 자동 접속을 보장하는 tunnel/relay/P2P 계층은 아직 없다.
- 사용자의 로컬 서버 포트 `8000`을 인터넷에 직접 여는 방식은 기본 경로에서 제외한다.
- 기본 외부 접속은 `on-demand secure relay session`으로 설계한다.
- AWS 무료 티어는 "완전 무료 보장"이 아니라 "무료 한도 친화형"으로만 본다.
- 상시 연결은 목표가 아니므로 작업할 때만 relay 세션을 열고, idle이면 닫는다.
- AWS relay는 control 메시지만 다루고, 이미지/DB/대용량 결과를 중계하지 않는다.

## 연결 모드 정의

| Mode | 용도 | 기본 노출 |
|---|---|---|
| `direct_local` | 같은 PC | `localhost` |
| `direct_lan` | 같은 LAN | `192.168.x.x` |
| `manual_external` | 고급 사용자 포트포워딩/터널 | 사용자 책임 |
| `relay_session` | 권장 외부 접속/클라우드 워커 | AWS relay 443 |
| `cloud_server` | 서버 자체를 클라우드에 설치 | 별도 배포 옵션 |

## 현재 코드 기준 확인 지점

- Firebase lookup은 `public_ip`, `lan_ip`, `port`로 URL 후보를 만든다: `frontend/src/api/firebase.js`
- 클라이언트는 URL 후보에 `/api/v1/health`를 직접 호출한다: `frontend/src/contexts/AuthContext.jsx`
- 서버는 Firestore에 `group_name`, `lan_ip`, `public_ip`, `port`, `updated_at`을 등록한다: `backend/server/firebase_registry.py`
- headless worker는 `IMAGINE_SERVER_URL`로 직접 서버 API에 붙는다: `backend/worker/cli.py`, `backend/worker/transport.py`
- tunnel/relay/WebRTC/STUN/TURN 구현은 현재 없다.

## Phase 0: Worker Runtime Foundation

**목표:** 워커를 Electron 부속 기능이 아니라 공유 worker core를 쓰는 세 가지 실행 유형으로 명확히 분리한다.

**Status:** 현재 작업 트리에 대부분 구현되어 있음. 실행 전 재검증 필요.

**Files:**
- Modify: `backend/db/sqlite_schema_auth.sql`
- Modify: `backend/db/sqlite_migrations.py`
- Modify: `backend/server/routers/workers.py`
- Modify: `backend/server/embedded_worker.py`
- Modify: `backend/worker/transport.py`
- Modify: `backend/worker/worker_daemon.py`
- Modify: `backend/worker/worker_ipc.py`
- Create: `backend/worker/cli.py`
- Modify: `scripts/cloud_worker_boot.sh`
- Test: `tests/test_worker_runtime_origin.py`
- Test: `tests/test_worker_headless_cli.py`
- Test: `tests/test_worker_bootstrap.py`
- Create: `tools/check_worker_runtime_contract.py`

- [ ] **Step 1: 현재 워커 계약 재검증**

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_bootstrap.py tests/test_worker_runtime_origin.py tests/test_worker_headless_cli.py -q
.venv/bin/python tools/check_worker_runtime_contract.py
.venv/bin/python -m py_compile backend/server/routers/workers.py backend/worker/transport.py backend/server/embedded_worker.py backend/worker/worker_daemon.py backend/worker/worker_ipc.py backend/worker/cli.py backend/db/sqlite_migrations.py backend/server/config.py
bash -n scripts/cloud_worker_boot.sh scripts/run-server.sh scripts/run-full.sh
```

Expected:

```text
8 passed
worker runtime contract static check: OK
no shell syntax errors
```

- [ ] **Step 2: 문서와 실제 코드 차이 확인**

Run:

```bash
git diff -- docs/worker_runtime_contract_ko.md docs/worker_runtime_audit_2026-05-25.md backend/worker/cli.py backend/server/routers/workers.py
```

Expected:

```text
worker origin values: server-local, client-launched, headless
worker launcher values: server, electron, cli, service, cloud
headless command endpoint present
linux bootstrap endpoint present
```

## Phase 1: Connection Mode Contract

**목표:** direct, manual tunnel, relay session을 서버/Firebase/UI가 같은 용어로 이해하게 만든다.

**Files:**
- Modify: `backend/server/firebase_registry.py`
- Modify: `frontend/src/api/firebase.js`
- Modify: `docs/worker_runtime_contract_ko.md`
- Create: `docs/external_access_security_contract_ko.md`
- Test: `tests/test_firebase_registry_connection_modes.py`

- [ ] **Step 1: Firebase registry 필드 확장 테스트 작성**

Create `tests/test_firebase_registry_connection_modes.py` with assertions for:

```python
def test_registry_payload_contains_connection_modes(monkeypatch):
    from backend.server import firebase_registry

    captured = {}

    def fake_urlopen(req, timeout):
        captured["body"] = req.data.decode("utf-8")

        class Resp:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False

        return Resp()

    monkeypatch.setattr(firebase_registry.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(firebase_registry, "_get_lan_ip", lambda: "192.168.0.8")
    monkeypatch.setattr(firebase_registry, "_get_public_ip", lambda: "")

    assert firebase_registry.register_group("studio", 8000) is True
    body = captured["body"]
    assert '"connect_mode"' in body
    assert '"lan_url"' in body
    assert '"external_url"' in body
    assert '"relay_endpoint"' in body
    assert '"relay_online"' in body
```

Run:

```bash
.venv/bin/python -m pytest tests/test_firebase_registry_connection_modes.py -q
```

Expected:

```text
FAIL because new fields are not registered yet
```

- [ ] **Step 2: Backend registry 필드 추가**

Modify `backend/server/firebase_registry.py` so Firestore fields include:

```text
connect_mode: direct_lan
lan_url: http://<lan_ip>:<port>
external_url: empty string
tunnel_url: empty string
relay_endpoint: empty string
relay_online: false
reachable_status: unknown
updated_at: ISO timestamp
```

Run:

```bash
.venv/bin/python -m pytest tests/test_firebase_registry_connection_modes.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 3: Frontend Firebase lookup 확장**

Modify `frontend/src/api/firebase.js` to read:

```text
connect_mode
lan_url
external_url
tunnel_url
relay_endpoint
relay_online
reachable_status
```

URL priority:

```text
external_url
tunnel_url
public_ip:port
lan_url
lan_ip:port
localhost:port
```

Run:

```bash
cd frontend && npm run build
```

Expected:

```text
build succeeds
```

## Phase 2: Server Connection Info API

**목표:** Admin UI와 worker command 생성기가 같은 서버 접속 후보 목록을 사용하게 만든다.

**Files:**
- Create: `backend/server/connection_info.py`
- Modify: `backend/server/app.py` or create router `backend/server/routers/connection_info.py`
- Modify: `backend/server/routers/workers.py`
- Test: `tests/test_server_connection_info.py`

- [ ] **Step 1: connection-info 테스트 작성**

Create `tests/test_server_connection_info.py` with cases:

```text
GET /api/v1/server/connection-info returns request_origin
returns localhost_url
returns lan_url candidates
marks direct public as not verified
does not return tokens
```

Run:

```bash
.venv/bin/python -m pytest tests/test_server_connection_info.py -q
```

Expected:

```text
FAIL because endpoint does not exist
```

- [ ] **Step 2: connection-info endpoint 구현**

Response shape:

```json
{
  "server_id": "local-or-generated-server-id",
  "group_name": "group",
  "request_origin": "http://localhost:8000",
  "connect_modes": [
    {
      "mode": "direct_local",
      "url": "http://localhost:8000",
      "reachable_scope": "same-machine",
      "security": "safe-local"
    },
    {
      "mode": "direct_lan",
      "url": "http://192.168.0.8:8000",
      "reachable_scope": "same-lan",
      "security": "lan-only"
    },
    {
      "mode": "relay_session",
      "url": "",
      "reachable_scope": "external",
      "security": "recommended-external",
      "available": false
    }
  ]
}
```

Run:

```bash
.venv/bin/python -m pytest tests/test_server_connection_info.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 3: headless command가 선택 URL을 받게 변경**

Modify `HeadlessWorkerCommandRequest` in `backend/server/routers/workers.py`:

```text
server_url: optional string
connect_mode: direct_lan | manual_external | relay_session
```

Rules:

```text
direct_lan/manual_external: linux_command uses IMAGINE_SERVER_URL
relay_session: linux_command uses IMAGINE_RELAY_ENDPOINT and IMAGINE_SERVER_ID
```

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_bootstrap.py -q
```

Expected:

```text
PASS
```

## Phase 3: Admin UI For Connection And Worker Install

**목표:** 관리자가 현재 서버 주소 후보, 위험도, 외부 접속 방식, Linux worker 설치 명령을 한 화면에서 볼 수 있게 만든다.

**Files:**
- Modify: `frontend/src/api/admin.js`
- Modify: `frontend/src/components/admin/WorkersPanel.jsx`
- Create: `frontend/src/components/admin/ConnectionInfoPanel.jsx`
- Modify: `frontend/src/i18n/locales/ko-KR.json`
- Modify: `frontend/src/i18n/locales/en-US.json`

- [ ] **Step 1: API client 추가**

Add functions:

```text
getConnectionInfo()
createHeadlessWorkerCommand({ worker_name, launcher, expires_minutes, server_url, connect_mode })
```

- [ ] **Step 2: UI 상태 표시**

Panel sections:

```text
현재 서버 주소
LAN 주소
외부 주소 상태
권장 relay 상태
Linux cloud worker command
보안 경고
```

Security labels:

```text
안전: localhost
보통: 같은 LAN
주의: 수동 외부 URL
권장: relay session
위험: 공인 IP 포트 직접 공개
```

- [ ] **Step 3: build 검증**

Run:

```bash
cd frontend && npm run build
```

Expected:

```text
build succeeds
```

## Phase 4: AWS On-Demand Control Relay MVP

**목표:** 서버/클라/워커가 작업할 때만 AWS relay에 붙고, control 메시지만 교환하게 만든다.

**Files:**
- Create: `infra/aws-relay/template.yaml`
- Create: `infra/aws-relay/src/connect.py`
- Create: `infra/aws-relay/src/disconnect.py`
- Create: `infra/aws-relay/src/message.py`
- Create: `infra/aws-relay/src/auth.py`
- Create: `infra/aws-relay/README.md`
- Create: `backend/server/relay_client.py`
- Create: `backend/worker/relay_transport.py`
- Test: `tests/test_relay_protocol_contract.py`

- [ ] **Step 1: Relay protocol 문서화**

Create `docs/relay_protocol_contract_ko.md` with message types:

```text
server.register
server.heartbeat
client.attach
worker.attach
worker.claim
worker.complete
session.close
error
```

Message hard limits:

```text
max message size: 16KB for MVP
heartbeat interval: 60 seconds
idle timeout: 10 minutes
session max duration: 2 hours, reconnect required
payload file bytes: forbidden
```

- [ ] **Step 2: AWS SAM/CloudFormation skeleton**

Resources:

```text
ApiGateway WebSocket API
Lambda Connect
Lambda Disconnect
Lambda Message
DynamoDB RelayConnections table
DynamoDB RelaySessions table
DynamoDB TTL enabled
CloudWatch retention 3 days
```

- [ ] **Step 3: Relay auth rules**

Rules:

```text
server.register requires server_id + server_secret
client.attach requires Firebase/server session proof or server-issued attach token
worker.attach requires worker enrollment token
relay never receives server password
relay never receives admin refresh token
```

- [ ] **Step 4: Local contract tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_relay_protocol_contract.py -q
```

Expected:

```text
PASS
```

## Phase 5: Server Relay Connector

**목표:** 사용자 Imagine 서버가 자기 포트를 인터넷에 열지 않고 relay에 outbound 연결을 만들게 한다.

**Files:**
- Create: `backend/server/relay_client.py`
- Modify: `backend/server/app.py`
- Modify: `backend/server/config.py`
- Modify: `backend/server/firebase_registry.py`
- Test: `tests/test_server_relay_client.py`

- [ ] **Step 1: relay client 설정 추가**

Config keys:

```text
relay.enabled
relay.endpoint
relay.server_id
relay.server_secret
relay.auto_connect
relay.heartbeat_interval_seconds
relay.idle_timeout_seconds
```

- [ ] **Step 2: relay online 상태를 Firebase에 반영**

When connected:

```text
connect_mode: relay_session
relay_endpoint: configured endpoint
relay_online: true
updated_at: now
```

When disconnected:

```text
relay_online: false
updated_at: now
```

- [ ] **Step 3: 서버 시작/활성화 흐름 연결**

Rules:

```text
server starts relay connector only after admin activation or explicit relay enable
relay connector reconnects with backoff
relay connector stops on shutdown
```

Run:

```bash
.venv/bin/python -m pytest tests/test_server_relay_client.py -q
```

Expected:

```text
PASS
```

## Phase 6: Worker Relay Mode

**목표:** cloud/headless worker가 direct `IMAGINE_SERVER_URL` 없이 relay session으로 서버에 붙을 수 있게 한다.

**Files:**
- Modify: `backend/worker/cli.py`
- Create: `backend/worker/relay_transport.py`
- Modify: `backend/worker/worker_daemon.py`
- Modify: `scripts/cloud_worker_boot.sh`
- Test: `tests/test_worker_relay_mode.py`

- [ ] **Step 1: CLI config 확장**

Add env/args:

```text
IMAGINE_CONNECT_MODE=direct|relay
IMAGINE_RELAY_ENDPOINT
IMAGINE_SERVER_ID
IMAGINE_WORKER_ENROLLMENT_TOKEN
```

Rules:

```text
direct mode requires IMAGINE_SERVER_URL
relay mode requires IMAGINE_RELAY_ENDPOINT and IMAGINE_SERVER_ID
worker admin token is never accepted
```

- [ ] **Step 2: RelayTransport 추가**

RelayTransport should implement the same worker transport contract:

```text
connect
heartbeat
claim
report_start
report_complete
get_thumbnail
get_mc_data
save_vision
save_vv
save_mv
disconnect
```

For MVP:

```text
only control and small JSON payloads are allowed
file bytes through relay return explicit unsupported error
```

- [ ] **Step 3: Worker relay tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_relay_mode.py -q
```

Expected:

```text
PASS
```

## Phase 7: Data Plane Policy

**목표:** AWS relay 비용과 개인정보 위험을 막기 위해 대용량 데이터 이동 방식을 명시한다.

**Files:**
- Create: `docs/data_plane_policy_ko.md`
- Modify: `backend/server/routers/upload.py`
- Modify: `backend/server/routers/analysis.py`
- Test: `tests/test_worker_data_access_scope.py`

- [ ] **Step 1: 데이터 경로 정책 문서화**

Allowed through relay:

```text
task metadata
heartbeat
phase status
small JSON result
error summary
```

Forbidden through relay:

```text
original image bytes
bulk thumbnails
database files
model weights
large vectors in bulk
```

- [ ] **Step 2: 할당 task 기반 파일 접근 재검증**

Test cases:

```text
worker can download only assigned file
worker cannot download another user's file
worker cannot update task not assigned to itself
blocked worker cannot claim
```

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_data_access_scope.py -q
```

Expected:

```text
PASS
```

## Phase 8: Security Hardening

**목표:** 서버/클라/워커/relay/Firebase 권한을 분리하고, 공격면을 줄인다.

**Files:**
- Modify: `backend/server/auth/router.py`
- Modify: `backend/server/routers/workers.py`
- Modify: `backend/db/sqlite_schema_auth.sql`
- Modify: `backend/db/sqlite_migrations.py`
- Create: `backend/server/security/audit_log.py`
- Test: `tests/test_worker_token_scope.py`
- Test: `tests/test_audit_log_security_events.py`

- [ ] **Step 1: 토큰 역할 분리**

Roles/scopes:

```text
admin: server management
member: search/use server
worker: claim/complete assigned work
server: relay registration
enrollment: one-time worker registration
```

- [ ] **Step 2: worker enrollment token 추가**

Rules:

```text
short-lived: default 15 minutes
one-time or limited-use
revocable
created only by admin
never grants admin API access
```

- [ ] **Step 3: audit log 이벤트 추가**

Events:

```text
server_initialized
firebase_group_registered
external_url_changed
relay_connected
relay_disconnected
worker_enrollment_token_created
worker_connected
worker_blocked
worker_token_revoked
admin_login
failed_login
```

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_token_scope.py tests/test_audit_log_security_events.py -q
```

Expected:

```text
PASS
```

## Phase 9: AWS Cost Guardrails

**목표:** 무료 한도 친화형 구조를 유지하되, 0원 보장이 아님을 전제로 비용 폭주를 막는다.

**Files:**
- Create: `infra/aws-relay/budget.md`
- Modify: `infra/aws-relay/template.yaml`
- Create: `infra/aws-relay/scripts/deploy.sh`
- Create: `infra/aws-relay/scripts/teardown.sh`
- Create: `infra/aws-relay/scripts/usage_report.sh`

- [ ] **Step 1: 비용 정책 문서화**

Policy:

```text
AWS is control relay only
no original image relay
no database relay
no always-on target for MVP
monthly budget alarm at $1, $5, $10
CloudWatch retention 1 to 3 days
DynamoDB TTL enabled
message size limit enforced
```

- [ ] **Step 2: 배포 스크립트**

Commands:

```bash
bash infra/aws-relay/scripts/deploy.sh
bash infra/aws-relay/scripts/usage_report.sh
bash infra/aws-relay/scripts/teardown.sh
```

Expected:

```text
deploy prints WebSocket endpoint
usage_report prints connection/message counters
teardown removes relay resources
```

## Phase 10: Product UX And Warnings

**목표:** 사용자가 연결 방식과 위험도를 오해하지 않게 한다.

**Files:**
- Modify: `frontend/src/components/admin/ConnectionInfoPanel.jsx`
- Modify: `frontend/src/pages/LoginPageV2.jsx`
- Modify: `frontend/src/i18n/locales/ko-KR.json`
- Modify: `frontend/src/i18n/locales/en-US.json`
- Modify: `docs/quick_start_guide.md`

- [ ] **Step 1: 연결 방식 표시 문구**

Korean labels:

```text
같은 PC 연결
같은 네트워크 연결
외부 수동 연결
안전 Relay 세션
클라우드 서버 모드
```

Warnings:

```text
공인 IP 포트를 직접 공개하면 공격을 받을 수 있습니다.
외부 클라우드 워커는 Relay 세션을 권장합니다.
AWS 무료 한도는 0원 보장이 아닙니다.
원본 이미지 데이터는 Relay를 통과하지 않도록 설계되어 있습니다.
```

- [ ] **Step 2: build 검증**

Run:

```bash
cd frontend && npm run build
```

Expected:

```text
build succeeds
```

## Phase 11: End-To-End Validation

**목표:** 서울/부산처럼 다른 네트워크에 있다고 가정한 외부 worker flow를 재현한다.

**Files:**
- Create: `scripts/e2e_relay_smoke.sh`
- Create: `docs/relay_e2e_runbook_ko.md`

- [ ] **Step 1: Direct mode baseline**

Run:

```bash
bash scripts/run-server.sh 8000
curl -s http://localhost:8000/api/v1/health
curl -s http://<lan-ip>:8000/api/v1/health
```

Expected:

```text
localhost returns 200
LAN returns 200 only on same network
public IP direct access is not required and should not be advertised as safe
```

- [ ] **Step 2: Relay mode smoke**

Flow:

```text
server enables relay session
Firebase shows relay_online=true
admin creates worker enrollment command
headless worker connects with relay mode
worker appears as origin=headless launcher=cloud
worker heartbeat updates
worker can claim assigned tasks
worker cannot access unassigned files
idle session closes
```

Expected:

```text
cloud worker works without opening user PC public port
AWS relay sees only control messages
```

## Commit Strategy

Suggested commits:

```text
feat: persist worker runtime origin
feat: add headless worker runner
feat: document secure external access modes
feat: expose server connection info
feat: add admin connection panel
feat: add relay protocol contract
feat: add aws relay mvp
feat: add server relay connector
feat: add worker relay mode
feat: harden worker token scopes
feat: add relay cost guardrails
```

## Done Criteria

- Worker has three explicit origins: `server-local`, `client-launched`, `headless`.
- Admin UI shows connection modes and security level.
- Direct public port exposure is never the default recommendation.
- Firebase discovery supports `connect_mode` and relay metadata.
- Headless worker can be installed with a command generated by the server.
- Relay mode works without opening the user's local server port to the internet.
- AWS relay passes only control messages under size limits.
- Worker tokens are separate from admin/member tokens.
- Worker can access only assigned tasks and files.
- Audit log records external access and worker security events.
- Cost guardrails and teardown scripts exist before live relay testing.
