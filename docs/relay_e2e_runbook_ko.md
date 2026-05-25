# Relay E2E Runbook

> **상태:** Phase 11. 외부 워커가 다른 네트워크(예: 클라우드 GPU
> 인스턴스)에 있고, 메인 서버가 사용자 PC에 있는 시나리오를
> 처음부터 끝까지 재현하기 위한 절차.

## 사전 준비

| 항목                                     | 비고 |
|------------------------------------------|------|
| AWS 계정, 1개 region 선택 (`us-west-2`)  | budget alarm 셋업 (`infra/aws-relay/budget.md`) |
| `aws-sam-cli`, `aws`, `jq`               | 로컬 워크스테이션 |
| `websocat` (선택)                         | 직접 WS 메시지 디버깅 |
| Imagine 서버가 실행 중 (`localhost:8000`) | admin 계정 활성 |
| Imagine 서버의 `server_id`               | `GET /api/v1/server/connection-info` 응답의 `server_id` 필드 |
| 외부 GPU 머신 (Linux)                     | NVIDIA + Python 3.12 + 인터넷 |

## 1단계: Direct mode baseline

먼저 relay 없이도 워커가 붙는지 확인한다. relay 문제와 워커 문제를
분리하기 위함이다.

```bash
# 로컬 서버
bash scripts/run-server.sh 8000
curl -s http://localhost:8000/api/v1/health         # 200
curl -s http://<LAN-IP>:8000/api/v1/health          # 같은 네트워크면 200
```

- `localhost` 200: 항상 가능해야 함.
- `LAN-IP` 200: 같은 네트워크에서만. 외부에서 안 됨이 정상.
- **공인 IP 포트 직접 노출은 권장하지 않는다.** 이 단계에서 공인 IP는
  안전 경로로 안내되지 않는다(Phase 3 UI의 manual_external 라벨 참조).

## 2단계: Relay 배포

```bash
cd infra/aws-relay
bash scripts/deploy.sh
# 출력의 RelayEndpoint(wss://...) 보관
```

서버별 시크릿을 SSM에 1회 등록:

```bash
SERVER_ID=$(curl -s -H "Authorization: Bearer $ADMIN" \
            http://localhost:8000/api/v1/server/connection-info | jq -r .server_id)
SECRET=$(openssl rand -hex 32)

aws ssm put-parameter --region us-west-2 \
    --name "/imagine/relay/${SERVER_ID}/secret" \
    --type SecureString --value "$SECRET"
```

## 3단계: 서버 측 relay connector 활성화

서버를 실행하는 셸에 환경 변수를 export하고 재시작한다.

```bash
export IMAGINE_RELAY_ENDPOINT='wss://<api_id>.execute-api.us-west-2.amazonaws.com/prod'
export IMAGINE_RELAY_SERVER_SECRET="$SECRET"
export IMAGINE_RELAY_ENABLED=1
bash scripts/run-server.sh 8000
```

로그에 다음이 보여야 한다:

```
Relay connector started (endpoint=wss://...)
Registered group 'studio' to Firestore (mode=relay_session ... relay=True)
```

Firebase 그룹 문서에서도 확인 가능:

```bash
curl -s "https://firestore.googleapis.com/v1/projects/imagine-b1e9c/databases/(default)/documents/groups/<group>" \
    | jq '.fields.connect_mode, .fields.relay_online'
```

→ `"relay_session"`, `true`

## 4단계: Headless worker 명령 발급

관리자 UI ("연결" 탭) 또는 직접 호출:

```bash
curl -s -X POST -H "Authorization: Bearer $ADMIN" \
    -H 'Content-Type: application/json' \
    http://localhost:8000/api/v1/admin/workers/headless-command \
    -d '{
        "worker_name": "cloud-a100-1",
        "launcher": "cloud",
        "expires_minutes": 120,
        "connect_mode": "relay_session"
    }' | jq .linux_command
```

응답의 `linux_command` 한 줄을 외부 GPU 머신에서 실행한다. 명령에는
`IMAGINE_CONNECT_MODE=relay`, `IMAGINE_RELAY_ENDPOINT`,
`IMAGINE_SERVER_ID`, `IMAGINE_WORKER_ENROLLMENT_TOKEN` 이 포함된다.

## 5단계: 외부에서 검증

외부 GPU 머신에 SSH 후 위 명령을 한 줄 실행. 정상이면:

```
[worker-bootstrap] server= launcher=cloud
...
[worker] relay attached (Phase 6 MVP — control plane only)
```

서버 측 관리자 UI의 "워커" 탭에 새 세션이 표시되어야 한다
(`origin=headless launcher=cloud`).

## 6단계: 안전성 단언

- **공인 IP 노출 없음.** 사용자 PC의 8000 포트는 LAN/localhost에만 열려
  있고 인터넷에서 직접 접근되지 않는다. `nmap` 등으로 외부에서 8000
  포트가 안 보이는지 확인.
- **AWS는 control 메시지만 본다.** CloudWatch Logs에서 `MessageFunction`
  로그를 확인하면 task 메타데이터/하트비트만 보이고 파일 byte는 없다.
- **idle 정리.** 워커를 30분 가만히 두면 relay가 `IDLE_TIMEOUT`을 보내
  세션을 닫는다.

## 7단계: 정리

작업이 끝나면:

```bash
# 외부 GPU 머신에서: Ctrl-C 또는 systemctl stop ...
# 로컬에서:
unset IMAGINE_RELAY_ENABLED
bash scripts/run-server.sh 8000        # relay 미사용 재시작

# 더 안 쓸 거면 stack 정리:
bash infra/aws-relay/scripts/teardown.sh
```

## 자동 스모크

`scripts/e2e_relay_smoke.sh`는 위 1~4 단계의 사전 조건을 자동 검증한다:

```bash
ADMIN_BEARER='<admin access token>' \
IMAGINE_RELAY_ENDPOINT='wss://...' \
bash scripts/e2e_relay_smoke.sh
```

체크 항목:

1. `localhost:8000/api/v1/health` 응답
2. `/api/v1/server/connection-info` 가 direct_local/direct_lan/
   relay_session 모드를 모두 노출
3. `IMAGINE_RELAY_ENDPOINT` 환경 변수 존재 여부
4. (`websocat` 설치돼 있으면) relay에 잘못된 envelope을 보내
   `BAD_ENVELOPE` 응답이 오는지

## 자주 보이는 실패

| 증상 | 원인 | 조치 |
|------|------|------|
| `relay_online=false` 가 유지됨 | `SECRET` 불일치 | SSM 값과 `IMAGINE_RELAY_SERVER_SECRET` 일치 확인 |
| Worker가 `AUTH_FAILED` 받음 | `enrollment_token` 누락 | headless-command를 새로 발급 |
| Worker 즉시 끊김 | idle_timeout (600s) | 서버 측 relay heartbeat 스레드 살아있는지 확인 (Phase 5 수정) |
| AWS Budgets 알람 | 비정상 트래픽 | `bash infra/aws-relay/scripts/usage_report.sh` 로 원인 추적 |
