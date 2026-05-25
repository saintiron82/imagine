# Imagine Relay Protocol Contract

> **상태:** Phase 4 MVP (on-demand control relay). AWS API Gateway
> WebSocket + Lambda + DynamoDB 위에서 동작한다. 본 문서는 로컬
> 구현과 Lambda 핸들러가 공유하는 단일 출처(single source of truth)다.

## 설계 원칙

1. **Control plane only.** Relay는 task 메타데이터, 하트비트, 작은
   JSON 결과만 중계한다. 원본 이미지·DB 파일·모델 가중치·대용량 벡터
   bulk는 절대 통과시키지 않는다(자세한 데이터 평면 규칙은 Phase 7
   `data_plane_policy_ko.md` 참조).
2. **On-demand.** 항상 켜두는 대상이 아니다. 작업이 있을 때만 세션을
   열고 idle이면 닫는다.
3. **No secrets at rest in relay.** Relay는 서버 패스워드/관리자
   refresh token을 받거나 저장하지 않는다.
4. **Outbound from owner.** 사용자 PC의 8000 포트를 인터넷에 직접
   여는 것을 대체한다. 서버/클라/워커가 relay 443 endpoint로
   outbound WebSocket을 연다.

## 전체 흐름

```text
Server  ──register──>  Relay  <──attach──  Client
                          ▲
                          └────attach────  Worker
```

## 메시지 타입

| Type                | Direction         | 누가 보내나           | 목적 |
|---------------------|-------------------|------------------------|------|
| `server.register`   | Server → Relay    | 사용자 PC의 Imagine 서버 | 자기 server_id를 relay에 등록 |
| `server.heartbeat`  | Server → Relay    | 등록된 서버            | 세션 유지(60s) |
| `client.attach`     | Client → Relay    | 외부 클라이언트         | 특정 server_id에 붙기 |
| `worker.attach`     | Worker → Relay    | 외부 워커              | 특정 server_id에 워커로 붙기 |
| `worker.claim`      | Worker → Server   | 워커 (relay 경유)      | task 할당 요청 |
| `worker.complete`   | Worker → Server   | 워커 (relay 경유)      | task 완료/실패 보고 |
| `session.close`     | 양방향            | 누구나                 | 세션을 명시적으로 닫음 |
| `error`             | Relay → Endpoint  | Relay                  | 인증 실패/포맷 오류 등 |

## 메시지 봉투(envelope)

모든 메시지는 동일한 JSON 봉투에 담는다.

```json
{
  "v": 1,
  "type": "server.register",
  "msg_id": "uuid4-string",
  "server_id": "imagine-xxxxxxxxxxxx",
  "session_id": null,
  "ts": "2026-05-25T12:00:00Z",
  "body": { /* type별 페이로드 */ }
}
```

규칙:

- `v`: 정수, 현재 `1`. 호환 안 되는 변경 시 증가.
- `type`: 위 표의 문자열. 알 수 없는 type은 relay가 `error`로 반려.
- `msg_id`: 클라이언트가 만든 UUID. relay는 그대로 응답에 echo한다.
- `server_id`: relay가 라우팅에 사용하는 키. `client.attach` /
  `worker.attach` / `server.heartbeat`에서 필수.
- `session_id`: relay가 발급한 세션 식별자. 첫 메시지에서는 null이
  될 수 있다. `error` 응답에서도 그대로 전달된다.
- `ts`: ISO-8601 UTC. drift가 5분 이상이면 relay가 `error` 응답.
- `body`: type별 페이로드.

## 페이로드 정의(필수 필드만)

### `server.register`

```json
{ "server_secret": "...", "group_name": "studio" }
```

### `server.heartbeat`

```json
{ "online": true, "stats": { "workers_online": 1 } }
```

### `client.attach`

```json
{ "attach_token": "..." }
```

`attach_token`은 서버가 Firebase ID token 또는 server password 확인
후 발급한 짧은 수명(15분) 토큰이다. relay는 절대로 Firebase ID
token이나 server password 자체를 받지 않는다.

### `worker.attach`

```json
{ "enrollment_token": "..." }
```

Worker enrollment token은 관리자가 `/admin/workers/headless-command`
경로로 발급한 1회용/제한 사용 토큰이다(Phase 8에서 별도 scope 분리).

### `worker.claim`

```json
{ "phase": "mc", "max_batch": 4 }
```

### `worker.complete`

```json
{ "task_id": 123, "phase": "mc", "status": "done", "summary": { "ms": 4321 } }
```

### `session.close`

```json
{ "reason": "idle-timeout" }
```

### `error`

```json
{ "code": "AUTH_FAILED", "detail": "server_secret mismatch" }
```

표준 에러 코드:

| code             | 의미 |
|------------------|------|
| `BAD_ENVELOPE`   | 봉투 필드가 빠지거나 잘못됨 |
| `UNSUPPORTED_V`  | 메시지 버전이 relay가 모르는 값 |
| `UNKNOWN_TYPE`   | `type`이 정의되지 않은 값 |
| `AUTH_FAILED`    | 토큰/시크릿 검증 실패 |
| `PAYLOAD_TOO_BIG`| 16KB 초과 |
| `RATE_LIMITED`   | 속도 제한 |
| `NO_SUCH_SERVER` | 라우팅 대상 server_id 미등록 |
| `FORBIDDEN_TYPE` | 데이터 평면 정책 위반 (예: file bytes) |
| `IDLE_TIMEOUT`   | idle 한도 초과로 세션 종료 |
| `MAX_DURATION`   | 세션 최대 길이 초과 |

## 하드 리밋

| 항목                  | 값       | 비고 |
|-----------------------|----------|------|
| Max message size      | 16 KB    | API Gateway WebSocket frame 한도 직전 |
| Heartbeat interval    | 60 s     | 미수신 시 idle 카운팅 시작 |
| Idle timeout          | 600 s    | 마지막 메시지 기준 |
| Session max duration  | 2 h      | 이후 reconnect 필요 |
| Payload file bytes    | 금지     | base64 첨부 형태도 거부 |

위 한도는 `infra/aws-relay/src/auth.py`의 `RelayLimits` 상수와
`tests/test_relay_protocol_contract.py`의 단언과 동기화된다.

## Relay가 절대로 하지 않는 것

- 서버 패스워드 / 서버 패스워드 해시를 받는다 — 거절.
- 관리자 refresh token을 받는다 — 거절.
- 16KB가 넘는 frame을 전달한다 — `PAYLOAD_TOO_BIG`.
- 파일 byte를 라우팅한다 — `FORBIDDEN_TYPE`.
- 등록되지 않은 server_id로 라우팅을 시도한다 — `NO_SUCH_SERVER`.

## 인증 흐름 요약

1. **Server.** `server.register`는 `server_id` + `server_secret`을
   보낸다. `server_secret`은 Imagine 서버가 최초 활성화 시 생성해
   `system_meta` 와 환경변수로 보관한다.
2. **Client.** 외부 클라이언트는 Firebase ID token으로 사용자 본인
   서버 origin과 통신해서 attach token을 받고, 그 토큰만 relay에
   제시한다.
3. **Worker.** 워커는 관리자가 발급한 enrollment token으로 relay에
   붙는다. enrollment token은 admin API 권한이 없다(Phase 8).

상세 인증 규칙은 `infra/aws-relay/src/auth.py`에 정의되어 있다.
