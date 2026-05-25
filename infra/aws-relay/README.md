# Imagine Relay (AWS) — Phase 4 MVP

이 디렉터리는 **사용자 PC의 8000 포트를 인터넷에 직접 노출하지 않고도** 외부
클라이언트/워커가 Imagine 서버에 붙을 수 있게 하는 on-demand control relay의
AWS 구현이다. 상세 프로토콜은 [`docs/relay_protocol_contract_ko.md`](../../docs/relay_protocol_contract_ko.md)
를 참조한다.

## 무엇이 들어 있는가

```
infra/aws-relay/
├── template.yaml          AWS SAM 템플릿
├── README.md              이 문서
└── src/
    ├── auth.py            프로토콜 검증/인증/라우팅 (Lambda 공통)
    ├── connect.py         $connect 핸들러
    ├── disconnect.py      $disconnect 핸들러
    └── message.py         $default 핸들러
```

## 인프라 요약

| 자원                  | 역할 |
|-----------------------|------|
| API Gateway WebSocket | 외부 endpoint, 443 wss |
| Lambda (Connect)      | 새 connectionId 기록 |
| Lambda (Disconnect)   | connectionId/세션 정리 |
| Lambda (Message)      | 봉투 검증 → 인증 → 라우팅 |
| DynamoDB `RelayConnections` | connectionId 인덱스, TTL |
| DynamoDB `RelaySessions`    | server/client/worker 세션 매핑, TTL |
| CloudWatch Logs       | 보존 1~3일 (cost guardrail) |
| SSM Parameter Store   | `/imagine/relay/<server_id>/secret` |

## Relay가 절대로 하지 않는 것

- 원본 이미지 byte 전송 → `FORBIDDEN_TYPE`
- 모델 가중치 / DB 파일 전송 → `FORBIDDEN_TYPE`
- 서버 패스워드, 관리자 refresh token 수신 → `FORBIDDEN_TYPE`
- 16KB가 넘는 frame 전달 → `PAYLOAD_TOO_BIG`
- 등록되지 않은 server_id 라우팅 → `NO_SUCH_SERVER`

## 배포 (Phase 9에서 자동화 예정)

Phase 4 MVP에서는 **수동 배포만 지원**한다. Phase 9에서 deploy/teardown/
usage_report 스크립트가 추가된다.

수동 배포 절차:

```bash
cd infra/aws-relay

# 0. 사전 준비
aws configure  # us-west-2 등 1개 region 고정 권장

# 1. 빌드 + 배포
sam build
sam deploy --guided  # 첫 배포만 --guided, 이후는 sam deploy

# 2. server_secret 등록 (서버마다 1회)
aws ssm put-parameter \
    --name "/imagine/relay/imagine-xxxxxxxxxxxx/secret" \
    --type SecureString \
    --value "$(openssl rand -hex 32)"

# 3. 출력에서 RelayEndpoint(wss://...) 확보
sam list stack-outputs --stack-name <stack-name>
```

배포 후 Imagine 서버 측에서 `server.relay_endpoint` 설정에 위 wss URL을
저장하고, `server.relay_server_secret` 설정에 동일한 시크릿을 보관하면
Phase 5의 relay client가 자동으로 연결을 시도한다.

## 비용 가드레일 (Phase 9에서 강화)

- 모든 DynamoDB 테이블은 `PAY_PER_REQUEST` + TTL 사용
- CloudWatch Logs는 최대 3일 보존
- API Gateway에 throttling (25 rps / 50 burst)
- AWS Budgets 알람: 월 $1 / $5 / $10 임계치 (Phase 9에서 추가)
- "AWS 무료 한도는 0원 보장이 아니다" — 사용량 보고서를 주기적으로 점검할 것

## 로컬 단위 테스트

이 디렉터리는 별도 패키지로 배포되지만, 검증 로직(`auth.py`)은
프로젝트 루트의 `tests/test_relay_protocol_contract.py`에서 단위
테스트된다. 변경 시:

```bash
.venv/bin/python -m pytest tests/test_relay_protocol_contract.py -q
```

만약 `auth.py`의 상수(`RelayLimits`, `ALLOWED_TYPES`, `ERROR_CODES`)를
바꿀 경우 위 문서와 컨트랙트 테스트, Phase 5/6의 클라이언트도 함께
업데이트해야 한다.
