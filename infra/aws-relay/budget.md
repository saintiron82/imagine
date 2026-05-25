# Imagine Relay — Cost Guardrails

> **상태:** Phase 9. AWS 무료 한도는 **0원 보장이 아니다.** 비용 폭주를
> 막기 위한 정책과 모니터링 셋업 절차를 정리한다.

## 정책 요약

| 항목                       | 값/규칙 |
|----------------------------|---------|
| 리소스 범위                | API Gateway WebSocket + Lambda + DynamoDB + CloudWatch Logs + SSM Parameter Store |
| 원본 이미지 relay 통과     | 금지 (`FORBIDDEN_BODY_KEYS`) |
| DB 파일 relay 통과         | 금지 |
| 메시지 크기 제한           | 16 KB |
| Lambda Memory              | 256 MB |
| Lambda Timeout             | 10 s |
| Architecture               | arm64 (Graviton — 비용 ↓) |
| CloudWatch Logs 보존       | 3일 |
| DynamoDB billing           | PAY_PER_REQUEST |
| DynamoDB TTL               | 모든 테이블에 `expires_at` |
| API Gateway throttling     | 25 rps / 50 burst |
| Always-on 워커 가정        | 없음 (MVP) |
| 월 Budgets 알람 임계치     | $1 / $5 / $10 (수동 설정) |

## Budgets 알람 (1회 수동 셋업)

`aws budgets` API는 stack과 분리해서 관리하는 것이 안전하다(스택을 지웠을
때 알람도 같이 사라지는 사고를 막기 위함).

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

cat > /tmp/imagine-relay-budget.json <<EOF
{
  "BudgetName": "imagine-relay-monthly",
  "BudgetType": "COST",
  "TimeUnit": "MONTHLY",
  "BudgetLimit": {"Amount": "10", "Unit": "USD"},
  "CostFilters": {
    "Service": ["AmazonApiGateway", "AWSLambda", "AmazonDynamoDB",
                "AmazonCloudWatch", "AmazonSSM"]
  }
}
EOF

cat > /tmp/imagine-relay-notifications.json <<EOF
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 10
    },
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "you@example.com"}]
  },
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 50
    },
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "you@example.com"}]
  },
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 100
    },
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "you@example.com"}]
  }
]
EOF

aws budgets create-budget \
  --account-id "$ACCOUNT_ID" \
  --budget file:///tmp/imagine-relay-budget.json \
  --notifications-with-subscribers file:///tmp/imagine-relay-notifications.json
```

알람은 한도의 10% / 50% / 100% 지점에서 발화된다. 즉, $1 / $5 / $10
임계치에 대응한다.

## 무엇이 비용을 만들 수 있나

| 항목                          | 가격 단위 (us-west-2 기준, 2026-05) |
|-------------------------------|-------------------------------------|
| API Gateway WebSocket 메시지  | 백만건당 약 $1, 첫 1M / month 무료 |
| API Gateway WebSocket 분-시간 | GB-hr, 매우 적음                   |
| Lambda 호출                   | 백만건당 $0.20, 첫 1M / month 무료 |
| Lambda GB-second              | 256MB × 100ms = 0.025 GB-sec       |
| DynamoDB on-demand            | 백만 WCU/RCU 단위 비용              |
| CloudWatch Logs ingest        | GB당 약 $0.50                       |
| SSM SecureString get          | 무료(표준 파라미터)                  |

가장 큰 위험은 (a) 무한 reconnect 루프, (b) 무한 heartbeat 폭주,
(c) 거대한 envelope 시도 → 16KB 차단으로 무력화. 이미 protocol에서
대응한다. throttling을 추가로 걸어둔 이유는 (a)/(b)에 대한 보험이다.

## 정기 점검 권장

| 주기   | 항목 |
|--------|------|
| 매주   | `bash infra/aws-relay/scripts/usage_report.sh` 실행 |
| 매월   | Budgets 알람이 안 울렸는지(예산 대비 % 확인) |
| 분기   | 미사용 워커 토큰 일괄 만료 (`refresh_tokens` 정리) |
| 새 워커 onboarding 시 | enrollment token이 단기여서 유출되어도 무력해지는지 확인 |
