#!/usr/bin/env bash
# Imagine relay — usage report
#
# Prints recent activity counters so it's easy to spot abuse or runaway
# usage. Designed to be run weekly (and as the first thing you do if a
# Budgets alarm fires).
#
# Environment overrides:
#   STACK_NAME   default: imagine-relay
#   REGION       default: us-west-2
#   DAYS         default: 7   (window for CloudWatch metrics)

set -euo pipefail

STACK_NAME="${STACK_NAME:-imagine-relay}"
REGION="${REGION:-us-west-2}"
DAYS="${DAYS:-7}"

command -v aws >/dev/null 2>&1 || {
    echo "[usage] error: aws CLI not found" >&2
    exit 1
}

START="$(date -u -v-"${DAYS}d" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || \
         date -u -d "-${DAYS} days" '+%Y-%m-%dT%H:%M:%SZ')"
END="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[usage] window: $START → $END"

echo "[usage] resolving CloudFormation outputs..."
API_ID=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='RelayApiId'].OutputValue" \
    --output text)
CONNS_TABLE=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='ConnectionsTable'].OutputValue" \
    --output text)
SESSIONS_TABLE=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='SessionsTable'].OutputValue" \
    --output text)

echo "[usage] api_id=$API_ID"
echo "[usage] tables: $CONNS_TABLE, $SESSIONS_TABLE"

_metric() {
    local namespace="$1"
    local name="$2"
    local dim="$3"
    local stat="$4"
    aws cloudwatch get-metric-statistics \
        --region "$REGION" \
        --namespace "$namespace" \
        --metric-name "$name" \
        --dimensions "$dim" \
        --start-time "$START" --end-time "$END" \
        --period 86400 \
        --statistics "$stat" \
        --query 'Datapoints[].'"$stat" \
        --output text 2>/dev/null | tr '\t' '\n' | \
        awk 'BEGIN{s=0} {s+=$1} END{printf "%.0f\n", s}'
}

echo
echo "── ApiGateway WebSocket (last ${DAYS} days) ──"
echo "  MessageCount        : $(_metric AWS/ApiGateway MessageCount Name=ApiId,Value=$API_ID Sum)"
echo "  ConnectCount        : $(_metric AWS/ApiGateway ConnectCount Name=ApiId,Value=$API_ID Sum)"
echo "  IntegrationError    : $(_metric AWS/ApiGateway IntegrationError Name=ApiId,Value=$API_ID Sum)"

echo
echo "── Lambda invocations (last ${DAYS} days, by function name) ──"
for fn_logical in ConnectFunction DisconnectFunction MessageFunction; do
    fn_name=$(aws cloudformation describe-stack-resource \
        --stack-name "$STACK_NAME" --region "$REGION" \
        --logical-resource-id "$fn_logical" \
        --query "StackResourceDetail.PhysicalResourceId" \
        --output text 2>/dev/null || echo "")
    if [ -n "$fn_name" ]; then
        inv=$(_metric AWS/Lambda Invocations Name=FunctionName,Value=$fn_name Sum)
        err=$(_metric AWS/Lambda Errors Name=FunctionName,Value=$fn_name Sum)
        printf "  %-22s invocations=%s errors=%s\n" "$fn_logical" "$inv" "$err"
    fi
done

echo
echo "── DynamoDB live rows (right now) ──"
echo "  $CONNS_TABLE: $(aws dynamodb scan --region "$REGION" --table-name "$CONNS_TABLE" --select COUNT --query Count --output text)"
echo "  $SESSIONS_TABLE: $(aws dynamodb scan --region "$REGION" --table-name "$SESSIONS_TABLE" --select COUNT --query Count --output text)"

echo
echo "[usage] done."
