#!/usr/bin/env bash
# Imagine relay — deploy
#
# Builds the SAM application and runs sam deploy. Prints the
# WebSocket endpoint so it can be copied into the Imagine server's
# `relay.endpoint` config.
#
# Required CLIs: sam, aws.
#
# Environment overrides:
#   STACK_NAME    default: imagine-relay
#   REGION        default: us-west-2
#   STAGE_NAME    default: prod
#
# This script intentionally does NOT create the Budgets alarm — see
# infra/aws-relay/budget.md for the one-time manual budget setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

STACK_NAME="${STACK_NAME:-imagine-relay}"
REGION="${REGION:-us-west-2}"
STAGE_NAME="${STAGE_NAME:-prod}"

command -v sam >/dev/null 2>&1 || {
    echo "[deploy] error: sam CLI not found (brew install aws-sam-cli)" >&2
    exit 1
}
command -v aws >/dev/null 2>&1 || {
    echo "[deploy] error: aws CLI not found" >&2
    exit 1
}

cd "$PROJECT_DIR"

echo "[deploy] sam build (stack=$STACK_NAME region=$REGION)"
sam build

echo "[deploy] sam deploy"
sam deploy \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --parameter-overrides "StageName=$STAGE_NAME" \
    --capabilities CAPABILITY_IAM \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

echo "[deploy] stack outputs"
aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs' \
    --output table

ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='RelayEndpoint'].OutputValue" \
    --output text)

cat <<EOF

[deploy] done.

  RelayEndpoint = $ENDPOINT

next steps:
  1. aws ssm put-parameter --region $REGION \\
       --name "/imagine/relay/<server_id>/secret" \\
       --type SecureString --value "\$(openssl rand -hex 32)"
  2. Set IMAGINE_RELAY_ENDPOINT=$ENDPOINT on your Imagine server
  3. Set IMAGINE_RELAY_SERVER_SECRET=<same secret> on your Imagine server
  4. See infra/aws-relay/budget.md for the manual Budgets alarm.
EOF
