#!/usr/bin/env bash
# Imagine relay — teardown
#
# Deletes the CloudFormation stack and removes the secret parameters
# created under /imagine/relay/* in SSM Parameter Store.
#
# This does NOT delete the Budgets alarm — by design, so cost ceilings
# survive an accidental stack drop. Remove the budget separately if
# you really want it gone (see infra/aws-relay/budget.md).
#
# Environment overrides:
#   STACK_NAME    default: imagine-relay
#   REGION        default: us-west-2

set -euo pipefail

STACK_NAME="${STACK_NAME:-imagine-relay}"
REGION="${REGION:-us-west-2}"

command -v aws >/dev/null 2>&1 || {
    echo "[teardown] error: aws CLI not found" >&2
    exit 1
}

echo "[teardown] this will delete CloudFormation stack '$STACK_NAME' in $REGION"
echo "[teardown] and remove all /imagine/relay/* SSM parameters."
read -r -p "Type 'teardown' to confirm: " confirm
if [ "$confirm" != "teardown" ]; then
    echo "[teardown] aborted."
    exit 1
fi

echo "[teardown] listing SSM parameters under /imagine/relay/..."
PARAMS=$(aws ssm describe-parameters \
    --region "$REGION" \
    --parameter-filters "Key=Name,Option=BeginsWith,Values=/imagine/relay/" \
    --query "Parameters[].Name" --output text || true)

if [ -n "$PARAMS" ]; then
    echo "[teardown] deleting parameters: $PARAMS"
    # delete-parameters accepts up to 10 at a time
    echo "$PARAMS" | tr '\t' '\n' | xargs -n 10 aws ssm delete-parameters --region "$REGION" --names || true
fi

echo "[teardown] deleting CloudFormation stack '$STACK_NAME'"
aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"

echo "[teardown] waiting for stack deletion (this can take ~3 minutes)..."
aws cloudformation wait stack-delete-complete \
    --stack-name "$STACK_NAME" --region "$REGION"

echo "[teardown] done."
