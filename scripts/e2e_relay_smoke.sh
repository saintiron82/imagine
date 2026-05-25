#!/usr/bin/env bash
# Imagine relay — end-to-end smoke
#
# Validates a *deployed* relay setup. Run on the workstation that has
# both the Imagine server and the AWS credentials configured.
#
# What it checks, top to bottom:
#   1. local server is reachable on http://localhost:<PORT>/api/v1/health
#   2. /api/v1/server/connection-info exposes the expected modes
#   3. relay endpoint env var or config is set
#   4. (if available) websocat is used to open the relay endpoint and
#      send a malformed envelope — relay must respond with BAD_ENVELOPE
#
# This script intentionally does not require Python deps; it's a
# pure-bash sanity check.
#
# Env:
#   IMAGINE_PORT          default: 8000
#   IMAGINE_RELAY_ENDPOINT  required for steps 3+
#   ADMIN_BEARER          required for step 2 (admin access token)

set -euo pipefail

PORT="${IMAGINE_PORT:-8000}"
BASE="http://localhost:${PORT}"
RELAY="${IMAGINE_RELAY_ENDPOINT:-}"
ADMIN="${ADMIN_BEARER:-}"

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }

step() { printf "\n── %s ──\n" "$*"; }

# 1. Local health
step "local server health"
if curl -fsS "${BASE}/api/v1/health" >/dev/null; then
    green "OK ${BASE}/api/v1/health"
else
    red "FAIL — local server not responding on ${BASE}"
    exit 1
fi

# 2. Connection info — needs admin token
step "connection-info"
if [ -z "$ADMIN" ]; then
    yellow "SKIP — set ADMIN_BEARER to verify /api/v1/server/connection-info"
else
    HTTP=$(curl -s -o /tmp/ci.json -w "%{http_code}" \
        -H "Authorization: Bearer ${ADMIN}" \
        "${BASE}/api/v1/server/connection-info")
    if [ "$HTTP" != "200" ]; then
        red "FAIL — connection-info returned HTTP $HTTP"
        cat /tmp/ci.json
        exit 1
    fi
    python3 - <<'PY'
import json, sys
data = json.load(open('/tmp/ci.json'))
modes = {m['mode']: m for m in data['connect_modes']}
assert 'direct_local' in modes, modes
assert modes['direct_local']['available'] is True
assert 'direct_lan' in modes
assert 'relay_session' in modes
print("connect modes:", ", ".join(sorted(modes)))
PY
    green "OK connect modes present"
fi

# 3. Relay endpoint configured
step "relay endpoint"
if [ -z "$RELAY" ]; then
    yellow "SKIP — IMAGINE_RELAY_ENDPOINT not set"
    exit 0
fi
green "endpoint: $RELAY"

# 4. websocat probe (optional)
step "websocat probe"
if ! command -v websocat >/dev/null 2>&1; then
    yellow "SKIP — websocat not installed (brew install websocat)"
    exit 0
fi

BAD='{"v":1,"type":"server.heartbeat","body":{}}'  # missing msg_id/server_id/ts
RESP=$(echo "$BAD" | websocat -n1 "$RELAY" || true)
if [ -z "$RESP" ]; then
    red "FAIL — relay closed without an error envelope"
    exit 1
fi
echo "relay response: $RESP"
echo "$RESP" | grep -q '"BAD_ENVELOPE"' || {
    red "FAIL — expected BAD_ENVELOPE in response"
    exit 1
}
green "OK relay rejected bad envelope as expected"
