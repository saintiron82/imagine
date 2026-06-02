#!/usr/bin/env bash
# Upload a signed Imagine release to R2.
#
# Usage:
#   PRIVKEY=~/.imagine/release-key.pem \
#   R2_PROFILE=imagine-r2 \
#   R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
#   operations/cloudflare/r2/upload-release.sh \
#     --channel stable --version 1.0.0 \
#     --darwin-arm64 dist/imagine-darwin-arm64.dmg \
#     --win32-x64 dist/imagine-win32-x64.exe \
#     --linux-x64 dist/imagine-linux-x64.AppImage
#
# Requires: aws (configured with R2 S3-compatible endpoint), openssl (sha256 + xxd),
# python3 with `cryptography` package (Ed25519 signing).

set -euo pipefail

CHANNEL=""
VERSION=""
declare -A ARTIFACTS

while [[ $# -gt 0 ]]; do
  case "$1" in
    --channel)       CHANNEL="$2"; shift 2 ;;
    --version)       VERSION="$2"; shift 2 ;;
    --darwin-arm64)  ARTIFACTS[darwin-arm64]="$2"; shift 2 ;;
    --darwin-x64)    ARTIFACTS[darwin-x64]="$2"; shift 2 ;;
    --win32-x64)     ARTIFACTS[win32-x64]="$2"; shift 2 ;;
    --linux-x64)     ARTIFACTS[linux-x64]="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

: "${CHANNEL:?--channel required}"
: "${VERSION:?--version required}"
: "${PRIVKEY:?PRIVKEY env required}"
: "${R2_PROFILE:?R2_PROFILE env required (AWS CLI profile pointed at R2)}"
: "${R2_ENDPOINT:?R2_ENDPOINT env required}"

BUCKET="imagine-releases"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# 1. hash + upload each artifact, build artifacts dict
artifacts_json='{}'
for platform in "${!ARTIFACTS[@]}"; do
  src="${ARTIFACTS[$platform]}"
  [[ -f "$src" ]] || { echo "missing $src" >&2; exit 1; }
  name=$(basename "$src")
  key="artifacts/${VERSION}/${name}"

  sha=$(openssl dgst -sha256 -binary "$src" | xxd -p -c 256)
  size=$(stat -f %z "$src" 2>/dev/null || stat -c %s "$src")
  url="https://releases.imagine.zeroechodaily.com/${key}"

  aws --endpoint-url "$R2_ENDPOINT" --profile "$R2_PROFILE" \
      s3 cp "$src" "s3://${BUCKET}/${key}"

  artifacts_json=$(python3 -c '
import json, sys
d = json.loads(sys.argv[1])
d[sys.argv[2]] = {"url": sys.argv[3], "sha256": sys.argv[4], "size_bytes": int(sys.argv[5])}
print(json.dumps(d))
' "$artifacts_json" "$platform" "$url" "$sha" "$size")
done

# 2. build canonical body (signature field omitted)
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
body=$(python3 -c '
import json, sys
artifacts = json.loads(sys.argv[5])
body = {
  "manifest_version": 1,
  "channel": sys.argv[1],
  "version": sys.argv[2],
  "released_at": sys.argv[3],
  "min_supported_version": sys.argv[4],
  "rollback_policy": "auto_on_health_check_failure",
  "artifacts": artifacts,
  "migration_plan": {"db_schema_target": 0, "scripts": []},
}
print(json.dumps(body, sort_keys=True, separators=(",", ":")))
' "$CHANNEL" "$VERSION" "$NOW" "0.0.0" "$artifacts_json")

# 3. sign canonical body
sig=$(python3 -c '
import base64, sys
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

with open(sys.argv[1], "rb") as f:
    priv = load_pem_private_key(f.read(), password=None)
assert isinstance(priv, Ed25519PrivateKey)
print(base64.b64encode(priv.sign(sys.argv[2].encode("utf-8"))).decode())
' "$PRIVKEY" "$body")

# 4. assemble signed manifest
signed=$(python3 -c '
import json, sys
body = json.loads(sys.argv[1])
body["signature"] = {"alg": "ed25519", "key_id": sys.argv[2], "value": sys.argv[3]}
print(json.dumps(body, sort_keys=True, indent=2))
' "$body" "imagine-release-key-2026" "$sig")

# 5. upload manifest as <version>.json and overwrite latest.json
echo "$signed" > "$TMP/${VERSION}.json"
echo "$signed" > "$TMP/latest.json"

for name in "${VERSION}.json" "latest.json"; do
  aws --endpoint-url "$R2_ENDPOINT" --profile "$R2_PROFILE" \
      s3 cp --content-type application/json \
      "$TMP/${name}" "s3://${BUCKET}/manifests/${CHANNEL}/${name}"
done

echo "uploaded: ${CHANNEL} ${VERSION} (${#ARTIFACTS[@]} artifacts, signed manifest)"
