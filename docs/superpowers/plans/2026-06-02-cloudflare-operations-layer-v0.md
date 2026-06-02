# Cloudflare Operations Layer v0 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the foundation of Imagine's BYO-Cloudflare operations layer — DNS plan, R2 release CDN, reference cloudflared tunnel, signed release manifest schema, and customer-setup wizard docs — without yet building any Workers code, license server, or auto-update client.

**Architecture:** All Cloudflare resources are scoped to **edge · CDN · routing only**. The Control-Plane backend (license/seat/token) hosting decision is deferred. Domain root: `imagine.zeroechodaily.com`. Release artifacts live in R2 behind a Cloudflare-managed custom subdomain. Customer Imagine nodes use BYO `cloudflared` for their own remote routing.

**Tech Stack:** Cloudflare DNS · Cloudflare R2 · cloudflared (BYO) · Ed25519 (release signing) · Bash/Python scripts · Markdown docs.

---

## Context

Sprint 1–3 closed the search-engine side at P@5 SLM-judge = 0.673. The next development phase is operations + commercial control, defined in `docs/imagine_operations_control_plane_2026-05-31.md`. The decision tracker (`docs/superpowers/plans/2026-05-31-control-plane-mvp-decisions.md`) has partial-locked Decision 2 to "Cloudflare edge·CDN·routing only" and a new sub-decision pins the domain at `imagine.zeroechodaily.com`.

This plan delivers **v0 of the operations layer** — the foundation that's safe to build regardless of the still-open Decisions 1 (MVP scope) and 2-backend (vendor choice). v0 outputs unblock both **P6 Auto-Update Agent** (which needs the release CDN) and **P8 BYO Cloudflare Wizard** (which needs the setup docs).

## Scope

**In v0:**
- Repository scaffold under `operations/cloudflare/`
- DNS plan + zone-config doc for `imagine.zeroechodaily.com`
- R2 release-CDN structure + upload script
- Signed release manifest schema (Ed25519) + key-management note
- Reference cloudflared tunnel config template + per-OS install instructions
- Customer BYO setup wizard (plain MD; not yet a UI wizard)
- E2E verification: download a test artifact from R2 via custom domain, and reach a local server through a cloudflared tunnel

**Out of v0 (separate plans):**
- Workers code (edge token validator) — defer to v1
- License/seat/token issuing server — separate plan
- Auto-update client integration — defer to P6 plan
- Customer-facing UI wizard — defer to P8 plan
- Control Plane backend hosting (Fly.io / Railway / etc.) — separate decision

## File Structure

```
operations/cloudflare/
├── README.md                              # purpose, layout, who runs what
├── dns/
│   └── imagine.zeroechodaily.com.md       # zone records + setup steps
├── r2/
│   ├── bucket-config.md                   # R2 bucket creation steps, public-read setup
│   ├── manifest-schema.md                 # human-readable spec of the manifest
│   ├── manifest-schema.json               # canonical JSON Schema (machine-readable)
│   └── upload-release.sh                  # release upload + manifest signing script
├── tunnel/
│   ├── cloudflared-template.yml           # reference tunnel config
│   └── install-tunnel.md                  # macOS / Linux / Windows install steps
└── wizard/
    └── byo-setup-guide.md                 # customer-facing BYO Cloudflare walkthrough
```

The user owns the actions inside Cloudflare (zone creation, R2 bucket creation, cloudflared install on the user's dev box). The dev session owns all the files above.

## Convention reused from existing code

- Ed25519 signing: stdlib `pynacl` or `cryptography` — no new dep beyond what `requirements*.txt` may already have. Verify before adding.
- Release manifest format follows the same pattern as `audit_log` entries in 11-phase Secure Worker Access (existing precedent for signed control-plane payloads). See `docs/superpowers/plans/2026-05-25-secure-external-worker-access.md`.

---

## Task 1: Operations scaffold + top-level README

**Files:**
- Create: `operations/cloudflare/README.md`
- Create: `operations/cloudflare/dns/.gitkeep` (empty)
- Create: `operations/cloudflare/r2/.gitkeep` (empty)
- Create: `operations/cloudflare/tunnel/.gitkeep` (empty)
- Create: `operations/cloudflare/wizard/.gitkeep` (empty)

- [ ] **Step 1: Write the README**

```markdown
# operations/cloudflare/

Cloudflare-side operations layer for Imagine. Scoped to **edge · CDN · routing only** per Decision 2.

## What lives here

- `dns/` — DNS plans and zone records for `imagine.zeroechodaily.com`
- `r2/` — Release-artifact CDN (bucket structure, manifest schema, upload script)
- `tunnel/` — Reference cloudflared tunnel config (used by us and shown to customers)
- `wizard/` — Customer BYO Cloudflare setup walkthroughs

## What does NOT live here

- License / seat / token issuing logic — separate Control-Plane backend
- Auto-update client (Electron updater integration) — see P6 plan
- Workers code — deferred to v1

## Who runs what

| Action | Run by |
|--------|--------|
| Add `imagine.zeroechodaily.com` to Cloudflare zone | Maintainer (owns the account) |
| Create R2 bucket `imagine-releases` | Maintainer |
| Install cloudflared on a host | Whoever owns that host (maintainer or customer) |
| Author / update config and docs in this dir | Anyone (PR-reviewed) |

## Reference

- Parent plan: `docs/superpowers/plans/2026-06-02-cloudflare-operations-layer-v0.md`
- Decision tracker: `docs/superpowers/plans/2026-05-31-control-plane-mvp-decisions.md`
- Architecture: `docs/imagine_operations_control_plane_2026-05-31.md`
```

- [ ] **Step 2: Commit**

```bash
git add operations/cloudflare/README.md operations/cloudflare/*/.gitkeep
git commit -m "ops: scaffold cloudflare operations layer (v0)"
```

---

## Task 2: DNS plan for `imagine.zeroechodaily.com`

**Files:**
- Create: `operations/cloudflare/dns/imagine.zeroechodaily.com.md`

- [ ] **Step 1: Write the zone record plan**

```markdown
# DNS — imagine.zeroechodaily.com

Domain root: `imagine.zeroechodaily.com` (subdomain of existing `zeroechodaily.com`).

## Record plan

| Name | Type | Target | Purpose |
|------|------|--------|---------|
| `imagine.zeroechodaily.com` | CNAME | apex landing / docs (TBD host) | Marketing landing + product docs |
| `releases.imagine.zeroechodaily.com` | CNAME | `imagine-releases.<account>.r2.cloudflarestorage.com` | R2 release CDN (custom domain) |
| `*.imagine.zeroechodaily.com` (later) | CNAME | tunnel routes | Customer-specific tunnels (managed-Cloudflare model only) |

## v0 actions (maintainer, in Cloudflare dashboard)

1. Add `zeroechodaily.com` (if not already) to the Imagine-owned Cloudflare account, or add `imagine.zeroechodaily.com` as a subzone if the apex stays with the parent account.
2. Create the `releases.imagine.zeroechodaily.com` CNAME pointing to the R2 bucket created in Task 3.
3. Issue a "Universal SSL" certificate for `*.imagine.zeroechodaily.com`.
4. Set the apex `imagine.zeroechodaily.com` to a temporary "Coming soon" page (Cloudflare Pages free, or a single static HTML file in R2).

## Out of scope for v0

- Customer-specific subdomains under `*.imagine.zeroechodaily.com` — these belong to P8 wizard.
- Email / MX records — not used by Imagine.

## Reference

- Cloudflare R2 custom domain: <https://developers.cloudflare.com/r2/buckets/public-buckets/#custom-domains-recommended>
```

- [ ] **Step 2: Commit**

```bash
git add operations/cloudflare/dns/imagine.zeroechodaily.com.md
git commit -m "ops: DNS plan for imagine.zeroechodaily.com (v0)"
```

---

## Task 3: R2 release-CDN bucket config

**Files:**
- Create: `operations/cloudflare/r2/bucket-config.md`

- [ ] **Step 1: Write the bucket setup doc**

```markdown
# R2 — imagine-releases bucket

## Purpose

Host signed release artifacts for Imagine auto-update. Served publicly via the custom domain `releases.imagine.zeroechodaily.com`.

## Bucket layout

```
imagine-releases/
├── manifests/
│   ├── stable/latest.json          # signed, points to current stable release
│   ├── stable/<version>.json       # historical
│   ├── beta/latest.json
│   └── nightly/latest.json
└── artifacts/
    └── <version>/
        ├── imagine-darwin-arm64.dmg
        ├── imagine-darwin-arm64.dmg.sig    # detached Ed25519 signature
        ├── imagine-win32-x64.exe
        ├── imagine-win32-x64.exe.sig
        ├── imagine-linux-x64.AppImage
        └── imagine-linux-x64.AppImage.sig
```

## v0 actions (maintainer)

1. Create R2 bucket named `imagine-releases` in the Imagine-owned Cloudflare account.
2. Attach the custom domain `releases.imagine.zeroechodaily.com` (DNS in Task 2).
3. Enable public read on the bucket. Object writes use an R2 API token (scoped to this bucket only).
4. Save the R2 API token in 1Password under "Imagine / R2 releases". Never commit it.
5. Place placeholder `manifests/stable/latest.json` pointing to version `0.0.0` so the URL works for verification (Task 7).

## Access control

- Read: public over HTTPS via the custom domain.
- Write: maintainer's R2 API token only. CI may later get a separately-scoped write token.

## Cost

R2 has 10 GB free egress + 10 GB free storage / month. Release binaries are ~50–200 MB; even with 100 downloads/day per release we stay near free tier for many months.

## Reference

- Cloudflare R2: <https://developers.cloudflare.com/r2/>
- Custom domain: <https://developers.cloudflare.com/r2/buckets/public-buckets/#custom-domains-recommended>
```

- [ ] **Step 2: Commit**

```bash
git add operations/cloudflare/r2/bucket-config.md
git commit -m "ops: R2 release bucket config (v0)"
```

---

## Task 4: Release manifest schema (signed)

**Files:**
- Create: `operations/cloudflare/r2/manifest-schema.md`
- Create: `operations/cloudflare/r2/manifest-schema.json`

- [ ] **Step 1: Write the human-readable spec**

`operations/cloudflare/r2/manifest-schema.md`:

```markdown
# Release Manifest Schema

Each release channel has one `latest.json` manifest plus per-version `<version>.json` archives. The auto-update agent downloads `latest.json`, verifies its signature, then downloads + verifies the artifact named inside.

## Example

```json
{
  "manifest_version": 1,
  "channel": "stable",
  "version": "1.0.0",
  "released_at": "2026-06-02T09:00:00Z",
  "min_supported_version": "0.9.0",
  "rollback_policy": "auto_on_health_check_failure",
  "artifacts": {
    "darwin-arm64": {
      "url": "https://releases.imagine.zeroechodaily.com/artifacts/1.0.0/imagine-darwin-arm64.dmg",
      "sha256": "abc123...",
      "size_bytes": 187654321
    },
    "darwin-x64":  { "url": "...", "sha256": "...", "size_bytes": 0 },
    "win32-x64":   { "url": "...", "sha256": "...", "size_bytes": 0 },
    "linux-x64":   { "url": "...", "sha256": "...", "size_bytes": 0 }
  },
  "migration_plan": {
    "db_schema_target": 19,
    "scripts": []
  },
  "signature": {
    "alg": "ed25519",
    "key_id": "imagine-release-key-2026",
    "value": "base64-of-signature-over-canonical-json-without-this-field"
  }
}
```

## Signing rules

1. Build the JSON body **excluding** the `signature` field.
2. Serialize to canonical JSON (sorted keys, no whitespace).
3. Sign with the maintainer's Ed25519 private key.
4. Append `signature.value` (base64) to the JSON.
5. Upload as `manifests/<channel>/<version>.json` and also overwrite `manifests/<channel>/latest.json`.

## Verification (client side, defer to P6)

The auto-update agent embeds the public key at compile time. On `latest.json` fetch it:
1. Strips `signature` field.
2. Recomputes canonical JSON.
3. Verifies `signature.value` against the public key.
4. On success: proceeds to download the artifact, then verifies the artifact's `sha256`.

## Key management

- **Private key** lives on the maintainer's machine, encrypted at rest (1Password, age, or yubikey). Never on a CI machine in v0.
- **Public key** (`imagine-release-key-2026`) is committed to the repo at `operations/cloudflare/r2/release-pubkey.pem` (created during P6 plan, not now).
- Key rotation: emit a new `key_id`, embed both old + new public keys in clients during a grace window, then drop the old.

## Reference

- Canonical JSON: RFC 8785 (JCS). For v0 we accept the simpler "sorted keys + no whitespace" form documented above.
```

- [ ] **Step 2: Write the JSON Schema (machine-readable validation)**

`operations/cloudflare/r2/manifest-schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://imagine.zeroechodaily.com/schemas/release-manifest-v1.json",
  "title": "Imagine release manifest",
  "type": "object",
  "required": [
    "manifest_version",
    "channel",
    "version",
    "released_at",
    "artifacts",
    "signature"
  ],
  "properties": {
    "manifest_version": { "const": 1 },
    "channel": { "enum": ["stable", "beta", "nightly"] },
    "version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+(-[a-z0-9.]+)?$" },
    "released_at": { "type": "string", "format": "date-time" },
    "min_supported_version": { "type": "string" },
    "rollback_policy": {
      "enum": ["auto_on_health_check_failure", "manual_only", "never"]
    },
    "artifacts": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["url", "sha256", "size_bytes"],
        "properties": {
          "url": { "type": "string", "format": "uri" },
          "sha256": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
          "size_bytes": { "type": "integer", "minimum": 0 }
        }
      }
    },
    "migration_plan": {
      "type": "object",
      "properties": {
        "db_schema_target": { "type": "integer" },
        "scripts": { "type": "array", "items": { "type": "string" } }
      }
    },
    "signature": {
      "type": "object",
      "required": ["alg", "key_id", "value"],
      "properties": {
        "alg": { "const": "ed25519" },
        "key_id": { "type": "string" },
        "value": { "type": "string" }
      }
    }
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add operations/cloudflare/r2/manifest-schema.md operations/cloudflare/r2/manifest-schema.json
git commit -m "ops: signed release manifest schema (v0)"
```

---

## Task 5: Release upload + signing script

**Files:**
- Create: `operations/cloudflare/r2/upload-release.sh`

- [ ] **Step 1: Write the script**

The script accepts a version + channel + per-platform artifact paths, hashes them, builds the canonical JSON manifest, signs it with the maintainer's local Ed25519 key, and uploads everything to R2 via `rclone` or `aws s3` (R2 is S3-compatible).

```bash
#!/usr/bin/env bash
# Upload a signed Imagine release to R2.
#
# Usage:
#   PRIVKEY=~/.imagine/release-key.pem \
#   R2_PROFILE=imagine-r2 \
#   operations/cloudflare/r2/upload-release.sh \
#     --channel stable --version 1.0.0 \
#     --darwin-arm64 dist/imagine-darwin-arm64.dmg \
#     --win32-x64 dist/imagine-win32-x64.exe \
#     --linux-x64 dist/imagine-linux-x64.AppImage
#
# Requires: aws (configured with R2 S3-compatible endpoint), openssl (for sha256),
# python3 (for canonical JSON + Ed25519 signing via pynacl or cryptography).

set -euo pipefail

# Argument parsing — keep it minimal in v0; tighten in P6.
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

BUCKET="imagine-releases"
ENDPOINT_URL_ENV="${R2_ENDPOINT:-}"
[[ -z "$ENDPOINT_URL_ENV" ]] && { echo "R2_ENDPOINT env required" >&2; exit 2; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# 1. hash + upload each artifact
artifacts_json='{}'
for platform in "${!ARTIFACTS[@]}"; do
  src="${ARTIFACTS[$platform]}"
  [[ -f "$src" ]] || { echo "missing $src" >&2; exit 1; }
  name=$(basename "$src")
  key="artifacts/${VERSION}/${name}"

  sha=$(openssl dgst -sha256 -binary "$src" | xxd -p -c 256)
  size=$(stat -f %z "$src" 2>/dev/null || stat -c %s "$src")
  url="https://releases.imagine.zeroechodaily.com/${key}"

  aws --endpoint-url "$ENDPOINT_URL_ENV" --profile "$R2_PROFILE" \
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
  aws --endpoint-url "$ENDPOINT_URL_ENV" --profile "$R2_PROFILE" \
      s3 cp --content-type application/json \
      "$TMP/${name}" "s3://${BUCKET}/manifests/${CHANNEL}/${name}"
done

echo "uploaded: ${CHANNEL} ${VERSION} (${#ARTIFACTS[@]} artifacts, signed manifest)"
```

- [ ] **Step 2: Mark executable + commit**

```bash
chmod +x operations/cloudflare/r2/upload-release.sh
git add operations/cloudflare/r2/upload-release.sh
git commit -m "ops: release upload + Ed25519 signing script (v0)"
```

---

## Task 6: Reference cloudflared tunnel template

**Files:**
- Create: `operations/cloudflare/tunnel/cloudflared-template.yml`
- Create: `operations/cloudflare/tunnel/install-tunnel.md`

- [ ] **Step 1: Write the config template**

`operations/cloudflare/tunnel/cloudflared-template.yml`:

```yaml
# cloudflared template — copy to ~/.cloudflared/config.yml on the host running
# the Imagine local server. Replace the placeholders before use.
#
# Setup once: `cloudflared tunnel login` (opens browser to authorize).
# Create tunnel:  `cloudflared tunnel create imagine-<host-nick>`
# Route DNS:      `cloudflared tunnel route dns imagine-<host-nick> <hostname>`
# Run service:    `cloudflared service install` (uses this config)

tunnel: REPLACE_WITH_TUNNEL_UUID
credentials-file: /etc/cloudflared/REPLACE_WITH_TUNNEL_UUID.json

ingress:
  # Public hostname → local Imagine HTTP API (default port 5174 in dev).
  - hostname: REPLACE_WITH_HOSTNAME
    service: http://localhost:5174
    originRequest:
      noTLSVerify: false
      connectTimeout: 30s

  # Required catch-all — anything that doesn't match above gets 404.
  - service: http_status:404
```

- [ ] **Step 2: Write per-OS install doc**

`operations/cloudflare/tunnel/install-tunnel.md`:

```markdown
# cloudflared install + run

Imagine uses **BYO cloudflared** for customer remote access. Each customer host that should be reachable from outside their LAN runs its own `cloudflared` instance pointing at the local Imagine server.

This doc is the reference for **our own** dev/demo host. The customer-facing version is in `operations/cloudflare/wizard/byo-setup-guide.md`.

## macOS (Apple Silicon)

```bash
brew install cloudflared
cloudflared tunnel login                                  # browser auth
cloudflared tunnel create imagine-dev                     # creates tunnel + credentials json
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/<UUID>.json /etc/cloudflared/
sudo cp operations/cloudflare/tunnel/cloudflared-template.yml /etc/cloudflared/config.yml
# Edit /etc/cloudflared/config.yml: fill REPLACE_WITH_* fields.
cloudflared tunnel route dns imagine-dev dev.imagine.zeroechodaily.com
sudo cloudflared service install
sudo launchctl start com.cloudflare.cloudflared
```

## Linux (systemd)

```bash
curl -fsSL https://pkg.cloudflare.com/install.sh | sudo bash
sudo apt install cloudflared
cloudflared tunnel login
cloudflared tunnel create imagine-dev
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/<UUID>.json /etc/cloudflared/
sudo cp operations/cloudflare/tunnel/cloudflared-template.yml /etc/cloudflared/config.yml
# Edit /etc/cloudflared/config.yml.
cloudflared tunnel route dns imagine-dev dev.imagine.zeroechodaily.com
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

## Windows

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel login
cloudflared tunnel create imagine-dev
# Copy config to C:\Windows\System32\config\systemprofile\.cloudflared\config.yml, fill placeholders.
cloudflared tunnel route dns imagine-dev dev.imagine.zeroechodaily.com
cloudflared service install
```

## Verifying

```bash
curl -fsS https://dev.imagine.zeroechodaily.com/health
```

Should return the local Imagine server's health response. If it 502s, check:
- Local Imagine server is running and listening on `localhost:5174`.
- `cloudflared` service is running (`systemctl status cloudflared` / `launchctl list | grep cloudflared`).
- The DNS record exists in the Cloudflare dashboard (auto-created by `tunnel route dns`).
```

- [ ] **Step 3: Commit**

```bash
git add operations/cloudflare/tunnel/cloudflared-template.yml \
        operations/cloudflare/tunnel/install-tunnel.md
git commit -m "ops: reference cloudflared tunnel config + install docs (v0)"
```

---

## Task 7: Customer BYO setup guide

**Files:**
- Create: `operations/cloudflare/wizard/byo-setup-guide.md`

- [ ] **Step 1: Write the customer-facing walkthrough**

```markdown
# BYO Cloudflare — Customer Setup Guide

This guide is for Imagine customers who want **remote access** to their local Imagine server (so they can reach it from outside the office LAN) using their own Cloudflare account.

> **Don't have a Cloudflare account?** Sign up free at <https://dash.cloudflare.com/sign-up>. The Free plan is sufficient.

## What this gets you

- A public HTTPS URL that points to your local Imagine server (e.g. `imagine.yourstudio.com`).
- TLS termination at Cloudflare's edge — no inbound port forwarding on your network.
- Cloudflare Access (optional) for identity-based gating.
- No traffic flows through Imagine's servers — your data stays on your machine.

## Prerequisites

1. A domain you control, added to Cloudflare (any plan).
2. Imagine installed and running on the host you want to expose. Default it listens on `localhost:5174`.

## Steps

### 1. Install cloudflared

(See platform-specific commands at the end.)

### 2. Authorize cloudflared to your Cloudflare account

```bash
cloudflared tunnel login
```

A browser opens. Pick the domain you want to use.

### 3. Create a named tunnel

```bash
cloudflared tunnel create imagine-<host-nickname>
```

`<host-nickname>` is anything — `studio-mac`, `home-nas`, etc.

The command outputs a tunnel UUID and creates a credentials file at `~/.cloudflared/<UUID>.json`. **Don't share this file** — it's a tunnel-specific secret.

### 4. Route a DNS record to the tunnel

```bash
cloudflared tunnel route dns imagine-<host-nickname> imagine.yourstudio.com
```

This creates a CNAME in Cloudflare automatically.

### 5. Configure the tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: <UUID-from-step-3>
credentials-file: ~/.cloudflared/<UUID-from-step-3>.json

ingress:
  - hostname: imagine.yourstudio.com
    service: http://localhost:5174
  - service: http_status:404
```

### 6. Run it

```bash
cloudflared tunnel run imagine-<host-nickname>
```

If that works, install as a service so it runs at boot:

```bash
sudo cloudflared service install
```

### 7. Verify

```bash
curl -fsS https://imagine.yourstudio.com/health
```

You should see your local Imagine's health response.

## Optional: gate it with Cloudflare Access

In the Cloudflare dashboard:

1. **Zero Trust → Access → Applications → Add an application**.
2. Type: **Self-hosted**.
3. Subdomain: `imagine`, Domain: `yourstudio.com`.
4. Add a policy: email = your team's emails.

Now `imagine.yourstudio.com` requires login before the request reaches your machine.

## Platform-specific install

(See `operations/cloudflare/tunnel/install-tunnel.md` for the maintainer-side reference.)

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| 502 Bad Gateway | Imagine not running, or `cloudflared` pointing at wrong local port | Check `cloudflared` logs; confirm `curl localhost:5174/health` works. |
| Tunnel "unhealthy" in dashboard | Credentials file wrong / tunnel deleted on the other end | Re-run `cloudflared tunnel create` and update `config.yml`. |
| DNS resolves but TLS fails | Cloudflare hasn't issued Universal SSL yet (can take 15 min) | Wait. Re-check after 15 min. |
| 1033 "Argo Tunnel error" | `cloudflared` not running | `systemctl status cloudflared` / `launchctl list \| grep cloudflared`. |
```

- [ ] **Step 2: Commit**

```bash
git add operations/cloudflare/wizard/byo-setup-guide.md
git commit -m "ops: customer BYO Cloudflare setup guide (v0)"
```

---

## Task 8: End-to-end verification

This is manual and requires maintainer Cloudflare account access. It is **the** validation that v0 actually works.

- [ ] **Step 1: Maintainer-side Cloudflare setup**

In the Cloudflare dashboard (maintainer):

1. Confirm `zeroechodaily.com` (or `imagine.zeroechodaily.com` subzone) is reachable in the Imagine-owned Cloudflare account.
2. Create R2 bucket `imagine-releases`.
3. Attach custom domain `releases.imagine.zeroechodaily.com` to the bucket (Task 3 doc).
4. Upload a 1 KB test file as `manifests/stable/test.txt` via the dashboard.

- [ ] **Step 2: Verify R2 release CDN**

```bash
curl -fsS https://releases.imagine.zeroechodaily.com/manifests/stable/test.txt
```

Expected: HTTP 200 with the file contents. If 404, check Custom Domain attachment in R2 dashboard.

- [ ] **Step 3: Maintainer-side tunnel setup (dev host)**

Follow `operations/cloudflare/tunnel/install-tunnel.md` to create a tunnel `imagine-dev` pointing `dev.imagine.zeroechodaily.com` → `http://localhost:5174` on the maintainer's machine. Start the local Imagine server on `:5174`.

- [ ] **Step 4: Verify the tunnel**

```bash
curl -fsS https://dev.imagine.zeroechodaily.com/health
```

Expected: HTTP 200 from the local Imagine server.

- [ ] **Step 5: Verify signed-manifest signing roundtrip**

Generate a maintainer Ed25519 key (one-time, save private key encrypted at rest):

```bash
mkdir -p ~/.imagine
openssl genpkey -algorithm ed25519 -out ~/.imagine/release-key.pem
openssl pkey -in ~/.imagine/release-key.pem -pubout -out ~/.imagine/release-pubkey.pem
chmod 600 ~/.imagine/release-key.pem
```

Run the upload script with a dummy artifact:

```bash
echo "test artifact" > /tmp/imagine-test.bin
PRIVKEY=~/.imagine/release-key.pem \
R2_PROFILE=imagine-r2 \
R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
operations/cloudflare/r2/upload-release.sh \
  --channel beta --version 0.0.1-test \
  --darwin-arm64 /tmp/imagine-test.bin
```

Then fetch the manifest and verify the signature with a small Python snippet (see `operations/cloudflare/r2/manifest-schema.md`).

Expected: signature verifies against the public key, artifact sha256 matches.

- [ ] **Step 6: Commit a record of the verification**

Create `operations/cloudflare/VERIFICATION-LOG.md` capturing the date and outcome of each verification step. (No code — just dated evidence.)

```bash
git add operations/cloudflare/VERIFICATION-LOG.md
git commit -m "ops: v0 end-to-end verification log"
```

---

## Completion criteria

v0 is complete when:

1. All 8 tasks are committed on `main` (or a feature branch merged via `superpowers:finishing-a-development-branch`).
2. `https://releases.imagine.zeroechodaily.com/manifests/stable/test.txt` returns 200.
3. `https://dev.imagine.zeroechodaily.com/health` returns 200 from the local Imagine server.
4. A test signed manifest exists at `https://releases.imagine.zeroechodaily.com/manifests/beta/latest.json` and its signature verifies against the published public key.
5. `operations/cloudflare/VERIFICATION-LOG.md` records the verification steps with timestamps.

After v0: P6 (Auto-Update Agent) can begin — it will consume the manifest format defined here. P8 (BYO Cloudflare Wizard, UI version) can wrap the doc from Task 7.
