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
