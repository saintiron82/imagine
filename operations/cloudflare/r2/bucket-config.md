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
5. Place placeholder `manifests/stable/latest.json` pointing to version `0.0.0` so the URL works for verification (Task 8).

## Access control

- Read: public over HTTPS via the custom domain.
- Write: maintainer's R2 API token only. CI may later get a separately-scoped write token.

## Cost

R2 has 10 GB free egress + 10 GB free storage / month. Release binaries are ~50–200 MB; even with 100 downloads/day per release we stay near free tier for many months.

## Reference

- Cloudflare R2: <https://developers.cloudflare.com/r2/>
- Custom domain: <https://developers.cloudflare.com/r2/buckets/public-buckets/#custom-domains-recommended>
