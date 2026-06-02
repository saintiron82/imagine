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
