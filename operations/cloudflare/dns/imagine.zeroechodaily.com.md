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
