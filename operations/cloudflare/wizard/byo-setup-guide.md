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
