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
