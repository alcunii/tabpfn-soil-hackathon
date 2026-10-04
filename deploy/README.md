# Deployment files for Soil2Crop

Everything here is **generic and secret-free** — safe to publish. Fill in your own domains/values.

| File | Purpose |
|---|---|
| `Caddyfile.example` | Reverse-proxy vhost (dedicated `soil2crop.<your-domain>` + secure headers). |
| `soil2crop.service` | systemd unit for the FastAPI app (uvicorn, host-agnostic paths). |
| `run_local.sh` | Create a venv, install deps, fetch the TabPFN-3.5 checkpoint, start the app locally. |
| `deploy.sh` | Copy the app to a server under `/opt/soil2crop` and (re)start the unit. |
| `soil2crop.env.example` | Environment variables (checkpoint path, port, optional LLM for the agent). |

## TL;DR (one-box, no container)

```bash
bash deploy/run_local.sh            # http://127.0.0.1:8877
```

## Behind Caddy (dedicated subdomain)

```bash
# 1. run the app on 127.0.0.1:8877 (systemd unit or run_local.sh)
# 2. adapt and install the vhost
sudo cp deploy/Caddyfile.example /etc/caddy/soil2crop.caddy   # edit the domain first
sudo caddy validate --config /etc/caddy/soil2crop.caddy
sudo systemctl reload caddy
```

See the top-level **README → “Deploy it yourself”** for the full walk-through and the security notes.
