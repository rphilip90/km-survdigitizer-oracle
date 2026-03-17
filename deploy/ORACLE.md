# Oracle Always Free Deployment

This app is designed for one Oracle Cloud Always Free VM with no managed services.

## Target shape

- Ubuntu on Ampere A1 Flex
- 2 OCPU / 12 GB RAM
- one public IP
- Caddy on port 80
- FastAPI on `127.0.0.1:8000`
- SQLite and local disk only

## System packages

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip r-base build-essential \
    libcurl4-openssl-dev libssl-dev libxml2-dev caddy
```

## App setup

```bash
sudo mkdir -p /opt/km-survdigitizer-oracle
sudo chown ubuntu:ubuntu /opt/km-survdigitizer-oracle
git clone <repo-url> /opt/km-survdigitizer-oracle
cd /opt/km-survdigitizer-oracle
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
Rscript scripts/bootstrap_r.R
```

Set `OPENAI_API_KEY` in `.env`, then install the service and reverse proxy:

```bash
sudo cp deploy/km-survdigitizer.service /etc/systemd/system/km-survdigitizer.service
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl daemon-reload
sudo systemctl enable --now km-survdigitizer
sudo systemctl restart caddy
```

The app will be available on the VM public IP.

## Always Free guardrails

- Keep everything on the VM filesystem.
- Do not add an Oracle load balancer or managed database.
- Keep batch uploads moderate; this is a small project footprint, not a large queue worker cluster.
