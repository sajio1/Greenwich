#!/usr/bin/env bash
set -euo pipefail

# Run once on a fresh Ubuntu 24.04 EC2 host.
if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this bootstrap with sudo." >&2
  exit 1
fi

apt-get update
apt-get install -y ca-certificates curl docker.io docker-compose-v2 rsync
systemctl enable --now docker
install -d -m 0755 \
  /srv/alphamotion/data \
  /srv/alphamotion/cache \
  /srv/alphamotion/data_studio \
  /srv/alphamotion/body_data \
  /opt/alphamotion-deploy

echo "Host ready. Copy this aws/ directory to /opt/alphamotion-deploy,"
echo "populate .env, copy the prepared .bundle, then run:"
echo "  docker compose up -d --build"
