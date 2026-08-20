# AWS Studio runbook

## Recommended small-demo host

- Ubuntu 24.04, `t3.large` (2 vCPU / 8 GB) to start. The UI and
  BodyDataStudio are CPU workloads; GENMO is remote. Add an 8 GB swap file on
  the EBS volume if the first full-corpus index briefly peaks above RAM.
- 150 GB encrypted gp3 EBS volume mounted at `/srv/alphamotion`.
- Security group: inbound 80/443 from the Internet; port 22 only from the
  operator IP. The application port 7860 is not public.
- An Elastic IP and a DNS A record for `STUDIO_DOMAIN`. If no domain is
  available, use `<ELASTIC_IP>.nip.io`; it resolves to the IP and lets Caddy
  issue the HTTPS certificate required for browser device keys.

Move to `t3.xlarge` only if measurements show sustained memory pressure or
several people process assets at once. Stop the instance between one-off
demonstrations; EBS and the public IPv4 address remain billable, but instance
compute stops.

## Build and upload

On the workstation:

```bash
python deploy/cloud/aws/prepare_bundle.py
rsync -az deploy/cloud/aws/ ubuntu@SERVER:/opt/alphamotion-deploy/
rsync -az --info=progress2 '/media/sajio/New Volume/body_data/' \
  ubuntu@SERVER:/srv/alphamotion/body_data/
```

On the EC2 host:

```bash
sudo bash /opt/alphamotion-deploy/bootstrap-ubuntu.sh
cd /opt/alphamotion-deploy
cp studio.env.example .env
# Edit .env locally on the server; do not paste secrets into chat.
sudo docker compose up -d --build
sudo docker compose logs -f studio
```

The first BodyDataStudio scan can take time. Its database, preview cache,
projects and the three-device registry persist on EBS across container updates.

## Updating

Prepare a fresh bundle, sync only the `aws/` directory, then run:

```bash
sudo docker compose up -d --build
```

No source dataset is copied into the image and the container mounts it
read-only.
