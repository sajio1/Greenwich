---
title: AlphaMotion Generation
emoji: 🦿
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: 3.10.13
app_file: app.py
startup_duration_timeout: 1h
---

# AlphaMotion generation worker

Private ZeroGPU worker for the AlphaMotion Studio. It exposes two named Gradio
API endpoints, `/generate_text` and `/generate_video`, and returns only a safe
NPZ containing local SMPL-22 rotations, root translation and FPS.

Text generation uses the AlphaMotion default of 300 frames (10 seconds at
30 FPS), while still accepting another requested length. Video generation
preserves the uploaded clip duration instead of truncating it. ZeroGPU jobs may
request up to 300 GPU seconds; exceptionally long inputs can still be stopped
by Hugging Face quota or runtime limits.

Required Space variable:

- `ALPHAMOTION_ASSET_REPO`: private Hugging Face model repository containing:
  - `hmr2.ckpt`
  - `vitpose.pth`
  - `smplx-neutral.npz`

Required Space secret:

- `HF_TOKEN`: a fine-grained read-only token scoped to the private asset
  repository. It is never sent to AlphaMotion browser clients.

The Studio calls this private Space with a separate read-only token stored only
on the AWS host.

The private Space repository also includes a compressed base64 bundle of the
small body-model lookup tables that are omitted from a filtered clone.

This integration is for the small non-commercial demo described by the
upstream model license. Review all upstream and checkpoint licenses before
changing that use.
