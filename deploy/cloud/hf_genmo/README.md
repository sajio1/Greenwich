---
title: AlphaMotion GENMO
emoji: 🦿
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: 3.10.13
app_file: app.py
startup_duration_timeout: 1h
---

# AlphaMotion GENMO worker

Private ZeroGPU worker for the AlphaMotion Studio. It exposes two named Gradio
API endpoints, `/generate_text` and `/generate_video`, and returns only a safe
NPZ containing local SMPL-22 rotations, root translation and FPS.

The free personal ZeroGPU tier is deliberately capped to three seconds / 90
frames per inference so each call requests at most 60 GPU seconds. AlphaMotion
can resample the returned motion to the project timeline length.

Required Space variable:

- `GENMO_ASSET_REPO`: private Hugging Face model repository containing:
  - `hmr2.ckpt`
  - `vitpose.pth`
  - `smplx-neutral.npz`

Required Space secret:

- `HF_TOKEN`: a fine-grained read-only token scoped to the private asset
  repository. It is never sent to AlphaMotion browser clients.

The Studio calls this private Space with a separate read-only token stored only
on the AWS host.

The private Space repository also includes a compressed base64 bundle of the
small GENMO `body_model` lookup tables that are omitted from a filtered clone.

This integration is for the small non-commercial demo described by the
upstream GENMO/GEM license. Review all upstream and checkpoint licenses before
changing that use.
