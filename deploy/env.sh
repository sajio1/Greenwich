#!/usr/bin/env bash

# Portable defaults. Set ALPHAMOTION_DATA/ALPHAMOTION_CACHE before sourcing
# this file when large assets should live on a dedicated disk.
deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$deploy_dir/.." && pwd)"
export ALPHAMOTION_ENV="${ALPHAMOTION_ENV:-$project_root/.venv}"
export ALPHAMOTION_DATA_STUDIO_PORT="${ALPHAMOTION_DATA_STUDIO_PORT:-8765}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PATH="$ALPHAMOTION_ENV/bin:$PATH"
