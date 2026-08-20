FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ffmpeg libegl1 libgl1 libglib2.0-0 libgomp1 nodejs npm p7zip-full \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
COPY alphamotion /opt/alphamotion
COPY body-data-studio /opt/body-data-studio

RUN python -m pip install --upgrade pip \
    && python -m pip install torch torchvision \
       --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install '/opt/alphamotion[perception]' \
       -r /opt/body-data-studio/requirements-preview.txt \
    && cd /opt/body-data-studio && npm ci --omit=dev

WORKDIR /opt/alphamotion
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
  CMD curl -fsS http://127.0.0.1:7860/access >/dev/null || exit 1
CMD ["alphamotion", "serve", "--host", "0.0.0.0", "--port", "7860"]
