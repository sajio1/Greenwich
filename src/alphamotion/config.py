"""Runtime configuration. Environment prefix: ALPHAMOTION_*."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .paths import data_dir


def _data_path(*parts: str) -> str:
    """Portable default under the user's AlphaMotion data directory."""
    return str(data_dir().joinpath(*parts))


def default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def setup_gl_backend() -> None:
    """Platform-conditional GL selection for MuJoCo offscreen rendering.

    Linux headless -> EGL; Windows -> leave unset (WGL default); macOS -> cgl.
    Must be called BEFORE importing mujoco. The API gateway calls this during
    module initialization because its persistent viewer and MP4 exporter both
    use MuJoCo in the warm process.
    """
    if "MUJOCO_GL" in os.environ:
        return
    if sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "egl"
    elif sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    # win32: WGL default


@dataclass
class AMConfig:
    device: str = field(default_factory=default_device)
    host: str = os.environ.get("ALPHAMOTION_HOST", "127.0.0.1")
    port: int = int(os.environ.get("ALPHAMOTION_PORT", "7860"))
    # target motion, body audit, and independent SMPL-X source comparison.
    viewer_ports: tuple[int, int, int] = (7871, 7876, 7877)
    hf_repo: str = os.environ.get("ALPHAMOTION_HF_REPO", "lloydlei/Greenwich")
    # Optional text/video generation runs in a separate environment so its
    # large weights never share AlphaMotion's warm process. Old variable names
    # remain fallback-only so existing private deployments keep working.
    genmo_python: str = os.environ.get(
        "ALPHAMOTION_GENERATION_PYTHON",
        os.environ.get("ALPHAMOTION_GENMO_PYTHON", ""))
    genmo_repo: str = os.environ.get(
        "ALPHAMOTION_GENERATION_REPO",
        os.environ.get("ALPHAMOTION_GENMO_REPO", ""))
    # Cloud deployments use a private Gradio/ZeroGPU Space instead of a
    # workstation-local subprocess. The token is a server-side secret and is
    # never exposed to the browser.
    genmo_space: str = os.environ.get(
        "ALPHAMOTION_GENERATION_SPACE",
        os.environ.get("ALPHAMOTION_GENMO_SPACE", ""))
    genmo_token: str = os.environ.get(
        "ALPHAMOTION_GENERATION_TOKEN",
        os.environ.get("ALPHAMOTION_GENMO_TOKEN", ""))
    genmo_timeout_s: int = int(os.environ.get(
        "ALPHAMOTION_GENERATION_TIMEOUT_S",
        os.environ.get("ALPHAMOTION_GENMO_TIMEOUT_S", "900")))
    # Kept for compatibility with deployments created while the experimental
    # MoMask/GVHMR adapter was available; neither field is used at runtime.
    motion_python: str = os.environ.get(
        "ALPHAMOTION_MOTION_PYTHON",
        os.environ.get("ALPHAMOTION_GENMO_PYTHON", ""),
    )
    momask_repo: str = os.environ.get("ALPHAMOTION_MOMASK_REPO", "")
    gvhmr_repo: str = os.environ.get("ALPHAMOTION_GVHMR_REPO", "")
    # BodyDataStudio remains its own worker/service because it owns a large
    # SQLite index and long-running augmentation jobs. AlphaMotion starts it,
    # proxies it under /data-studio/, and reads its published catalog.
    data_studio_repo: str = os.environ.get(
        "ALPHAMOTION_DATA_STUDIO_REPO",
        _data_path("vendor", "body-data-studio"),
    )
    data_studio_root: str = os.environ.get(
        "ALPHAMOTION_DATA_STUDIO_ROOT",
        _data_path("body_data"),
    )
    data_studio_cache: str = os.environ.get(
        "ALPHAMOTION_DATA_STUDIO_CACHE",
        _data_path("data_studio"),
    )
    data_studio_port: int = int(os.environ.get(
        "ALPHAMOTION_DATA_STUDIO_PORT", "8765"))

    @property
    def data_studio_db(self) -> Path:
        return Path(self.data_studio_cache) / "bodydata_studio.sqlite3"


CONFIG = AMConfig()
