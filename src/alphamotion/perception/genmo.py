"""GENMO perception adapters (optional extra): video->SMPL and text->SMPL.

The heavy models (GENMO checkpoint ~5.5 GB, T5) are NOT bundled or re-hosted;
they run in a separate python environment pointed to by
ALPHAMOTION_GENMO_PYTHON + ALPHAMOTION_GENMO_REPO, communicating through the
args + last-stdout-JSON worker protocol. Without that env configured these
functions raise with actionable instructions instead of pretending.
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..config import CONFIG
from ..engine.nets.rotations import matrix_to_rot6d, rot6d_to_matrix
from ..paths import cache_dir

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19]


def status() -> dict:
    """Cheap readiness probe used by health/UI; never loads the model itself."""
    if CONFIG.genmo_space:
        missing = [] if CONFIG.genmo_token else ["space_token"]
        ready = bool(CONFIG.genmo_token)
        return {"ready": ready, "text": ready, "video": ready,
                "backend": "AlphaMotion Cloud", "space": CONFIG.genmo_space,
                "missing": missing}
    python = Path(CONFIG.genmo_python) if CONFIG.genmo_python else None
    repo = Path(CONFIG.genmo_repo) if CONFIG.genmo_repo else None
    script = repo / "scripts" / "demo" / "demo_smpl.py" if repo else None
    reference = cache_dir() / "alphamotion_reference.mp4"
    if not reference.is_file():
        reference = cache_dir() / "genmo_reference.mp4"  # legacy deployment
    core_ready = bool(python and python.is_file() and script and
                      script.is_file())
    text_ready = core_ready and reference.is_file()
    video_ready = core_ready
    missing = []
    if not python or not python.is_file():
        missing.append("python")
    if not script or not script.is_file():
        missing.append("repo")
    if not reference.is_file():
        missing.append("reference_video")
    return {"ready": text_ready or video_ready,
            "text": text_ready,
            "video": video_ready,
            "backend": "AlphaMotion",
            "missing": missing}


def _require_env(*, needs_reference: bool = False):
    probe = status()
    capability = "text" if needs_reference else "video"
    if not probe[capability]:
        raise RuntimeError(
            "AlphaMotion generation is not configured on this host. "
            "Configure the AlphaMotion generation environment and place a short "
            "person reference clip in the configured cache. "
            f"Missing: {', '.join(probe['missing'])}.")


def _aa_to_R(aa: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.divide(aa, np.where(th < 1e-8, 1.0, th))
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    eye = np.broadcast_to(np.eye(3), K.shape).copy()
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def _segment_slice(smpl_params: dict, segment: str | None) -> slice:
    if not segment:
        return slice(None)
    segs = [x for x in smpl_params.get("segment_info", [])
            if x["type"] == segment]
    if not segs:
        raise ValueError(f"no '{segment}' segment in artifact")
    return slice(segs[0]["start"], segs[0]["end"])


def smpl_to_global_rot6d(smpl_params: dict, segment: str | None = None):
    """GENMO smpl_params dict -> GLOBAL rot6d [T,22,6] (torch, cpu).
    Local->global composition included — feeding local axis-angle straight to
    FK is the audited 11.6 cm silent bug."""
    p = smpl_params["body_params_global"]
    sl = _segment_slice(smpl_params, segment)
    go = p["global_orient"].float().numpy().reshape(-1, 1, 3)[sl]
    bp = p["body_pose"].float().numpy().reshape(-1, 21, 3)[sl]
    aa = np.concatenate([go, bp], 1)
    Rl = _aa_to_R(aa)
    Rg = np.zeros_like(Rl)
    for j, par in enumerate(SMPL_PARENTS):
        Rg[:, j] = Rl[:, j] if par < 0 else Rg[:, par] @ Rl[:, j]
    r6 = Rg[..., :, :2].transpose(0, 1, 3, 2).reshape(len(aa), 22, 6)
    return torch.from_numpy(r6).float()


def smpl_root_translation(smpl_params: dict,
                          segment: str | None = None) -> np.ndarray:
    """Return first-frame-anchored world root translation in centimeters.

    GENMO emits SMPL translation in meters with Y up. AlphaMotion traces use
    centimeters with the same canonical axes; anchoring every source clip at
    zero lets the temporal assembler concatenate it without coordinate drift.
    """
    p = smpl_params["body_params_global"]
    root = p["transl"].float().numpy()[_segment_slice(smpl_params, segment)]
    if root.ndim != 2 or root.shape[1] != 3 or len(root) == 0:
        raise ValueError("GENMO translation must have shape [frames, 3]")
    root = (root - root[:1]) * 100.0
    if not np.isfinite(root).all():
        raise ValueError("GENMO translation contains NaN or infinity")
    return root.astype(np.float64, copy=False)


def global_to_local_rot6d(global_rot6d: torch.Tensor) -> torch.Tensor:
    """Convert GENMO's composed SMPL rotations back to local joint rotations.

    Data Studio's portable SMPL asset contract stores local rotations; the
    timeline adapter deliberately consumes the composed form above.
    """
    global_R = rot6d_to_matrix(global_rot6d)
    local_R = global_R.clone()
    for joint, parent in enumerate(SMPL_PARENTS):
        if parent >= 0:
            local_R[:, joint] = global_R[:, parent].transpose(-1, -2) @ global_R[:, joint]
    return matrix_to_rot6d(local_R).float()


def local_to_global_rot6d(local_rot6d: torch.Tensor) -> torch.Tensor:
    """Compose portable local SMPL rotations for the timeline FK contract."""
    local_R = rot6d_to_matrix(local_rot6d)
    global_R = local_R.clone()
    for joint, parent in enumerate(SMPL_PARENTS):
        if parent >= 0:
            global_R[:, joint] = global_R[:, parent] @ local_R[:, joint]
    return matrix_to_rot6d(global_R).float()


@lru_cache(maxsize=2)
def _space_client(space: str, token: str):
    try:
        from gradio_client import Client
    except ImportError as exc:
        raise RuntimeError(
            "AlphaMotion Cloud generation requires the perception extra: "
            "pip install 'alphamotion[perception]'") from exc
    return Client(space, token=token)


def _space_result_path(result) -> Path:
    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("AlphaMotion Cloud returned an empty result")
        result = result[0]
    if isinstance(result, dict):
        result = result.get("path") or result.get("name")
    path = Path(str(result or ""))
    if not path.is_file():
        raise RuntimeError(f"AlphaMotion Cloud result is missing: {path}")
    return path


def _load_space_motion(result) -> tuple[torch.Tensor, np.ndarray, float]:
    path = _space_result_path(result)
    try:
        with np.load(path, allow_pickle=False) as payload:
            local = np.asarray(payload["local_rot6d"], np.float32)
            root = np.asarray(payload["root_cm"], np.float64)
            fps = float(np.asarray(payload.get("fps", 30.0)).reshape(()))
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"invalid AlphaMotion Cloud artifact: {exc}") from exc
    if local.ndim != 3 or local.shape[1:] != (22, 6) or len(local) == 0:
        raise RuntimeError("AlphaMotion Cloud local_rot6d must have shape [T,22,6]")
    if root.shape != (len(local), 3):
        raise RuntimeError("AlphaMotion Cloud root_cm must have shape [T,3]")
    if not np.isfinite(local).all() or not np.isfinite(root).all():
        raise RuntimeError("AlphaMotion Cloud artifact contains NaN or infinity")
    if fps <= 0 or not np.isfinite(fps):
        raise RuntimeError("AlphaMotion Cloud fps must be positive")
    root = root - root[:1]
    return local_to_global_rot6d(torch.from_numpy(local)), root, fps


def _space_job_result(job, operation: str):
    try:
        return job.result(timeout=CONFIG.genmo_timeout_s)
    except Exception as exc:
        detail = str(exc).strip() or repr(exc)
        try:
            status = getattr(job.status(), "code", None)
            status = getattr(status, "value", status)
        except Exception:  # status is diagnostic only
            status = None
        suffix = f"; queue status: {status}" if status else ""
        raise RuntimeError(
            f"AlphaMotion Cloud {operation} failed "
            f"({type(exc).__name__}): {detail}{suffix}"
        ) from exc


def _space_prompt(text: str, frames: int) -> tuple[torch.Tensor, np.ndarray]:
    client = _space_client(CONFIG.genmo_space, CONFIG.genmo_token)
    job = client.submit(text, int(frames), api_name="/generate_text")
    rotations, root, _fps = _load_space_motion(
        _space_job_result(job, "text generation"))
    return _resample(rotations, root, frames)


def _space_video(video_path: str,
                 frames: int | None) -> tuple[torch.Tensor, np.ndarray]:
    try:
        from gradio_client import handle_file
    except ImportError as exc:
        raise RuntimeError(
            "AlphaMotion Cloud generation requires the perception extra: "
            "pip install 'alphamotion[perception]'") from exc
    client = _space_client(CONFIG.genmo_space, CONFIG.genmo_token)
    requested = int(frames) if frames is not None else 0
    job = client.submit(handle_file(video_path), requested,
                        api_name="/generate_video")
    rotations, root, _fps = _load_space_motion(
        _space_job_result(job, "video generation"))
    return (_resample(rotations, root, frames)
            if frames is not None else (rotations, root))


def _run_genmo(inputs: list[str], text_len: int,
               *, needs_reference: bool = False) -> dict:
    _require_env(needs_reference=needs_reference)
    staging = cache_dir() / "alphamotion_generation"
    staging.mkdir(parents=True, exist_ok=True)
    cmd = [CONFIG.genmo_python, "scripts/demo/demo_smpl.py",
           "--input_list", *inputs, "--no_render", "--static_cam",
           "--text_length", str(text_len), "--output_root", str(staging)]
    proc = subprocess.run(cmd, cwd=CONFIG.genmo_repo, text=True,
                          capture_output=True, timeout=1900, check=False)
    if proc.returncode != 0:
        raise RuntimeError("AlphaMotion generation failed: "
                           + (proc.stderr[-1500:] or proc.stdout[-1500:]))
    stem = Path(inputs[0]).stem
    art = staging / f"{stem}_mix" / "smpl_params.pt"
    if not art.is_file():
        raise RuntimeError(f"AlphaMotion generation artifact missing: {art}")
    return torch.load(art, map_location="cpu", weights_only=False)


def reference_video() -> Path:
    """The cached reference clip whose preprocessing is prewarmed — GENMO needs
    one video for camera intrinsics even in text mode."""
    ref = cache_dir() / "alphamotion_reference.mp4"
    if not ref.exists():
        ref = cache_dir() / "genmo_reference.mp4"  # legacy deployment
    if not ref.exists():
        raise RuntimeError(
            "no reference video cached; copy any short person video to "
            f"{ref} once (its perception cache warms on first run)")
    return ref


def motion_from_prompt(text: str, seconds: float = 10.0
                       ) -> tuple[torch.Tensor, np.ndarray]:
    frames = max(30, int(seconds * 30))
    if CONFIG.genmo_space:
        return _space_prompt(text, frames)
    art = _run_genmo([str(reference_video()), f"text:{text}"], frames,
                     needs_reference=True)
    return (smpl_to_global_rot6d(art, segment="text"),
            smpl_root_translation(art, segment="text"))


def _resample(rot6d: torch.Tensor, root: np.ndarray,
              frames: int) -> tuple[torch.Tensor, np.ndarray]:
    """Resample GENMO output to the 30 FPS timeline contract when requested."""
    frames = max(1, int(frames))
    if len(rot6d) == frames:
        return rot6d, root
    x = rot6d.permute(1, 2, 0).reshape(1, -1, len(rot6d))
    x = torch.nn.functional.interpolate(
        x, size=frames, mode="linear", align_corners=True)
    out = x.reshape(22, 6, frames).permute(2, 0, 1)
    out = matrix_to_rot6d(rot6d_to_matrix(out)).float()
    src = np.linspace(0.0, 1.0, len(root))
    dst = np.linspace(0.0, 1.0, frames)
    root_out = np.stack([np.interp(dst, src, root[:, axis])
                         for axis in range(3)], axis=1)
    return out, root_out.astype(np.float64)


def motion_from_video(video_path: str, frames: int | None = None
                      ) -> tuple[torch.Tensor, np.ndarray]:
    src = Path(video_path)
    if not src.is_file():
        raise RuntimeError(f"video not found: {src}")
    if CONFIG.genmo_space:
        return _space_video(str(src), frames)
    staged = cache_dir() / "alphamotion_generation" / "uploads" / src.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    if not staged.exists():
        shutil.copy2(src, staged)
    art = _run_genmo([str(staged)], 60)
    rot, root = smpl_to_global_rot6d(art), smpl_root_translation(art)
    return _resample(rot, root, frames) if frames is not None else (rot, root)
