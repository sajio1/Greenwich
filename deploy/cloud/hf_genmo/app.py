"""Private Gradio/ZeroGPU worker for AlphaMotion generation."""
# ruff: noqa: I001
from __future__ import annotations

# ZeroGPU must patch torch before torch itself is imported.
import spaces

import base64
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
GENMO_DIR = ROOT / "GENMO"
GENMO_REVISION = "16bebf402d8893184249ee206d957b8248cd8310"
ASSET_REPO = os.environ.get(
    "ALPHAMOTION_ASSET_REPO", os.environ.get("GENMO_ASSET_REPO", "")).strip()
HF_TOKEN = os.environ.get("HF_TOKEN")
DEFAULT_TEXT_FRAMES = 300


def _prepare_source() -> None:
    if not (GENMO_DIR / ".git").is_dir():
        subprocess.run([
            "git", "clone", "--filter=blob:none",
            "https://github.com/NVlabs/GENMO.git", str(GENMO_DIR),
        ], check=True)
    subprocess.run(["git", "checkout", "--force", GENMO_REVISION],
                   cwd=GENMO_DIR, check=True)
    patch = ROOT / "genmo-zero-gpu.patch"
    check = subprocess.run(["git", "apply", "--check", str(patch)],
                           cwd=GENMO_DIR, check=False)
    if check.returncode == 0:
        subprocess.run(["git", "apply", str(patch)],
                       cwd=GENMO_DIR, check=True)


def _prepare_body_model() -> None:
    """Restore small upstream auxiliary tables omitted from the Git clone."""
    encoded = ROOT / "body-model-assets.tar.gz.b64"
    destination = GENMO_DIR / "gem" / "utils" / "body_model"
    destination.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(encoded.read_bytes(), validate=True)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not (root / member.name).resolve().is_relative_to(root):
                raise RuntimeError("Unsafe body-model archive member")
        archive.extractall(destination)


def _asset(filename: str, destination: Path) -> Path:
    if not ASSET_REPO:
        raise RuntimeError(
            "Set the private Space variable ALPHAMOTION_ASSET_REPO before startup.")
    source = Path(hf_hub_download(
        ASSET_REPO, filename=filename, token=HF_TOKEN))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination


_prepare_source()
_prepare_body_model()
sys.path.insert(0, str(GENMO_DIR))
sys.path.insert(0, str(GENMO_DIR / "scripts" / "demo"))
os.chdir(GENMO_DIR)

HMR2_PATH = _asset(
    "hmr2.ckpt",
    GENMO_DIR / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt")
_asset(
    "vitpose.pth",
    GENMO_DIR / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth")
_asset(
    "smplx-neutral.npz",
    GENMO_DIR / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz")
CHECKPOINT_PATH = Path(hf_hub_download(
    "nvidia/GEM-X", filename="gem_smpl.ckpt", token=HF_TOKEN))

from demo_smpl import (
    assemble_mixed_data,
    create_text_segment,
    parse_input_list,
    preprocess_video_segment,
)
from demo_utils import get_camera_static, load_model, run_inference


# ZeroGPU's CUDA emulation permits module-level placement and avoids a full
# 5.5 GB model reload on every queued request.
MODEL = load_model(str(CHECKPOINT_PATH), load_text_encoder=True)


def _axis_angle_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = np.divide(axis_angle, np.where(angle < 1e-8, 1.0, angle))
    skew = np.zeros(axis.shape[:-1] + (3, 3), dtype=np.float32)
    skew[..., 0, 1], skew[..., 0, 2] = -axis[..., 2], axis[..., 1]
    skew[..., 1, 0], skew[..., 1, 2] = axis[..., 2], -axis[..., 0]
    skew[..., 2, 0], skew[..., 2, 1] = -axis[..., 1], axis[..., 0]
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), skew.shape).copy()
    sine = np.sin(angle)[..., None]
    cosine = np.cos(angle)[..., None]
    return eye + sine * skew + (1.0 - cosine) * (skew @ skew)


def _safe_artifact(pred: dict, segment_info: list[dict],
                   segment_type: str | None) -> str:
    body = pred["body_params_global"]
    selected = slice(None)
    if segment_type:
        info = next((item for item in segment_info
                     if item["type"] == segment_type), None)
        if info is None:
            raise RuntimeError(f"AlphaMotion did not return a {segment_type} segment")
        selected = slice(int(info["start"]), int(info["end"]))
    orient = body["global_orient"].detach().float().cpu().numpy()
    pose = body["body_pose"].detach().float().cpu().numpy()
    orient = orient.reshape(-1, 1, 3)[selected]
    pose = pose.reshape(-1, 21, 3)[selected]
    local_matrix = _axis_angle_matrix(np.concatenate([orient, pose], axis=1))
    local_rot6d = np.concatenate(
        [local_matrix[..., :, 0], local_matrix[..., :, 1]], axis=-1)
    root = body["transl"].detach().float().cpu().numpy()[selected]
    if len(local_rot6d) == 0 or root.shape != (len(local_rot6d), 3):
        raise RuntimeError("AlphaMotion returned an empty or malformed motion")
    root_cm = (root - root[:1]) * 100.0
    if not np.isfinite(local_rot6d).all() or not np.isfinite(root_cm).all():
        raise RuntimeError("AlphaMotion returned NaN or infinity")
    output = Path(tempfile.mkstemp(prefix="alphamotion-motion-",
                                   suffix=".npz")[1])
    np.savez_compressed(
        output, local_rot6d=local_rot6d.astype(np.float32),
        root_cm=root_cm.astype(np.float64), fps=np.float32(30.0))
    return str(output)


def _normalized_video(path: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise gr.Error("Upload a readable video file.")
    output = Path(tempfile.mkstemp(prefix="alphamotion-input-",
                                   suffix=".mp4")[1])
    subprocess.run([
        "ffmpeg", "-y", "-i", str(source), "-vf", "fps=30",
        "-an", "-c:v", "libx264", "-preset", "veryfast", str(output),
    ], check=True, capture_output=True)
    return str(output)


def _predict(inputs: list[str], text_frames: int,
             segment_type: str | None) -> str:
    input_segments = parse_input_list(inputs)
    processed = []
    reference_k = None
    segment_info: list[dict]
    for segment in input_segments:
        if segment["type"] == "video":
            work = Path(tempfile.mkdtemp(prefix="alphamotion-preprocess-"))
            item = preprocess_video_segment(
                segment["path"], str(work), str(HMR2_PATH), True)
            item["video_path"] = segment["path"]
            reference_k = item["K_fullimg"][0]
            processed.append(item)
        else:
            processed.append(segment)
    if reference_k is None:
        raise RuntimeError("AlphaMotion requires one reference video")
    for index, item in enumerate(processed):
        if item.get("type") == "text" and "K_fullimg" not in item:
            processed[index] = create_text_segment(
                item["caption"], text_frames, reference_k)
    data, segment_info = assemble_mixed_data(processed, True)
    prediction = run_inference(MODEL, data, static_cam=True)
    return _safe_artifact(prediction, segment_info, segment_type)


def _predict_text(prompt: str, text_frames: int) -> str:
    # Text generation only needs plausible camera intrinsics.  Do not prepend
    # a dummy video frame: compute_cam_angvel emits no sample for a one-frame
    # clip, which leaves camera features one frame shorter than every mask.
    width = height = 720
    _rotation, _angular, _velocity, intrinsics = get_camera_static(
        2, width, height)
    text = create_text_segment(prompt, text_frames, intrinsics[0])
    data, segment_info = assemble_mixed_data([text], True)
    prediction = run_inference(MODEL, data, static_cam=True)
    return _safe_artifact(prediction, segment_info, "text")


@spaces.GPU(duration=300)
def generate_text(prompt: str, frames: int = DEFAULT_TEXT_FRAMES) -> str:
    prompt = str(prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a motion prompt.")
    frames = max(30, int(frames or DEFAULT_TEXT_FRAMES))
    return _predict_text(prompt, frames)


@spaces.GPU(duration=300)
def generate_video(video_path: str, _requested_frames: int = 0) -> str:
    normalized = _normalized_video(video_path)
    return _predict([normalized], 60, None)


with gr.Blocks(title="AlphaMotion Generation") as demo:
    gr.Markdown(
        "# AlphaMotion generation worker\nPrivate ZeroGPU inference service. "
        "Use the main AlphaMotion Studio for the complete workflow.")
    with gr.Tab("Text → SMPL"):
        text = gr.Textbox(label="Motion prompt")
        frames = gr.Number(value=DEFAULT_TEXT_FRAMES, minimum=30, step=1,
                           precision=0,
                           label="Frames at 30 FPS (AlphaMotion default: 300)")
        text_output = gr.File(label="AlphaMotion NPZ")
        gr.Button("Generate").click(
            generate_text, [text, frames], text_output,
            api_name="generate_text")
    with gr.Tab("Video → SMPL"):
        video = gr.Video(label="Person video", format="mp4")
        video_output = gr.File(label="AlphaMotion NPZ")
        gr.Button("Estimate motion").click(
            generate_video, [video, gr.State(0)], video_output,
            api_name="generate_video")

demo.queue(default_concurrency_limit=1, max_size=6).launch()
