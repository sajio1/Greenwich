from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import threading
import zipfile

import numpy as np


from bodydata_config import CACHE_ROOT, DATA_ROOT, SEVEN_ZIP


PREPARED_ROOT = CACHE_ROOT / "prepared"
MODEL_ROOT = CACHE_ROOT / "models"
PREVIEW_CACHE_VERSION = "v4"
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_SMPLH_MODEL = None
_SMPLH_MODEL_KEY = None


class PreviewError(RuntimeError):
    pass


def _lock_for(identity: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(identity, threading.Lock())


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)[-180:]


def _asset_dir(asset: dict) -> Path:
    # Preview files are derived data. Version the cache so decoder fixes never
    # reuse stale geometry while the original dataset remains untouched.
    result = PREPARED_ROOT / PREVIEW_CACHE_VERSION / asset["source"].replace("/", "_") / asset["id"]
    result.mkdir(parents=True, exist_ok=True)
    return result


def _copy_zip_member(archive: Path, inner: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _safe_name(PurePosixPath(inner).name)
    if target.exists() and target.stat().st_size > 0:
        return target
    with zipfile.ZipFile(archive) as zf:
        try:
            info = zf.getinfo(inner)
        except KeyError as exc:
            raise PreviewError(f"Archive member is missing: {inner}") from exc
        with zf.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    return target


def _copy_tar_member(archive: Path, inner: str, target_dir: Path) -> Path:
    target = target_dir / _safe_name(PurePosixPath(inner).name)
    if target.exists() and target.stat().st_size > 0:
        return target
    mode = "r|bz2" if archive.name.lower().endswith(".tar.bz2") else "r|gz"
    found = False
    with tarfile.open(archive, mode) as tf:
        for info in tf:
            if info.name != inner:
                continue
            src = tf.extractfile(info)
            if src is None:
                raise PreviewError(f"Archive member is not a regular file: {inner}")
            with src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            found = True
            break
    if not found:
        raise PreviewError(f"Archive member is missing: {inner}")
    return target


def _copy_7z_member(archive: Path, inner: str, target_dir: Path) -> Path:
    target = target_dir / _safe_name(PurePosixPath(inner.replace("\\", "/")).name)
    if target.exists() and target.stat().st_size > 0:
        return target
    if SEVEN_ZIP is None or not SEVEN_ZIP.exists():
        raise PreviewError("7-Zip is required but is not installed.")
    command = [str(SEVEN_ZIP), "e", str(archive), inner.replace("/", "\\"), f"-o{target_dir}", "-y"]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    if proc.returncode != 0 or not target.exists():
        raise PreviewError(f"7-Zip extraction failed: {proc.stdout[-800:]}")
    return target


def materialize(asset: dict) -> Path:
    locator = asset["locator_type"]
    container = Path(asset["container"])
    target_dir = _asset_dir(asset)
    if locator == "direct":
        return container
    if locator in {"zip_member", "zip_rig"}:
        return _copy_zip_member(container, asset["inner_path"], target_dir)
    if locator == "tar_member":
        return _copy_tar_member(container, asset["inner_path"], target_dir)
    if locator == "7z_rig":
        return _copy_7z_member(container, asset["inner_path"], target_dir)
    raise PreviewError(f"Unsupported locator: {locator}")


def _write_skeleton(
    path: Path,
    frames: np.ndarray,
    parents,
    names=None,
    fps: float = 30.0,
    metadata: dict | None = None,
) -> Path:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 2:
        frames = frames[None, ...]
    parents = [-1 if p is None else int(p) for p in list(parents)]
    joint_count = min(frames.shape[1], len(parents))
    frames = frames[:, :joint_count]
    parents = parents[:joint_count]
    if names is None:
        names = [f"joint_{i}" for i in range(joint_count)]
    else:
        names = [str(x) for x in list(names)[:joint_count]]
    payload = {
        "fps": float(fps or 30.0),
        "frame_count": int(frames.shape[0]),
        "joint_count": int(joint_count),
        "parents": parents,
        "names": names,
        "frames": np.round(frames, 6).reshape(frames.shape[0], -1).tolist(),
        "metadata": metadata or {},
    }
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
    return path


def _export_mesh(path: Path, vertices, faces, normals=None) -> Path:
    import trimesh

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    kwargs = {"vertices": vertices, "faces": faces, "process": False}
    if normals is not None and np.asarray(normals).shape == vertices.shape:
        kwargs["vertex_normals"] = np.asarray(normals, dtype=np.float32)
    mesh = trimesh.Trimesh(**kwargs)
    mesh.visual.face_colors = [116, 169, 255, 255]
    mesh.export(path, file_type="glb")
    return path


def _z_up_to_y_up(points: np.ndarray) -> np.ndarray:
    """Rotate right-handed Z-up coordinates into Three.js' Y-up space."""
    points = np.asarray(points, dtype=np.float32)
    converted = np.empty_like(points)
    converted[..., 0] = points[..., 0]
    converted[..., 1] = points[..., 2]
    converted[..., 2] = -points[..., 1]
    return converted


def _decode_rig_npz(asset: dict, npz_path: Path) -> dict:
    target_dir = _asset_dir(asset)
    mesh_path = target_dir / "mesh.glb"
    skeleton_path = target_dir / "skeleton.json.gz"
    if not mesh_path.exists() or not skeleton_path.exists():
        # These are downloaded research datasets and their object arrays hold parents/classes.
        with np.load(npz_path, allow_pickle=True) as data:
            required = {"vertices", "faces", "joints", "parents"}
            if not required.issubset(data.files):
                raise PreviewError(f"Rig NPZ is missing fields: {sorted(required - set(data.files))}")
            if not mesh_path.exists():
                _export_mesh(mesh_path, data["vertices"], data["faces"], data["vertex_normals"] if "vertex_normals" in data.files else None)
            if not skeleton_path.exists():
                _write_skeleton(
                    skeleton_path,
                    data["joints"],
                    data["parents"],
                    data["names"] if "names" in data.files else None,
                    metadata={"skin_weights": "present" if "skin" in data.files else "absent"},
                )
    return {
        "viewer": "model+skeleton",
        "model_format": "glb",
        "model_path": str(mesh_path),
        "skeleton_path": str(skeleton_path),
        "animated": False,
    }


def _ensure_smplh_models() -> Path:
    expected = MODEL_ROOT / "smplh" / "SMPLH_NEUTRAL.npz"
    if expected.exists():
        return MODEL_ROOT
    archive = DATA_ROOT / "smpl_smplh" / "smplh_300.zip"
    if not archive.exists():
        raise PreviewError("SMPL-H model archive is missing from D:\\body_data\\smpl_smplh.")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for gender in ("FEMALE", "MALE", "NEUTRAL"):
            _copy_zip_member(archive, f"smplh/SMPLH_{gender}.npz", MODEL_ROOT / "smplh")
    return MODEL_ROOT


def _get_smplh_model(gender: str, num_betas: int):
    global _SMPLH_MODEL, _SMPLH_MODEL_KEY
    import smplx

    gender = gender.lower()
    if gender not in {"male", "female", "neutral"}:
        gender = "neutral"
    num_betas = max(1, min(int(num_betas), 300))
    key = (gender, num_betas)
    if _SMPLH_MODEL is None or _SMPLH_MODEL_KEY != key:
        _SMPLH_MODEL = smplx.create(
            str(_ensure_smplh_models()),
            model_type="smplh",
            gender=gender,
            ext="npz",
            use_pca=False,
            num_betas=num_betas,
            batch_size=1,
        )
        _SMPLH_MODEL.eval()
        _SMPLH_MODEL_KEY = key
    return _SMPLH_MODEL


def _decode_smplh_motion(asset: dict, motion_path: Path) -> dict:
    import torch
    from bodydata_gltf import write_smplh_animation_glb

    target = _asset_dir(asset) / "smplh_preview.glb"
    metadata_path = _asset_dir(asset) / "smplh_preview.json"
    if target.exists() and target.stat().st_size > 0:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return {
            "viewer": "model",
            "model_format": "glb",
            "model_path": str(target),
            "animated": True,
            "preview_metadata": metadata,
        }
    with np.load(motion_path, allow_pickle=True) as data:
        if "poses" not in data.files:
            raise PreviewError("This NPZ has no SMPL-H 'poses' array.")
        poses = np.asarray(data["poses"], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] < 156:
            raise PreviewError(f"Expected SMPL-H poses [frames,156], got {poses.shape}.")
        trans = np.asarray(data["trans"], dtype=np.float32) if "trans" in data.files else np.zeros((len(poses), 3), np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1) if "betas" in data.files else np.zeros(16, np.float32)
        gender = str(np.asarray(data["gender"]).item()) if "gender" in data.files else "neutral"
        fps = 30.0
        for key in ("mocap_framerate", "mocap_frame_rate", "fps"):
            if key in data.files:
                fps = float(np.asarray(data[key]).item())
                break
    model = _get_smplh_model(gender, len(betas))
    parents = model.parents.detach().cpu().numpy()
    with torch.no_grad():
        shaped = model(
            betas=torch.from_numpy(betas[: model.num_betas][None].astype(np.float32)),
            return_verts=True,
        )
    metadata = write_smplh_animation_glb(
        target,
        vertices=shaped.vertices[0].detach().cpu().numpy(),
        faces=model.faces,
        rest_joints=shaped.joints[0, : len(parents)].detach().cpu().numpy(),
        parents=parents,
        weights=model.lbs_weights.detach().cpu().numpy(),
        poses=poses[:, : len(parents) * 3],
        translations=trans,
        source_fps=fps,
        max_preview_fps=30.0,
        title=asset["title"],
    )
    metadata.update({
        "gender": gender,
        "content": "SMPL-H skinned motion proxy",
        "accuracy": "linear blend skinning; pose corrective blend shapes omitted for fast review",
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "viewer": "model",
        "model_format": "glb",
        "model_path": str(target),
        "animated": True,
        "preview_metadata": metadata,
    }


def _decode_smplh_model(asset: dict) -> dict:
    import torch

    title = asset["title"].upper()
    gender = "female" if "FEMALE" in title else "male" if "MALE" in title else "neutral"
    target_dir = _asset_dir(asset)
    mesh_path = target_dir / "neutral.glb"
    skeleton_path = target_dir / "neutral.skeleton.json.gz"
    model = _get_smplh_model(gender, 16)
    if not mesh_path.exists() or not skeleton_path.exists():
        with torch.no_grad():
            result = model(return_verts=True)
        vertices = result.vertices[0].detach().cpu().numpy()
        joints = result.joints[0, : len(model.parents)].detach().cpu().numpy()
        if not mesh_path.exists():
            _export_mesh(mesh_path, vertices, model.faces)
        if not skeleton_path.exists():
            _write_skeleton(skeleton_path, joints, model.parents.detach().cpu().numpy(), fps=30)
    return {
        "viewer": "model+skeleton",
        "model_format": "glb",
        "model_path": str(mesh_path),
        "skeleton_path": str(skeleton_path),
        "animated": False,
    }


def _decode_articulation_record(asset: dict) -> dict:
    target_dir = _asset_dir(asset)
    mesh_path = target_dir / "mesh.glb"
    skeleton_path = target_dir / "skeleton.json.gz"
    if not mesh_path.exists() or not skeleton_path.exists():
        index = int(asset["inner_path"])
        # Articulation-XL stores a pickled object array. Loading the containing
        # shard is unavoidable upstream; the derived asset is cached immediately.
        with np.load(asset["container"], allow_pickle=True) as source:
            records = source["arr_0"]
            if index < 0 or index >= len(records):
                raise PreviewError(f"Articulation record index is out of range: {index}")
            item = records[index]
            vertices = np.asarray(item["vertices"])
            faces = np.asarray(item["faces"])
            joints = np.asarray(item["joints"])
            bones = np.asarray(item["bones"], dtype=np.int64)
            names = item.get("joint_names")
            normals = item.get("normals")
            uuid = str(item.get("uuid", asset["title"]))
        parents = [-1] * len(joints)
        for parent, child in bones:
            if 0 <= child < len(parents) and 0 <= parent < len(parents):
                parents[int(child)] = int(parent)
        if not mesh_path.exists():
            _export_mesh(mesh_path, vertices, faces, normals)
        if not skeleton_path.exists():
            _write_skeleton(skeleton_path, joints, parents, names, metadata={"uuid": uuid})
    return {
        "viewer": "model+skeleton",
        "model_format": "glb",
        "model_path": str(mesh_path),
        "skeleton_path": str(skeleton_path),
        "animated": False,
    }


def _decode_anymate_record(asset: dict) -> dict:
    import torch

    target_dir = _asset_dir(asset)
    mesh_path = target_dir / "mesh.glb"
    skeleton_path = target_dir / "skeleton.json.gz"
    schema_path = target_dir / "anymate_schema.json"
    if not mesh_path.exists() or not skeleton_path.exists() or not schema_path.exists():
        records = torch.load(asset["container"], map_location="cpu", weights_only=True, mmap=True)
        index = int(asset["inner_path"])
        if index < 0 or index >= len(records):
            raise PreviewError(f"Anymate record index is out of range: {index}")
        item = records[index]
        schema = []
        for name, value in item.items():
            if torch.is_tensor(value):
                schema.append({
                    "name": str(name),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "elements": int(value.numel()),
                    "bytes": int(value.numel() * value.element_size()),
                })
            else:
                schema.append({"name": str(name), "shape": [], "dtype": type(value).__name__, "sample": [str(value)[:160]]})
        mesh_pc = item["mesh_pc"].detach().cpu().numpy()
        faces = item["mesh_face"].detach().cpu().numpy()
        joints = item["joints"].detach().cpu().numpy()
        parents = item["conns"].detach().cpu().numpy().astype(np.int64)
        parents = [-1 if int(parent) == i else int(parent) for i, parent in enumerate(parents)]
        if not mesh_path.exists():
            normals = mesh_pc[:, 3:6] if mesh_pc.shape[1] >= 6 else None
            # Anymate stores the triangle mesh in a unit-height display frame
            # (Y in 0..1), while pc/joints/bones use the training frame centered
            # around the origin (Y in -1..1). The published normalization is an
            # exact scale-and-translate, not an asset-specific camera correction.
            vertices = np.asarray(mesh_pc[:, :3], dtype=np.float32) * 2.0
            vertices[:, 1] -= 1.0
            _export_mesh(mesh_path, vertices, faces, normals)
        if not skeleton_path.exists():
            _write_skeleton(
                skeleton_path,
                joints,
                parents,
                metadata={
                    "name": str(item.get("name", asset["title"])),
                    "content": "static_rig",
                    "mesh_normalization": "mesh_xyz * 2; mesh_y -= 1",
                },
            )
        if not schema_path.exists():
            schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        del records
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        schema = []
    return {
        "viewer": "model+skeleton",
        "model_format": "glb",
        "model_path": str(mesh_path),
        "skeleton_path": str(skeleton_path),
        "animated": False,
        "data_schema": schema,
        "edit_policy": {
            "source_storage": "PyTorch list shard; one record is not an independent file",
            "read": "memory-mapped record access",
            "non_destructive": ["display rotation", "display scale", "axis conversion", "mirror recipe", "left/right semantic recipe"],
            "derived_copy_required": ["mesh vertices", "mesh topology", "joint positions", "parent hierarchy", "skinning indices", "skinning weights"],
            "in_place_source_edit": False,
            "reason": "Changing one record in place would require serializing the complete multi-gigabyte shard.",
        },
    }


def _prepare_split_obj(asset: dict) -> dict:
    target_dir = _asset_dir(asset)
    archive = Path(asset["container"])
    inner = asset["inner_path"].replace("/", "\\")
    folder = str(Path(inner).parent)
    marker = target_dir / ".complete"
    if not marker.exists():
        if SEVEN_ZIP is None or not SEVEN_ZIP.exists():
            raise PreviewError("7-Zip is required to preview this split archive. Install 7-Zip and restart Body Data Studio.")
        proc = subprocess.run(
            [str(SEVEN_ZIP), "e", str(archive), f"{folder}\\*", f"-o{target_dir}", "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise PreviewError(f"Split ZIP extraction failed: {proc.stdout[-800:]}")
        marker.write_text("ok", encoding="ascii")
    obj_path = target_dir / PurePosixPath(asset["inner_path"]).name
    if not obj_path.exists():
        raise PreviewError("The OBJ member was not extracted from the split ZIP.")
    mtl_candidate = obj_path.with_suffix(".mtl")
    mtl = mtl_candidate if mtl_candidate.exists() else next(target_dir.glob("*.mtl"), None)
    return {
        "viewer": "model",
        "model_format": "obj",
        "model_path": str(obj_path),
        "mtl_path": str(mtl) if mtl else "",
        "resource_root_path": str(target_dir),
        "animated": False,
    }


def prepare(asset: dict) -> dict:
    with _lock_for(asset["id"]):
        kind = asset["kind"]
        if asset["status"] != "ready":
            note = json.loads(asset.get("metadata_json") or "{}").get("note", "")
            raise PreviewError(f"This asset is indexed but not yet previewable ({asset['status']}). {note}".strip())
        if asset["locator_type"] == "split_zip_group":
            return _prepare_split_obj(asset)
        if kind == "rig_npz":
            return _decode_rig_npz(asset, materialize(asset))
        if kind == "smplh_motion":
            return _decode_smplh_motion(asset, materialize(asset))
        if kind == "smplh_model":
            return _decode_smplh_model(asset)
        if kind == "articulation_record":
            return _decode_articulation_record(asset)
        if kind == "anymate_record":
            return _decode_anymate_record(asset)
        path = materialize(asset)
        fmt = asset["format"].lower().lstrip(".")
        if fmt == "blend":
            raise PreviewError(".blend preview needs Blender. Install Blender 4.5 LTS from blender.org, then rescan.")
        if fmt not in {"glb", "gltf", "fbx", "dae", "obj", "ply", "bvh"}:
            raise PreviewError(f"No interactive viewer is registered for .{fmt}.")
        return {
            "viewer": "model" if fmt != "bvh" else "bvh",
            "model_format": fmt,
            "model_path": str(path),
            "resource_root_path": str(path.parent),
            "animated": bool(asset["animated"]),
        }
