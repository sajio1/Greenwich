from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from bodydata_decode import materialize


MAX_STAT_SAMPLES = 200_000


def _safe_scalar(value):
    try:
        item = np.asarray(value).reshape(-1)[0]
        return item.item() if hasattr(item, "item") else item
    except Exception:
        return None


def _array_summary(name: str, array: np.ndarray) -> dict:
    value = np.asarray(array)
    result = {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "elements": int(value.size),
        "bytes": int(value.nbytes),
    }
    if value.dtype.kind in "biufc" and value.size:
        flat = value.reshape(-1)
        if len(flat) > MAX_STAT_SAMPLES:
            flat = flat[:: max(1, len(flat) // MAX_STAT_SAMPLES)][:MAX_STAT_SAMPLES]
            result["stats_sampled"] = True
        finite = np.isfinite(flat)
        result["nan_count"] = int(np.count_nonzero(np.isnan(flat))) if value.dtype.kind in "fc" else 0
        result["inf_count"] = int(np.count_nonzero(np.isinf(flat))) if value.dtype.kind in "fc" else 0
        if np.any(finite):
            valid = np.real(flat[finite]).astype(np.float64, copy=False)
            result.update({
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
            })
        sample = flat[: min(8, len(flat))]
        result["sample"] = [float(np.real(item)) for item in sample]
    elif value.size == 1 and value.dtype.kind not in "OV":
        result["sample"] = [str(_safe_scalar(value))]
    return result


def _npz_structure(path: Path) -> tuple[list[dict], dict, list[dict]]:
    arrays: list[dict] = []
    motion: dict = {}
    warnings: list[dict] = []
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            try:
                arrays.append(_array_summary(name, archive[name]))
            except ValueError as exc:
                arrays.append({"name": name, "dtype": "object/pickled", "shape": [], "unavailable": str(exc)})
        fps = None
        fps_field = ""
        for name in ("mocap_framerate", "mocap_frame_rate", "fps"):
            if name in archive.files:
                try:
                    fps = float(_safe_scalar(archive[name]))
                    fps_field = name
                    break
                except (TypeError, ValueError):
                    pass
        poses = None
        if "poses" in archive.files:
            try:
                poses = np.asarray(archive["poses"])
            except ValueError:
                pass
        trans = None
        if "trans" in archive.files:
            try:
                trans = np.asarray(archive["trans"])
            except ValueError:
                pass
        frame_count = int(len(poses)) if poses is not None and poses.ndim else int(len(trans)) if trans is not None and trans.ndim else 0
        motion.update({
            "frame_count": frame_count,
            "fps": fps,
            "fps_field": fps_field,
            "duration_seconds": (frame_count - 1) / fps if frame_count > 1 and fps and fps > 0 else None,
            "pose_dimensions": int(poses.shape[1]) if poses is not None and poses.ndim == 2 else None,
        })
        if frame_count and not fps:
            warnings.append({"severity": "error", "code": "missing_fps", "message": "Motion FPS is missing; playback timing cannot be trusted."})
        if poses is not None:
            if not np.all(np.isfinite(poses)):
                warnings.append({"severity": "error", "code": "non_finite_pose", "message": "Pose data contains NaN or Inf values."})
            if len(poses) > 1:
                delta = np.diff(poses.astype(np.float64), axis=0)
                duplicate_ratio = float(np.mean(np.max(np.abs(delta), axis=1) < 1e-10))
                motion["duplicate_frame_ratio"] = duplicate_ratio
                if duplicate_ratio > 0.1:
                    warnings.append({"severity": "warning", "code": "duplicate_frames", "message": f"{duplicate_ratio:.1%} of adjacent pose frames are identical."})
                if poses.ndim == 2 and poses.shape[1] >= 3:
                    try:
                        from scipy.spatial.transform import Rotation
                        joint_count = poses.shape[1] // 3
                        rotations = Rotation.from_rotvec(poses[:, : joint_count * 3].reshape(-1, 3))
                        matrices = rotations.as_matrix().reshape(len(poses), joint_count, 3, 3)
                        relative = np.einsum("fjki,fjkl->fjil", matrices[:-1], matrices[1:])
                        steps_deg = np.degrees(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(len(poses) - 1, joint_count)
                        per_frame = np.max(steps_deg, axis=1)
                        motion["rotation_step_p99_deg"] = float(np.percentile(per_frame, 99))
                        motion["rotation_step_max_deg"] = float(np.max(per_frame))
                        spike_frames = (np.flatnonzero(per_frame > 60.0) + 1).astype(int).tolist()
                        if spike_frames:
                            warnings.append({"severity": "warning", "code": "rotation_spike", "message": f"{len(spike_frames)} frames contain a joint rotation step above 60°.", "frames": spike_frames[:500]})
                    except Exception:
                        pass
        if trans is not None:
            if not np.all(np.isfinite(trans)):
                warnings.append({"severity": "error", "code": "non_finite_translation", "message": "Root translation contains NaN or Inf values."})
            if len(trans) > 1:
                steps = np.linalg.norm(np.diff(trans.astype(np.float64), axis=0), axis=-1)
                motion["root_step_p99_m"] = float(np.percentile(steps, 99))
                motion["root_step_max_m"] = float(np.max(steps))
                median = float(np.median(steps))
                mad = float(np.median(np.abs(steps - median)))
                jump_limit = max(0.25, median + 10.0 * max(mad, 1e-8))
                jump_frames = (np.flatnonzero(steps > jump_limit) + 1).astype(int).tolist()
                if jump_frames:
                    warnings.append({"severity": "warning", "code": "root_jump", "message": f"{len(jump_frames)} frames exceed the robust root-motion jump threshold ({jump_limit:.3f} m/frame).", "frames": jump_frames[:500]})
    return arrays, motion, warnings


def inspect_asset(asset: dict, preview: dict | None = None, recipe: dict | None = None) -> dict:
    container = Path(asset.get("container", ""))
    stat = None
    try:
        stat = container.stat()
    except OSError:
        pass
    structure: list[dict] = []
    motion: dict = {}
    warnings: list[dict] = []
    inspection_error = ""
    if str(asset.get("format", "")).lower().lstrip(".") == "npz":
        try:
            structure, motion, warnings = _npz_structure(materialize(asset))
        except Exception as exc:
            inspection_error = f"{type(exc).__name__}: {exc}"
    elif asset.get("kind") == "anymate_record" and (preview or {}).get("data_schema"):
        structure = list(preview.get("data_schema") or [])
        warnings.append({
            "severity": "info",
            "code": "static_rig",
            "message": "This Anymate record contains a rest mesh, skeleton hierarchy, and skinning weights, but no animation frames.",
        })
    metadata = dict(asset.get("metadata") or {})
    preview_metadata = dict((preview or {}).get("preview_metadata") or {})
    if preview_metadata.get("quality_warning"):
        warnings.append({"severity": "warning", "code": "rotation_spikes", "message": str(preview_metadata["quality_warning"])})
    source_coord = {
        "up_axis": "Z" if asset.get("kind") == "smplh_motion" else "unknown",
        "forward_axis": "unknown",
        "handedness": "right" if asset.get("kind") == "smplh_motion" else "unknown",
        "unit": "meters" if asset.get("kind") in {"smplh_motion", "smplh_model"} else "unknown",
    }
    conversion = preview_metadata.get("coordinate_conversion") or ("(x, z, -y) · source Z-up to viewer Y-up" if asset.get("kind") == "smplh_motion" else "No decoder-declared conversion")
    return {
        "generated_at": time.time(),
        "summary": {
            "asset_id": asset.get("id"),
            "title": asset.get("title"),
            "source": asset.get("source"),
            "folder": asset.get("folder"),
            "format": str(asset.get("format", "")).upper(),
            "kind": asset.get("kind"),
            "animated": bool(asset.get("animated")),
            "size": int(asset.get("size", 0)),
            "source_path": str(container),
            "archive_member": asset.get("inner_path", ""),
            "locator": asset.get("locator_type", ""),
            "modified_at": stat.st_mtime if stat else None,
            "source_exists": bool(stat),
        },
        "structure": structure,
        "motion": {**motion, **preview_metadata},
        "coordinates": {
            "source": source_coord,
            "viewer": {"up_axis": "Y", "handedness": "right", "unit": source_coord["unit"]},
            "decoder_conversion": conversion,
            "recipe": recipe or {},
        },
        "provenance": {
            "source_read_only": True,
            "preview_cached": bool(preview),
            "preview_decoder": "Body Data Studio",
            "preview_cache_version": "v4",
            "metadata": metadata,
            "edit_policy": dict((preview or {}).get("edit_policy") or {}),
        },
        "warnings": warnings,
        "inspection_error": inspection_error,
    }
