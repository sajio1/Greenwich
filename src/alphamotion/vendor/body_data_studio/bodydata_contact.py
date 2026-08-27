from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import numpy as np

from bodydata_config import CACHE_ROOT
from bodydata_decode import _get_smplh_model, materialize
from bodydata_index import connect


CONTACT_LABEL_VERSION = 4
CONTACT_ROOT = CACHE_ROOT / "contact_labels"
CONTACT_EFFECTORS = ("left_heel", "left_forefoot", "right_heel", "right_forefoot")
LEGACY_EFFECTORS = ("left_foot", "right_foot")
METHOD = "SMPL-H sole patches · ground/elevated-support hysteresis · temporal cleanup"


def _frame_speed(points: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    speed = np.zeros(values.shape[:-1], dtype=np.float64)
    if len(values) < 2:
        return speed
    speed[0] = np.linalg.norm(values[1] - values[0], axis=-1) * fps
    speed[-1] = np.linalg.norm(values[-1] - values[-2], axis=-1) * fps
    if len(values) > 2:
        speed[1:-1] = np.linalg.norm(values[2:] - values[:-2], axis=-1) * (fps * 0.5)
    return speed


def _fill_short_gaps(values: np.ndarray, maximum_gap: int) -> np.ndarray:
    output = np.asarray(values, dtype=bool).copy()
    start = 0
    while start < len(output):
        if output[start]:
            start += 1
            continue
        stop = start
        while stop < len(output) and not output[stop]:
            stop += 1
        if start > 0 and stop < len(output) and stop - start <= maximum_gap:
            output[start:stop] = True
        start = stop
    return output


def _remove_short_runs(values: np.ndarray, minimum_run: int) -> np.ndarray:
    output = np.asarray(values, dtype=bool).copy()
    start = 0
    while start < len(output):
        if not output[start]:
            start += 1
            continue
        stop = start
        while stop < len(output) and output[stop]:
            stop += 1
        if stop - start < minimum_run:
            output[start:stop] = False
        start = stop
    return output


def _segments(values: np.ndarray) -> list[dict]:
    result = []
    start = None
    for index, active in enumerate(np.asarray(values, dtype=bool)):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(values) - 1):
            stop = index if active and index == len(values) - 1 else index - 1
            result.append({"start_frame": int(start), "end_frame": int(stop)})
            start = None
    return result


def detect_foot_contacts(
    foot_positions: np.ndarray,
    fps: float,
    body_height: float,
    up_axis: int = 2,
    effector_names: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Detect ground and inferred elevated support from world-space foot regions."""
    feet = np.asarray(foot_positions, dtype=np.float64)
    if feet.ndim != 3 or feet.shape[2] != 3 or feet.shape[1] not in (2, 4):
        raise ValueError(f"foot_positions must have shape [frames,2,3] or [frames,4,3], got {feet.shape}")
    if len(feet) < 1 or not np.isfinite(feet).all():
        raise ValueError("Foot positions must be non-empty and finite")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite")
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    names = tuple(effector_names or (CONTACT_EFFECTORS if feet.shape[1] == 4 else LEGACY_EFFECTORS))
    if len(names) != feet.shape[1]:
        raise ValueError("effector_names must match the number of position channels")
    height_scale = max(float(body_height), 0.5)
    ground_height = float(np.percentile(feet[:, :, up_axis], 2.0))
    height_above_ground = feet[:, :, up_axis] - ground_height
    speed = _frame_speed(feet, fps)

    # Scale thresholds with the subject but keep conservative metric bounds.
    enter_height = float(np.clip(0.025 * height_scale, 0.025, 0.050))
    exit_height = float(np.clip(0.050 * height_scale, 0.050, 0.090))
    enter_speed = float(np.clip(0.14 * height_scale, 0.16, 0.30))
    exit_speed = float(np.clip(0.24 * height_scale, 0.28, 0.48))

    confidence_height = 1.0 / (1.0 + np.exp((height_above_ground - exit_height) / max(exit_height * 0.18, 1e-4)))
    confidence_speed = 1.0 / (1.0 + np.exp((speed - exit_speed) / max(exit_speed * 0.16, 1e-4)))
    ground_confidence = np.sqrt(confidence_height * confidence_speed)
    support_enter_speed = float(np.clip(0.070 * height_scale, 0.080, 0.160))
    support_exit_speed = float(np.clip(0.110 * height_scale, 0.120, 0.220))
    support_confidence = 1.0 / (
        1.0 + np.exp((speed - support_exit_speed) / max(support_exit_speed * 0.16, 1e-4))
    )

    # 0 = air, 1 = ground contact, 2 = elevated support inferred without a scene mesh.
    states = np.zeros((len(feet), feet.shape[1]), dtype=np.uint8)
    for foot in range(feet.shape[1]):
        ground_active = False
        for frame in range(len(feet)):
            if ground_active:
                ground_active = bool(height_above_ground[frame, foot] <= exit_height and speed[frame, foot] <= exit_speed)
            else:
                ground_active = bool(height_above_ground[frame, foot] <= enter_height and speed[frame, foot] <= enter_speed)
            states[frame, foot] = 1 if ground_active else 0

        support_active = False
        elevated = np.zeros(len(feet), dtype=bool)
        for frame in range(len(feet)):
            threshold = support_exit_speed if support_active else support_enter_speed
            support_active = bool(
                height_above_ground[frame, foot] > exit_height
                and speed[frame, foot] <= threshold
            )
            elevated[frame] = support_active
        if feet.shape[1] == 4:
            partner = foot + 1 if foot % 2 == 0 else foot - 1
            elevated &= feet[:, foot, up_axis] <= feet[:, partner, up_axis] + 0.020 * height_scale
        elevated = _fill_short_gaps(elevated, max(1, int(round(0.04 * fps))))
        elevated = _remove_short_runs(elevated, max(1, int(round(0.12 * fps))))
        states[elevated & (states[:, foot] == 0), foot] = 2

        cleaned = _fill_short_gaps(states[:, foot] > 0, max(1, int(round(0.04 * fps))))
        cleaned = _remove_short_runs(cleaned, max(1, int(round(0.06 * fps))))
        states[~cleaned, foot] = 0

    labels = states > 0
    elevated_mask = states == 2
    confidence = ground_confidence.astype(np.float32)
    confidence[elevated_mask] = np.minimum(0.85, 0.70 * support_confidence[elevated_mask] + 0.15)

    contact_speeds = speed[labels]
    minimum_height = float(np.min(height_above_ground))
    transitions = int(np.count_nonzero(np.diff(labels.astype(np.int8), axis=0))) if len(labels) > 1 else 0
    duration = max((len(labels) - 1) / fps, 1.0 / fps)
    transition_rate = transitions / duration
    contact_speed_p95 = float(np.percentile(contact_speeds, 95)) if len(contact_speeds) else None
    confident_rate = float(np.mean(confidence[labels] >= 0.65)) if np.any(labels) else 0.0
    checks = {
        "finite_kinematics": bool(np.isfinite(feet).all() and np.isfinite(confidence).all()),
        "binary_labels": bool(np.isin(labels, (False, True)).all()),
        "contact_speed_consistent": contact_speed_p95 is None or contact_speed_p95 <= exit_speed * 1.20,
        "ground_penetration_bounded": minimum_height >= -0.025 * height_scale,
        "transition_rate_plausible": transition_rate <= 12.0,
        "contact_confidence_acceptable": confident_rate >= 0.75 if np.any(labels) else True,
    }
    return {
        "labels": labels,
        "contact_type": states,
        "confidence": confidence.astype(np.float32),
        "speed_mps": speed.astype(np.float32),
        "height_above_ground_m": height_above_ground.astype(np.float32),
        "ground_height_m": ground_height,
        "up_axis": "XYZ"[up_axis],
        "thresholds": {
            "enter_height_m": enter_height,
            "exit_height_m": exit_height,
            "enter_speed_mps": enter_speed,
            "exit_speed_mps": exit_speed,
            "support_enter_speed_mps": support_enter_speed,
            "support_exit_speed_mps": support_exit_speed,
            "gap_fill_seconds": 0.04,
            "minimum_contact_seconds": 0.06,
            "minimum_elevated_support_seconds": 0.12,
        },
        "validation": {
            "passed": all(checks.values()),
            "checks": checks,
            "metrics": {
                "effector_contact_ratio": {name: float(np.mean(labels[:, index])) for index, name in enumerate(names)},
                "left_contact_ratio": float(np.mean(np.any(labels[:, : feet.shape[1] // 2], axis=1))),
                "right_contact_ratio": float(np.mean(np.any(labels[:, feet.shape[1] // 2 :], axis=1))),
                "double_support_ratio": float(np.mean(
                    np.any(labels[:, : feet.shape[1] // 2], axis=1)
                    & np.any(labels[:, feet.shape[1] // 2 :], axis=1)
                )),
                "flight_ratio": float(np.mean(~np.any(labels, axis=1))),
                "elevated_support_ratio": float(np.mean(elevated_mask)),
                "contact_speed_p95_mps": contact_speed_p95,
                "minimum_height_above_ground_m": minimum_height,
                "transitions_per_second": transition_rate,
                "confident_contact_ratio": confident_rate,
            },
        },
    }


def _sole_patch_indices(vertices: np.ndarray, joints: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build heel/forefoot sole patches from shaped SMPL-H rest geometry."""
    patches: list[np.ndarray] = []
    for ankle_index, forefoot_index in ((7, 10), (8, 11)):
        ankle, forefoot = joints[ankle_index], joints[forefoot_index]
        side = 1.0 if forefoot[0] >= 0 else -1.0
        min_z = min(ankle[2], forefoot[2]) - 0.10
        max_z = max(ankle[2], forefoot[2]) + 0.10
        candidates = np.flatnonzero(
            (vertices[:, 0] * side > 0.025)
            & (np.abs(vertices[:, 0] - forefoot[0]) < 0.12)
            & (vertices[:, 1] < min(ankle[1], forefoot[1]) + 0.025)
            & (vertices[:, 2] >= min_z)
            & (vertices[:, 2] <= max_z)
        )
        if len(candidates) < 16:
            raise ValueError("Could not identify a stable SMPL-H sole patch")
        split = float((ankle[2] + forefoot[2]) * 0.5)
        for region in (candidates[vertices[candidates, 2] <= split], candidates[vertices[candidates, 2] > split]):
            if len(region) < 6:
                raise ValueError("SMPL-H heel/forefoot patch is too small")
            # Retain the lowest vertices in rest pose; averaging a patch is more
            # stable than relying on one topology-specific landmark vertex.
            count = min(32, max(8, len(region) // 2))
            patches.append(region[np.argsort(vertices[region, 1])[:count]])
    return tuple(patches)


def _smplh_foot_kinematics(motion: dict[str, np.ndarray]) -> tuple[np.ndarray, float, tuple[np.ndarray, ...]]:
    import torch

    poses = np.asarray(motion["poses"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 156:
        raise ValueError(f"Expected SMPL-H poses [frames,156], got {poses.shape}")
    trans = np.asarray(motion.get("trans", np.zeros((len(poses), 3))), dtype=np.float32)
    betas = np.asarray(motion.get("betas", np.zeros(16)), dtype=np.float32).reshape(-1)
    gender = str(np.asarray(motion.get("gender", "neutral")).reshape(-1)[0])
    model = _get_smplh_model(gender, len(betas))
    beta_batch = betas[: model.num_betas]
    chunks = []
    with torch.no_grad():
        shaped = model(betas=torch.from_numpy(beta_batch[None]), return_verts=True)
        rest_joints = shaped.joints[0].detach().cpu().numpy()
        rest_vertices = shaped.vertices[0].detach().cpu().numpy()
        body_height = float(rest_joints[:, 1].max() - rest_joints[:, 1].min())
        patches = _sole_patch_indices(rest_vertices, rest_joints)
        for start in range(0, len(poses), 256):
            stop = min(start + 256, len(poses))
            batch_poses = torch.from_numpy(poses[start:stop])
            result = model(
                betas=torch.from_numpy(np.repeat(beta_batch[None], stop - start, axis=0)),
                global_orient=batch_poses[:, :3],
                body_pose=batch_poses[:, 3:66],
                left_hand_pose=batch_poses[:, 66:111],
                right_hand_pose=batch_poses[:, 111:156],
                transl=torch.from_numpy(trans[start:stop]),
                return_verts=True,
            )
            vertices = result.vertices
            region_positions = [vertices[:, indices].mean(dim=1) for indices in patches]
            chunks.append(torch.stack(region_positions, dim=1).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0), body_height, patches


def _read_motion(asset: dict) -> tuple[dict[str, np.ndarray], float]:
    source = materialize(asset)
    with np.load(source, allow_pickle=True) as archive:
        motion = {key: np.array(archive[key], copy=True) for key in archive.files}
    for key in ("mocap_framerate", "mocap_frame_rate"):
        if key in motion:
            fps = float(np.asarray(motion[key]).reshape(-1)[0])
            if np.isfinite(fps) and fps > 0:
                return motion, fps
    raise ValueError("AMASS source frame rate is missing; expected mocap_framerate or mocap_frame_rate")


def _label_paths(asset_id: str) -> tuple[Path, Path]:
    folder = CONTACT_ROOT / asset_id
    return folder / f"v{CONTACT_LABEL_VERSION}.npz", folder / f"v{CONTACT_LABEL_VERSION}.json"


def load_contact_labels(asset_id: str) -> dict | None:
    _, metadata_path = _label_paths(asset_id)
    candidates = [metadata_path]
    if not metadata_path.is_file():
        folder = CONTACT_ROOT / asset_id
        candidates.extend(sorted(
            (path for path in folder.glob("v*.json") if path.is_file()),
            key=lambda path: int(path.stem[1:]) if path.stem[1:].isdigit() else -1,
            reverse=True,
        ))
    for candidate in candidates:
        try:
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
    return None


def load_contact_preview(asset_id: str) -> dict | None:
    """Load the canonical sidecar with every channel needed by the viewer."""
    result = load_contact_labels(asset_id)
    if not result:
        return None
    sidecar = Path(result.get("sidecar_npz", ""))
    if not sidecar.is_file():
        return result
    output = dict(result)
    with np.load(sidecar, allow_pickle=False) as archive:
        stored_effectors = [str(value) for value in archive["effectors"].tolist()] if "effectors" in archive else list(output.get("available_effectors") or LEGACY_EFFECTORS)
        stored_map = {name: index for index, name in enumerate(stored_effectors)}

        def expanded(key: str, tail: tuple[int, ...] = (), dtype=np.float32) -> np.ndarray | None:
            if key not in archive:
                return None
            source = np.asarray(archive[key])
            if source.ndim >= 2 and source.shape[1] == len(CONTACT_EFFECTORS):
                return source.astype(dtype, copy=False)
            values = np.zeros((len(source), len(CONTACT_EFFECTORS), *tail), dtype=dtype)
            for column, name in enumerate(CONTACT_EFFECTORS):
                source_index = stored_map.get(name)
                if source_index is None:
                    source_index = stored_map.get("left_foot" if name.startswith("left_") else "right_foot")
                if source_index is not None:
                    values[:, column] = source[:, source_index]
            return values

        contact = expanded("contact", dtype=np.uint8)
        contact_type = expanded("contact_type", dtype=np.uint8)
        confidence = expanded("contact_confidence")
        position = expanded("foot_position_world_m", (3,))
        height = expanded("foot_height_above_ground_m")
        speed = expanded("foot_speed_mps")
        if contact is not None:
            output["contact"] = contact.tolist()
        if contact_type is not None:
            output["contact_type"] = contact_type.tolist()
        if confidence is not None:
            output["confidence"] = np.round(confidence.astype(np.float64), 4).tolist()
        if position is not None:
            output["position_world_m"] = np.round(position.astype(np.float64), 6).tolist()
        if height is not None:
            output["height_above_ground_m"] = np.round(height.astype(np.float64), 6).tolist()
        if speed is not None:
            output["speed_mps"] = np.round(speed.astype(np.float64), 6).tolist()
    output["legacy_compatibility"] = int(output.get("version", CONTACT_LABEL_VERSION)) < CONTACT_LABEL_VERSION
    output["selected_effectors"] = list(CONTACT_EFFECTORS)
    output["selected_channels"] = list(output.get("available_channels") or ["binary", "confidence", "segments"])
    return output


def _source_signature(asset: dict) -> dict:
    return {
        "locator_type": str(asset.get("locator_type", "")),
        "container": str(asset.get("container", "")),
        "inner_path": str(asset.get("inner_path", "")),
        "size": int(asset.get("size", 0)),
    }


def _cache_matches(asset: dict, cached: dict | None) -> bool:
    return bool(
        cached
        and int(cached.get("version", -1)) == CONTACT_LABEL_VERSION
        and cached.get("source_signature") == _source_signature(asset)
    )


def label_asset(asset: dict, force: bool = False) -> dict:
    npz_path, metadata_path = _label_paths(asset["id"])
    if not force:
        cached = load_contact_labels(asset["id"])
        if _cache_matches(asset, cached):
            return cached
    motion, fps = _read_motion(asset)
    feet, body_height, patches = _smplh_foot_kinematics(motion)
    detected = detect_foot_contacts(feet, fps, body_height, effector_names=CONTACT_EFFECTORS)
    labels = detected.pop("labels")
    contact_type = detected.pop("contact_type")
    confidence = detected.pop("confidence")
    speed = detected.pop("speed_mps")
    height = detected.pop("height_above_ground_m")
    inherited = np.asarray(motion.get("augmentation_contact_labels", []), dtype=np.uint8)
    inherited_type = np.asarray(motion.get("augmentation_contact_type", []), dtype=np.uint8)
    inherited_confidence = np.asarray(motion.get("augmentation_contact_confidence", []), dtype=np.float32)
    contact_source = "detected"
    if inherited.shape == labels.shape and np.isin(inherited, (0, 1)).all():
        labels = inherited.astype(bool)
        if inherited_type.shape == contact_type.shape:
            contact_type = inherited_type
        if inherited_confidence.shape == confidence.shape and np.isfinite(inherited_confidence).all():
            confidence = inherited_confidence
        contact_source = "inherited_from_shape_augmentation"
        metrics = detected.get("validation", {}).get("metrics", {})
        metrics["effector_contact_ratio"] = {
            name: float(np.mean(labels[:, index])) for index, name in enumerate(CONTACT_EFFECTORS)
        }
        metrics["left_contact_ratio"] = float(np.mean(np.any(labels[:, :2], axis=1)))
        metrics["right_contact_ratio"] = float(np.mean(np.any(labels[:, 2:], axis=1)))
        metrics["double_support_ratio"] = float(np.mean(np.any(labels[:, :2], axis=1) & np.any(labels[:, 2:], axis=1)))
        metrics["flight_ratio"] = float(np.mean(~np.any(labels, axis=1)))
        detected["geometric_detector_validation"] = detected.get("validation", {})
        inherited_checks = {
            "finite_kinematics": bool(np.isfinite(feet).all() and np.isfinite(confidence).all()),
            "binary_labels": bool(np.isin(labels, (False, True)).all()),
            "contact_phase_length_matches_motion": bool(len(labels) == len(feet)),
            "shape_augmentation_contact_phase_inherited": True,
        }
        detected["validation"] = {
            "passed": bool(all(inherited_checks.values())),
            "checks": inherited_checks,
            "metrics": metrics,
        }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        contact=labels.astype(np.uint8),
        contact_type=contact_type.astype(np.uint8),
        left_contact=np.any(labels[:, :2], axis=1).astype(np.uint8),
        right_contact=np.any(labels[:, 2:], axis=1).astype(np.uint8),
        contact_confidence=confidence,
        foot_position_world_m=feet.astype(np.float32),
        foot_speed_mps=speed,
        foot_height_above_ground_m=height,
        fps=np.asarray(fps, dtype=np.float64),
        method=np.asarray(METHOD),
        contact_source=np.asarray(contact_source),
        source_asset_id=np.asarray(asset["id"]),
        effectors=np.asarray(CONTACT_EFFECTORS),
    )
    segments = {name: _segments(labels[:, index]) for index, name in enumerate(CONTACT_EFFECTORS)}
    segments["left"] = _segments(np.any(labels[:, :2], axis=1))
    segments["right"] = _segments(np.any(labels[:, 2:], axis=1))
    result = {
        "version": CONTACT_LABEL_VERSION,
        "asset_id": asset["id"],
        "title": asset["title"],
        "source": asset["source"],
        "folder": asset.get("folder", ""),
        "frame_count": int(len(labels)),
        "fps": fps,
        "method": METHOD,
        "contact_source": contact_source,
        "created_at": time.time(),
        "storage_state": "cache_sidecar",
        "source_signature": _source_signature(asset),
        "sidecar_npz": str(npz_path),
        "contact": labels.astype(np.uint8).tolist(),
        "contact_type": contact_type.astype(np.uint8).tolist(),
        "confidence": np.round(confidence, 4).tolist(),
        "segments": segments,
        "available_effectors": list(CONTACT_EFFECTORS),
        "contact_modes": {"0": "air", "1": "ground", "2": "elevated_inferred"},
        "region_definition": "mean of stable SMPL-H sole vertex patches",
        "available_channels": ["binary", "confidence", "segments", "position", "height", "speed"],
        **detected,
    }
    metadata_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def label_assets(asset_ids: list[str], force: bool = False, options: dict | None = None) -> dict:
    unique_ids = list(dict.fromkeys(str(value) for value in asset_ids if value))
    if not unique_ids:
        raise ValueError("Select at least one SMPL-H motion asset")
    placeholders = ",".join("?" for _ in unique_ids)
    db = connect()
    rows = db.execute(
        f"SELECT * FROM assets WHERE id IN ({placeholders}) AND kind='smplh_motion' AND format='npz' AND animated=1",
        unique_ids,
    ).fetchall()
    db.close()
    assets = {row["id"]: dict(row) for row in rows}
    run_id = uuid.uuid4().hex[:20]
    completed, failures, cached_count = [], [], 0
    for asset_id in unique_ids:
        asset = assets.get(asset_id)
        if asset is None:
            failures.append({"asset_id": asset_id, "error": "Asset is not an indexed SMPL-H NPZ motion"})
            continue
        was_cached = _cache_matches(asset, load_contact_labels(asset_id)) and not force
        try:
            result = label_asset(asset, force=force)
            completed.append({
                "asset_id": asset_id,
                "title": asset["title"],
                "cached": was_cached,
                "validation_passed": result["validation"]["passed"],
                "contact_source": result.get("contact_source", "detected"),
                "left_segments": len(result["segments"]["left"]),
                "right_segments": len(result["segments"]["right"]),
                "effector_segments": {name: len(result["segments"][name]) for name in CONTACT_EFFECTORS},
            })
            cached_count += int(was_cached)
        except Exception as exc:
            failures.append({"asset_id": asset_id, "title": asset["title"], "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "run_id": run_id,
        "status": "ready" if completed else "failed",
        "method": METHOD,
        "requested": len(unique_ids),
        "completed": len(completed),
        "cached": cached_count,
        "passed": sum(bool(item["validation_passed"]) for item in completed),
        "failed": len(failures) + sum(not bool(item["validation_passed"]) for item in completed),
        "items": completed,
        "failures": failures,
        "storage_state": "cache_sidecar",
        "selection": options or {
            "effectors": list(CONTACT_EFFECTORS),
            "channels": ["binary", "confidence", "segments"],
            "scope": "original",
        },
    }
    run_path = CONTACT_ROOT / "runs" / f"{run_id}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
