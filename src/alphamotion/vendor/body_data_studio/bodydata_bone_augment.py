from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from bodydata_augment import AUGMENT_ROOT, _copy_motion, _safe_name, read_source_fps
from bodydata_contact import CONTACT_EFFECTORS, _smplh_foot_kinematics, detect_foot_contacts
from bodydata_decode import _get_smplh_model, materialize
from bodydata_index import connect, row


AUGMENTATION_TYPE = "bone"
METHOD = "SMPL-H shape-manifold bone length sampling + contact-anchored root correction"
DEFAULT_VARIANTS = 1
MAX_VARIANTS = 8
LEG_SCALE_RANGE = (0.94, 1.06)
ARM_SCALE_RANGE = (0.94, 1.06)
TORSO_SCALE_RANGE = (0.97, 1.03)
MAX_BETA_ABS = 3.0


def _shape_metrics(joints: np.ndarray) -> np.ndarray:
    """Return symmetric leg, arm, and torso lengths for one or more SMPL bodies."""
    values = np.asarray(joints, dtype=np.float64)
    leg_left = np.linalg.norm(values[..., 1, :] - values[..., 4, :], axis=-1) + np.linalg.norm(
        values[..., 4, :] - values[..., 7, :], axis=-1
    )
    leg_right = np.linalg.norm(values[..., 2, :] - values[..., 5, :], axis=-1) + np.linalg.norm(
        values[..., 5, :] - values[..., 8, :], axis=-1
    )
    arm_left = np.linalg.norm(values[..., 16, :] - values[..., 18, :], axis=-1) + np.linalg.norm(
        values[..., 18, :] - values[..., 20, :], axis=-1
    )
    arm_right = np.linalg.norm(values[..., 17, :] - values[..., 19, :], axis=-1) + np.linalg.norm(
        values[..., 19, :] - values[..., 21, :], axis=-1
    )
    torso = np.linalg.norm(values[..., 0, :] - values[..., 12, :], axis=-1)
    return np.stack(((leg_left + leg_right) * 0.5, (arm_left + arm_right) * 0.5, torso), axis=-1)


def _rest_joints(model, beta_batch: np.ndarray) -> np.ndarray:
    import torch

    betas = torch.from_numpy(np.asarray(beta_batch, dtype=np.float32))
    batch = len(betas)
    zeros3 = torch.zeros((batch, 3), dtype=betas.dtype)
    with torch.no_grad():
        result = model(
            betas=betas,
            global_orient=zeros3,
            body_pose=torch.zeros((batch, 63), dtype=betas.dtype),
            left_hand_pose=torch.zeros((batch, 45), dtype=betas.dtype),
            right_hand_pose=torch.zeros((batch, 45), dtype=betas.dtype),
            transl=zeros3,
            return_verts=False,
        )
    return result.joints[:, :22].detach().cpu().numpy()


def _sample_shaped_betas(
    source_betas: np.ndarray,
    gender: str,
    random_seed: int,
    leg_range: tuple[float, float] = LEG_SCALE_RANGE,
    arm_range: tuple[float, float] = ARM_SCALE_RANGE,
    torso_range: tuple[float, float] = TORSO_SCALE_RANGE,
) -> tuple[np.ndarray, dict]:
    """Move within the learned SMPL-H shape space toward sampled segment ratios."""
    source = np.asarray(source_betas, dtype=np.float32).reshape(-1)
    model = _get_smplh_model(gender, len(source))
    dimensions = min(16, model.num_betas, len(source))
    base = source[: model.num_betas].copy()
    base_joints = _rest_joints(model, base[None])[0]
    base_metrics = _shape_metrics(base_joints)

    # A local finite-difference Jacobian maps beta changes to log segment-length
    # changes. Ridge regularization keeps the solution close to the source body.
    epsilon = 0.35
    probes = np.repeat(base[None], dimensions * 2, axis=0)
    for index in range(dimensions):
        probes[index * 2, index] += epsilon
        probes[index * 2 + 1, index] -= epsilon
    probe_metrics = _shape_metrics(_rest_joints(model, probes))
    jacobian = np.empty((3, dimensions), dtype=np.float64)
    for index in range(dimensions):
        positive = np.maximum(probe_metrics[index * 2], 1e-8)
        negative = np.maximum(probe_metrics[index * 2 + 1], 1e-8)
        jacobian[:, index] = (np.log(positive) - np.log(negative)) / (2.0 * epsilon)

    rng = np.random.default_rng(int(random_seed))
    requested = np.asarray((rng.uniform(*leg_range), rng.uniform(*arm_range), rng.uniform(*torso_range)))
    target = np.log(requested)
    # The three measurements are well-conditioned enough for a light ridge.
    # A stronger penalty barely changes visible proportions, defeating the
    # purpose of this augmentation.
    ridge = 1e-4
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + ridge * np.eye(3), target)
    delta = np.clip(delta, -1.35, 1.35)
    candidate = base.copy()
    candidate[:dimensions] = np.clip(candidate[:dimensions] + delta, -MAX_BETA_ABS, MAX_BETA_ABS)
    actual_metrics = _shape_metrics(_rest_joints(model, candidate[None])[0])
    actual = actual_metrics / np.maximum(base_metrics, 1e-8)

    output = source.copy()
    output[: model.num_betas] = candidate
    return output, {
        "requested_scales": {"legs": float(requested[0]), "arms": float(requested[1]), "torso": float(requested[2])},
        "actual_scales": {"legs": float(actual[0]), "arms": float(actual[1]), "torso": float(actual[2])},
        "source_lengths_m": {"legs": float(base_metrics[0]), "arms": float(base_metrics[1]), "torso": float(base_metrics[2])},
        "output_lengths_m": {"legs": float(actual_metrics[0]), "arms": float(actual_metrics[1]), "torso": float(actual_metrics[2])},
        "optimized_beta_dimensions": int(dimensions),
        "beta_delta_l2": float(np.linalg.norm(candidate - base)),
    }


def _contact_root_correction(
    source_feet: np.ndarray,
    output_feet: np.ndarray,
    source_contact: np.ndarray,
) -> np.ndarray:
    frames = len(source_feet)
    correction = np.zeros((frames, 3), dtype=np.float64)
    anchored = np.any(source_contact, axis=1)
    differences = np.asarray(source_feet, dtype=np.float64) - np.asarray(output_feet, dtype=np.float64)
    for frame in np.flatnonzero(anchored):
        correction[frame] = np.mean(differences[frame, source_contact[frame]], axis=0)
    anchors = np.flatnonzero(anchored)
    if len(anchors):
        timeline = np.arange(frames)
        for axis in range(3):
            correction[:, axis] = np.interp(timeline, anchors, correction[anchors, axis])
    else:
        # Airborne-only clips still get a constant vertical alignment so their
        # world-space floor reference does not change with the sampled body.
        correction[:, 2] = float(np.percentile(source_feet[..., 2], 2) - np.percentile(output_feet[..., 2], 2))
    return correction


def randomize_bone_lengths(
    original: dict[str, np.ndarray],
    random_seed: int = 0,
    leg_range: tuple[float, float] = LEG_SCALE_RANGE,
    arm_range: tuple[float, float] = ARM_SCALE_RANGE,
    torso_range: tuple[float, float] = TORSO_SCALE_RANGE,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    poses = np.asarray(original.get("poses"))
    if poses.ndim != 2 or poses.shape[1] < 156 or len(poses) < 1:
        raise ValueError(f"Random bone length requires SMPL-H poses [frames,156], got {poses.shape}")
    fps = read_source_fps(original)
    source_betas = np.asarray(original.get("betas", np.zeros(16)), dtype=np.float32).reshape(-1)
    gender = str(np.asarray(original.get("gender", "neutral")).reshape(-1)[0])
    sampled_betas, shape = _sample_shaped_betas(source_betas, gender, random_seed, leg_range, arm_range, torso_range)

    output = _copy_motion(original)
    output["betas"] = sampled_betas.reshape(np.asarray(original.get("betas", source_betas)).shape).astype(
        np.asarray(original.get("betas", source_betas)).dtype, copy=False
    )
    source_feet, source_height, _ = _smplh_foot_kinematics(original)
    raw_output_feet, output_height, _ = _smplh_foot_kinematics(output)
    source_detection = detect_foot_contacts(source_feet, fps, source_height, effector_names=CONTACT_EFFECTORS)
    source_contact = np.asarray(source_detection["labels"], dtype=bool)
    correction = _contact_root_correction(source_feet, raw_output_feet, source_contact)
    source_trans = np.asarray(original.get("trans", np.zeros((len(poses), 3))), dtype=np.float64)
    output["trans"] = (source_trans + correction).astype(source_trans.dtype, copy=False)
    corrected_feet = raw_output_feet + correction[:, None, :]
    output_detection = detect_foot_contacts(corrected_feet, fps, output_height, effector_names=CONTACT_EFFECTORS)
    output_contact = np.asarray(output_detection["labels"], dtype=bool)
    # Shape-only augmentation does not change the action timeline or joint
    # rotations, so its contact phases are inherited exactly. The geometric
    # detector is still run below as an independent sanity check.
    output["augmentation_contact_labels"] = source_contact.astype(np.uint8)
    output["augmentation_contact_type"] = np.asarray(source_detection["contact_type"], dtype=np.uint8)
    output["augmentation_contact_confidence"] = np.asarray(source_detection["confidence"], dtype=np.float32)
    output["augmentation_contact_effectors"] = np.asarray(CONTACT_EFFECTORS)

    true_positive = int(np.count_nonzero(source_contact & output_contact))
    source_positive = int(np.count_nonzero(source_contact))
    output_positive = int(np.count_nonzero(output_contact))
    recall = true_positive / source_positive if source_positive else 1.0
    precision = true_positive / output_positive if output_positive else 1.0
    agreement = float(np.mean(source_contact == output_contact))
    active_residual = np.linalg.norm(corrected_feet - source_feet, axis=-1)[source_contact]
    residual_p95 = float(np.percentile(active_residual, 95)) if len(active_residual) else 0.0
    checks = {
        "finite_numeric_values": bool(all(np.isfinite(np.asarray(value)).all() for value in output.values() if np.issubdtype(np.asarray(value).dtype, np.number))),
        "poses_unchanged": bool(np.array_equal(output["poses"], original["poses"])),
        "frame_count_unchanged": bool(len(output["poses"]) == len(original["poses"])),
        "source_fps_preserved": bool(read_source_fps(output) == fps),
        "betas_constant_over_clip": bool(np.asarray(output["betas"]).ndim == 1 or (np.asarray(output["betas"]).ndim == 2 and np.asarray(output["betas"]).shape[0] == 1)),
        "human_shape_bounds_respected": bool(np.max(np.abs(np.asarray(output["betas"], dtype=np.float64))) <= MAX_BETA_ABS + 1e-6),
        "requested_proportions_reached": bool(
            leg_range[0] - 0.01 <= shape["actual_scales"]["legs"] <= leg_range[1] + 0.01
            and arm_range[0] - 0.01 <= shape["actual_scales"]["arms"] <= arm_range[1] + 0.01
            and torso_range[0] - 0.01 <= shape["actual_scales"]["torso"] <= torso_range[1] + 0.01
        ),
        "source_contacts_preserved": bool(np.array_equal(output["augmentation_contact_labels"], source_contact)),
        "contact_geometry_consistent": bool(residual_p95 <= 0.02),
    }
    validation = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": {
            "source_frames": int(len(poses)),
            "output_frames": int(len(poses)),
            "source_fps": float(fps),
            "output_fps": float(fps),
            "leg_scale": shape["actual_scales"]["legs"],
            "arm_scale": shape["actual_scales"]["arms"],
            "torso_scale": shape["actual_scales"]["torso"],
            "contact_recall": float(recall),
            "semantic_contact_recall": 1.0,
            "contact_precision": float(precision),
            "contact_agreement": agreement,
            "contact_anchor_residual_p95_m": residual_p95,
            "root_correction_max_m": float(np.max(np.linalg.norm(correction, axis=1))),
        },
        "warnings": [],
    }
    provenance = {
        "augmentation_type": AUGMENTATION_TYPE,
        "method": METHOD,
        "random_seed": int(random_seed),
        "source_fps": float(fps),
        "output_fps": float(fps),
        "shape": shape,
        "ranges": {"legs": list(leg_range), "arms": list(arm_range), "torso": list(torso_range)},
        "contact_correction": "per-frame least-squares root translation over source contact sole patches",
    }
    return output, provenance, validation


def bone_augmented_filename(original_name: str, variant_index: int) -> str:
    return f"augmented__{_safe_name(Path(original_name).stem)}__bl{variant_index + 1:02d}.npz"


def _variant_seed(random_seed: int, asset_id: str, variant_index: int) -> int:
    digest = hashlib.sha256(f"{int(random_seed)}:{asset_id}:{variant_index}".encode()).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _ensure_output_table(db) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS augmentation_bone_outputs(
            run_id TEXT NOT NULL,
            original_asset_id TEXT NOT NULL,
            variant_index INTEGER NOT NULL,
            augmented_asset_id TEXT,
            validation_json TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(run_id, original_asset_id, variant_index)
        )"""
    )


def augment_bone_assets(asset_ids: list[str], variants: int = DEFAULT_VARIANTS, random_seed: int = 0) -> dict:
    unique_ids = list(dict.fromkeys(str(value) for value in asset_ids if value))
    if not unique_ids:
        raise ValueError("Select at least one motion asset")
    variants = int(variants)
    if variants < 1 or variants > MAX_VARIANTS:
        raise ValueError(f"Random bone length variants must be between 1 and {MAX_VARIANTS}")
    run_id = uuid.uuid4().hex[:20]
    created_at = time.time()
    output_dir = AUGMENT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    params = {
        "type": AUGMENTATION_TYPE,
        "method": METHOD,
        "variants": variants,
        "leg_scale_range": list(LEG_SCALE_RANGE),
        "arm_scale_range": list(ARM_SCALE_RANGE),
        "torso_scale_range": list(TORSO_SCALE_RANGE),
        "random_seed": int(random_seed),
        "source_fps_preserved": True,
    }
    db = connect()
    with db:
        _ensure_output_table(db)
        db.execute(
            "INSERT INTO augmentation_runs(id,augmentation_type,params_json,status,created_at) VALUES(?,?,?,?,?)",
            (run_id, AUGMENTATION_TYPE, json.dumps(params, ensure_ascii=False), "building", created_at),
        )
    placeholders = ",".join("?" for _ in unique_ids)
    rows = db.execute(f"SELECT * FROM assets WHERE id IN ({placeholders})", unique_ids).fetchall()
    assets = {item["id"]: dict(item) for item in rows}
    pairs, failures = [], []
    asset_fields = "id source title folder kind format locator_type container inner_path aux_json animated status size metadata_json".split()
    for original_id in unique_ids:
        asset = assets.get(original_id)
        if asset is None:
            failures.append({"original_asset_id": original_id, "error": "Asset not found in index"})
            continue
        for variant_index in range(variants):
            seed = _variant_seed(random_seed, original_id, variant_index)
            try:
                if asset["kind"] != "smplh_motion" or asset["format"].lower() != "npz":
                    raise ValueError("Random bone length supports indexed SMPL-H/AMASS NPZ motion clips")
                source_path = materialize(asset)
                with np.load(source_path, allow_pickle=True) as archive:
                    original = {key: np.array(archive[key], copy=True) for key in archive.files}
                augmented, provenance, validation = randomize_bone_lengths(original, seed)
                provenance["source_asset_id"] = asset["id"]
                provenance["source_file"] = str(source_path)
                output_name = bone_augmented_filename(Path(asset.get("inner_path") or source_path.name).name, variant_index)
                output_path = output_dir / output_name
                augmented["augmentation_type"] = np.asarray(AUGMENTATION_TYPE)
                augmented["augmentation_method"] = np.asarray(METHOD)
                augmented["augmentation_parent_asset_id"] = np.asarray(asset["id"])
                augmented["augmentation_random_seed"] = np.asarray(seed, dtype=np.int64)
                augmented["augmentation_provenance_json"] = np.asarray(json.dumps(provenance, ensure_ascii=False))
                augmented["augmentation_validation_json"] = np.asarray(json.dumps(validation, ensure_ascii=False))
                np.savez_compressed(output_path, **augmented)
                derived = row(
                    "Augmentation Cache / Bone Length",
                    output_path.stem,
                    asset["kind"],
                    "npz",
                    "direct",
                    output_path,
                    folder=f"{run_id}/{asset['source']}/{asset['folder']}",
                    animated=True,
                    size=output_path.stat().st_size,
                    metadata={
                        "augmentation_type": AUGMENTATION_TYPE,
                        "augmentation_draft": True,
                        "augmentation_run_id": run_id,
                        "parent_asset_id": original_id,
                        "variant_index": variant_index,
                        "random_seed": seed,
                        "actual_scales": provenance["shape"]["actual_scales"],
                        "validation_passed": validation["passed"],
                    },
                )
                with db:
                    db.execute(
                        f"INSERT OR REPLACE INTO assets({','.join(asset_fields)}) VALUES({','.join('?' for _ in asset_fields)})",
                        [derived[field] for field in asset_fields],
                    )
                    db.execute(
                        "INSERT OR REPLACE INTO augmentation_bone_outputs(run_id,original_asset_id,variant_index,augmented_asset_id,validation_json) VALUES(?,?,?,?,?)",
                        (run_id, original_id, variant_index, derived["id"], json.dumps(validation, ensure_ascii=False)),
                    )
                pairs.append({
                    "original_asset_id": original_id,
                    "augmented_asset_id": derived["id"],
                    "augmentation_type": AUGMENTATION_TYPE,
                    "variant_index": variant_index,
                    "variant_label": f"Bone length {variant_index + 1}",
                    "augmented_filename": output_name,
                    "cache_path": str(output_path),
                    "provenance": provenance,
                    "validation": validation,
                })
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures.append({"original_asset_id": original_id, "title": asset.get("title", original_id), "variant_index": variant_index, "error": error})
                with db:
                    db.execute(
                        "INSERT OR REPLACE INTO augmentation_bone_outputs(run_id,original_asset_id,variant_index,error) VALUES(?,?,?,?)",
                        (run_id, original_id, variant_index, error),
                    )
    completed_at = time.time()
    status = "ready" if pairs else "failed"
    with db:
        db.execute("UPDATE augmentation_runs SET status=?,completed_at=? WHERE id=?", (status, completed_at, run_id))
    db.close()
    result = {
        "run_id": run_id,
        "run_ids": [run_id],
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "params": params,
        "pairs": pairs,
        "failures": failures,
        "output_directory": str(output_dir),
        "output_directories": [str(output_dir)],
        "storage_state": "cache_draft",
        "saved": False,
        "passed": sum(bool(pair["validation"]["passed"]) for pair in pairs),
        "failed": len(failures) + sum(not bool(pair["validation"]["passed"]) for pair in pairs),
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
