from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import time
import uuid

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import butter, sosfiltfilt, welch
from scipy.spatial.transform import Rotation, RotationSpline, Slerp
from scipy.stats import wasserstein_distance

from bodydata_decode import materialize
from bodydata_index import CACHE_ROOT, DATA_ROOT, connect, row


AUGMENT_ROOT = CACHE_ROOT / "augmentations"
AUGMENTATION_TYPE = "time"
DEFAULT_DURATION_SCALES = (1.15,)
MIN_DURATION_SCALE = 1.00
MAX_DURATION_SCALE = 1.50
MAX_VARIANTS = 8
METHOD = "Duration multiplier · SO(3) RotationSpline · cubic translation · source FPS preserved"


def _safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value)).strip(" .")
    return value[:150] or "motion"


def read_source_fps(motion: dict[str, np.ndarray]) -> float:
    for key in ("mocap_framerate", "mocap_frame_rate"):
        if key in motion:
            fps = float(np.asarray(motion[key]).reshape(-1)[0])
            if np.isfinite(fps) and fps > 0:
                return fps
            raise ValueError(f"Invalid source frame rate in {key}: {fps}")
    raise ValueError("AMASS source frame rate is missing; expected mocap_framerate or mocap_frame_rate")


def _copy_motion(motion: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in motion.items()}


def _rotation_increments(poses: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] % 3:
        raise ValueError(f"poses must have shape (frames, joints*3), got {values.shape}")
    if len(values) < 2:
        return np.empty((0, values.shape[1] // 3, 3), dtype=np.float64)
    joints = values.shape[1] // 3
    left = Rotation.from_rotvec(values[:-1].reshape(-1, 3))
    right = Rotation.from_rotvec(values[1:].reshape(-1, 3))
    return (left.inv() * right).as_rotvec().reshape(len(values) - 1, joints, 3)


def _motion_psd(poses: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    increments = _rotation_increments(poses)
    if len(increments) < 2:
        return np.asarray([0.0, fps / 2.0]), np.asarray([1.0, 0.0])
    angular_velocity = increments * fps
    nperseg = min(256, len(angular_velocity))
    frequencies, power = welch(
        angular_velocity,
        fs=fps,
        nperseg=nperseg,
        axis=0,
        detrend="constant",
        scaling="spectrum",
    )
    return frequencies, np.sum(power, axis=(1, 2))


def estimate_motion_bandwidth_hz(poses: np.ndarray, fps: float, energy_quantile: float = 0.99) -> float:
    frequencies, power = _motion_psd(poses, fps)
    total = float(np.sum(power))
    if not np.isfinite(total) or total <= np.finfo(np.float64).eps:
        return 0.0
    cumulative = np.cumsum(power) / total
    index = min(int(np.searchsorted(cumulative, energy_quantile, side="left")), len(frequencies) - 1)
    return float(frequencies[index])


def _zero_phase_lowpass(values: np.ndarray, cutoff_hz: float, fps: float) -> tuple[np.ndarray, str | None]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 4:
        return np.array(array, copy=True), "Sequence is too short for zero-phase filtering; conservative short-sequence fallback was used"
    nyquist = fps / 2.0
    if cutoff_hz >= nyquist * 0.999:
        return np.array(array, copy=True), None
    normalized = float(np.clip(cutoff_hz / nyquist, 1e-4, 0.999))
    sos = butter(4, normalized, btype="lowpass", output="sos")
    padlen = min(len(array) - 1, 3 * (2 * len(sos) + 1))
    return sosfiltfilt(sos, array, axis=0, padlen=padlen), None


def _smootherstep(length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros(length, dtype=np.float64)
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lowpass_rotations(poses: np.ndarray, cutoff_hz: float, fps: float) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(poses)
    if len(values) < 2:
        return np.array(values, copy=True), []
    increments = _rotation_increments(values)
    filtered, warning = _zero_phase_lowpass(increments, cutoff_hz, fps)
    joints = values.shape[1] // 3
    source_frames = Rotation.from_rotvec(values.reshape(-1, 3))
    current = source_frames[:joints]
    frames = [current]
    for step in filtered:
        current = current * Rotation.from_rotvec(step)
        frames.append(current)
    end_target = source_frames[-joints:]
    end_error = frames[-1].inv() * end_target
    end_error_vectors = end_error.as_rotvec()
    progress = _smootherstep(len(frames))
    corrected = []
    for index, frame in enumerate(frames):
        correction = Rotation.from_rotvec(end_error_vectors * progress[index])
        corrected.append((frame * correction).as_rotvec())
    output = np.stack(corrected, axis=0).reshape(values.shape).astype(values.dtype, copy=False)
    return output, [warning] if warning else []


def _lowpass_translation(trans: np.ndarray, cutoff_hz: float, fps: float) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(trans)
    if len(values) < 2:
        return np.array(values, copy=True), []
    velocity = np.diff(values.astype(np.float64), axis=0) * fps
    filtered, warning = _zero_phase_lowpass(velocity, cutoff_hz, fps)
    output = np.empty_like(values, dtype=np.float64)
    output[0] = values[0]
    output[1:] = values[0] + np.cumsum(filtered / fps, axis=0)
    endpoint_error = values[-1] - output[-1]
    output += _smootherstep(len(output)).reshape((-1,) + (1,) * (output.ndim - 1)) * endpoint_error
    return output.astype(values.dtype, copy=False), [warning] if warning else []


def _prefilter_motion(
    poses: np.ndarray,
    trans: np.ndarray | None,
    cutoff_hz: float,
    fps: float,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    filtered_poses, rotation_warnings = _lowpass_rotations(poses, cutoff_hz, fps)
    filtered_trans, translation_warnings = (None, []) if trans is None else _lowpass_translation(trans, cutoff_hz, fps)
    return filtered_poses, filtered_trans, list(dict.fromkeys(rotation_warnings + translation_warnings))


def _rotation_resample_smooth(poses: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    values = np.asarray(poses)
    frames, columns = values.shape
    joints = columns // 3
    if frames == 1:
        return np.repeat(values, len(source_positions), axis=0)
    source_times = np.arange(frames, dtype=np.float64)
    rotations = values.reshape(frames, joints, 3)
    output = np.empty((len(source_positions), joints, 3), dtype=np.float64)
    for joint in range(joints):
        samples = Rotation.from_rotvec(rotations[:, joint].astype(np.float64))
        curve = RotationSpline(source_times, samples) if frames >= 3 else Slerp(source_times, samples)
        output[:, joint] = curve(source_positions).as_rotvec()
    return output.reshape(len(source_positions), columns).astype(values.dtype, copy=False)


def _cubic_resample(array: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if len(values) == 1:
        return np.repeat(values, len(source_positions), axis=0)
    source_times = np.arange(len(values), dtype=np.float64)
    curve = CubicSpline(source_times, values.astype(np.float64), axis=0, bc_type="natural") if len(values) >= 4 else PchipInterpolator(source_times, values.astype(np.float64), axis=0)
    return np.asarray(curve(source_positions)).astype(values.dtype, copy=False)


def _nearest_resample(array: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    indices = np.clip(np.rint(source_positions).astype(np.int64), 0, len(array) - 1)
    return np.asarray(array)[indices]


def _resample_arrays(
    original: dict[str, np.ndarray],
    source_positions: np.ndarray,
    pose_source: np.ndarray | None = None,
    trans_source: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    source_frames = int(np.asarray(original["poses"]).shape[0])
    output: dict[str, np.ndarray] = {}
    for key, value in original.items():
        array = np.asarray(value)
        if key == "poses":
            output[key] = _rotation_resample_smooth(pose_source if pose_source is not None else array, source_positions)
        elif key == "trans" and array.ndim > 0 and array.shape[0] == source_frames:
            output[key] = _cubic_resample(trans_source if trans_source is not None else array, source_positions)
        elif key == "betas":
            output[key] = np.array(array, copy=True)
        elif array.ndim > 0 and array.shape[0] == source_frames:
            output[key] = _cubic_resample(array, source_positions) if np.issubdtype(array.dtype, np.floating) else _nearest_resample(array, source_positions)
        else:
            output[key] = np.array(array, copy=True)
    return output


def _set_output_fps(motion: dict[str, np.ndarray], fps: float) -> None:
    found = False
    for key in ("mocap_framerate", "mocap_frame_rate"):
        if key in motion:
            motion[key] = np.asarray(fps, dtype=np.asarray(motion[key]).dtype)
            found = True
    if not found:
        raise ValueError("Cannot write output frame rate because the source frame-rate field is missing")


def retime_motion(
    original: dict[str, np.ndarray],
    duration_multiplier: float,
    training_fps: float | None = None,
    random_seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    if "poses" not in original:
        raise ValueError("Motion does not contain poses")
    poses = np.asarray(original["poses"])
    if poses.ndim != 2 or poses.shape[1] % 3:
        raise ValueError(f"poses must have shape (frames, joints*3), got {poses.shape}")
    source_frames = len(poses)
    if source_frames < 1:
        raise ValueError("Motion contains no pose frames")
    source_fps = read_source_fps(original)
    target_fps = source_fps if training_fps is None else float(training_fps)
    multiplier = float(duration_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Duration multiplier must be positive and finite")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("Training frame rate must be positive and finite")

    source_duration = (source_frames - 1) / source_fps if source_frames > 1 else 0.0
    motion_bandwidth = estimate_motion_bandwidth_hz(poses, source_fps)
    minimum_safe_multiplier = 2.0 * motion_bandwidth / source_fps
    source_stage_frames = int(round((source_frames - 1) * multiplier)) + 1
    source_stage_scale = (source_stage_frames - 1) / max(source_frames - 1, 1)

    if multiplier == 1.0 and target_fps == source_fps:
        return _copy_motion(original), {
            "requested_duration_multiplier": multiplier,
            "actual_duration_multiplier": 1.0,
            "source_fps": source_fps,
            "output_fps": source_fps,
            "source_frames": source_frames,
            "output_frames": source_frames,
            "motion_bandwidth_hz_99pct": motion_bandwidth,
            "minimum_safe_duration_multiplier": minimum_safe_multiplier,
            "aliasing_safe": True,
            "method": METHOD,
            "random_seed": int(random_seed),
            "filters": [],
            "identity_fast_path": True,
        }

    if multiplier < 1.0 and source_stage_scale + 1e-12 < minimum_safe_multiplier:
        raise ValueError(
            f"Requested duration multiplier {multiplier:.4f} is below the measured anti-alias limit "
            f"{minimum_safe_multiplier:.4f} (f_motion={motion_bandwidth:.3f} Hz, source_fps={source_fps:.3f})"
        )

    warnings: list[str] = []
    filters: list[dict] = []
    trans = np.asarray(original["trans"]) if "trans" in original and np.asarray(original["trans"]).shape[0] == source_frames else None
    pose_source = poses
    trans_source = trans
    if multiplier < 1.0 and source_frames >= 2:
        cutoff = source_fps * source_stage_scale * 0.49
        pose_source, trans_source, stage_warnings = _prefilter_motion(poses, trans, cutoff, source_fps)
        warnings.extend(stage_warnings)
        filters.append({"stage": "source_rate_speedup", "type": "zero_phase_tangent_butterworth", "cutoff_hz": cutoff})

    source_positions = np.linspace(0.0, source_frames - 1.0, source_stage_frames, dtype=np.float64)
    source_stage = _resample_arrays(original, source_positions, pose_source=pose_source, trans_source=trans_source)
    _set_output_fps(source_stage, source_fps)

    output = source_stage
    if target_fps != source_fps and source_stage_frames > 1:
        final_frames = int(round(source_duration * multiplier * target_fps)) + 1
        final_frames = max(final_frames, 1)
        stage_poses = np.asarray(source_stage["poses"])
        stage_trans = np.asarray(source_stage["trans"]) if "trans" in source_stage and np.asarray(source_stage["trans"]).shape[0] == source_stage_frames else None
        filtered_poses = stage_poses
        filtered_trans = stage_trans
        if target_fps < source_fps:
            cutoff = target_fps * 0.49
            filtered_poses, filtered_trans, stage_warnings = _prefilter_motion(stage_poses, stage_trans, cutoff, source_fps)
            warnings.extend(stage_warnings)
            filters.append({"stage": "training_rate_downsample", "type": "zero_phase_tangent_butterworth", "cutoff_hz": cutoff})
        final_positions = np.linspace(0.0, source_stage_frames - 1.0, final_frames, dtype=np.float64)
        output = _resample_arrays(source_stage, final_positions, pose_source=filtered_poses, trans_source=filtered_trans)
        _set_output_fps(output, target_fps)

    output_frames = int(np.asarray(output["poses"]).shape[0])
    output_duration = (output_frames - 1) / target_fps if output_frames > 1 else 0.0
    actual_multiplier = output_duration / source_duration if source_duration else 1.0
    provenance = {
        "requested_duration_multiplier": multiplier,
        "actual_duration_multiplier": actual_multiplier,
        "source_fps": source_fps,
        "output_fps": target_fps,
        "source_frames": source_frames,
        "source_rate_retime_frames": source_stage_frames,
        "output_frames": output_frames,
        "motion_bandwidth_hz_99pct": motion_bandwidth,
        "minimum_safe_duration_multiplier": minimum_safe_multiplier,
        "aliasing_safe": multiplier >= 1.0 or source_stage_scale + 1e-12 >= minimum_safe_multiplier,
        "method": METHOD,
        "random_seed": int(random_seed),
        "filters": filters,
        "warnings": list(dict.fromkeys(warnings)),
        "identity_fast_path": False,
    }
    return output, provenance


def _rotation_errors_degrees(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    left = Rotation.from_rotvec(np.asarray(reference).reshape(-1, 3))
    right = Rotation.from_rotvec(np.asarray(candidate).reshape(-1, 3))
    return np.rad2deg((left.inv() * right).magnitude())


def _rotation_step_degrees(poses: np.ndarray) -> np.ndarray:
    increments = _rotation_increments(poses)
    return np.rad2deg(np.linalg.norm(increments, axis=-1))


def _angular_acceleration(poses: np.ndarray, fps: float) -> np.ndarray:
    velocity = _rotation_increments(poses) * fps
    if len(velocity) < 2:
        return np.asarray([0.0])
    return np.linalg.norm(np.diff(velocity, axis=0) * fps, axis=-1).reshape(-1)


def _rotation_matrix_quality(poses: np.ndarray) -> tuple[float, float]:
    flat = np.asarray(poses).reshape(-1, 3)
    det_error = 0.0
    orthogonal_error = 0.0
    identity = np.eye(3)
    for start in range(0, len(flat), 200_000):
        matrices = Rotation.from_rotvec(flat[start:start + 200_000]).as_matrix()
        determinants = np.linalg.det(matrices)
        det_error = max(det_error, float(np.max(np.abs(determinants - 1.0), initial=0.0)))
        products = np.swapaxes(matrices, -1, -2) @ matrices
        orthogonal_error = max(orthogonal_error, float(np.max(np.abs(products - identity), initial=0.0)))
    return det_error, orthogonal_error


def _normalized_psd_alias_ratio(
    source_poses: np.ndarray,
    output_poses: np.ndarray,
    source_fps: float,
    output_fps: float,
    duration_multiplier: float,
    motion_bandwidth_hz: float,
) -> tuple[float, float, float]:
    source_frequency, source_power = _motion_psd(source_poses, source_fps)
    output_frequency, output_power = _motion_psd(output_poses, output_fps)
    output_frequency = output_frequency * duration_multiplier
    high_start = max(motion_bandwidth_hz * 0.8, min(source_fps, output_fps * duration_multiplier) * 0.20)

    def ratio(frequency, power):
        total = float(np.sum(power))
        if total <= np.finfo(np.float64).eps:
            return 0.0
        return float(np.sum(power[frequency >= high_start]) / total)

    source_ratio = ratio(source_frequency, source_power)
    output_ratio = ratio(output_frequency, output_power)
    return source_ratio, output_ratio, high_start


def _smplh_foot_positions(motion: dict[str, np.ndarray]) -> np.ndarray | None:
    poses = np.asarray(motion.get("poses"))
    if poses.ndim != 2 or poses.shape[1] < 156 or len(poses) < 1:
        return None
    try:
        import torch
        from bodydata_decode import _get_smplh_model

        betas = np.asarray(motion.get("betas", np.zeros(16)), dtype=np.float32).reshape(-1)
        gender = str(np.asarray(motion.get("gender", "neutral")).reshape(-1)[0])
        trans = np.asarray(motion.get("trans", np.zeros((len(poses), 3))), dtype=np.float32)
        model = _get_smplh_model(gender, len(betas))
        chunks = []
        with torch.no_grad():
            for start in range(0, len(poses), 256):
                stop = min(start + 256, len(poses))
                batch_poses = torch.from_numpy(poses[start:stop].astype(np.float32))
                batch_betas = torch.from_numpy(np.repeat(betas[None, : model.num_betas], stop - start, axis=0))
                result = model(
                    betas=batch_betas,
                    global_orient=batch_poses[:, :3],
                    body_pose=batch_poses[:, 3:66],
                    left_hand_pose=batch_poses[:, 66:111],
                    right_hand_pose=batch_poses[:, 111:156],
                    transl=torch.from_numpy(trans[start:stop]),
                    return_verts=False,
                )
                chunks.append(result.joints[:, [10, 11]].detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)
    except Exception:
        return None


def _foot_slide_series(
    motion: dict[str, np.ndarray],
    fps: float,
    foot_positions: np.ndarray | None = None,
) -> np.ndarray | None:
    if foot_positions is not None:
        feet = np.asarray(foot_positions)
        if feet.ndim != 3 or feet.shape[1:] != (2, 3) or len(feet) < 2:
            return None
        displacement = np.linalg.norm(np.diff(feet, axis=0), axis=-1)
        height = feet[:-1, :, 1]
        vertical_speed = np.abs(np.diff(feet[:, :, 1], axis=0)) * fps
        contact = (height <= np.percentile(height, 30, axis=0, keepdims=True)) & (vertical_speed <= 0.15)
        return displacement[contact] * fps
    key = next((name for name in ("joints", "joint_positions", "joints_world") if name in motion), None)
    if key is None:
        return None
    joints = np.asarray(motion[key])
    if joints.ndim != 3 or joints.shape[1] < 12 or joints.shape[2] != 3 or len(joints) < 2:
        return None
    feet = joints[:, [10, 11]]
    displacement = np.linalg.norm(np.diff(feet, axis=0), axis=-1)
    height = feet[:-1, :, 1]
    contact = height <= np.percentile(height, 25, axis=0, keepdims=True)
    return displacement[contact] * fps


def validate_retime(
    original: dict[str, np.ndarray],
    augmented: dict[str, np.ndarray],
    provenance: dict,
    source_feet: np.ndarray | None = None,
    output_feet: np.ndarray | None = None,
) -> dict:
    source_poses = np.asarray(original["poses"])
    output_poses = np.asarray(augmented["poses"])
    source_frames = len(source_poses)
    output_frames = len(output_poses)
    source_fps = float(provenance["source_fps"])
    output_fps = float(provenance["output_fps"])
    actual_scale = float(provenance["actual_duration_multiplier"])
    back_positions = np.linspace(0.0, output_frames - 1.0, source_frames, dtype=np.float64)
    roundtrip_poses = _rotation_resample_smooth(output_poses, back_positions)
    rotation_errors = _rotation_errors_degrees(source_poses, roundtrip_poses)
    endpoint_errors = np.concatenate((
        _rotation_errors_degrees(source_poses[0:1], output_poses[0:1]),
        _rotation_errors_degrees(source_poses[-1:], output_poses[-1:]),
    ))
    determinant_error, orthogonal_error = _rotation_matrix_quality(output_poses)

    trans_rms = 0.0
    trans_endpoint_max = 0.0
    if "trans" in original and "trans" in augmented:
        source_trans = np.asarray(original["trans"])
        output_trans = np.asarray(augmented["trans"])
        roundtrip_trans = _cubic_resample(output_trans, back_positions)
        trans_rms = float(np.sqrt(np.mean(np.square(roundtrip_trans - source_trans))))
        trans_endpoint_max = float(max(np.max(np.abs(output_trans[0] - source_trans[0])), np.max(np.abs(output_trans[-1] - source_trans[-1]))))

    source_acceleration = _angular_acceleration(source_poses, source_fps)
    output_acceleration = _angular_acceleration(output_poses, output_fps) * (actual_scale ** 2)
    acceleration_distance = float(wasserstein_distance(source_acceleration, output_acceleration))
    acceleration_reference = float(np.percentile(np.abs(source_acceleration), 95)) + 1e-9
    normalized_acceleration_distance = acceleration_distance / acceleration_reference
    source_high, output_high, high_start = _normalized_psd_alias_ratio(
        source_poses,
        output_poses,
        source_fps,
        output_fps,
        actual_scale,
        float(provenance["motion_bandwidth_hz_99pct"]),
    )
    source_slide = _foot_slide_series(original, source_fps, source_feet)
    output_slide = _foot_slide_series(augmented, output_fps, output_feet)
    foot_slide_ratio = None
    if source_slide is not None and output_slide is not None:
        source_p95 = float(np.percentile(source_slide, 95)) + 1e-9
        foot_slide_ratio = float(np.percentile(output_slide * actual_scale, 95) / source_p95)

    numeric_arrays = [value for value in augmented.values() if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)]
    finite = all(bool(np.isfinite(value).all()) for value in numeric_arrays)
    shape_unchanged = "betas" not in original or np.array_equal(original["betas"], augmented["betas"])
    identity_exact = True
    if float(provenance["requested_duration_multiplier"]) == 1.0 and source_fps == output_fps:
        identity_exact = set(original) == set(augmented) and all(np.array_equal(original[key], augmented[key]) for key in original)
    checks = {
        "finite_numeric_values": finite,
        "rotation_endpoints_preserved": float(np.max(endpoint_errors, initial=0.0)) < 1e-7,
        "translation_endpoints_preserved": trans_endpoint_max < 1e-7,
        "rotation_matrices_orthogonal": orthogonal_error < 1e-9,
        "rotation_determinants_positive_one": determinant_error < 1e-9,
        "shape_parameters_unchanged": shape_unchanged,
        "aliasing_limit_respected": bool(provenance["aliasing_safe"]),
        "normalized_high_frequency_energy_not_increased": output_high <= source_high + 0.02,
        "normalized_angular_acceleration_distance_acceptable": normalized_acceleration_distance <= 0.35,
    }
    if float(provenance["requested_duration_multiplier"]) == 1.0 and source_fps == output_fps:
        checks["identity_multiplier_is_elementwise_exact"] = identity_exact
    if foot_slide_ratio is not None:
        checks["foot_slide_not_significantly_increased"] = foot_slide_ratio <= 1.25
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "source_frames": source_frames,
            "output_frames": output_frames,
            "source_fps": source_fps,
            "output_fps": output_fps,
            "source_duration_seconds": (source_frames - 1) / source_fps if source_frames > 1 else 0.0,
            "output_duration_seconds": (output_frames - 1) / output_fps if output_frames > 1 else 0.0,
            "requested_duration_scale": float(provenance["requested_duration_multiplier"]),
            "measured_duration_scale": actual_scale,
            "motion_bandwidth_hz_99pct": float(provenance["motion_bandwidth_hz_99pct"]),
            "minimum_safe_duration_multiplier": float(provenance["minimum_safe_duration_multiplier"]),
            "roundtrip_rotation_mean_degrees": float(np.mean(rotation_errors)),
            "roundtrip_rotation_p95_degrees": float(np.percentile(rotation_errors, 95)),
            "roundtrip_rotation_p99_degrees": float(np.percentile(rotation_errors, 99)),
            "roundtrip_rotation_max_degrees": float(np.max(rotation_errors, initial=0.0)),
            "roundtrip_translation_rms_meters": trans_rms,
            "endpoint_rotation_max_degrees": float(np.max(endpoint_errors, initial=0.0)),
            "endpoint_translation_max_meters": trans_endpoint_max,
            "rotation_determinant_max_error": determinant_error,
            "rotation_orthogonality_max_error": orthogonal_error,
            "angular_acceleration_wasserstein": acceleration_distance,
            "angular_acceleration_wasserstein_normalized": normalized_acceleration_distance,
            "psd_high_band_start_hz_normalized": high_start,
            "source_high_frequency_energy_ratio": source_high,
            "output_high_frequency_energy_ratio": output_high,
            "foot_slide_p95_ratio": foot_slide_ratio,
        },
        "warnings": list(provenance.get("warnings", [])) + (["Foot-slide validation unavailable because world-space joints are not stored in this NPZ"] if foot_slide_ratio is None else []),
        "provenance": provenance,
        "guarantees": [
            "Source frame rate is read explicitly from AMASS metadata",
            "Speed-up is rejected below the measured PSD anti-alias limit",
            "Retime occurs at source rate before zero-phase training-rate downsampling",
            "Joint rotations use an SO(3) RotationSpline with continuous angular velocity",
            "Root translation uses a cubic trajectory",
            "Body shape and source files remain unchanged",
        ],
    }


def _quality_plot(
    original: dict[str, np.ndarray],
    augmented: dict[str, np.ndarray],
    provenance: dict,
    output_path: Path,
    source_feet: np.ndarray | None = None,
    output_feet: np.ndarray | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_fps = float(provenance["source_fps"])
    output_fps = float(provenance["output_fps"])
    scale = float(provenance["actual_duration_multiplier"])
    source_acc = _angular_acceleration(original["poses"], source_fps)
    output_acc = _angular_acceleration(augmented["poses"], output_fps) * (scale ** 2)
    source_frequency, source_power = _motion_psd(original["poses"], source_fps)
    output_frequency, output_power = _motion_psd(augmented["poses"], output_fps)
    source_slide = _foot_slide_series(original, source_fps, source_feet)
    output_slide = _foot_slide_series(augmented, output_fps, output_feet)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    upper = max(float(np.percentile(source_acc, 99)), float(np.percentile(output_acc, 99)), 1e-6)
    bins = np.linspace(0.0, upper, 60)
    axes[0].hist(source_acc, bins=bins, density=True, alpha=0.48, label="source")
    axes[0].hist(output_acc, bins=bins, density=True, alpha=0.48, label="retimed (normalized)")
    axes[0].set_title("Angular acceleration")
    axes[0].set_xlabel("rad/s²")
    axes[0].legend()
    axes[1].semilogy(source_frequency, source_power + 1e-14, label="source")
    axes[1].semilogy(output_frequency * scale, output_power + 1e-14, label="retimed (normalized time)")
    axes[1].set_title("Angular-velocity PSD")
    axes[1].set_xlabel("normalized Hz")
    axes[1].legend()
    axes[2].set_title("Foot slide during contact")
    if source_slide is None or output_slide is None:
        axes[2].text(0.5, 0.5, "World-space joints unavailable", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_axis_off()
    else:
        axes[2].plot(source_slide, label="source")
        axes[2].plot(output_slide * scale, label="retimed (normalized)")
        axes[2].set_ylabel("m/s")
        axes[2].legend()
    figure.suptitle(f"Temporal augmentation quality · requested ×{provenance['requested_duration_multiplier']:.3f} · actual ×{scale:.3f}")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _scale_token(duration_scale: float) -> str:
    return f"{duration_scale:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def augmented_filename(original_name: str, duration_scale: float) -> str:
    """Compact, sortable name: lineage first and the transform last."""
    stem = _safe_name(Path(original_name).stem)
    return f"augmented__{stem}__t{_scale_token(duration_scale)}.npz"


def _augment_one(
    asset: dict,
    output_dir: Path,
    duration_scale: float,
    training_fps: float | None,
    random_seed: int,
) -> tuple[Path, dict, dict, Path]:
    if asset["kind"] != "smplh_motion" or asset["format"].lower() != "npz":
        raise ValueError("Temporal augmentation supports indexed SMPL-H/AMASS NPZ motion clips")
    source_path = materialize(asset)
    with np.load(source_path, allow_pickle=True) as archive:
        original = {key: np.array(archive[key], copy=True) for key in archive.files}
    augmented, provenance = retime_motion(original, duration_scale, training_fps=training_fps, random_seed=random_seed)
    provenance["source_asset_id"] = asset["id"]
    provenance["source_file"] = str(source_path)
    source_feet = _smplh_foot_positions(original)
    output_feet = _smplh_foot_positions(augmented)
    validation = validate_retime(original, augmented, provenance, source_feet=source_feet, output_feet=output_feet)
    original_name = Path(asset.get("inner_path") or source_path.name).name
    output_name = augmented_filename(original_name, duration_scale)
    output_path = output_dir / output_name
    plot_path = output_dir / f"{output_path.stem}__quality.png"
    _quality_plot(original, augmented, provenance, plot_path, source_feet=source_feet, output_feet=output_feet)
    augmented["augmentation_type"] = np.asarray("time")
    augmented["augmentation_method"] = np.asarray(METHOD)
    augmented["augmentation_duration_scale"] = np.asarray(duration_scale, dtype=np.float64)
    augmented["augmentation_actual_duration_scale"] = np.asarray(provenance["actual_duration_multiplier"], dtype=np.float64)
    augmented["augmentation_parent_asset_id"] = np.asarray(asset["id"])
    augmented["augmentation_random_seed"] = np.asarray(random_seed, dtype=np.int64)
    augmented["augmentation_provenance_json"] = np.asarray(json.dumps(provenance, ensure_ascii=False))
    augmented["augmentation_validation_json"] = np.asarray(json.dumps(validation, ensure_ascii=False))
    np.savez_compressed(output_path, **augmented)
    return output_path, validation, provenance, plot_path


def augment_assets(
    asset_ids: list[str],
    duration_scales: list[float] | tuple[float, ...] | None = None,
    training_fps: float | None = None,
    random_seed: int = 0,
) -> dict:
    unique_ids = list(dict.fromkeys(str(value) for value in asset_ids if value))
    if not unique_ids:
        raise ValueError("Select at least one motion asset")
    scales = list(dict.fromkeys(round(float(value), 4) for value in (duration_scales or DEFAULT_DURATION_SCALES)))
    if not scales or len(scales) > MAX_VARIANTS:
        raise ValueError(f"Select between 1 and {MAX_VARIANTS} duration multipliers")
    if any(not np.isfinite(value) or value < MIN_DURATION_SCALE or value > MAX_DURATION_SCALE for value in scales):
        raise ValueError(f"Duration multipliers must be between {MIN_DURATION_SCALE:.2f} and {MAX_DURATION_SCALE:.2f}")
    if training_fps is not None and (not np.isfinite(training_fps) or training_fps <= 0):
        raise ValueError("Training frame rate must be positive")
    run_id = uuid.uuid4().hex[:20]
    created_at = time.time()
    output_dir = AUGMENT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    params = {
        "type": AUGMENTATION_TYPE,
        "method": METHOD,
        "duration_scales": scales,
        "speed_scales": [1.0 / value for value in scales],
        "training_fps": training_fps if training_fps is not None else "same_as_source",
        "rotation_interpolation": "SO(3) RotationSpline",
        "translation_interpolation": "cubic spline",
        "anti_alias": "zero-phase Butterworth on tangent angular velocity and root velocity",
        "random_seed": int(random_seed),
    }
    db = connect()
    with db:
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
        for variant_index, duration_scale in enumerate(scales):
            variant_seed = int(random_seed) + variant_index
            try:
                output_path, validation, provenance, plot_path = _augment_one(asset, output_dir, duration_scale, training_fps, variant_seed)
                derived = row(
                    "Augmentation Cache / Time",
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
                        "augmentation_saved": False,
                        "augmentation_run_id": run_id,
                        "parent_asset_id": original_id,
                        "duration_scale": duration_scale,
                        "actual_duration_scale": provenance["actual_duration_multiplier"],
                        "source_fps": provenance["source_fps"],
                        "output_fps": provenance["output_fps"],
                        "random_seed": variant_seed,
                        "validation_passed": validation["passed"],
                    },
                )
                with db:
                    db.execute(
                        f"INSERT OR REPLACE INTO assets({','.join(asset_fields)}) VALUES({','.join('?' for _ in asset_fields)})",
                        [derived[field] for field in asset_fields],
                    )
                    db.execute(
                        "INSERT INTO augmentation_outputs(run_id,original_asset_id,duration_scale,augmented_asset_id,validation_json) VALUES(?,?,?,?,?)",
                        (run_id, original_id, duration_scale, derived["id"], json.dumps(validation, ensure_ascii=False)),
                    )
                pairs.append({
                    "original_asset_id": original_id,
                    "augmented_asset_id": derived["id"],
                    "duration_scale": duration_scale,
                    "actual_duration_scale": provenance["actual_duration_multiplier"],
                    "augmented_filename": output_path.name,
                    "quality_plot_path": str(plot_path),
                    "provenance": provenance,
                    "validation": validation,
                })
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures.append({"original_asset_id": original_id, "title": asset["title"], "duration_scale": duration_scale, "error": error})
                with db:
                    db.execute(
                        "INSERT INTO augmentation_outputs(run_id,original_asset_id,duration_scale,error) VALUES(?,?,?,?)",
                        (run_id, original_id, duration_scale, error),
                    )
    completed_at = time.time()
    status = "ready" if pairs else "failed"
    with db:
        db.execute("UPDATE augmentation_runs SET status=?,completed_at=? WHERE id=?", (status, completed_at, run_id))
    db.close()
    result = {
        "run_id": run_id,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "params": params,
        "pairs": pairs,
        "failures": failures,
        "output_directory": str(output_dir),
        "storage_state": "cache_draft",
        "saved": False,
        "passed": sum(bool(pair["validation"]["passed"]) for pair in pairs),
        "failed": len(failures) + sum(not bool(pair["validation"]["passed"]) for pair in pairs),
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def save_augmentation(run_id: str) -> dict:
    """Promote a cached draft run into the user's persistent data library."""
    result = load_augmentation(run_id)
    if not result:
        raise FileNotFoundError(f"Augmentation run not found: {run_id}")
    if result.get("saved"):
        return result
    if result.get("status") != "ready" or not result.get("pairs"):
        raise ValueError("Only a completed augmentation run can be saved")

    cache_dir = AUGMENT_ROOT / run_id
    destination = DATA_ROOT / "augmented" / "time" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    asset_fields = "id source title folder kind format locator_type container inner_path aux_json animated status size metadata_json".split()
    db = connect()
    saved_pairs = []
    try:
        for pair in result["pairs"]:
            cached_path = cache_dir / pair["augmented_filename"]
            if not cached_path.is_file():
                raise FileNotFoundError(f"Cached augmentation is missing: {cached_path.name}")
            saved_path = destination / cached_path.name
            shutil.copy2(cached_path, saved_path)
            quality_path = Path(pair.get("quality_plot_path", ""))
            saved_quality_path = None
            if quality_path.is_file():
                saved_quality_path = destination / quality_path.name
                shutil.copy2(quality_path, saved_quality_path)
            provenance = pair.get("provenance", {})
            validation = pair.get("validation", {})
            formal = row(
                "Augmented / Time",
                saved_path.stem,
                "smplh_motion",
                "npz",
                "direct",
                saved_path,
                folder=run_id,
                animated=True,
                size=saved_path.stat().st_size,
                metadata={
                    "augmentation_type": AUGMENTATION_TYPE,
                    "augmentation_run_id": run_id,
                    "augmentation_draft": False,
                    "augmentation_saved": True,
                    "parent_asset_id": pair["original_asset_id"],
                    "duration_scale": pair["duration_scale"],
                    "actual_duration_scale": pair.get("actual_duration_scale"),
                    "source_fps": provenance.get("source_fps"),
                    "output_fps": provenance.get("output_fps"),
                    "random_seed": provenance.get("random_seed"),
                    "validation_passed": validation.get("passed", False),
                },
            )
            with db:
                db.execute(
                    f"INSERT OR REPLACE INTO assets({','.join(asset_fields)}) VALUES({','.join('?' for _ in asset_fields)})",
                    [formal[field] for field in asset_fields],
                )
            pair["saved_asset_id"] = formal["id"]
            pair["saved_path"] = str(saved_path)
            if saved_quality_path:
                pair["saved_quality_plot_path"] = str(saved_quality_path)
            saved_pairs.append(pair)
    finally:
        db.close()

    result["pairs"] = saved_pairs
    result["saved"] = True
    result["storage_state"] = "saved"
    result["saved_at"] = time.time()
    result["saved_directory"] = str(destination)
    result_path = cache_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "augmentation_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def load_augmentation(run_id: str) -> dict | None:
    path = AUGMENT_ROOT / run_id / "result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
