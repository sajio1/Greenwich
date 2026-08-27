from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


_COMPONENT = {
    np.dtype("float32"): 5126,
    np.dtype("uint32"): 5125,
    np.dtype("uint16"): 5123,
    np.dtype("uint8"): 5121,
}


def _pad4(data: bytes, fill: bytes = b"\0") -> bytes:
    return data + fill * ((-len(data)) % 4)


class _Builder:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, array: np.ndarray, gltf_type: str, *, target: int | None = None, bounds: bool = False) -> int:
        array = np.ascontiguousarray(array)
        padding = (-len(self.binary)) % 4
        if padding:
            self.binary.extend(b"\0" * padding)
        offset = len(self.binary)
        payload = array.tobytes()
        self.binary.extend(payload)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        view_index = len(self.views)
        self.views.append(view)
        count = int(array.shape[0]) if array.ndim else 1
        accessor = {
            "bufferView": view_index,
            "componentType": _COMPONENT[array.dtype],
            "count": count,
            "type": gltf_type,
        }
        if bounds:
            values = array.reshape(count, -1)
            accessor["min"] = values.min(axis=0).astype(float).tolist()
            accessor["max"] = values.max(axis=0).astype(float).tolist()
        index = len(self.accessors)
        self.accessors.append(accessor)
        return index


def write_smplh_animation_glb(
    path: Path,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    weights: np.ndarray,
    poses: np.ndarray,
    translations: np.ndarray,
    source_fps: float,
    max_preview_fps: float = 30.0,
    title: str = "SMPL-H motion",
) -> dict:
    """Write a compact, portable skinned SMPL-H animation without changing source data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conversion = np.asarray(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=np.float32)
    vertices = np.asarray(vertices, np.float32) @ conversion.T
    rest_joints = np.asarray(rest_joints, np.float32) @ conversion.T
    translations = np.asarray(translations, np.float32) @ conversion.T
    parents = np.asarray(parents, np.int64)
    faces = np.asarray(faces, np.uint32)

    source_poses = np.asarray(poses[:, : len(parents) * 3], np.float32).reshape(-1, len(parents), 3)
    source_frame_count = len(source_poses)
    # Quality signal from the raw source, before preview resampling. Large adjacent
    # rotations around elbows/wrists usually indicate mocap/fitting spikes.
    arm_slice = source_poses[:, 18:22] if len(parents) >= 22 else source_poses
    arm_q = Rotation.from_rotvec((arm_slice @ conversion.T).reshape(-1, 3)).as_quat().reshape(source_frame_count, -1, 4)
    if source_frame_count > 1:
        arm_delta = (Rotation.from_quat(arm_q[:-1].reshape(-1, 4)).inv() * Rotation.from_quat(arm_q[1:].reshape(-1, 4))).magnitude()
        arm_delta_deg = np.degrees(arm_delta)
        max_arm_step = float(arm_delta_deg.max(initial=0.0))
        arm_spikes = int(np.count_nonzero(arm_delta_deg > 60.0))
    else:
        max_arm_step, arm_spikes = 0.0, 0

    stride = max(1, int(np.ceil(max(float(source_fps), 1.0) / max_preview_fps)))
    frame_indices = np.arange(0, source_frame_count, stride, dtype=np.int64)
    if source_frame_count and frame_indices[-1] != source_frame_count - 1:
        frame_indices = np.append(frame_indices, source_frame_count - 1)
    poses = source_poses[frame_indices]
    translations = translations[frame_indices]
    times = (frame_indices.astype(np.float32) / max(float(source_fps), 1.0)).astype(np.float32)

    # Axis-angle vectors transform like ordinary vectors under this proper basis rotation.
    converted_rotvec = poses @ conversion.T
    quaternions = Rotation.from_rotvec(converted_rotvec.reshape(-1, 3)).as_quat().astype(np.float32)
    quaternions = quaternions.reshape(len(frame_indices), len(parents), 4)
    # q and -q encode the same rotation, but keeping one hemisphere prevents
    # importers that linearly interpolate components from taking a long arc.
    for frame in range(1, len(quaternions)):
        flip = np.sum(quaternions[frame - 1] * quaternions[frame], axis=1) < 0
        quaternions[frame, flip] *= -1

    top = np.argpartition(weights, -4, axis=1)[:, -4:]
    top_weights = np.take_along_axis(np.asarray(weights, np.float32), top, axis=1)
    order = np.argsort(-top_weights, axis=1)
    top = np.take_along_axis(top, order, axis=1).astype(np.uint8)
    top_weights = np.take_along_axis(top_weights, order, axis=1)
    sums = top_weights.sum(axis=1, keepdims=True)
    top_weights = (top_weights / np.maximum(sums, 1e-8)).astype(np.float32)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, np.float32)
    indices_dtype = np.uint16 if len(vertices) <= 65535 else np.uint32
    indices = faces.reshape(-1).astype(indices_dtype)

    b = _Builder()
    position_acc = b.add(vertices, "VEC3", target=34962, bounds=True)
    normal_acc = b.add(normals, "VEC3", target=34962)
    joints_acc = b.add(top, "VEC4", target=34962)
    weights_acc = b.add(top_weights, "VEC4", target=34962)
    indices_acc = b.add(indices, "SCALAR", target=34963)

    inverse_bind = np.repeat(np.eye(4, dtype=np.float32)[None], len(parents), axis=0)
    inverse_bind[:, :3, 3] = -rest_joints
    # glTF matrices are column-major.
    inverse_bind_acc = b.add(np.ascontiguousarray(inverse_bind.transpose(0, 2, 1)), "MAT4")
    time_acc = b.add(times, "SCALAR", bounds=True)

    nodes: list[dict] = []
    joint_nodes: list[int] = []
    for index, parent in enumerate(parents):
        local = rest_joints[index] if parent < 0 else rest_joints[index] - rest_joints[parent]
        joint_nodes.append(len(nodes))
        nodes.append({"name": f"joint_{index:02d}", "translation": local.astype(float).tolist()})
    roots: list[int] = []
    for index, parent in enumerate(parents):
        if parent < 0:
            roots.append(joint_nodes[index])
        else:
            nodes[joint_nodes[int(parent)]].setdefault("children", []).append(joint_nodes[index])

    root_joint = int(np.where(parents < 0)[0][0]) if np.any(parents < 0) else 0
    root_translation = translations + rest_joints[root_joint]
    root_translation_acc = b.add(root_translation.astype(np.float32), "VEC3")
    samplers = [{"input": time_acc, "output": root_translation_acc, "interpolation": "LINEAR"}]
    channels = [{"sampler": 0, "target": {"node": joint_nodes[root_joint], "path": "translation"}}]
    for joint in range(len(parents)):
        rotation_acc = b.add(quaternions[:, joint], "VEC4")
        sampler_index = len(samplers)
        samplers.append({"input": time_acc, "output": rotation_acc, "interpolation": "LINEAR"})
        channels.append({"sampler": sampler_index, "target": {"node": joint_nodes[joint], "path": "rotation"}})

    mesh_node = len(nodes)
    nodes.append({"name": "SMPL-H body", "mesh": 0, "skin": 0})
    document = {
        "asset": {"version": "2.0", "generator": "Body Data Studio"},
        "scene": 0,
        "scenes": [{"name": title, "nodes": [mesh_node, *roots]}],
        "nodes": nodes,
        "buffers": [{"byteLength": len(b.binary)}],
        "bufferViews": b.views,
        "accessors": b.accessors,
        "materials": [{
            "name": "Body",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.72, 0.78, 0.88, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.78,
            },
            "doubleSided": True,
        }],
        "meshes": [{"name": "SMPL-H", "primitives": [{
            "attributes": {"POSITION": position_acc, "NORMAL": normal_acc, "JOINTS_0": joints_acc, "WEIGHTS_0": weights_acc},
            "indices": indices_acc,
            "material": 0,
        }]}],
        "skins": [{"name": "SMPL-H rig", "inverseBindMatrices": inverse_bind_acc, "joints": joint_nodes, "skeleton": joint_nodes[root_joint]}],
        "animations": [{"name": title, "samplers": samplers, "channels": channels}],
        "extras": {
            "sourceFps": float(source_fps),
            "previewFrames": int(len(frame_indices)),
            "sourceFrames": int(source_frame_count),
            "previewStride": stride,
            "coordinateConversion": "source Z-up to viewer Y-up: (x,z,-y)",
            "accuracy": "SMPL-H linear blend skinning proxy; pose corrective blend shapes omitted",
        },
    }
    json_chunk = _pad4(json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), b" ")
    bin_chunk = _pad4(bytes(b.binary))
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    payload = (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
        + struct.pack("<I4s", len(bin_chunk), b"BIN\0") + bin_chunk
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {
        "source_frames": int(frame_indices[-1] + 1 if len(frame_indices) else 0),
        "preview_frames": int(len(frame_indices)),
        "source_fps": float(source_fps),
        "preview_stride": stride,
        "duration": float(times[-1]) if len(times) else 0.0,
        "max_arm_rotation_step_degrees": max_arm_step,
        "arm_rotation_spikes_over_60deg": arm_spikes,
        "quality_warning": "raw arm rotation spikes detected" if arm_spikes else "",
    }
