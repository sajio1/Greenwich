import numpy as np
import torch

from alphamotion.perception.genmo import (
    _load_space_motion,
    global_to_local_rot6d,
    local_to_global_rot6d,
    smpl_root_translation,
    smpl_to_global_rot6d,
)


def _artifact():
    frames = 5
    return {
        "body_params_global": {
            "global_orient": torch.zeros(frames, 3),
            "body_pose": torch.zeros(frames, 63),
            "transl": torch.tensor([
                [9.0, 1.0, -2.0], [9.1, 1.0, -2.2],
                [9.2, 1.0, -2.4], [9.3, 1.0, -2.6],
                [9.4, 1.0, -2.8],
            ]),
        },
        "segment_info": [{"start": 2, "end": 5, "type": "text"}],
    }


def test_genmo_motion_keeps_rotation_and_anchored_world_translation():
    artifact = _artifact()
    rot = smpl_to_global_rot6d(artifact, "text")
    root = smpl_root_translation(artifact, "text")
    assert rot.shape == (3, 22, 6)
    assert root.shape == (3, 3)
    np.testing.assert_allclose(root[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(root[-1], [20.0, 0.0, -40.0], atol=1e-4)


def test_local_global_rot6d_round_trip():
    global_rot = smpl_to_global_rot6d(_artifact())
    local = global_to_local_rot6d(global_rot)
    restored = local_to_global_rot6d(local)
    np.testing.assert_allclose(restored.numpy(), global_rot.numpy(), atol=1e-5)


def test_remote_space_npz_is_safe_and_anchored(tmp_path):
    global_rot = smpl_to_global_rot6d(_artifact())
    local = global_to_local_rot6d(global_rot).numpy()
    root = np.asarray([[8.0, 1.0, -4.0], [9.0, 1.0, -6.0],
                       [10.0, 1.0, -8.0], [11.0, 1.0, -10.0],
                       [12.0, 1.0, -12.0]], np.float32)
    path = tmp_path / "motion.npz"
    np.savez_compressed(path, local_rot6d=local, root_cm=root,
                        fps=np.asarray(30.0, np.float32))
    restored, anchored, fps = _load_space_motion(str(path))
    np.testing.assert_allclose(restored.numpy(), global_rot.numpy(), atol=1e-5)
    np.testing.assert_allclose(anchored[0], 0.0)
    np.testing.assert_allclose(anchored[-1], [4.0, 0.0, -8.0])
    assert fps == 30.0
