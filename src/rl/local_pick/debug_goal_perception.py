"""Compare ground-truth goal platform position vs camera estimate."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig


def perception_com(model, data, cam_id, xs, ys, depth, render_h, render_w):
    fovy = model.cam_fovy[cam_id]
    f = 0.5 * render_h / np.tan(np.radians(fovy / 2))
    cx, cy = render_w / 2.0, render_h / 2.0
    cam_pos = data.cam_xpos[cam_id]
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
    d = depth[ys, xs].astype(float)
    x_cam = (xs - cx) * d / f
    y_cam = -(ys - cy) * d / f
    z_cam = -d
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    points_world = cam_pos + points_cam @ cam_rot.T
    return points_world.mean(axis=0)


def camera_estimate(model, data, cam_id, renderer, render_h, render_w, target_color, threshold=50):
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    diff = np.linalg.norm(rgb.astype(float) - np.array(target_color, dtype=float), axis=2)
    ys, xs = np.where(diff < threshold)
    if len(xs) == 0:
        return None, 0
    return perception_com(model, data, cam_id, xs, ys, depth, render_h, render_w), len(xs)


def main():
    config = LocalPickConfig()
    model = mujoco.MjModel.from_xml_path(str(config.xml_path))
    data = mujoco.MjData(model)

    render_h, render_w = 120, 160
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "perception_cam")
    renderer = mujoco.Renderer(model, height=render_h, width=render_w)

    goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_platform")
    goal_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "goal_freejoint")
    goal_qpos_idx = model.jnt_qposadr[goal_joint_id]

    test_positions = [
        [0.43, 0.10, 0.1],
        [0.50, 0.15, 0.1],
        [0.57, 0.20, 0.1],
        [0.45, 0.12, 0.1],
        [0.55, 0.22, 0.1],
    ]

    print("Goal platform — raw CoM estimate (no z correction applied)")
    print("-" * 90)

    for pos in test_positions:
        mujoco.mj_resetData(model, data)
        data.qpos[goal_qpos_idx:goal_qpos_idx + 3] = pos
        data.qpos[goal_qpos_idx + 3:goal_qpos_idx + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        gt = data.xpos[goal_body_id].copy()
        cam_pos, n_pixels = camera_estimate(
            model, data, cam_id, renderer, render_h, render_w,
            target_color=[26, 217, 26], threshold=60
        )

        if cam_pos is None:
            print(f"  GT={np.array2string(gt, precision=3)}  CAMERA: goal not detected")
            continue

        err = cam_pos - gt
        print(
            f"  GT  [{gt[0]:+.3f} {gt[1]:+.3f} {gt[2]:+.3f}]"
            f"  CAM [{cam_pos[0]:+.3f} {cam_pos[1]:+.3f} {cam_pos[2]:+.3f}]"
            f"  err [{err[0]:+.4f} {err[1]:+.4f} {err[2]:+.4f}]"
            f"  |err|={np.linalg.norm(err):.4f}m"
            f"  pixels={n_pixels}"
        )

    renderer.close()


if __name__ == "__main__":
    main()
