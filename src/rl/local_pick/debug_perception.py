"""Quick script to compare ground-truth object position vs camera estimate."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig


def pixel_to_world(model, data, cam_id, px, py, depth, render_h, render_w):
    fovy = model.cam_fovy[cam_id]
    f = 0.5 * render_h / np.tan(np.radians(fovy / 2))
    d = float(depth[py, px])
    cx, cy = render_w / 2.0, render_h / 2.0
    x_cam = (px - cx) * d / f
    y_cam = -(py - cy) * d / f
    z_cam = -d
    cam_pos = data.cam_xpos[cam_id]
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
    return cam_pos + cam_rot @ np.array([x_cam, y_cam, z_cam])


def perception_com(model, data, cam_id, xs, ys, depth, render_h, render_w):
    """Unproject all detected pixels to 3D and return their mean."""
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
    com = points_world.mean(axis=0)
    com[2] -= 0.029
    return com


def camera_estimate(model, data, cam_id, renderer, render_h, render_w):
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    target = np.array([230, 38, 38], dtype=float)
    diff = np.linalg.norm(rgb.astype(float) - target, axis=2)
    ys, xs = np.where(diff < 50)
    if len(xs) == 0:
        return None, None, None

    world_pos = perception_com(model, data, cam_id, xs, ys, depth, render_h, render_w)
    return world_pos, None, int(len(xs))


def main():
    config = LocalPickConfig()
    model = mujoco.MjModel.from_xml_path(str(config.xml_path))
    data = mujoco.MjData(model)

    render_h, render_w = 120, 160
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "perception_cam")
    renderer = mujoco.Renderer(model, height=render_h, width=render_w)

    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_object")
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pick_freejoint")
    object_qpos_idx = model.jnt_qposadr[object_joint_id]

    test_positions = [
        [0.43, -0.20, 0.125],
        [0.50, -0.15, 0.125],
        [0.57, -0.10, 0.125],
        [0.45, -0.18, 0.125],
        [0.55, -0.12, 0.125],
    ]

    print(f"{'Position (GT)':>30}  {'Camera Est':>30}  {'Error (m)':>10}  {'Pixels':>8}")
    print("-" * 90)

    for pos in test_positions:
        mujoco.mj_resetData(model, data)
        data.qpos[object_qpos_idx:object_qpos_idx + 3] = pos
        data.qpos[object_qpos_idx + 3:object_qpos_idx + 7] = [0.7071, 0.7071, 0, 0]
        mujoco.mj_forward(model, data)

        gt = data.xpos[object_body_id].copy()
        cam_pos, pixel, n_pixels = camera_estimate(model, data, cam_id, renderer, render_h, render_w)

        if cam_pos is None:
            print(f"  GT={np.array2string(gt, precision=3)}  CAMERA: object not detected")
            continue

        err = cam_pos - gt
        total_err = np.linalg.norm(err)
        print(
            f"  GT  [{gt[0]:+.3f} {gt[1]:+.3f} {gt[2]:+.3f}]"
            f"  CAM [{cam_pos[0]:+.3f} {cam_pos[1]:+.3f} {cam_pos[2]:+.3f}]"
            f"  err [{err[0]:+.4f} {err[1]:+.4f} {err[2]:+.4f}]"
            f"  |err|={total_err:.4f}m"
            f"  pixels={n_pixels}"
        )

    renderer.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
