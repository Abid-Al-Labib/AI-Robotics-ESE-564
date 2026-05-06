"""Measure camera accuracy with arm in approach pose across multiple object positions."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_env import LocalPickEnv

_OUT_DIR = Path(__file__).resolve().parent / "debug_frames"


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ImportError:
        # fallback: write raw PPM (no dependency)
        h, w = rgb.shape[:2]
        with open(path.with_suffix(".ppm"), "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(rgb.tobytes())


def perception_com(model, data, cam_id, renderer, render_h, render_w, save_path=None):
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    if save_path is not None:
        save_rgb(rgb, save_path)

    target = np.array([230, 38, 38], dtype=float)
    diff = np.linalg.norm(rgb.astype(float) - target, axis=2)
    ys, xs = np.where(diff < 50)
    if len(xs) == 0:
        return None, 0, rgb

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
    return points_world.mean(axis=0), len(xs), rgb


def main():
    test_positions = [
        [0.43, -0.20, 0.125],
        [0.50, -0.15, 0.125],
        [0.57, -0.10, 0.125],
        [0.45, -0.18, 0.125],
        [0.55, -0.12, 0.125],
    ]

    render_h, render_w = 240, 320  # larger for saved images

    _OUT_DIR.mkdir(exist_ok=True)

    for cam_name in ["perception_cam", "wrist_cam"]:
        print(f"\n=== {cam_name} — arm in approach pose ===")

        for i, pos in enumerate(test_positions):
            config = LocalPickConfig(
                fixed_object_position=tuple(pos),
                approach_xy_noise=0.0,
                object_xy_noise=0.0,
            )
            env = LocalPickEnv(config=config, render_mode=None)
            env.reset(seed=0)

            model = env.model
            data = env.data
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            renderer = mujoco.Renderer(model, height=render_h, width=render_w)

            # Print camera world pose for first position
            if i == 0:
                cam_pos_w = data.cam_xpos[cam_id].copy()
                cam_rot_w = data.cam_xmat[cam_id].reshape(3, 3)
                # Camera looks along its -Z axis in world frame
                look_dir = -cam_rot_w[:, 2]
                print(f"  cam world pos  : {cam_pos_w}")
                print(f"  cam look dir   : {look_dir}  (should point toward table for wrist_cam)")

            save_path = _OUT_DIR / f"{cam_name}_pos{i}.png" if i == 0 else None
            gt = data.xpos[env.object_body_id].copy()
            result = perception_com(model, data, cam_id, renderer, render_h, render_w, save_path)
            cam_est, n_px, rgb = result

            if i == 0 and save_path:
                ext = "png" if save_path.exists() else "ppm"
                print(f"  frame saved    : {save_path.with_suffix('.' + ext)}")

            if cam_est is None:
                print(f"  GT [{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}]  NOT DETECTED")
            else:
                err = cam_est - gt
                print(
                    f"  GT  [{gt[0]:+.3f} {gt[1]:+.3f} {gt[2]:+.3f}]"
                    f"  CAM [{cam_est[0]:+.3f} {cam_est[1]:+.3f} {cam_est[2]:+.3f}]"
                    f"  err [{err[0]:+.4f} {err[1]:+.4f} {err[2]:+.4f}]"
                    f"  |err| {np.linalg.norm(err):.4f}m  px {n_px}"
                )

            renderer.close()
            env.close()


if __name__ == "__main__":
    main()
