"""Try different joint7 values at approach pose and save wrist cam images."""
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
_OUT_DIR.mkdir(exist_ok=True)

OBJECT_POS = [0.5, -0.15, 0.125]
J7_VALUES = np.linspace(2.70, 2.8973, 10)  # fine sweep near joint limit for alignment


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ImportError:
        h, w = rgb.shape[:2]
        with open(path.with_suffix(".ppm"), "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(rgb.tobytes())


config = LocalPickConfig(
    fixed_object_position=tuple(OBJECT_POS),
    approach_xy_noise=0.0,
    object_xy_noise=0.0,
)
env = LocalPickEnv(config=config, render_mode=None)

render_h, render_w = 240, 320
renderer = mujoco.Renderer(env.model, height=render_h, width=render_w)
cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

for j7 in J7_VALUES:
    # Force joint7 to this value, keep other joints from normal reset
    env.reset(seed=0)
    env.data.qpos[env.qpos_idx[-1]] = j7
    env.data.ctrl[env.actuator_ids[-1]] = j7
    mujoco.mj_forward(env.model, env.data)

    renderer.update_scene(env.data, camera=cam_id)
    rgb = renderer.render().copy()

    label = f"j7_{j7:+.2f}"
    save_rgb(rgb, _OUT_DIR / f"wrist_{label}.png")
    print(f"j7={j7:+.3f} rad ({np.degrees(j7):+.1f}°) → saved wrist_{label}.png")

renderer.close()
env.close()
print(f"\nAll images saved to {_OUT_DIR}")
print("Open them and find which j7 has fingers aligned with the bottle.")
