"""Watch the trained pixel SAC policy pick the bottle in the MuJoCo viewer."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/local_pick_drq_2m"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds per step (slow down to watch).")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    import mujoco
    import mujoco.viewer as mj_viewer
    from stable_baselines3 import SAC
    from rl.local_pick.config import LocalPickConfig
    from rl.local_pick.local_pick_cartesian_pixel_env import LocalPickCartesianPixelEnv

    args = parse_args()
    config = LocalPickConfig(
        seed=args.seed,
        fixed_object_position=(0.5, -0.15, 0.125),
    )

    # Create env headless — viewer runs separately so no OpenGL conflict.
    env = LocalPickCartesianPixelEnv(config=config, render_mode=None, aug_pad=0)
    model = SAC.load(args.model_path, device="auto")
    print(f"Loaded model from {args.model_path}")

    n_stack = 3
    successes = 0

    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135

        for ep in range(args.episodes):
            obs, _ = env.reset()

            # Initialise frame stack with the first observation repeated.
            frames = [obs.copy() for _ in range(n_stack)]

            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                # Stack frames (84,84,9) HWC → (9,84,84) CHW → add batch dim.
                stacked = np.concatenate(frames, axis=2)
                stacked_chw = stacked.transpose(2, 0, 1)[np.newaxis]

                action, _ = model.predict(stacked_chw, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action[0])

                frames.pop(0)
                frames.append(obs.copy())

                total_reward += reward
                steps += 1
                done = terminated or truncated

                viewer.sync()
                time.sleep(args.delay)

            success = bool(info.get("success", False))
            if success:
                successes += 1
            print(
                f"Episode {ep + 1}/{args.episodes}  "
                f"reward={total_reward:.1f}  steps={steps}  success={success}"
            )
            time.sleep(0.5)  # pause between episodes so you can see the reset

    env.close()
    print(f"\n=== Success rate: {successes}/{args.episodes} ===")


if __name__ == "__main__":
    main()
