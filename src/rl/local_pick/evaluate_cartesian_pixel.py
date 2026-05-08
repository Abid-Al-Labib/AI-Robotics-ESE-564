from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_pixel_env import LocalPickCartesianPixelEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained pixel SAC model.")
    parser.add_argument("--model-path", type=Path, default=Path("models/local_pick_option_b_pixel_2m"))
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-stack", type=int, default=3)
    parser.add_argument("--object-noise", type=float, default=0.0)
    parser.add_argument("--approach-noise", type=float, default=0.0)
    parser.add_argument("--approach-z-noise", type=float, default=0.0)
    parser.add_argument("--random-j7", action="store_true")
    parser.add_argument("--full-table-random", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
    except ImportError as exc:
        raise ImportError("Run: pip install 'stable-baselines3[extra]>=2.0'") from exc

    args = parse_args()
    config = LocalPickConfig(
        seed=args.seed,
        fixed_object_position=None if args.full_table_random else (0.5, -0.15, 0.125),
        object_xy_noise=args.object_noise,
        approach_xy_noise=args.approach_noise,
        approach_z_noise=args.approach_z_noise,
        random_j7=args.random_j7,
    )

    # Always run headless — render_mode="human" conflicts with the pixel renderer.
    # Visualization is handled via mujoco passive viewer below.
    env = DummyVecEnv([lambda: LocalPickCartesianPixelEnv(config=config, render_mode=None)])
    env = VecFrameStack(env, n_stack=args.n_stack)
    env = VecTransposeImage(env)  # HWC -> CHW, must match training stack order

    vecnormalize_path = args.vecnormalize_path
    if vecnormalize_path is None:
        vecnormalize_path = args.model_path.with_name(args.model_path.name + "_vecnormalize.pkl")
    if vecnormalize_path.exists():
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize from {vecnormalize_path}")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=False)
        print("No VecNormalize found — evaluating with raw observations.")

    model = SAC.load(args.model_path, device="auto")

    # Launch passive viewer if requested — separate from the pixel renderer so no conflict.
    viewer = None
    if not args.no_render:
        try:
            import mujoco
            import mujoco.viewer as mj_viewer
            inner = env
            while hasattr(inner, 'venv'):
                inner = inner.venv
            raw_env = inner.envs[0]
            viewer = mj_viewer.launch_passive(raw_env.model, raw_env.data)
            print("Viewer launched. Watch the simulation window.")
        except Exception as e:
            print(f"Could not launch viewer ({e}). Running headless.")

    reward_keys = [
        "reach_distance", "object_lift", "progress_reward", "task_score",
        "best_score", "reached_box", "gripper_grasping",
        "premature_close_penalty", "action_penalty", "table_penalty",
        "success_hold_count",
    ]

    successes = 0
    for episode in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        totals = {k: 0.0 for k in reward_keys}
        steps = 0
        done = [False]

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = env.step(action)
            info = infos[0]
            total_reward += float(rewards[0])
            steps += 1
            for k in reward_keys:
                totals[k] += info.get(k, 0.0)
            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.05)  # slow down so you can watch

        success = bool(info.get("success", False))
        if success:
            successes += 1

        print(
            f"\nEpisode {episode + 1}/{args.episodes}  "
            f"reward={total_reward:.2f}  steps={steps}  success={success}"
        )
        print(f"  avg reach_dist:        {totals['reach_distance'] / steps:.4f} m")
        print(f"  total object_lift:     {totals['object_lift']:.4f} m*steps")
        print(f"  progress_reward:       {totals['progress_reward']:.2f}")
        print(f"  best_score (final):    {totals['best_score'] / steps:.2f}")
        print(f"  reached_box steps:     {totals['reached_box']:.0f}")
        print(f"  gripper_grasping steps:{totals['gripper_grasping']:.0f}")
        print(f"  premature_close_pen:   {totals['premature_close_penalty']:.2f}")
        print(f"  table_penalty:         {totals['table_penalty']:.2f}")

    print(f"\n=== Success rate: {successes}/{args.episodes} ===")
    if viewer is not None:
        viewer.close()
    env.close()


if __name__ == "__main__":
    main()
