from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_env import LocalPickCartesianEnv
from rl.local_pick.local_pick_cartesian_lift_env import LocalPickCartesianLiftEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Cartesian local pick SAC model.")
    parser.add_argument("--model-path", type=Path, default=Path("models/local_pick_cartesian_sac"))
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-noise", type=float, default=0.0)
    parser.add_argument("--approach-noise", type=float, default=0.0)
    parser.add_argument("--approach-z-noise", type=float, default=0.0)
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument(
        "--full-table-random",
        action="store_true",
        help="Evaluate object positions across the configured table bounds.",
    )
    parser.add_argument(
        "--side-grasp",
        action="store_true",
        help="Use side grasp approach instead of top-down.",
    )
    parser.add_argument("--lift-reward", action="store_true", help="Evaluate with the lift-focused Cartesian env.")
    parser.add_argument("--no-render", action="store_true", help="Disable viewer window.")
    return parser.parse_args()


def main():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires stable-baselines3. Install dependencies with "
            "`pip install -r src/requirements.txt`."
        ) from exc

    args = parse_args()
    config = LocalPickConfig(
        seed=args.seed,
        fixed_object_position=None if args.full_table_random else (0.5, -0.15, 0.125),
        object_xy_noise=args.object_noise,
        approach_xy_noise=args.approach_noise,
        approach_z_noise=args.approach_z_noise,
        joint_noise=args.joint_noise,
        side_grasp=args.side_grasp,
    )
    render_mode = None if args.no_render else "human"
    env_cls = LocalPickCartesianLiftEnv if args.lift_reward else LocalPickCartesianEnv

    def make_env():
        return env_cls(config=config, render_mode=render_mode)

    env = DummyVecEnv([make_env])
    vecnormalize_path = args.vecnormalize_path
    if vecnormalize_path is None:
        vecnormalize_path = args.model_path.with_name(args.model_path.name + "_vecnormalize.pkl")
    if vecnormalize_path.exists():
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats from {vecnormalize_path}")
    else:
        print(f"VecNormalize stats not found at {vecnormalize_path}; evaluating with raw observations.")

    model = SAC.load(args.model_path)

    reward_keys = [
        "reach_distance", "object_lift", "align_close_bonus",
        "contact_bonus", "hold_bonus", "premature_close_penalty",
        "action_penalty", "table_penalty",
    ]

    successes = 0
    for episode in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        totals = {k: 0.0 for k in reward_keys}
        ik_failures = 0
        steps = 0
        done = [False]

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = env.step(action)
            info = infos[0]
            total_reward += float(rewards[0])
            steps += 1
            ik_failures += int(bool(info.get("ik_failed", False)))
            for k in reward_keys:
                totals[k] += info.get(k, 0.0)
            if render_mode == "human":
                unwrapped = env.envs[0]
                if hasattr(unwrapped, "render"):
                    unwrapped.render()
                time.sleep(0.02)

        success = bool(info.get("success", False))
        if success:
            successes += 1
        print(
            f"\nEpisode {episode + 1}/{args.episodes}  "
            f"reward={total_reward:.2f}  steps={steps}  success={success}  ik_failures={ik_failures}"
        )
        print(f"  avg reach_dist:        {totals['reach_distance'] / steps:.4f} m")
        print(f"  total lift:            {totals['object_lift']:.4f} m*steps")
        print(f"  align_close_bonus:     {totals['align_close_bonus']:.2f}")
        print(f"  contact_bonus:         {totals['contact_bonus']:.2f}")
        print(f"  hold_bonus:            {totals['hold_bonus']:.2f}")
        print(f"  premature_close_pen:   {totals['premature_close_penalty']:.2f}")
        print(f"  action_penalty:        {totals['action_penalty']:.4f}")
        print(f"  table_penalty:         {totals['table_penalty']:.2f}")

    print(f"\n=== Success rate: {successes}/{args.episodes} ===")
    env.close()


if __name__ == "__main__":
    main()
