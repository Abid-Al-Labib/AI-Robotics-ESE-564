from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_env import LocalPickEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained local pick SAC model.")
    parser.add_argument("--model-path", type=Path, default=Path("models/local_pick_sac"))
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
    parser.add_argument("--no-render", action="store_true", help="Disable viewer window.")
    return parser.parse_args()


def main():
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires stable-baselines3. Install dependencies with "
            "`pip install -r src/requirements.txt`."
        ) from exc

    args = parse_args()
    config = LocalPickConfig(
        seed=args.seed,
        fixed_object_position=None if args.full_table_random else (0.4, -0.15, 0.13),
        object_xy_noise=args.object_noise,
        approach_xy_noise=args.approach_noise,
        approach_z_noise=args.approach_z_noise,
        joint_noise=args.joint_noise,
    )
    render_mode = None if args.no_render else "human"
    env = LocalPickEnv(config=config, render_mode=render_mode)
    model = SAC.load(args.model_path)

    reward_keys = [
        "reach_distance", "object_lift", "align_close_bonus",
        "contact_bonus", "hold_bonus", "premature_close_penalty",
        "action_penalty", "table_penalty",
    ]

    successes = 0
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        totals = {k: 0.0 for k in reward_keys}
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            for k in reward_keys:
                totals[k] += info.get(k, 0.0)
            if render_mode == "human":
                env.render()
                time.sleep(0.02)
            done = terminated or truncated

        success = info.get("success", False)
        if success:
            successes += 1
        print(
            f"\nEpisode {episode + 1}/{args.episodes}  "
            f"reward={total_reward:.2f}  steps={steps}  success={success}"
        )
        print(f"  avg reach_dist:        {totals['reach_distance'] / steps:.4f} m")
        print(f"  total lift:            {totals['object_lift']:.4f} m·steps")
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

