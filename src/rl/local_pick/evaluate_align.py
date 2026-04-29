from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_align_env import LocalPickAlignEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained local pick alignment model.")
    parser.add_argument("--model-path", type=Path, default=Path("models/local_pick_align"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-noise", type=float, default=0.02)
    parser.add_argument("--approach-noise", type=float, default=0.01)
    parser.add_argument("--joint-noise", type=float, default=0.01)
    parser.add_argument("--full-table-random", action="store_true")
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
        joint_noise=args.joint_noise,
        max_episode_steps=40,
    )
    env = LocalPickAlignEnv(config=config, render_mode="human")
    model = SAC.load(args.model_path)

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.03)
            done = terminated or truncated
        print(
            f"Episode {episode + 1}: reward={total_reward:.2f}, "
            f"success={info.get('success', False)}, "
            f"reach_distance={info.get('reach_distance', float('nan')):.4f}"
        )
    env.close()


if __name__ == "__main__":
    main()

