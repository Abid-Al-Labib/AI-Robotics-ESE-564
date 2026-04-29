from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_align_env import LocalPickAlignEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC for local pick alignment only.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=Path("models/local_pick_align"))
    parser.add_argument("--object-noise", type=float, default=0.0)
    parser.add_argument("--approach-noise", type=float, default=0.0)
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument("--full-table-random", action="store_true")
    return parser.parse_args()


def main():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError(
            "Training requires stable-baselines3. Install dependencies with "
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
    env = Monitor(LocalPickAlignEnv(config=config, render_mode=None))

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.95,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
    )
    model.learn(total_timesteps=args.timesteps)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    env.close()
    print(f"Saved local pick align model to {args.model_path}")


if __name__ == "__main__":
    main()

