from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT_STR = str(SRC_ROOT)
if _SRC_ROOT_STR not in sys.path:
    sys.path.append(_SRC_ROOT_STR)

from rl.local_pick.config import LocalPickConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC for the local pick task.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=Path("models/local_pick_sac"))
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--object-noise", type=float, default=0.0)
    parser.add_argument("--approach-noise", type=float, default=0.0)
    parser.add_argument("--approach-z-noise", type=float, default=0.0)
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument(
        "--full-table-random",
        action="store_true",
        help="Sample object positions across the configured table bounds instead of near the fixed object.",
    )
    parser.add_argument(
        "--side-grasp",
        action="store_true",
        help="Use side grasp approach instead of top-down.",
    )
    return parser.parse_args()


def _make_env(config: LocalPickConfig):
    """Module-level factory so SubprocVecEnv can pickle it on Windows."""
    import sys
    if _SRC_ROOT_STR not in sys.path:
        sys.path.insert(0, _SRC_ROOT_STR)
    from stable_baselines3.common.monitor import Monitor
    from rl.local_pick.local_pick_env import LocalPickEnv
    return Monitor(LocalPickEnv(config=config, render_mode=None))


def main():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "Training requires stable-baselines3. Install dependencies with "
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

    env_fns = [partial(_make_env, config) for _ in range(args.n_envs)]
    try:
        env = SubprocVecEnv(env_fns)
        print(f"Using SubprocVecEnv with {args.n_envs} parallel environments.")
    except Exception as e:
        print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv.")
        env = DummyVecEnv(env_fns)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device="cuda",
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=512,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=8,
        learning_starts=5_000,
        use_sde=True,
        sde_sample_freq=4,
    )
    model.learn(total_timesteps=args.timesteps)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    env.close()
    print(f"Saved local pick model to {args.model_path}")


if __name__ == "__main__":
    main()
