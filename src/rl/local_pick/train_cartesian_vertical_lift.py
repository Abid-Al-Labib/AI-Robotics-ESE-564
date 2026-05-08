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
    parser = argparse.ArgumentParser(description="Train SAC for vertical Cartesian local pick lift.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=Path("models/local_pick_cartesian_vertical_lift_sac"))
    parser.add_argument("--pretrained", type=Path, default=None, help="Existing SAC .zip to continue from.")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--object-noise", type=float, default=0.0)
    parser.add_argument("--approach-noise", type=float, default=0.0)
    parser.add_argument("--approach-z-noise", type=float, default=0.0)
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument("--full-table-random", action="store_true")
    parser.add_argument("--side-grasp", action="store_true")
    parser.add_argument("--dummy-vec", action="store_true")
    return parser.parse_args()


def _make_env(config: LocalPickConfig):
    import sys

    if _SRC_ROOT_STR not in sys.path:
        sys.path.insert(0, _SRC_ROOT_STR)

    from stable_baselines3.common.monitor import Monitor
    from rl.local_pick.local_pick_cartesian_vertical_lift_env import LocalPickCartesianVerticalLiftEnv

    return Monitor(LocalPickCartesianVerticalLiftEnv(config=config, render_mode=None))


def main():
    try:
        import torch
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
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
    if args.dummy_vec:
        env = DummyVecEnv(env_fns)
        print(f"Using DummyVecEnv with {args.n_envs} environments.")
    else:
        try:
            env = SubprocVecEnv(env_fns)
            print(f"Using SubprocVecEnv with {args.n_envs} parallel environments.")
        except Exception as e:
            print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv.")
            env = DummyVecEnv(env_fns)

    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.pretrained is not None:
        model = SAC.load(args.pretrained, env=env, device=device)
        model.num_timesteps = 0
        model._episode_num = 0
        model.replay_buffer.reset()
        print(f"Loaded pretrained policy from {args.pretrained}")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device=device,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            learning_starts=2_000,
            use_sde=True,
            sde_sample_freq=4,
        )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(args.model_path.parent),
        name_prefix=args.model_path.name,
    )
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)
    model.save(args.model_path)
    env.save(str(args.model_path.with_name(args.model_path.name + "_vecnormalize.pkl")))
    env.close()
    print(f"Saved vertical Cartesian lift model to {args.model_path}")


if __name__ == "__main__":
    main()
