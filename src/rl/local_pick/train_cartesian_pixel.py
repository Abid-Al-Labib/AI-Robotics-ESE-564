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
    parser = argparse.ArgumentParser(description="Train DrQ-style pixel SAC for Cartesian local pick.")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=Path("models/local_pick_drq_2m"))
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-stack", type=int, default=3)
    parser.add_argument("--aug-pad", type=int, default=4, help="Random shift pad in pixels. 0 = no aug.")
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--object-noise", type=float, default=0.0)
    parser.add_argument("--approach-noise", type=float, default=0.0)
    parser.add_argument("--approach-z-noise", type=float, default=0.0)
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument("--full-table-random", action="store_true")
    parser.add_argument("--random-j7", action="store_true",
                        help="Randomise wrist joint (j7) at episode start so policy handles any pipeline approach.")
    parser.add_argument("--min-lift", type=float, default=0.02,
                        help="Minimum lift height (m) required for success.")
    parser.add_argument("--hold-steps", type=int, default=3,
                        help="Consecutive steps meeting lift+contact criteria to count as success.")
    parser.add_argument("--require-both-fingers", action="store_true",
                        help="Require both fingers contacting object for success (forces centered grasp).")
    parser.add_argument("--both-finger-bonus", type=float, default=3.0,
                        help="Reward for both fingers contacting object simultaneously.")
    parser.add_argument("--drop-penalty", type=float, default=2.0,
                        help="Penalty for dropping object after lifting it.")
    parser.add_argument("--hold-lift-threshold", type=float, default=0.04,
                        help="Min lift (m) before hold reward and drop penalty activate.")
    parser.add_argument("--load-from", type=Path, default=None,
                        help="Path to existing model zip to fine-tune from instead of training from scratch.")
    parser.add_argument("--dummy-vec", action="store_true")
    return parser.parse_args()


def _make_env(config: LocalPickConfig, aug_pad: int):
    import sys
    if _SRC_ROOT_STR not in sys.path:
        sys.path.insert(0, _SRC_ROOT_STR)
    from stable_baselines3.common.monitor import Monitor
    from rl.local_pick.local_pick_cartesian_pixel_env import LocalPickCartesianPixelEnv
    return Monitor(LocalPickCartesianPixelEnv(config=config, render_mode=None, aug_pad=aug_pad))


def main():
    try:
        import torch
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
    except ImportError as exc:
        raise ImportError("Run: pip install 'stable-baselines3[extra]>=2.0'") from exc

    args = parse_args()
    config = LocalPickConfig(
        seed=args.seed,
        fixed_object_position=None if args.full_table_random else (0.5, -0.15, 0.125),
        object_xy_noise=args.object_noise,
        approach_xy_noise=args.approach_noise,
        approach_z_noise=args.approach_z_noise,
        joint_noise=args.joint_noise,
        random_j7=args.random_j7,
        min_lift_for_success=args.min_lift,
        success_hold_steps=args.hold_steps,
        require_both_fingers=args.require_both_fingers,
        both_finger_bonus=args.both_finger_bonus,
        drop_penalty=args.drop_penalty,
        hold_lift_threshold=args.hold_lift_threshold,
    )

    env_fns = [partial(_make_env, config, args.aug_pad) for _ in range(args.n_envs)]
    if args.dummy_vec:
        vec_env = DummyVecEnv(env_fns)
        print(f"Using DummyVecEnv with {args.n_envs} environments.")
    else:
        try:
            vec_env = SubprocVecEnv(env_fns)
            print(f"Using SubprocVecEnv with {args.n_envs} parallel environments.")
        except Exception as e:
            print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv.")
            vec_env = DummyVecEnv(env_fns)

    # Stack 3 frames: (84,84,3) -> (84,84,9) for temporal context.
    env = VecFrameStack(vec_env, n_stack=args.n_stack)
    # Transpose HWC -> CHW *before* VecNormalize so get_original_obs() returns CHW.
    # SB3 would add VecTransposeImage automatically but only after VecNormalize,
    # which causes a shape mismatch in the replay buffer.
    env = VecTransposeImage(env)
    # Reward normalisation only — obs already CHW uint8, CnnPolicy handles it.
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    aug_label = f"random-shift pad={args.aug_pad}px" if args.aug_pad > 0 else "none"
    print(f"Device: {device}  |  Augmentation: {aug_label}  |  Buffer: {args.buffer_size:,}")

    if args.load_from is not None:
        model = SAC.load(args.load_from, env=env, device=device, verbose=1)
        print(f"Fine-tuning from {args.load_from}")
    else:
        model = SAC(
            "CnnPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device=device,
            learning_rate=1e-4,
            buffer_size=args.buffer_size,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            learning_starts=1_000,
            use_sde=False,
        )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=str(args.model_path.parent),
        name_prefix=args.model_path.name,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)
    model.save(args.model_path)
    env.save(str(args.model_path.with_name(args.model_path.name + "_vecnormalize.pkl")))
    env.close()
    print(f"Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
