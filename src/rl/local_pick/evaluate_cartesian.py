from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_balanced_lift_env import LocalPickCartesianBalancedLiftEnv
from rl.local_pick.local_pick_cartesian_colift_env import LocalPickCartesianCoLiftEnv
from rl.local_pick.local_pick_cartesian_direct_lift_env import LocalPickCartesianDirectLiftEnv
from rl.local_pick.local_pick_cartesian_env import LocalPickCartesianEnv
from rl.local_pick.local_pick_cartesian_hold_env import LocalPickCartesianHoldEnv
from rl.local_pick.local_pick_cartesian_lift_env import LocalPickCartesianLiftEnv
from rl.local_pick.local_pick_cartesian_progress_env import LocalPickCartesianProgressEnv
from rl.local_pick.local_pick_cartesian_stable_lift_env import LocalPickCartesianStableLiftEnv
from rl.local_pick.local_pick_cartesian_vertical_lift_env import LocalPickCartesianVerticalLiftEnv


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
    parser.add_argument("--hold-reward", action="store_true", help="Evaluate with the stable-hold Cartesian env.")
    parser.add_argument("--stable-lift", action="store_true", help="Evaluate with the stable-lift Cartesian env.")
    parser.add_argument("--balanced-lift", action="store_true", help="Evaluate with the balanced-lift Cartesian env.")
    parser.add_argument("--vertical-lift", action="store_true", help="Evaluate with the vertical-lift Cartesian env.")
    parser.add_argument("--colift", action="store_true", help="Evaluate with the co-lift Cartesian env.")
    parser.add_argument("--direct-lift", action="store_true", help="Evaluate with the direct-lift Cartesian env.")
    parser.add_argument("--progress", action="store_true", help="Evaluate with the progress-reward Cartesian env.")
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
    if args.progress:
        env_cls = LocalPickCartesianProgressEnv
    elif args.direct_lift:
        env_cls = LocalPickCartesianDirectLiftEnv
    elif args.colift:
        env_cls = LocalPickCartesianCoLiftEnv
    elif args.vertical_lift:
        env_cls = LocalPickCartesianVerticalLiftEnv
    elif args.balanced_lift:
        env_cls = LocalPickCartesianBalancedLiftEnv
    elif args.stable_lift:
        env_cls = LocalPickCartesianStableLiftEnv
    elif args.hold_reward:
        env_cls = LocalPickCartesianHoldEnv
    elif args.lift_reward:
        env_cls = LocalPickCartesianLiftEnv
    else:
        env_cls = LocalPickCartesianEnv

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
        "contact_bonus", "hold_bonus", "lift_action_bonus", "premature_close_penalty",
        "action_penalty", "table_penalty", "no_lift_contact_penalty",
        "invalid_lift_penalty", "object_motion_penalty", "pop_lift_penalty",
        "bad_contact_penalty", "hold_count",
        "lift_delta", "object_motion", "stability_penalty", "stable_lift_count",
        "ee_z_delta", "co_lift_reward", "lift_mismatch_penalty",
        "success_hold_count",
        "object_xy_motion", "object_xy_drift", "xy_action_penalty",
        "post_grasp_xy_penalty", "object_xy_motion_penalty", "xy_drift_penalty",
        "progress_reward", "task_score", "best_score", "reach_score",
        "close_score", "grasp_score", "lift_score", "stable_score",
        "reached_box", "gripper_grasping",
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
        print(f"  lift_action_bonus:     {totals['lift_action_bonus']:.2f}")
        print(f"  premature_close_pen:   {totals['premature_close_penalty']:.2f}")
        print(f"  action_penalty:        {totals['action_penalty']:.4f}")
        print(f"  table_penalty:         {totals['table_penalty']:.2f}")
        print(f"  no_lift_contact_pen:   {totals['no_lift_contact_penalty']:.2f}")
        print(f"  invalid_lift_pen:      {totals['invalid_lift_penalty']:.2f}")
        print(f"  object_motion_pen:     {totals['object_motion_penalty']:.2f}")
        print(f"  bad_contact_pen:       {totals['bad_contact_penalty']:.2f}")
        print(f"  pop_lift_pen:          {totals['pop_lift_penalty']:.2f}")
        print(f"  max/total hold_count:  {totals['hold_count']:.2f}")
        print(f"  total lift_delta:      {totals['lift_delta']:.4f} m")
        print(f"  total ee_z_delta:      {totals['ee_z_delta']:.4f} m")
        print(f"  total object_motion:   {totals['object_motion']:.4f} m")
        print(f"  stability_penalty:     {totals['stability_penalty']:.2f}")
        print(f"  stable_lift_count:     {totals['stable_lift_count']:.2f}")
        print(f"  co_lift_reward:        {totals['co_lift_reward']:.2f}")
        print(f"  lift_mismatch_pen:     {totals['lift_mismatch_penalty']:.2f}")
        print(f"  object_xy_motion:      {totals['object_xy_motion']:.4f} m")
        print(f"  object_xy_drift:       {totals['object_xy_drift']:.4f} m")
        print(f"  xy_action_penalty:     {totals['xy_action_penalty']:.4f}")
        print(f"  post_grasp_xy_pen:     {totals['post_grasp_xy_penalty']:.2f}")
        print(f"  object_xy_motion_pen:  {totals['object_xy_motion_penalty']:.2f}")
        print(f"  xy_drift_pen:          {totals['xy_drift_penalty']:.2f}")
        print(f"  progress_reward:       {totals['progress_reward']:.2f}")
        print(f"  task_score total:      {totals['task_score']:.2f}")
        print(f"  best_score total:      {totals['best_score']:.2f}")
        print(f"  reach_score total:     {totals['reach_score']:.2f}")
        print(f"  close_score total:     {totals['close_score']:.2f}")
        print(f"  grasp_score total:     {totals['grasp_score']:.2f}")
        print(f"  lift_score total:      {totals['lift_score']:.2f}")
        print(f"  stable_score total:    {totals['stable_score']:.2f}")
        print(f"  reached_box steps:     {totals['reached_box']:.0f}")
        print(f"  gripper_grasping steps:{totals['gripper_grasping']:.0f}")

    print(f"\n=== Success rate: {successes}/{args.episodes} ===")
    env.close()


if __name__ == "__main__":
    main()
