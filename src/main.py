import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.mujoco_env import MujocoEnv
from pipeline.controller.kinematics import Kinematics
from pipeline.controller.gripper_controls import GripperController
from pipeline.controller.arm_controller import ArmController
from pipeline.perception.perception import Perception
from pipeline.planner.rrt_planner import RRTPlanner
from pipeline.localPick.rl_pixel_pick import RLPixelPickController

PICK_POSE_CORRECTION = np.array([0.0, 0.020, -0.026], dtype=float)
CLEARANCE_Z = 0.38
PLACE_Z_OFFSET = 0.25
PICK_RETRIES = 3
TRIAL_RETRIES = 3
TARGET_ROT = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(description="Pick-and-place pipeline.")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to run.")
    return parser.parse_args()


def wait(viewer, seconds):
    start = time.time()
    while viewer.is_running() and time.time() - start < seconds:
        viewer.sync()
        time.sleep(0.01)


def get_ik_goal(kin, planner, target_pos, z_offset=0.15):
    approach_pos = kin.approach_position(target_pos, z_offset=z_offset)
    solutions = kin.ik(approach_pos, TARGET_ROT, free_joint_samples=100)
    if not solutions:
        return None
    q_current = planner.get_joint_positions()
    valid = [q for q in solutions if planner.is_config_valid(q)]
    candidates = valid if valid else solutions
    return min(candidates, key=lambda q: np.linalg.norm(q - q_current))


def move_to_pose(kin, planner, arm, viewer, target_pos, label, z_offset=0.15, max_joint_step=0.04):
    q_goal = get_ik_goal(kin, planner, target_pos, z_offset=z_offset)
    if q_goal is None:
        print(f"  ERROR: IK failed — {label}")
        return None
    path = planner.plan(q_goal)
    if not path:
        print(f"  ERROR: RRT failed — {label}")
        return None
    arm.execute_path(path, viewer, max_joint_step=max_joint_step)
    approach_pos = kin.approach_position(target_pos, z_offset=z_offset)
    ee_pos, _ = kin.fk(arm.get_joint_positions())
    if np.linalg.norm(ee_pos - approach_pos) > 0.04:
        arm.move_to_smooth(q_goal, viewer, max_joint_step=0.015, tol=0.008, vel_tol=0.04, max_steps_per_target=4000)
    return q_goal


def run_rl_pick(local_pick, gripper, arm, planner, kin, perception, pick_pos, q_approach, viewer):
    for attempt in range(PICK_RETRIES):
        gripper.open(viewer)
        wait(viewer, 0.25)
        success = local_pick.execute(viewer)
        if not success:
            print(f"  Pick {attempt + 1}/{PICK_RETRIES}: policy failed")
            return False

        clearance_pos = pick_pos.copy()
        clearance_pos[2] = CLEARANCE_Z
        q_clearance = get_ik_goal(kin, planner, clearance_pos, z_offset=0.0)
        if q_clearance is not None:
            arm.move_to_smooth(q_clearance, viewer, max_joint_step=0.01)
        wait(viewer, 0.5)

        obj_pos = perception.get_pick_object_position()
        object_z = float(obj_pos[2]) if obj_pos is not None else 0.0
        holding = object_z > 0.20
        print(f"  Pick {attempt + 1}/{PICK_RETRIES}: {'holding' if holding else 'DROPPED'} (z={object_z:.3f})")

        if holding:
            return True
        if attempt < PICK_RETRIES - 1:
            arm.move_to_smooth(q_approach, viewer, max_joint_step=0.01)
            wait(viewer, 0.3)

    return False


def run_pick_and_place(env, kin, perception, planner, gripper, arm, local_pick, viewer, goal_pos):
    planner.set_ignore_held_object(False)

    pick_pos, place_pos = perception.get_object_positions()
    if pick_pos is None:
        print("  ERROR: perception — pick object not visible")
        return False
    print(f"  Perception  pick={np.round(pick_pos, 3)}  place={np.round(place_pos, 3) if place_pos is not None else 'n/a'}")

    pick_pos_for_handoff = pick_pos + PICK_POSE_CORRECTION
    q_pick_approach = move_to_pose(kin, planner, arm, viewer, pick_pos_for_handoff, "pick approach")
    if q_pick_approach is None:
        return False

    holding = run_rl_pick(local_pick, gripper, arm, planner, kin, perception, pick_pos_for_handoff, q_pick_approach, viewer)
    if not holding:
        return False

    planner.set_ignore_held_object(True)

    if place_pos is None:
        print("  ERROR: perception — place target not visible")
        planner.set_ignore_held_object(False)
        return False

    if move_to_pose(kin, planner, arm, viewer, place_pos, "place approach", z_offset=PLACE_Z_OFFSET, max_joint_step=0.01) is None:
        planner.set_ignore_held_object(False)
        return False

    planner.set_ignore_held_object(False)
    gripper.open(viewer)
    wait(viewer, 1.0)

    obj_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pick_object")
    obj_pos = env.data.xpos[obj_body_id]
    dist = float(np.linalg.norm(obj_pos[:2] - goal_pos[:2]))
    print(f"  Place: dist to goal = {dist:.3f} m")
    return dist < 0.10


def run_trial(env, kin, perception, planner, gripper, arm, local_pick, viewer, obj_pos, goal_pos, trial_num, num_trials):
    print(f"\n[Trial {trial_num}/{num_trials}]  obj={np.round(obj_pos, 3)}  goal={np.round(goal_pos, 3)}")
    with viewer.lock():
        env.reset_to_config(obj_pos, goal_pos)
    viewer.sync()
    wait(viewer, 2)

    t_start = time.time()
    for attempt in range(TRIAL_RETRIES):
        if attempt > 0:
            print(f"  -- retry {attempt + 1}/{TRIAL_RETRIES} --")
        if run_pick_and_place(env, kin, perception, planner, gripper, arm, local_pick, viewer, goal_pos):
            print(f"  >> SUCCESS  ({time.time() - t_start:.1f}s)")
            return True

    print(f"  >> FAILED  ({time.time() - t_start:.1f}s)")
    return False


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    env = MujocoEnv(str(project_root / "assets" / "mujoco" / "scene.xml"))
    kin = Kinematics()
    perception = Perception(env.model, env.data)
    planner = RRTPlanner(env.model, env.data)
    gripper = GripperController(env.model, env.data)
    arm = ArmController(env.model, env.data)

    model_path = project_root / "models" / "local_pick_drq_stage4_700000_steps.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    local_pick = RLPixelPickController(env.model, env.data, model_path)

    obj_positions, goal_positions = env.generate_random_configs(args.trials)

    successes = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        wait(viewer, 3)
        for trial in range(args.trials):
            if run_trial(
                env, kin, perception, planner, gripper, arm, local_pick, viewer,
                obj_positions[trial], goal_positions[trial], trial + 1, args.trials,
            ):
                successes += 1
            wait(viewer, 2)

        print(f"\nResults: {successes}/{args.trials} succeeded")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
