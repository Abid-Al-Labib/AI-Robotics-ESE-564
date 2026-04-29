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
from pipeline.localPick.rl_local_pick import RLLocalPickController
from pipeline.localPick.rl_align_pick import RLAlignPickController


def set_joints(env, joints, viewer):
    with viewer.lock():
        for i in range(7):
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            qpos_idx = env.model.jnt_qposadr[joint_id]
            env.data.qpos[qpos_idx] = joints[i]
        mujoco.mj_forward(env.model, env.data)
    viewer.sync()


def get_ik_goal(kin, target_pos, planner, side_grasp=False):
    if side_grasp:
        target_rot = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
    else:
        target_rot = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

    approach_pos = get_approach_position(target_pos, side_grasp=side_grasp)

    solutions = kin.ik(approach_pos, target_rot, free_joint_samples=100)
    if not solutions:
        print(f"No IK solution for position {approach_pos}")
        return None

    q_current = planner.get_joint_positions()
    best = None
    best_dist = float('inf')
    for idx, sol in enumerate(solutions):
        valid = planner.is_config_valid(sol, debug=(idx < 3))
        if valid:
            dist = np.linalg.norm(sol - q_current)
            if dist < best_dist:
                best_dist = dist
                best = sol

    if best is not None:
        print(f"Found collision-free IK solution")
        return best

    print("No collision-free IK solution, using closest")
    dists = [np.linalg.norm(sol - q_current) for sol in solutions]
    return solutions[np.argmin(dists)]


def get_approach_position(target_pos, side_grasp=False):
    approach_pos = target_pos.copy()
    if side_grasp:
        approach_pos[1] -= 0.10
        approach_pos[2] += 0.03
    else:
        approach_pos[2] += 0.15
    return approach_pos


def execute_path(env, waypoints, viewer, delay=0.3):
    """Execute a list of joint-space waypoints with animation (kinematic)."""
    for wp in waypoints:
        set_joints(env, wp, viewer)
        wait(viewer, delay)


def execute_path_motor(arm, waypoints, viewer):
    """Execute a list of joint-space waypoints using motor controls (physics)."""
    for idx, wp in enumerate(waypoints):
        pos_err, vel_err = arm.move_to_smooth(wp, viewer, debug=(idx == len(waypoints) - 1))
        if pos_err > 0.08:
            print(f"  warning: waypoint {idx + 1}/{len(waypoints)} ended with joint error {pos_err:.4f}")
            print("  retrying waypoint with slower motor interpolation...")
            pos_err, vel_err = arm.move_to_smooth(
                wp,
                viewer,
                max_joint_step=0.02,
                tol=0.01,
                vel_tol=0.05,
                max_steps_per_target=3000,
                debug=(idx == len(waypoints) - 1),
            )
            if pos_err > 0.08:
                print(f"  warning: retry still ended with joint error {pos_err:.4f}")


def report_ee_error(kin, arm, label, target_pos):
    ee_pos, _ = kin.fk(arm.get_joint_positions())
    err = np.linalg.norm(ee_pos - target_pos)
    print(f"{label} FK end-effector: {ee_pos}")
    print(f"{label} target position: {target_pos}")
    print(f"{label} Cartesian error: {err:.4f} m")
    return err


def retry_goal_if_needed(kin, arm, viewer, q_goal, target_pos, label, threshold=0.04):
    err = report_ee_error(kin, arm, label, target_pos)
    if err > threshold:
        print(f"{label} missed approach by {err:.4f} m; retrying final goal slowly...")
        arm.move_to_smooth(
            q_goal,
            viewer,
            max_joint_step=0.015,
            tol=0.008,
            vel_tol=0.04,
            max_steps_per_target=4000,
            debug=True,
        )
        err = report_ee_error(kin, arm, f"{label} RETRY", target_pos)
    return err


def wait(viewer, seconds):
    start = time.time()
    while viewer.is_running() and time.time() - start < seconds:
        viewer.sync()
        time.sleep(0.01)


def main():
    project_root = Path(__file__).resolve().parent.parent
    xml_path = project_root / "assets" / "mujoco" / "scene.xml"

    env = MujocoEnv(str(xml_path))
    kin = Kinematics()
    perception = Perception(env.model, env.data)
    planner = RRTPlanner(env.model, env.data)
    gripper = GripperController(env.model, env.data)
    arm = ArmController(env.model, env.data)
    local_align_model_path = project_root / "models" / "local_pick_align_full_table.zip"
    local_pick_model_path = project_root / "models" / "local_pick_sac_randomized_100k.zip"
    local_align_pick = None
    local_pick = None
    if local_align_model_path.exists():
        local_align_pick = RLAlignPickController(env.model, env.data, local_align_model_path)
        print(f"Loaded RL align-pick model: {local_align_model_path}")
    elif local_pick_model_path.exists():
        local_pick = RLLocalPickController(env.model, env.data, local_pick_model_path)
        print(f"Loaded RL local pick model: {local_pick_model_path}")
    else:
        print(f"RL align-pick model not found: {local_align_model_path}")
        print(f"RL local pick model not found: {local_pick_model_path}")
        print("Pipeline will still move to pick approach, but will skip RL local pick.")

    # Generate random configurations
    num_trials = 10
    obj_positions, goal_positions = env.generate_random_configs(num_trials)
    print(f"Generated {num_trials} random configurations")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for trial in range(num_trials):
            print(f"\n{'='*50}")
            print(f"Trial {trial + 1}/{num_trials}")
            print(f"{'='*50}")
            planner.set_ignore_held_object(False)

            # Reset environment
            with viewer.lock():
                env.reset_to_config(obj_positions[trial], goal_positions[trial])
            viewer.sync()

            print(f"Object at: {obj_positions[trial]}")
            print(f"Goal at: {goal_positions[trial]}")
            wait(viewer, 2)

            # Perceive objects
            pick_pos, place_pos = perception.get_object_positions()
            print(f"Perceived pick: {pick_pos}")
            print(f"Perceived place: {place_pos}")

            # Move to pick approach (0.15 m above object) and test gripper
            if pick_pos is not None:
                print("\nPlanning path to PICK APPROACH...")
                q_pick_approach = get_ik_goal(kin, pick_pos, planner, side_grasp=False)
                if q_pick_approach is not None:
                    path = planner.plan(q_pick_approach)
                    if path:
                        print("Executing path to pick approach...")
                        execute_path_motor(arm, path, viewer)
                        pick_approach_pos = get_approach_position(pick_pos, side_grasp=False)
                        retry_goal_if_needed(
                            kin, arm, viewer, q_pick_approach, pick_approach_pos, "PICK APPROACH"
                        )
                        if local_align_pick is not None:
                            print("Running RL align, scripted close, and motor lift...")
                            gripper.open(viewer)
                            wait(viewer, 0.25)
                            success = local_align_pick.execute(arm, gripper, viewer, q_pick_approach)
                            print(f"RL align-pick success: {success}")
                            planner.set_ignore_held_object(success)
                        elif local_pick is not None:
                            print("Running RL local pick...")
                            gripper.open(viewer)
                            wait(viewer, 0.25)
                            success = local_pick.execute(viewer)
                            print(f"RL local pick success: {success}")
                            planner.set_ignore_held_object(success)
                        else:
                            print("Skipping RL local pick because no trained model was found.")

            wait(viewer, 1)

            # Move to place approach (0.15 m above goal) and test gripper
            if place_pos is not None:
                print("\nPlanning path to PLACE APPROACH...")
                q_place_approach = get_ik_goal(kin, place_pos, planner)
                if q_place_approach is not None:
                    path = planner.plan(q_place_approach)
                    if path:
                        print("Executing path to place approach...")
                        execute_path_motor(arm, path, viewer)
                        place_approach_pos = place_pos.copy()
                        place_approach_pos[2] += 0.15
                        retry_goal_if_needed(
                            kin, arm, viewer, q_place_approach, place_approach_pos, "PLACE APPROACH"
                        )
                        print("Testing gripper open at place approach...")
                        gripper.open(viewer)

            wait(viewer, 2)

        print("\nAll trials complete. Viewer still running...")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)

    print("Viewer closed")


if __name__ == "__main__":
    main()
