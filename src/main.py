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
from pipeline.localPick.rl_cartesian_vertical_pick import RLCartesianVerticalPickController
from pipeline.localPick.rl_local_pick import RLLocalPickController


PICK_POSE_CORRECTION = np.array([0.0, 0.020, -0.026], dtype=float)


def get_ik_goal(kin, target_pos, planner, side_grasp=False, z_offset=0.15, free_joint_range=None):
    if side_grasp:
        target_rot = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ])
    else:
        target_rot = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

    approach_pos = get_approach_position(target_pos, side_grasp=side_grasp, z_offset=z_offset)

    if free_joint_range is not None:
        solutions = kin.ik(approach_pos, target_rot, free_joint_range=free_joint_range)
    else:
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


def get_approach_position(target_pos, side_grasp=False, z_offset=0.15):
    approach_pos = target_pos.copy()
    if side_grasp:
        approach_pos[0] -= 0.10
        approach_pos[2] += 0.03
    else:
        approach_pos[2] += z_offset
    return approach_pos


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
    pixel_pick_model_path = project_root / "models" / "local_pick_drq_2m.zip"
    cartesian_pick_model_path = project_root / "models" / "local_pick_cartesian_colift_from_zbonus_sac_500k_500000_steps.zip"
    cartesian_pick_vecnormalize_path = project_root / "models" / "local_pick_cartesian_colift_from_zbonus_sac_500k_vecnormalize.pkl"
    local_pick_model_path = project_root / "models" / "perception_model_v1.zip"
    local_pick = None
    if pixel_pick_model_path.exists():
        local_pick = RLPixelPickController(env.model, env.data, pixel_pick_model_path)
        print(f"Loaded pixel SAC local pick model: {pixel_pick_model_path}")
    elif cartesian_pick_model_path.exists():
        local_pick = RLCartesianVerticalPickController(
            env.model,
            env.data,
            cartesian_pick_model_path,
            vecnormalize_path=cartesian_pick_vecnormalize_path,
        )
        print(f"Loaded Cartesian RL local pick model: {cartesian_pick_model_path}")
    elif local_pick_model_path.exists():
        local_pick = RLLocalPickController(env.model, env.data, local_pick_model_path)
        print(f"Loaded RL local pick model: {local_pick_model_path}")
    else:
        print(f"No RL local pick model found.")
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
            pick_pos_for_handoff = None
            if pick_pos is not None:
                pick_pos_for_handoff = pick_pos + PICK_POSE_CORRECTION
                print(f"Corrected pick for handoff: {pick_pos_for_handoff}")

            # Move to pick approach, then hand off to RL local pick
            if pick_pos_for_handoff is not None:
                print("\nPlanning path to PICK APPROACH...")
                q_pick_approach = get_ik_goal(kin, pick_pos_for_handoff, planner, side_grasp=False)
                if q_pick_approach is not None:
                    path = planner.plan(q_pick_approach)
                    if path:
                        print("Executing path to pick approach...")
                        execute_path_motor(arm, path, viewer)
                        pick_approach_pos = get_approach_position(pick_pos_for_handoff, side_grasp=False)
                        retry_goal_if_needed(
                            kin, arm, viewer, q_pick_approach, pick_approach_pos, "PICK APPROACH"
                        )
                        if local_pick is not None:
                            print("Running RL local pick...")
                            gripper.open(viewer)
                            wait(viewer, 0.25)
                            success = local_pick.execute(viewer)
                            print(f"RL local pick success: {success}")
                            planner.set_ignore_held_object(success)

                            if success and place_pos is not None:
                                # Lift straight up to clear the obstacle before handing to RRT.
                                print("\nLifting to clearance height...")
                                clearance_pos = pick_pos_for_handoff.copy()
                                clearance_pos[2] = 0.38
                                q_clearance = get_ik_goal(kin, clearance_pos, planner, side_grasp=False, z_offset=0.0)
                                if q_clearance is not None:
                                    arm.move_to_smooth(q_clearance, viewer)
                                wait(viewer, 0.5)

                                # RRT from clearance height to place approach.
                                print("\nPlanning path to PLACE APPROACH...")
                                q_place_approach = get_ik_goal(kin, place_pos, planner, z_offset=0.25)
                                if q_place_approach is not None:
                                    path = planner.plan(q_place_approach)
                                    if path:
                                        print("Executing path to place approach...")
                                        execute_path_motor(arm, path, viewer)
                                        place_approach_pos = place_pos.copy()
                                        place_approach_pos[2] += 0.25
                                        retry_goal_if_needed(
                                            kin, arm, viewer, q_place_approach, place_approach_pos, "PLACE APPROACH"
                                        )
                                        print("Dropping object...")
                                        gripper.open(viewer)
                                        wait(viewer, 1.0)
                        else:
                            print("Skipping RL local pick because no trained model was found.")

            wait(viewer, 2)

        print("\nAll trials complete. Viewer still running...")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)

    print("Viewer closed")


if __name__ == "__main__":
    main()
