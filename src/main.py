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
from pipeline.perception.perception import Perception
from pipeline.planner.rrt_planner import RRTPlanner


def set_joints(env, joints, viewer):
    with viewer.lock():
        for i in range(7):
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            qpos_idx = env.model.jnt_qposadr[joint_id]
            env.data.qpos[qpos_idx] = joints[i]
        mujoco.mj_forward(env.model, env.data)
    viewer.sync()


def get_ik_goal(kin, target_pos, planner):
    target_rot = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    approach_pos = target_pos.copy()
    approach_pos[2] += 0.15

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


def execute_path(env, waypoints, viewer, delay=0.3):
    """Execute a list of joint-space waypoints with animation."""
    for wp in waypoints:
        set_joints(env, wp, viewer)
        wait(viewer, delay)


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

    # Generate random configurations
    num_trials = 3
    obj_positions, goal_positions = env.generate_random_configs(num_trials)
    print(f"Generated {num_trials} random configurations")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for trial in range(num_trials):
            print(f"\n{'='*50}")
            print(f"Trial {trial + 1}/{num_trials}")
            print(f"{'='*50}")

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

            # Plan and move to pick position
            if pick_pos is not None:
                print("\nPlanning path to PICK position...")
                q_pick = get_ik_goal(kin, pick_pos, planner)
                if q_pick is not None:
                    path = planner.plan(q_pick)
                    if path:
                        print("Executing path to pick...")
                        execute_path(env, path, viewer)

            wait(viewer, 2)

            # Plan and move to place position
            if place_pos is not None:
                print("\nPlanning path to PLACE position...")
                q_place = get_ik_goal(kin, place_pos, planner)
                if q_place is not None:
                    path = planner.plan(q_place)
                    if path:
                        print("Executing path to place...")
                        execute_path(env, path, viewer)

            wait(viewer, 2)

        print("\nAll trials complete. Viewer still running...")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)

    print("Viewer closed")


if __name__ == "__main__":
    main()