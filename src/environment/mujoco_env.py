import mujoco
import numpy as np
from pathlib import Path


class MujocoEnv:
    def __init__(self, xml_path: str):
        if not Path(xml_path).exists():
            raise FileNotFoundError()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Table bounds with margin
        self.table_x_min = 0.25
        self.table_x_max = 0.75
        self.table_y_min = -0.25
        self.table_y_max = 0.25
        self.pick_z = 0.12
        self.goal_z = 0.1
        self.min_dist = 0.15

        # Joint IDs
        self.pick_qpos_idx = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pick_freejoint")
        ]
        self.goal_qpos_idx = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "goal_freejoint")
        ]

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_time(self):
        return self.data.time

    def generate_random_configs(self, n):
        """Generate n random (obj, goal) position pairs on the table."""
        obj_positions = []
        goal_positions = []

        for _ in range(n):
            while True:
                obj_x = np.random.uniform(self.table_x_min, self.table_x_max)
                obj_y = np.random.uniform(self.table_y_min, self.table_y_max)
                goal_x = np.random.uniform(self.table_x_min, self.table_x_max)
                goal_y = np.random.uniform(self.table_y_min, self.table_y_max)

                dist = np.sqrt((obj_x - goal_x)**2 + (obj_y - goal_y)**2)
                if dist >= self.min_dist:
                    break

            obj_positions.append([obj_x, obj_y, self.pick_z])
            goal_positions.append([goal_x, goal_y, self.goal_z])

        return np.array(obj_positions), np.array(goal_positions)

    def reset_to_config(self, obj_pos, goal_pos):
        """Reset environment with specific object and goal positions."""
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set pick object (freejoint: x, y, z, qw, qx, qy, qz)
        self.data.qpos[self.pick_qpos_idx:self.pick_qpos_idx + 3] = obj_pos
        self.data.qpos[self.pick_qpos_idx + 3:self.pick_qpos_idx + 7] = [1, 0, 0, 0]

        # Set goal platform (freejoint: x, y, z, qw, qx, qy, qz)
        self.data.qpos[self.goal_qpos_idx:self.goal_qpos_idx + 3] = goal_pos
        self.data.qpos[self.goal_qpos_idx + 3:self.goal_qpos_idx + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)