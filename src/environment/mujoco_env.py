import mujoco
import numpy as np
from pathlib import Path


class MujocoEnv:
    def __init__(self, xml_path: str):
        if not Path(xml_path).exists():
            raise FileNotFoundError()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Table bounds — object side centered, goal side opposite
        self.table_x_min = 0.43
        self.table_x_max = 0.57
        self.pick_y_min = -0.20
        self.pick_y_max = -0.10
        self.goal_y_min = 0.10
        self.goal_y_max = 0.24
        self.pick_z = 0.125
        self.goal_z = 0.1
        self.min_dist = 0.10

        # Joint IDs
        self.pick_qpos_idx = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pick_freejoint")
        ]
        self.goal_qpos_idx = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "goal_freejoint")
        ]
        self.arm_home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853], dtype=float)
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            for i in range(7)
        ]
        self.arm_qpos_idx = [self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids]
        self.arm_qvel_idx = [self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids]
        self.arm_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i+1}")
            for i in range(7)
        ]
        self.finger_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
        ]
        self.finger_qpos_idx = [self.model.jnt_qposadr[jid] for jid in self.finger_joint_ids]
        self.finger_qvel_idx = [self.model.jnt_dofadr[jid] for jid in self.finger_joint_ids]
        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self._reset_robot_home()
        mujoco.mj_forward(self.model, self.data)

    def _reset_robot_home(self):
        """Reset arm/gripper to the motor-held home pose used by the Panda XML."""
        self.data.qpos[self.arm_qpos_idx] = self.arm_home_qpos
        self.data.qvel[self.arm_qvel_idx] = 0.0
        for act_id, q in zip(self.arm_actuator_ids, self.arm_home_qpos):
            self.data.ctrl[act_id] = q

        for qidx, vidx in zip(self.finger_qpos_idx, self.finger_qvel_idx):
            self.data.qpos[qidx] = 0.04
            self.data.qvel[vidx] = 0.0
        self.data.ctrl[self.gripper_actuator_id] = 255.0

    def get_time(self):
        return self.data.time

    def generate_random_configs(self, n):
        """Generate n random (obj, goal) position pairs on the table."""
        obj_positions = []
        goal_positions = []

        for _ in range(n):
            while True:
                obj_x = np.random.uniform(self.table_x_min, self.table_x_max)
                obj_y = np.random.uniform(self.pick_y_min, self.pick_y_max)
                goal_x = np.random.uniform(self.table_x_min, self.table_x_max)
                goal_y = np.random.uniform(self.goal_y_min, self.goal_y_max)

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
        self.data.qvel[:] = 0.0
        self._reset_robot_home()

        # Set pick object lying on its side (90deg rotation around Y axis)
        self.data.qpos[self.pick_qpos_idx:self.pick_qpos_idx + 3] = obj_pos
        self.data.qpos[self.pick_qpos_idx + 3:self.pick_qpos_idx + 7] = [0.7071, 0.7071, 0, 0]

        # Set goal platform (freejoint: x, y, z, qw, qx, qy, qz)
        self.data.qpos[self.goal_qpos_idx:self.goal_qpos_idx + 3] = goal_pos
        self.data.qpos[self.goal_qpos_idx + 3:self.goal_qpos_idx + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)
