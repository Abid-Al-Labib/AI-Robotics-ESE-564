# pipeline/controller/arm_controller.py
import numpy as np
import mujoco


class ArmController:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i+1}")
            for i in range(7)
        ]
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            for i in range(7)
        ]
        self.qpos_idx = [model.jnt_qposadr[jid] for jid in joint_ids]
        self.qvel_idx = [model.jnt_dofadr[jid] for jid in joint_ids]

    def get_joint_positions(self):
        return np.array([self.data.qpos[idx] for idx in self.qpos_idx])

    def get_joint_velocities(self):
        return np.array([self.data.qvel[idx] for idx in self.qvel_idx])

    def move_to(self, q_target, viewer, tol=0.01, vel_tol=0.05, max_steps=5000, steps_per_frame=10):
        """Drive arm to q_target using motor controls (mj_step).
        Waits until both position error and joint velocities are small to avoid overshoot."""
        for i, act_id in enumerate(self.act_ids):
            self.data.ctrl[act_id] = q_target[i]

        for _ in range(max_steps // steps_per_frame):
            for _ in range(steps_per_frame):
                mujoco.mj_step(self.model, self.data)
            viewer.sync()
            pos_err = np.linalg.norm(self.get_joint_positions() - q_target)
            vel_err = np.linalg.norm(self.get_joint_velocities())
            if pos_err < tol and vel_err < vel_tol:
                break
