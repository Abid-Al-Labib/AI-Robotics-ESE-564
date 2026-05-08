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
        self.joint_limits = np.array([model.jnt_range[jid] for jid in joint_ids])

    def get_joint_positions(self):
        return np.array([self.data.qpos[idx] for idx in self.qpos_idx])

    def get_joint_velocities(self):
        return np.array([self.data.qvel[idx] for idx in self.qvel_idx])

    def _set_motor_targets(self, q_target):
        for i, act_id in enumerate(self.act_ids):
            self.data.ctrl[act_id] = q_target[i]

    def move_to(self, q_target, viewer, tol=0.01, vel_tol=0.05, max_steps=5000, steps_per_frame=10):
        """Drive arm to q_target using motor controls (mj_step).
        Waits until both position error and joint velocities are small to avoid overshoot."""
        q_target = np.asarray(q_target, dtype=float)
        q_target = np.clip(q_target, self.joint_limits[:, 0], self.joint_limits[:, 1])
        self._set_motor_targets(q_target)

        for _ in range(max_steps // steps_per_frame):
            for _ in range(steps_per_frame):
                mujoco.mj_step(self.model, self.data)
            viewer.sync()
            pos_err = np.linalg.norm(self.get_joint_positions() - q_target)
            vel_err = np.linalg.norm(self.get_joint_velocities())
            if pos_err < tol and vel_err < vel_tol:
                break
        return pos_err, vel_err

    def move_to_smooth(
        self,
        q_target,
        viewer,
        max_joint_step=0.04,
        tol=0.015,
        vel_tol=0.08,
        max_steps_per_target=1500,
        steps_per_frame=10,
        debug=False,
    ):
        """Move through small joint-space setpoints to avoid large target jumps."""
        q_start = self.get_joint_positions()
        q_target = np.asarray(q_target, dtype=float)
        q_target = np.clip(q_target, self.joint_limits[:, 0], self.joint_limits[:, 1])

        distance = np.max(np.abs(q_target - q_start))
        n_segments = max(1, int(np.ceil(distance / max_joint_step)))
        final_pos_err = 0.0
        final_vel_err = 0.0

        for i in range(1, n_segments + 1):
            alpha = i / n_segments
            q_mid = q_start + alpha * (q_target - q_start)
            final_pos_err, final_vel_err = self.move_to(
                q_mid,
                viewer,
                tol=tol,
                vel_tol=vel_tol,
                max_steps=max_steps_per_target,
                steps_per_frame=steps_per_frame,
            )

        if debug:
            print(
                f"  motor target segments={n_segments}, "
                f"final_pos_err={final_pos_err:.4f}, final_vel_err={final_vel_err:.4f}"
            )
        return final_pos_err, final_vel_err

    def execute_path(self, path, viewer, max_joint_step=0.04):
        for wp in path:
            pos_err, _ = self.move_to_smooth(wp, viewer, max_joint_step=max_joint_step)
            if pos_err > 0.08:
                self.move_to_smooth(
                    wp, viewer, max_joint_step=0.02,
                    tol=0.01, vel_tol=0.05, max_steps_per_target=3000,
                )
