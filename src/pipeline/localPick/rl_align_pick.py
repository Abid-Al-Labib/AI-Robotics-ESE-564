from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig


class RLAlignPickController:
    """Align with RL, then use scripted gripper close and motor lift."""

    def __init__(
        self,
        model,
        data,
        model_path: str | Path = "models/local_pick_align_full_table.zip",
        config: LocalPickConfig | None = None,
        align_success_distance: float = 0.035,
    ):
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise ImportError(
                "RLAlignPickController requires stable-baselines3. Install dependencies with "
                "`pip install -r src/requirements.txt`."
            ) from exc

        self.model = model
        self.data = data
        self.config = config or LocalPickConfig(max_episode_steps=40)
        self.align_success_distance = align_success_distance
        self.policy = SAC.load(model_path)

        self.joint_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ])
        self.actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
            for i in range(1, 8)
        ])
        self.qpos_idx = model.jnt_qposadr[self.joint_ids]
        self.qvel_idx = model.jnt_dofadr[self.joint_ids]
        self.joint_limits = np.array([model.jnt_range[jid] for jid in self.joint_ids])

        finger_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.finger_qpos_idx = model.jnt_qposadr[finger_joint_id]
        self.object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_object")
        self.left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    def execute(self, arm, gripper, viewer, q_lift) -> bool:
        """Run RL alignment, close gripper, then lift to q_lift using motor control."""
        initial_object_z = float(self._object_pos()[2])
        aligned = self._run_alignment(viewer)
        print(f"RL local align success: {aligned}")

        gripper.close(viewer)
        arm.move_to_smooth(
            q_lift,
            viewer,
            max_joint_step=0.02,
            tol=0.01,
            vel_tol=0.05,
            max_steps_per_target=3000,
            debug=True,
        )
        return bool(self._object_pos()[2] > initial_object_z + 0.05)

    def _run_alignment(self, viewer=None) -> bool:
        aligned = False
        for step_count in range(self.config.max_episode_steps):
            obs = self._get_obs(step_count)
            action, _ = self.policy.predict(obs, deterministic=True)
            self._apply_action(action)

            for _ in range(self.config.frame_skip):
                mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

            aligned = self._reach_distance() <= self.align_success_distance
            if aligned:
                break
        return bool(aligned)

    def _apply_action(self, action) -> None:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        q_target = self._joint_positions() + action * self.config.action_scale
        q_target = np.clip(q_target, self.joint_limits[:, 0], self.joint_limits[:, 1])
        for act_id, value in zip(self.actuator_ids, q_target):
            self.data.ctrl[act_id] = float(value)

    def _get_obs(self, step_count: int) -> np.ndarray:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        obs = np.concatenate([
            self._joint_positions(),
            self._joint_velocities(),
            ee_pos - object_pos,
            np.array([0.0], dtype=float),
            np.array([self._gripper_opening()], dtype=float),
            np.array([0.0], dtype=float),
        ])
        return obs.astype(np.float32)

    def _reach_distance(self) -> float:
        return float(np.linalg.norm(self._ee_pos() - self._object_pos()))

    def _joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def _joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    def _object_pos(self) -> np.ndarray:
        return self.data.xpos[self.object_body_id].copy()

    def _ee_pos(self) -> np.ndarray:
        left = self.data.xpos[self.left_finger_body_id]
        right = self.data.xpos[self.right_finger_body_id]
        return ((left + right) * 0.5).copy()

    def _gripper_opening(self) -> float:
        return float(self.data.qpos[self.finger_qpos_idx])
