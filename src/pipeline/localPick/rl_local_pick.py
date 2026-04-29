from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig


class RLLocalPickController:
    """Run a trained local-pick SAC policy inside an existing MuJoCo scene.

    Use this after the global pipeline has already moved the robot to the pick
    approach pose. The observation/action layout mirrors LocalPickEnv exactly.
    """

    def __init__(
        self,
        model,
        data,
        model_path: str | Path = "models/local_pick_sac.zip",
        config: LocalPickConfig | None = None,
    ):
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise ImportError(
                "RLLocalPickController requires stable-baselines3. Install dependencies with "
                "`pip install -r src/requirements.txt`."
            ) from exc

        self.model = model
        self.data = data
        self.config = config or LocalPickConfig()
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

        self.gripper_actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        finger_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.finger_qpos_idx = model.jnt_qposadr[finger_joint_id]

        self.object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_object")
        self.left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    def execute(self, viewer=None) -> bool:
        """Run the trained local-pick policy from the current approach state."""
        initial_object_z = float(self._object_pos()[2])
        success = False

        for step_count in range(self.config.max_episode_steps):
            obs = self._get_obs(initial_object_z, step_count)
            action, _ = self.policy.predict(obs, deterministic=True)
            self._apply_action(action, step_count)

            for _ in range(self.config.frame_skip):
                mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

            success = self._object_pos()[2] >= self.config.lift_success_z
            if success:
                break

        return bool(success)

    def _apply_action(self, action, step_count: int) -> None:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        q_target = self._joint_positions() + action * self.config.action_scale
        q_target = np.clip(q_target, self.joint_limits[:, 0], self.joint_limits[:, 1])

        for act_id, value in zip(self.actuator_ids, q_target):
            self.data.ctrl[act_id] = float(value)

        self.data.ctrl[self.gripper_actuator_id] = (
            self.config.open_gripper_ctrl
            if step_count < self.config.gripper_close_step
            else self.config.close_gripper_ctrl
        )

    def _get_obs(self, initial_object_z: float, step_count: int) -> np.ndarray:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        phase = 0.0 if step_count < self.config.gripper_close_step else 1.0
        obs = np.concatenate([
            self._joint_positions(),
            self._joint_velocities(),
            ee_pos - object_pos,
            np.array([object_pos[2] - initial_object_z], dtype=float),
            np.array([self._gripper_opening()], dtype=float),
            np.array([phase], dtype=float),
        ])
        return obs.astype(np.float32)

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
