from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_env import LocalPickEnv


class LocalPickCartesianEnv(LocalPickEnv):
    """Local pick env with Cartesian delta actions and camera-estimated object pose.

    Observation layout is inherited from LocalPickEnv, so the policy still sees
    perception-derived object-relative state rather than MuJoCo object position.
    The action is smaller and task-aligned:

        [dx, dy, dz, gripper]

    The Cartesian target is converted to joint targets through IK each step.
    """

    def __init__(self, config: LocalPickConfig | None = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self._target_ee_pos: np.ndarray | None = None
        self._target_rot = self._topdown_grasp_rotation()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        obs, info = super().reset(seed=seed, options=options)
        self._target_ee_pos = self._ee_pos()
        self._target_rot = self._side_grasp_rotation() if self.config.side_grasp else self._topdown_grasp_rotation()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        cart_action = action[:3]
        gripper_cmd = float(action[3])

        if self._target_ee_pos is None:
            self._target_ee_pos = self._ee_pos()

        self._target_ee_pos = self._target_ee_pos + cart_action * self.config.action_scale
        self._target_ee_pos = self._clip_target(self._target_ee_pos)

        ik_failed = not self._command_cartesian_target(self._target_ee_pos)
        self._set_gripper(open_gripper=(gripper_cmd <= 0))

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        pos = self._render_object_pos()
        if pos is not None:
            self._cam_object_pos = pos

        self.step_count += 1
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(cart_action, gripper_cmd)
        if ik_failed:
            reward -= 1.0
        success = self._is_success()
        terminated = success
        truncated = self.step_count >= self.config.max_episode_steps
        info = {"success": success, "ik_failed": ik_failed, **reward_info}
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _topdown_grasp_rotation() -> np.ndarray:
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])

    @staticmethod
    def _side_grasp_rotation() -> np.ndarray:
        return np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        clipped = target.copy()
        clipped[0] = np.clip(clipped[0], self.config.table_x_min - 0.08, self.config.table_x_max + 0.08)
        clipped[1] = np.clip(clipped[1], self.config.table_y_min - 0.08, self.config.table_y_max + 0.08)
        clipped[2] = np.clip(clipped[2], self.config.object_z - 0.06, self.config.object_z + 0.25)
        return clipped

    def _command_cartesian_target(self, target_pos: np.ndarray) -> bool:
        current_q = self._joint_positions()
        free_joint_range = np.linspace(
            current_q[-1] - 0.8,
            current_q[-1] + 0.8,
            31,
        )
        solutions = self.kinematics.ik(
            target_pos,
            self._target_rot,
            free_joint_range=free_joint_range,
        )
        valid = [q for q in solutions if self._within_joint_limits(q)]
        if not valid:
            return False

        q_target = min(valid, key=lambda q: np.linalg.norm(q - current_q))
        self._command_arm(np.asarray(q_target, dtype=float))
        return True
