from __future__ import annotations

import numpy as np
import mujoco

from rl.local_pick.local_pick_env import LocalPickEnv


class LocalPickAlignEnv(LocalPickEnv):
    """Train only the local alignment part of pick.

    The policy moves the arm from an approach pose toward a grasp-ready pose.
    Gripper closing and lifting are intentionally scripted outside this env.
    """

    def __init__(self, *args, align_success_distance=0.035, **kwargs):
        super().__init__(*args, **kwargs)
        self.align_success_distance = align_success_distance
        self.config.max_episode_steps = min(self.config.max_episode_steps, 40)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        q_current = self._joint_positions()
        q_target = q_current + action * self.config.action_scale
        q_target = np.clip(q_target, self.joint_limits[:, 0], self.joint_limits[:, 1])

        self._command_arm(q_target)
        self._set_gripper(open_gripper=True)

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        success = self._is_success()
        terminated = success
        truncated = self.step_count >= self.config.max_episode_steps
        info = {"success": success, **reward_info}
        return obs, reward, terminated, truncated, info

    def _phase(self) -> float:
        return 0.0

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        reach_distance = float(np.linalg.norm(ee_pos - object_pos))
        action_penalty = float(np.linalg.norm(action) ** 2)
        success = reach_distance <= self.align_success_distance

        reward = -10.0 * reach_distance - self.config.action_penalty_weight * action_penalty
        if success:
            reward += 25.0

        return float(reward), {
            "reach_distance": reach_distance,
            "object_lift": 0.0,
            "action_penalty": action_penalty,
            "table_penalty": 0.0,
        }

    def _is_success(self) -> bool:
        return bool(np.linalg.norm(self._ee_pos() - self._object_pos()) <= self.align_success_distance)
