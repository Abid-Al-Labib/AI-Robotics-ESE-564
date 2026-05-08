from __future__ import annotations

import numpy as np

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_env import LocalPickCartesianEnv


class LocalPickCartesianProgressEnv(LocalPickCartesianEnv):
    """Cartesian local pick with progress-based staged reward.

    Inspired by progress rewards: compute a task score and only pay positive
    improvements over the best score seen in the episode. This discourages
    camping at alignment/contact states.
    """

    def __init__(self, config: LocalPickConfig | None = None, render_mode: str | None = None):
        config = config or LocalPickConfig()
        config.action_scale = 0.01
        config.max_episode_steps = max(config.max_episode_steps, 150)
        super().__init__(config=config, render_mode=render_mode)
        self.config.action_scale = 0.01
        self.config.max_episode_steps = max(self.config.max_episode_steps, 150)
        self._best_score = 0.0
        self._success_hold_count = 0

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._best_score = 0.0
        self._success_hold_count = 0
        return obs, info

    def _compute_reward(self, cart_action: np.ndarray, gripper_cmd: float) -> tuple[float, dict[str, float]]:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        reach_distance = float(np.linalg.norm(ee_pos - object_pos))
        object_lift = float(max(0.0, self.data.xpos[self.object_body_id][2] - self.initial_object_z))
        action_penalty = float(np.linalg.norm(cart_action) ** 2)
        table_penalty = self._table_collision_penalty()
        left_touch, right_touch = self._finger_object_contact()
        both_touching = left_touch and right_touch
        gripper_closed = self._gripper_opening() <= 0.012
        aligned = reach_distance <= self.config.align_close_threshold

        reach_score = 1.0 - np.tanh(10.0 * reach_distance)
        close_score = 1.0 if aligned and gripper_closed else 0.0
        grasp_score = 1.0 if aligned and gripper_closed and both_touching else 0.0
        lift_fraction = min(object_lift / self.config.min_lift_for_success, 1.0)
        lift_score = lift_fraction if gripper_closed else 0.0
        stable_score = 1.0 if self._is_success() else 0.0

        score = (
            1.5 * reach_score
            + 1.0 * close_score
            + 2.0 * grasp_score
            + 8.0 * lift_score
            + 2.0 * stable_score
        )

        progress_reward = max(score - self._best_score, 0.0)
        self._best_score = max(self._best_score, score)

        premature_close_penalty = 0.0
        if gripper_cmd > 0 and reach_distance > self.config.premature_close_threshold:
            premature_close_penalty = 0.05

        reward = (
            progress_reward
            - premature_close_penalty
            - 0.005 * action_penalty
            - 0.25 * table_penalty
        )
        if self._is_success():
            reward += 2.0

        return float(reward), {
            "reach_distance": reach_distance,
            "object_lift": object_lift,
            "action_penalty": action_penalty,
            "table_penalty": table_penalty,
            "align_close_bonus": close_score,
            "contact_bonus": 0.0,
            "hold_bonus": lift_score,
            "premature_close_penalty": premature_close_penalty,
            "progress_reward": progress_reward,
            "task_score": score,
            "best_score": self._best_score,
            "reach_score": reach_score,
            "close_score": close_score,
            "grasp_score": grasp_score,
            "lift_score": lift_score,
            "stable_score": stable_score,
            "success_hold_count": float(self._success_hold_count),
        }

    def _is_success(self) -> bool:
        object_lift = float(max(0.0, self.data.xpos[self.object_body_id][2] - self.initial_object_z))
        left_touch, right_touch = self._finger_object_contact()
        if object_lift >= self.config.min_lift_for_success and self._gripper_opening() <= 0.012 and left_touch and right_touch:
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        return self._success_hold_count >= 3
