from __future__ import annotations

import numpy as np

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_env import LocalPickCartesianEnv


class LocalPickCartesianHoldEnv(LocalPickCartesianEnv):
    """Cartesian local-pick env for learning a stable grasp/hold.

    This variant stops before transport. The policy should align, close the
    gripper, and hold two-finger contact briefly. The full pipeline can then use
    scripted smooth lifting and placement.
    """

    def __init__(self, config: LocalPickConfig | None = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
        self._hold_count = 0
        self._was_stable_grasp = False
        self._prev_object_pos = np.zeros(3, dtype=float)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._hold_count = 0
        self._was_stable_grasp = False
        self._prev_object_pos = self.data.xpos[self.object_body_id].copy()
        return obs, info

    def step(self, action):
        prev_object_pos = self.data.xpos[self.object_body_id].copy()
        obs, reward, terminated, truncated, info = super().step(action)
        self._prev_object_pos = prev_object_pos
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, cart_action: np.ndarray, gripper_cmd: float) -> tuple[float, dict[str, float]]:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        reach_distance = float(np.linalg.norm(ee_pos - object_pos))
        object_lift = float(max(0.0, self.data.xpos[self.object_body_id][2] - self.initial_object_z))
        action_penalty = float(np.linalg.norm(cart_action) ** 2)
        table_penalty = self._table_collision_penalty()
        left_touch, right_touch = self._finger_object_contact()
        both_touching = left_touch and right_touch
        # Use gripper_cmd > 0 rather than opening <= 0.012: a bottle (~25 mm dia)
        # physically prevents full finger closure so the 12 mm threshold is never met.
        gripper_closing = gripper_cmd > 0
        object_motion = float(np.linalg.norm(self.data.xpos[self.object_body_id] - self._prev_object_pos))

        reach_reward = 1.0 - np.tanh(12.0 * reach_distance)

        close_aligned_bonus = 0.0
        if gripper_closing and reach_distance <= self.config.align_close_threshold:
            close_aligned_bonus = 0.5

        grasp_alignment_threshold = 0.045
        premature_close_penalty = 0.0
        if gripper_closing and reach_distance > grasp_alignment_threshold:
            premature_close_penalty = 2.0

        aligned_contact = bool(reach_distance <= grasp_alignment_threshold)
        stable_grasp = bool(gripper_closing and both_touching and aligned_contact and object_motion < 0.006)
        if stable_grasp:
            self._hold_count += 1
        else:
            self._hold_count = 0

        first_grasp_bonus = 0.0
        if stable_grasp and not self._was_stable_grasp:
            first_grasp_bonus = 4.0

        hold_bonus = 0.35 * min(self._hold_count, 20)

        object_motion_penalty = 0.0
        if gripper_closing and both_touching:
            object_motion_penalty = 20.0 * object_motion

        bad_contact_penalty = 0.0
        if gripper_closing and both_touching and not aligned_contact:
            bad_contact_penalty = 6.0

        pop_lift_penalty = 0.0
        if object_lift > 0.015:
            pop_lift_penalty = 2.0 * min(object_lift / self.config.min_lift_for_success, 1.0)

        reward = (
            reach_reward
            + close_aligned_bonus
            + first_grasp_bonus
            + hold_bonus
            - premature_close_penalty
            - object_motion_penalty
            - bad_contact_penalty
            - pop_lift_penalty
            - self.config.action_penalty_weight * action_penalty
            - 2.0 * table_penalty
        )
        if self._is_success():
            reward += self.config.success_bonus

        self._was_stable_grasp = stable_grasp
        return float(reward), {
            "reach_distance": reach_distance,
            "object_lift": object_lift,
            "action_penalty": action_penalty,
            "table_penalty": table_penalty,
            "align_close_bonus": close_aligned_bonus,
            "contact_bonus": first_grasp_bonus,
            "hold_bonus": hold_bonus,
            "premature_close_penalty": premature_close_penalty,
            "object_motion_penalty": object_motion_penalty,
            "bad_contact_penalty": bad_contact_penalty,
            "pop_lift_penalty": pop_lift_penalty,
            "hold_count": float(self._hold_count),
        }

    def _is_success(self) -> bool:
        return self._hold_count >= 10
