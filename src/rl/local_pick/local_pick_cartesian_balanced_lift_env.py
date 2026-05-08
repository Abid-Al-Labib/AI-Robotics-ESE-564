from __future__ import annotations

import numpy as np

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_stable_lift_env import LocalPickCartesianStableLiftEnv


class LocalPickCartesianBalancedLiftEnv(LocalPickCartesianStableLiftEnv):
    """Middle-ground lift env: pick up reliably without over-constraining stability."""

    def __init__(self, config: LocalPickConfig | None = None, render_mode: str | None = None):
        config = config or LocalPickConfig()
        config.action_scale = min(config.action_scale, 0.015)
        super().__init__(config=config, render_mode=render_mode)
        self.config.action_scale = 0.015

    def _compute_reward(self, cart_action: np.ndarray, gripper_cmd: float) -> tuple[float, dict[str, float]]:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        reach_distance = float(np.linalg.norm(ee_pos - object_pos))
        object_world_pos = self.data.xpos[self.object_body_id].copy()
        object_lift = float(max(0.0, object_world_pos[2] - self.initial_object_z))
        lift_delta = max(0.0, object_lift - self._prev_object_lift)
        object_motion = float(np.linalg.norm(object_world_pos - self._prev_object_pos))
        action_penalty = float(np.linalg.norm(cart_action) ** 2)
        table_penalty = self._table_collision_penalty()
        left_touch, right_touch = self._finger_object_contact()
        both_touching = left_touch and right_touch
        gripper_closed = self._gripper_opening() <= 0.012
        valid_grasp = bool(gripper_closed and both_touching)

        reach_reward = 1.0 - np.tanh(12.0 * reach_distance)

        close_aligned_bonus = 0.0
        if gripper_cmd > 0 and reach_distance <= self.config.align_close_threshold:
            close_aligned_bonus = 0.5

        premature_close_penalty = 0.0
        if gripper_cmd > 0 and reach_distance > self.config.premature_close_threshold:
            premature_close_penalty = self.config.premature_close_penalty

        grasp_bonus = 0.0
        if valid_grasp:
            grasp_bonus = 0.35

        lift_progress_reward = 0.0
        lifted_hold_reward = 0.0
        stability_penalty = 0.0
        if valid_grasp:
            lift_progress_reward = 120.0 * lift_delta
            lift_fraction = min(object_lift / self.config.min_lift_for_success, 1.0)
            lifted_hold_reward = 3.0 * lift_fraction
            if object_lift > 0.005:
                stability_penalty = 5.0 * object_motion

        invalid_lift_penalty = 0.0
        if object_lift > 0.008 and not valid_grasp:
            invalid_lift_penalty = 5.0 * min(object_lift / self.config.min_lift_for_success, 1.0)

        if valid_grasp and object_lift >= self.config.min_lift_for_success and object_motion < 0.02:
            self._stable_lift_count += 1
        else:
            self._stable_lift_count = 0

        reward = (
            reach_reward
            + close_aligned_bonus
            + grasp_bonus
            + lift_progress_reward
            + lifted_hold_reward
            - stability_penalty
            - invalid_lift_penalty
            - premature_close_penalty
            - 0.02 * action_penalty
            - 2.0 * table_penalty
        )
        if self._is_success():
            reward += self.config.success_bonus

        return float(reward), {
            "reach_distance": reach_distance,
            "object_lift": object_lift,
            "lift_delta": lift_delta,
            "object_motion": object_motion,
            "action_penalty": action_penalty,
            "table_penalty": table_penalty,
            "align_close_bonus": close_aligned_bonus,
            "contact_bonus": grasp_bonus,
            "hold_bonus": lifted_hold_reward,
            "premature_close_penalty": premature_close_penalty,
            "stability_penalty": stability_penalty,
            "invalid_lift_penalty": invalid_lift_penalty,
            "stable_lift_count": float(self._stable_lift_count),
        }

    def _is_success(self) -> bool:
        return self._stable_lift_count >= 5
