from __future__ import annotations

import numpy as np

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_env import LocalPickCartesianEnv


class LocalPickCartesianLiftEnv(LocalPickCartesianEnv):
    """Cartesian local-pick env with reward focused on lifting after contact.

    The first Cartesian experiment learned to farm per-step contact reward without
    lifting. This reward keeps contact useful, but makes sustained contact without
    lift much less valuable than actual lift progress.
    """

    def __init__(self, config: LocalPickConfig | None = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
        self._was_both_touching = False
        self._success_hold_count = 0

    def reset(self, *args, **kwargs):
        self._was_both_touching = False
        self._success_hold_count = 0
        return super().reset(*args, **kwargs)

    def _compute_reward(self, cart_action: np.ndarray, gripper_cmd: float) -> tuple[float, dict[str, float]]:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        reach_distance = float(np.linalg.norm(ee_pos - object_pos))
        object_lift = float(max(0.0, self.data.xpos[self.object_body_id][2] - self.initial_object_z))
        action_penalty = float(np.linalg.norm(cart_action) ** 2)
        table_penalty = self._table_collision_penalty()
        left_touch, right_touch = self._finger_object_contact()
        both_touching = left_touch and right_touch
        success = self._grasp_lift_condition(both_touching=both_touching, object_lift=object_lift)

        reach_reward = 1.0 - np.tanh(12.0 * reach_distance)

        close_aligned_bonus = 0.0
        if gripper_cmd > 0 and reach_distance <= self.config.align_close_threshold:
            close_aligned_bonus = 0.5

        first_contact_bonus = 0.0
        if gripper_cmd > 0 and both_touching and not self._was_both_touching:
            first_contact_bonus = 2.0

        grasp_hold_bonus = 0.0
        if gripper_cmd > 0 and both_touching:
            grasp_hold_bonus = 0.08

        lift_reward = 0.0
        if both_touching:
            lift_fraction = min(object_lift / self.config.min_lift_for_success, 1.0)
            lift_reward = 12.0 * lift_fraction

        no_lift_contact_penalty = 0.0
        if both_touching and object_lift < 0.003 and self.step_count > 15:
            no_lift_contact_penalty = 0.08

        invalid_lift_penalty = 0.0
        if object_lift > 0.006 and not both_touching:
            invalid_lift_penalty = 8.0 * min(object_lift / self.config.min_lift_for_success, 1.0)

        premature_close_penalty = 0.0
        if gripper_cmd > 0 and reach_distance > self.config.premature_close_threshold:
            premature_close_penalty = self.config.premature_close_penalty

        reward = (
            reach_reward
            + close_aligned_bonus
            + first_contact_bonus
            + grasp_hold_bonus
            + lift_reward
            - no_lift_contact_penalty
            - invalid_lift_penalty
            - premature_close_penalty
            - self.config.action_penalty_weight * action_penalty
            - 3.0 * table_penalty
        )
        if success:
            reward += self.config.success_bonus

        self._was_both_touching = both_touching
        return float(reward), {
            "reach_distance": reach_distance,
            "object_lift": object_lift,
            "action_penalty": action_penalty,
            "table_penalty": table_penalty,
            "align_close_bonus": close_aligned_bonus,
            "contact_bonus": first_contact_bonus + grasp_hold_bonus,
            "hold_bonus": lift_reward,
            "premature_close_penalty": premature_close_penalty,
            "no_lift_contact_penalty": no_lift_contact_penalty,
            "invalid_lift_penalty": invalid_lift_penalty,
        }

    def _grasp_lift_condition(self, both_touching: bool | None = None, object_lift: float | None = None) -> bool:
        if both_touching is None:
            left_touch, right_touch = self._finger_object_contact()
            both_touching = left_touch and right_touch
        if object_lift is None:
            object_lift = float(max(0.0, self.data.xpos[self.object_body_id][2] - self.initial_object_z))

        # 0.035 m allows for a bottle preventing full finger closure (~0.025 m).
        gripper_not_open = self._gripper_opening() <= 0.035
        lifted = object_lift >= self.config.min_lift_for_success
        return bool(both_touching and gripper_not_open and lifted)

    def _is_success(self) -> bool:
        if self._grasp_lift_condition():
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        return self._success_hold_count >= 3
