from __future__ import annotations

import numpy as np

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_env import LocalPickCartesianEnv


class LocalPickCartesianDirectLiftEnv(LocalPickCartesianEnv):
    """Option A: two-phase progress reward with reached_box_flag.

    Phase 1 (approach): score grows as GT distance to bottle shrinks, max 4.0.
    Phase 2 (grasp+lift): latches once EE gets within align_close_threshold of
        the bottle; adds contact bonus (2.0) and lift progress (0-8.0).

    Progress wrapper: only improvements over the episode-best score yield
    positive reward, eliminating all farming exploits.
    """

    def __init__(self, config: LocalPickConfig | None = None, render_mode: str | None = None):
        config = config or LocalPickConfig()
        config.action_scale = 0.015
        config.max_episode_steps = max(config.max_episode_steps, 125)
        super().__init__(config=config, render_mode=render_mode)
        self.config.action_scale = 0.015
        self.config.max_episode_steps = max(self.config.max_episode_steps, 125)
        self._best_score = 0.0
        self._reached_box = False
        self._success_hold_count = 0

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._best_score = 0.0
        self._reached_box = False
        self._success_hold_count = 0
        return obs, info

    def _compute_reward(self, cart_action: np.ndarray, gripper_cmd: float) -> tuple[float, dict]:
        ee_pos = self._ee_pos()
        gt_object_pos = self.data.xpos[self.object_body_id].copy()
        gt_distance = float(np.linalg.norm(ee_pos - gt_object_pos))
        object_lift = float(max(0.0, self.data.xpos[self.object_body_id][2] - self.initial_object_z))
        action_penalty = float(np.linalg.norm(cart_action) ** 2)
        table_penalty = self._table_collision_penalty()

        left_touch, right_touch = self._finger_object_contact()
        # EE must be at grasp height before contact counts toward grasping.
        ee_at_grasp_height = ee_pos[2] <= self.initial_object_z + 0.06
        gripper_grasping = (gripper_cmd > 0) and (left_touch or right_touch) and ee_at_grasp_height

        # Latch phase gate once EE is within reach of bottle.
        if not self._reached_box and gt_distance < self.config.align_close_threshold:
            self._reached_box = True

        # --- Two-phase task score ---
        if not self._reached_box:
            # Phase 1: approach. Score 0→4 as distance 0.20→0.
            score = 4.0 * (1.0 - np.tanh(5.0 * gt_distance))
        else:
            # Phase 2: grasp + lift. Score 4→14.
            lift_fraction = min(object_lift / self.config.min_lift_for_success, 1.0)
            score = 4.0                          # full approach credit held
            score += 2.0 if gripper_grasping else 0.0
            score += 8.0 * lift_fraction

        # Progress reward: only pay for improvements over episode best.
        progress_reward = max(score - self._best_score, 0.0)
        self._best_score = max(self._best_score, score)

        # Premature close penalty: discourage closing when still far and before phase 2.
        premature_close_penalty = 0.0
        if gripper_cmd > 0 and not self._reached_box and gt_distance > self.config.premature_close_threshold:
            premature_close_penalty = 0.5

        reward = (
            progress_reward
            - premature_close_penalty
            - 0.01 * action_penalty
            - 1.0 * table_penalty
        )
        if self._is_success():
            reward += self.config.success_bonus

        lift_fraction = min(object_lift / self.config.min_lift_for_success, 1.0)
        return float(reward), {
            "reach_distance": gt_distance,
            "object_lift": object_lift,
            "action_penalty": action_penalty,
            "table_penalty": table_penalty,
            "progress_reward": progress_reward,
            "task_score": score,
            "best_score": self._best_score,
            "reached_box": float(self._reached_box),
            "gripper_grasping": float(gripper_grasping),
            "premature_close_penalty": premature_close_penalty,
            "success_hold_count": float(self._success_hold_count),
            # Stub keys so evaluate_cartesian.py prints cleanly.
            "align_close_bonus": 0.0,
            "contact_bonus": 0.0,
            "hold_bonus": 8.0 * lift_fraction if gripper_grasping else 0.0,
            "lift_action_bonus": 0.0,
            "no_lift_contact_penalty": 0.0,
            "invalid_lift_penalty": 0.0,
            "lift_delta": 0.0,
        }

    def _is_success(self) -> bool:
        object_lift = float(max(0.0, self.data.xpos[self.object_body_id][2] - self.initial_object_z))
        left_touch, right_touch = self._finger_object_contact()
        if object_lift >= self.config.min_lift_for_success and (left_touch or right_touch):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        return self._success_hold_count >= 3
