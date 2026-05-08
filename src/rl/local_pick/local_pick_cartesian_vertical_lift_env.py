from __future__ import annotations

import numpy as np

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_balanced_lift_env import LocalPickCartesianBalancedLiftEnv


class LocalPickCartesianVerticalLiftEnv(LocalPickCartesianBalancedLiftEnv):
    """Balanced lift with post-grasp reward focused on straight-up motion."""

    def __init__(self, config: LocalPickConfig | None = None, render_mode: str | None = None):
        config = config or LocalPickConfig()
        config.action_scale = 0.01
        config.max_episode_steps = max(config.max_episode_steps, 125)
        super().__init__(config=config, render_mode=render_mode)
        self.config.action_scale = 0.01
        self.config.max_episode_steps = max(self.config.max_episode_steps, 125)
        self._gripper_ctrl_step = 15.0
        self._grasp_xy_anchor = np.zeros(2, dtype=float)
        self._has_grasp_anchor = False
        self._was_aligned = False
        self._was_valid_grasp = False
        self._prev_ee_z = 0.0

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.data.ctrl[self.gripper_actuator_id] = self.config.open_gripper_ctrl
        self._grasp_xy_anchor = self.data.xpos[self.object_body_id][:2].copy()
        self._has_grasp_anchor = False
        self._was_aligned = False
        self._was_valid_grasp = False
        self._prev_ee_z = self._ee_pos()[2]
        return obs, info

    def _set_gripper(self, open_gripper: bool) -> None:
        target = self.config.open_gripper_ctrl if open_gripper else self.config.close_gripper_ctrl
        current = float(self.data.ctrl[self.gripper_actuator_id])
        delta = target - current
        if abs(delta) <= self._gripper_ctrl_step:
            next_ctrl = target
        else:
            next_ctrl = current + np.sign(delta) * self._gripper_ctrl_step
        self.data.ctrl[self.gripper_actuator_id] = float(
            np.clip(next_ctrl, self.config.close_gripper_ctrl, self.config.open_gripper_ctrl)
        )

    def _compute_reward(self, cart_action: np.ndarray, gripper_cmd: float) -> tuple[float, dict[str, float]]:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        reach_distance = float(np.linalg.norm(ee_pos - object_pos))
        object_world_pos = self.data.xpos[self.object_body_id].copy()
        ee_z = float(ee_pos[2])
        ee_z_delta = max(0.0, ee_z - self._prev_ee_z)
        object_lift = float(max(0.0, object_world_pos[2] - self.initial_object_z))
        lift_delta = max(0.0, object_lift - self._prev_object_lift)
        object_motion_vec = object_world_pos - self._prev_object_pos
        object_motion = float(np.linalg.norm(object_motion_vec))
        object_xy_motion = float(np.linalg.norm(object_motion_vec[:2]))
        action_penalty = float(np.linalg.norm(cart_action) ** 2)
        xy_action_penalty = float(np.linalg.norm(cart_action[:2]) ** 2)
        table_penalty = self._table_collision_penalty()
        left_touch, right_touch = self._finger_object_contact()
        both_touching = left_touch and right_touch
        gripper_closed = self._gripper_opening() <= 0.012
        valid_grasp = bool(gripper_closed and both_touching)
        aligned = reach_distance <= self.config.align_close_threshold

        if valid_grasp and not self._has_grasp_anchor:
            self._grasp_xy_anchor = object_world_pos[:2].copy()
            self._has_grasp_anchor = True
        if not valid_grasp:
            self._has_grasp_anchor = False

        object_xy_drift = float(np.linalg.norm(object_world_pos[:2] - self._grasp_xy_anchor))

        reach_reward = 1.0 - np.tanh(12.0 * reach_distance)

        align_once_bonus = 0.0
        if aligned and not self._was_aligned:
            align_once_bonus = 1.0

        close_aligned_bonus = 0.0
        if gripper_cmd > 0 and aligned:
            close_progress = 1.0 - min(self._gripper_opening() / 0.04, 1.0)
            close_aligned_bonus = 1.5 * close_progress

        premature_close_penalty = 0.0
        if gripper_cmd > 0 and reach_distance > self.config.premature_close_threshold:
            premature_close_penalty = self.config.premature_close_penalty

        grasp_bonus = 0.0
        first_grasp_bonus = 0.0
        z_lift_reward = 0.0
        lifted_hold_reward = 0.0
        lift_action_bonus = 0.0
        co_lift_reward = 0.0
        lift_mismatch_penalty = 0.0
        post_grasp_xy_penalty = 0.0
        object_xy_motion_penalty = 0.0
        xy_drift_penalty = 0.0
        ready_to_lift = gripper_closed and aligned
        if ready_to_lift:
            lift_action_bonus = 2.0 * max(float(cart_action[2]), 0.0)
            co_lift_reward = 180.0 * min(lift_delta, ee_z_delta)
            if lift_delta > 0.001:
                lift_mismatch_penalty = 80.0 * max(0.0, lift_delta - ee_z_delta)
        if valid_grasp:
            z_lift_reward = 160.0 * lift_delta
            lift_fraction = min(object_lift / self.config.min_lift_for_success, 1.0)
            lifted_hold_reward = 3.0 * lift_fraction
            post_grasp_xy_penalty = 0.6 * xy_action_penalty
            object_xy_motion_penalty = 30.0 * object_xy_motion
            xy_drift_penalty = 12.0 * min(object_xy_drift, 0.05)

        invalid_lift_penalty = 0.0
        if object_lift > 0.008 and not valid_grasp:
            invalid_lift_penalty = 5.0 * min(object_lift / self.config.min_lift_for_success, 1.0)

        if valid_grasp and object_lift >= self.config.min_lift_for_success and object_xy_drift < 0.035:
            self._stable_lift_count += 1
        else:
            self._stable_lift_count = 0

        reward = (
            reach_reward
            + align_once_bonus
            + close_aligned_bonus
            + lift_action_bonus
            + co_lift_reward
            + z_lift_reward
            + lifted_hold_reward
            - post_grasp_xy_penalty
            - object_xy_motion_penalty
            - xy_drift_penalty
            - lift_mismatch_penalty
            - invalid_lift_penalty
            - premature_close_penalty
            - 0.015 * action_penalty
            - 2.0 * table_penalty
        )
        success = self._is_success()
        if success:
            reward += self.config.success_bonus

        self._was_aligned = aligned
        self._was_valid_grasp = valid_grasp
        self._prev_ee_z = ee_z

        return float(reward), {
            "reach_distance": reach_distance,
            "object_lift": object_lift,
            "lift_delta": lift_delta,
            "ee_z_delta": ee_z_delta,
            "object_motion": object_motion,
            "object_xy_motion": object_xy_motion,
            "object_xy_drift": object_xy_drift,
            "action_penalty": action_penalty,
            "xy_action_penalty": xy_action_penalty,
            "table_penalty": table_penalty,
            "align_close_bonus": align_once_bonus + close_aligned_bonus,
            "contact_bonus": 0.0,
            "hold_bonus": lifted_hold_reward,
            "lift_action_bonus": lift_action_bonus,
            "co_lift_reward": co_lift_reward,
            "lift_mismatch_penalty": lift_mismatch_penalty,
            "premature_close_penalty": premature_close_penalty,
            "post_grasp_xy_penalty": post_grasp_xy_penalty,
            "object_xy_motion_penalty": object_xy_motion_penalty,
            "xy_drift_penalty": xy_drift_penalty,
            "invalid_lift_penalty": invalid_lift_penalty,
            "stable_lift_count": float(self._stable_lift_count),
        }

    def _is_success(self) -> bool:
        return self._stable_lift_count >= 5
