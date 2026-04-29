from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only when deps are missing
    raise ImportError(
        "LocalPickEnv requires gymnasium. Install dependencies with "
        "`pip install -r src/requirements.txt`."
    ) from exc

# Allow running this file/scripts directly from the repo without installing as a package.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from pipeline.controller.kinematics import Kinematics
from rl.local_pick.config import LocalPickConfig


class LocalPickEnv(gym.Env):
    """Gymnasium environment for training only the local pick skill.

    Episodes start with the Panda gripper already near an approach pose above the
    object. The policy outputs small joint-space deltas; the gripper is scripted
    open before ``gripper_close_step`` and closed afterwards.
    """

    metadata = {"render_modes": ["human", None], "render_fps": 25}

    def __init__(self, config: LocalPickConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or LocalPickConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(self.config.seed)

        if not self.config.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.config.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.config.xml_path))
        self.data = mujoco.MjData(self.model)
        self.kinematics = Kinematics()
        self.viewer = None

        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.actuator_names = [f"actuator{i}" for i in range(1, 8)]
        self.joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ])
        self.actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.actuator_names
        ])
        self.qpos_idx = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_idx = self.model.jnt_dofadr[self.joint_ids]
        self.joint_limits = np.array([self.model.jnt_range[jid] for jid in self.joint_ids])

        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self.finger_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
        ]
        self.finger_qpos_idx = self.model.jnt_qposadr[self.finger_joint_ids[0]]
        self.finger_qpos_indices = [self.model.jnt_qposadr[jid] for jid in self.finger_joint_ids]
        self.finger_qvel_indices = [self.model.jnt_dofadr[jid] for jid in self.finger_joint_ids]

        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_object"
        )
        self.object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pick_freejoint"
        )
        self.object_qpos_idx = self.model.jnt_qposadr[self.object_joint_id]

        self.left_finger_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self.right_finger_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.step_count = 0
        self.initial_object_z = self.config.object_z
        self.last_object_pos = np.zeros(3)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        object_pos = None
        q_start = None
        requested_object_pos = options.get("object_pos")
        for _ in range(100):
            object_pos = np.array(
                requested_object_pos if requested_object_pos is not None else self._sample_object_position(),
                dtype=float,
            )
            q_start = self._sample_approach_joint_state(object_pos)
            if q_start is not None:
                break
            if requested_object_pos is not None:
                raise RuntimeError("Could not find an IK solution for the requested local-pick approach pose.")
        if object_pos is None or q_start is None:
            raise RuntimeError("Could not sample an IK-reachable local-pick approach pose.")

        self._set_object_pose(object_pos)
        self.initial_object_z = float(object_pos[2])

        self._set_arm_state(q_start)
        self._set_gripper(open_gripper=True)
        self._set_gripper_state(open_gripper=True)
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {"object_pos": self._object_pos().copy(), "success": False}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        q_current = self._joint_positions()
        q_target = q_current + action * self.config.action_scale
        q_target = np.clip(q_target, self.joint_limits[:, 0], self.joint_limits[:, 1])

        self._command_arm(q_target)
        self._set_gripper(open_gripper=self.step_count < self.config.gripper_close_step)

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

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _sample_object_position(self) -> np.ndarray:
        if self.config.fixed_object_position is None:
            x = self.rng.uniform(self.config.table_x_min, self.config.table_x_max)
            y = self.rng.uniform(self.config.table_y_min, self.config.table_y_max)
            base = np.array([x, y, self.config.object_z], dtype=float)
        else:
            base = np.array(self.config.fixed_object_position, dtype=float)

        if self.config.object_xy_noise > 0:
            base[:2] += self.rng.uniform(
                -self.config.object_xy_noise, self.config.object_xy_noise, size=2
            )
            base[0] = np.clip(base[0], self.config.table_x_min, self.config.table_x_max)
            base[1] = np.clip(base[1], self.config.table_y_min, self.config.table_y_max)
        return base

    def _sample_approach_joint_state(self, object_pos: np.ndarray) -> np.ndarray:
        target_rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
        home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

        for _ in range(50):
            approach_pos = object_pos.copy()
            approach_pos[2] += self.config.approach_height
            if self.config.approach_xy_noise > 0:
                approach_pos[:2] += self.rng.uniform(
                    -self.config.approach_xy_noise, self.config.approach_xy_noise, size=2
                )

            solutions = self.kinematics.ik(approach_pos, target_rot, free_joint_samples=100)
            valid = [q for q in solutions if self._within_joint_limits(q)]
            if valid:
                q = min(valid, key=lambda sol: np.linalg.norm(sol - home))
                q = np.array(q, dtype=float)
                if self.config.joint_noise > 0:
                    q += self.rng.normal(0.0, self.config.joint_noise, size=7)
                    q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])
                return q

        return None

    def _within_joint_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.joint_limits[:, 0]) and np.all(q <= self.joint_limits[:, 1]))

    def _set_object_pose(self, object_pos: np.ndarray) -> None:
        self.data.qpos[self.object_qpos_idx:self.object_qpos_idx + 3] = object_pos
        self.data.qpos[self.object_qpos_idx + 3:self.object_qpos_idx + 7] = [1.0, 0.0, 0.0, 0.0]

    def _set_arm_state(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_idx] = q
        self.data.qvel[self.qvel_idx] = 0.0
        self._command_arm(q)

    def _command_arm(self, q_target: np.ndarray) -> None:
        for act_id, value in zip(self.actuator_ids, q_target):
            self.data.ctrl[act_id] = float(value)

    def _set_gripper(self, open_gripper: bool) -> None:
        self.data.ctrl[self.gripper_actuator_id] = (
            self.config.open_gripper_ctrl if open_gripper else self.config.close_gripper_ctrl
        )

    def _set_gripper_state(self, open_gripper: bool) -> None:
        qpos = 0.04 if open_gripper else 0.0
        for qidx, vidx in zip(self.finger_qpos_indices, self.finger_qvel_indices):
            self.data.qpos[qidx] = qpos
            self.data.qvel[vidx] = 0.0

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

    def _phase(self) -> float:
        return 0.0 if self.step_count < self.config.gripper_close_step else 1.0

    def _get_obs(self) -> np.ndarray:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        obs = np.concatenate([
            self._joint_positions(),
            self._joint_velocities(),
            ee_pos - object_pos,
            np.array([object_pos[2] - self.initial_object_z], dtype=float),
            np.array([self._gripper_opening()], dtype=float),
            np.array([self._phase()], dtype=float),
        ])
        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        reach_distance = float(np.linalg.norm(ee_pos - object_pos))
        object_lift = float(max(0.0, object_pos[2] - self.initial_object_z))
        action_penalty = float(np.linalg.norm(action) ** 2)
        table_penalty = self._table_collision_penalty()
        success = self._is_success()

        reward = (
            -self.config.reach_weight * reach_distance
            + self.config.lift_weight * object_lift
            - self.config.action_penalty_weight * action_penalty
            - table_penalty
        )
        if success:
            reward += self.config.success_bonus

        return float(reward), {
            "reach_distance": reach_distance,
            "object_lift": object_lift,
            "action_penalty": action_penalty,
            "table_penalty": table_penalty,
        }

    def _table_collision_penalty(self) -> float:
        table_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
        robot_body_ids = set(range(1, 12))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if table_body_id in (body1, body2) and (body1 in robot_body_ids or body2 in robot_body_ids):
                return self.config.table_collision_penalty
        return 0.0

    def _is_success(self) -> bool:
        return bool(self._object_pos()[2] >= self.config.lift_success_z)

