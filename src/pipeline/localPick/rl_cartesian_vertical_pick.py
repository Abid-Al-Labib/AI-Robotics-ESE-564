from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from pipeline.controller.kinematics import Kinematics
from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_vertical_lift_env import LocalPickCartesianVerticalLiftEnv


class RLCartesianVerticalPickController:
    """Run a trained Cartesian local-pick policy in an existing MuJoCo scene."""

    def __init__(
        self,
        model,
        data,
        model_path: str | Path,
        vecnormalize_path: str | Path | None = None,
        config: LocalPickConfig | None = None,
    ):
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except ImportError as exc:
            raise ImportError(
                "RLCartesianVerticalPickController requires stable-baselines3. Install dependencies with "
                "`pip install -r src/requirements.txt`."
            ) from exc

        self.model = model
        self.data = data
        self.config = config or LocalPickConfig()
        self.config.action_scale = 0.01
        self.config.max_episode_steps = max(self.config.max_episode_steps, 125)
        self._gripper_ctrl_step = 15.0
        self.policy = SAC.load(model_path)
        self.kinematics = Kinematics()

        model_path = Path(model_path)
        vecnormalize_path = Path(vecnormalize_path) if vecnormalize_path is not None else self._default_vecnormalize_path(model_path)
        self.vecnormalize = None
        self._norm_env = None
        if vecnormalize_path.exists():
            self._norm_env = DummyVecEnv([
                lambda: LocalPickCartesianVerticalLiftEnv(config=LocalPickConfig(), render_mode=None)
            ])
            self.vecnormalize = VecNormalize.load(str(vecnormalize_path), self._norm_env)
            self.vecnormalize.training = False
            self.vecnormalize.norm_reward = False
            print(f"Loaded Cartesian pick VecNormalize stats: {vecnormalize_path}")
        else:
            print(f"Cartesian pick VecNormalize stats not found: {vecnormalize_path}; using raw observations.")

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

        self.left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_object")

        self._render_h, self._render_w = 120, 160
        self._cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        self._renderer = mujoco.Renderer(model, height=self._render_h, width=self._render_w)
        self._cam_object_pos: np.ndarray | None = None
        self._target_ee_pos: np.ndarray | None = None
        self._target_rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])

    @staticmethod
    def _default_vecnormalize_path(model_path: Path) -> Path:
        stem = model_path.stem if model_path.suffix == ".zip" else model_path.name
        return model_path.with_name(stem + "_vecnormalize.pkl")

    def execute(self, viewer=None) -> bool:
        self._cam_object_pos = None
        self._target_ee_pos = self._ee_pos()
        initial_object_z = float(self.data.xpos[self.object_body_id][2])
        success_count = 0

        for _ in range(self.config.max_episode_steps):
            obs = self._get_obs(initial_object_z)
            policy_obs = self._normalize_obs(obs)
            action, _ = self.policy.predict(policy_obs, deterministic=True)
            self._apply_action(action)

            for _ in range(self.config.frame_skip):
                mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

            object_lift = float(self.data.xpos[self.object_body_id][2] - initial_object_z)
            if object_lift >= self.config.min_lift_for_success:
                success_count += 1
                if success_count >= 5:
                    break
            else:
                success_count = 0

        return success_count >= 5

    def close(self) -> None:
        if self._norm_env is not None:
            self._norm_env.close()
        self._renderer.close()

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.vecnormalize is None:
            return obs
        return self.vecnormalize.normalize_obs(obs.reshape(1, -1))[0]

    def _apply_action(self, action) -> None:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        cart_action = action[:3]
        gripper_cmd = float(action[3])

        if self._target_ee_pos is None:
            self._target_ee_pos = self._ee_pos()
        self._target_ee_pos = self._target_ee_pos + cart_action * self.config.action_scale
        self._target_ee_pos = self._clip_target(self._target_ee_pos)
        self._command_cartesian_target(self._target_ee_pos)
        self._set_gripper(open_gripper=(gripper_cmd <= 0))

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        clipped = target.copy()
        clipped[0] = np.clip(clipped[0], self.config.table_x_min - 0.08, self.config.table_x_max + 0.08)
        clipped[1] = np.clip(clipped[1], self.config.table_y_min - 0.08, self.config.table_y_max + 0.08)
        clipped[2] = np.clip(clipped[2], self.config.object_z - 0.06, self.config.object_z + 0.25)
        return clipped

    def _command_cartesian_target(self, target_pos: np.ndarray) -> bool:
        current_q = self._joint_positions()
        free_joint_range = np.linspace(current_q[-1] - 0.8, current_q[-1] + 0.8, 31)
        solutions = self.kinematics.ik(target_pos, self._target_rot, free_joint_range=free_joint_range)
        valid = [q for q in solutions if self._within_joint_limits(q)]
        if not valid:
            return False
        q_target = min(valid, key=lambda q: np.linalg.norm(q - current_q))
        for act_id, value in zip(self.actuator_ids, q_target):
            self.data.ctrl[act_id] = float(value)
        return True

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

    def _get_obs(self, initial_object_z: float) -> np.ndarray:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        obs = np.concatenate([
            self._joint_positions(),
            self._joint_velocities(),
            ee_pos - object_pos,
            np.array([object_pos[2] - initial_object_z], dtype=float),
            np.array([self._gripper_opening()], dtype=float),
        ])
        return obs.astype(np.float32)

    def _within_joint_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.joint_limits[:, 0]) and np.all(q <= self.joint_limits[:, 1]))

    def _joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def _joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    def _render_object_pos(self) -> np.ndarray | None:
        self._renderer.update_scene(self.data, camera=self._cam_id)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=self._cam_id)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        target = np.array([230, 38, 38], dtype=float)
        diff = np.linalg.norm(rgb.astype(float) - target, axis=2)
        ys, xs = np.where(diff < 50)
        if len(xs) == 0:
            return None
        return self._perception_com(xs, ys, depth)

    def _perception_com(self, xs: np.ndarray, ys: np.ndarray, depth: np.ndarray) -> np.ndarray:
        fovy = self.model.cam_fovy[self._cam_id]
        f = 0.5 * self._render_h / np.tan(np.radians(fovy / 2))
        cx, cy = self._render_w / 2.0, self._render_h / 2.0
        cam_pos = self.data.cam_xpos[self._cam_id]
        cam_rot = self.data.cam_xmat[self._cam_id].reshape(3, 3)

        d = depth[ys, xs].astype(float)
        x_cam = (xs - cx) * d / f
        y_cam = -(ys - cy) * d / f
        z_cam = -d
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        points_world = cam_pos + points_cam @ cam_rot.T
        com = points_world.mean(axis=0)
        com[2] -= 0.0235
        com[1] -= 0.010
        return com

    def _object_pos(self) -> np.ndarray:
        pos = self._render_object_pos()
        if pos is not None:
            self._cam_object_pos = pos
        return self._cam_object_pos.copy() if self._cam_object_pos is not None else np.zeros(3)

    def _ee_pos(self) -> np.ndarray:
        left = self.data.xpos[self.left_finger_body_id]
        right = self.data.xpos[self.right_finger_body_id]
        return ((left + right) * 0.5).copy()

    def _gripper_opening(self) -> float:
        return float(self.data.qpos[self.finger_qpos_idx])
