from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig


class RLLocalPickController:
    """Run a trained local-pick SAC policy inside an existing MuJoCo scene.

    Use this after the global pipeline has already moved the robot to the pick
    approach pose. The observation/action layout mirrors LocalPickEnv exactly.
    """

    def __init__(
        self,
        model,
        data,
        model_path: str | Path = "models/local_pick_sac.zip",
        config: LocalPickConfig | None = None,
    ):
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise ImportError(
                "RLLocalPickController requires stable-baselines3. Install dependencies with "
                "`pip install -r src/requirements.txt`."
            ) from exc

        self.model = model
        self.data = data
        self.config = config or LocalPickConfig()
        self.policy = SAC.load(model_path)

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

        # Camera-based perception
        self._render_h, self._render_w = 120, 160
        self._cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "perception_cam")
        self._renderer = mujoco.Renderer(model, height=self._render_h, width=self._render_w)
        self._cam_object_pos: np.ndarray | None = None

    def execute(self, viewer=None) -> bool:
        """Run the trained local-pick policy from the current approach state."""
        self._cam_object_pos = None
        initial_pos = self._object_pos()
        initial_object_z = float(initial_pos[2])
        success_count = 0

        for _ in range(self.config.max_episode_steps):
            obs = self._get_obs(initial_object_z)
            action, _ = self.policy.predict(obs, deterministic=True)
            self._apply_action(action)

            for _ in range(self.config.frame_skip):
                mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

            self._cam_object_pos = None  # force fresh render each step at inference
            if self._object_pos()[2] - initial_object_z >= self.config.min_lift_for_success:
                success_count += 1
                if success_count >= 3:
                    break
            else:
                success_count = 0

        return success_count >= 3

    def _apply_action(self, action) -> None:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        joint_action = action[:7]
        gripper_cmd = float(action[7])

        q_target = self._joint_positions() + joint_action * self.config.action_scale
        q_target = np.clip(q_target, self.joint_limits[:, 0], self.joint_limits[:, 1])

        for act_id, value in zip(self.actuator_ids, q_target):
            self.data.ctrl[act_id] = float(value)

        self.data.ctrl[self.gripper_actuator_id] = (
            self.config.close_gripper_ctrl if gripper_cmd > 0 else self.config.open_gripper_ctrl
        )

    def _get_obs(self, initial_object_z: float) -> np.ndarray:
        object_pos = self._object_pos()
        ee_pos = self._ee_pos()
        phase = 0.0 if self._gripper_opening() > 0.01 else 1.0
        obs = np.concatenate([
            self._joint_positions(),
            self._joint_velocities(),
            ee_pos - object_pos,
            np.array([object_pos[2] - initial_object_z], dtype=float),
            np.array([self._gripper_opening()], dtype=float),
            np.array([phase], dtype=float),
        ])
        return obs.astype(np.float32)

    def _joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def _joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    def _render_object_pos(self) -> np.ndarray | None:
        """Estimate object CoM from camera RGB+depth by averaging all detected 3D points."""
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
        """Unproject every detected pixel to 3D and return their mean (true CoM estimate).

        A fixed -0.029m z correction compensates for the camera only seeing the top
        surface of the bottle — the occluded bottom half biases the raw estimate high.
        """
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
        com[2] -= 0.029  # camera sees top surface only; correct for occluded bottom half
        com[1] += 0.009  # systematic camera angle bias in Y
        return com

    def _pixel_to_world(self, px: int, py: int, depth: np.ndarray) -> np.ndarray:
        fovy = self.model.cam_fovy[self._cam_id]
        f = 0.5 * self._render_h / np.tan(np.radians(fovy / 2))
        d = float(depth[py, px])
        cx, cy = self._render_w / 2.0, self._render_h / 2.0
        x_cam = (px - cx) * d / f
        y_cam = -(py - cy) * d / f
        z_cam = -d
        cam_pos = self.data.cam_xpos[self._cam_id]
        cam_rot = self.data.cam_xmat[self._cam_id].reshape(3, 3)
        return cam_pos + cam_rot @ np.array([x_cam, y_cam, z_cam])

    def _object_pos(self) -> np.ndarray:
        if self._cam_object_pos is None:
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
