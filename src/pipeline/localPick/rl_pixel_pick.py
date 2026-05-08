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


class RLPixelPickController:
    """Run the trained DrQ-style pixel SAC policy in an existing MuJoCo scene.

    Drop-in replacement for RLCartesianVerticalPickController.
    Observation: 84x84 wrist-cam RGB, 3-frame stack, CHW uint8.
    Action:      4-D Cartesian delta [dx, dy, dz, gripper] via IK.
    No obs normalisation needed — policy was trained with norm_obs=False.
    """

    N_STACK = 3
    PIXEL_H = 84
    PIXEL_W = 84

    def __init__(
        self,
        model,
        data,
        model_path: str | Path,
        config: LocalPickConfig | None = None,
    ):
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise ImportError(
                "RLPixelPickController requires stable-baselines3. "
                "Run: pip install stable-baselines3"
            ) from exc

        self.model = model
        self.data = data
        self.config = config or LocalPickConfig()
        self.config.action_scale = 0.02
        self.config.max_episode_steps = 125

        self.policy = SAC.load(model_path, device="auto")
        self.kinematics = Kinematics()

        # IK / joint bookkeeping — same as other Cartesian controllers.
        self.joint_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ])
        self.actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
            for i in range(1, 8)
        ])
        self.qpos_idx = model.jnt_qposadr[self.joint_ids]
        self.joint_limits = np.array([model.jnt_range[jid] for jid in self.joint_ids])

        self.gripper_actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self.object_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pick_object"
        )
        self.left_finger_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self.right_finger_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )

        # Pixel renderer — separate from any viewer so no OpenGL conflict.
        self._cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
        )
        self._pixel_renderer = mujoco.Renderer(
            model, height=self.PIXEL_H, width=self.PIXEL_W
        )

        self._target_ee_pos: np.ndarray | None = None
        self._target_rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
        self._gripper_ctrl_step = 15.0

        print(f"Loaded pixel SAC model from {model_path}")

    def execute(self, viewer=None) -> bool:
        """Run the pixel policy from the current approach state."""
        self._target_ee_pos = self._ee_pos()
        initial_object_z = float(self.data.xpos[self.object_body_id][2])
        success_count = 0

        first_obs = self._get_pixel_obs()
        frames = [first_obs.copy() for _ in range(self.N_STACK)]

        for _ in range(self.config.max_episode_steps):
            stacked = np.concatenate(frames, axis=2)
            obs_chw = stacked.transpose(2, 0, 1)[np.newaxis]

            action, _ = self.policy.predict(obs_chw, deterministic=True)
            self._apply_action(action[0])

            for _ in range(self.config.frame_skip):
                mujoco.mj_step(self.model, self.data)

            if viewer is not None:
                viewer.sync()

            frames.pop(0)
            frames.append(self._get_pixel_obs())

            object_lift = float(self.data.xpos[self.object_body_id][2] - initial_object_z)
            left_touch, right_touch = self._finger_contact()
            if object_lift >= self.config.min_lift_for_success and (left_touch or right_touch):
                success_count += 1
                if success_count >= 3:
                    break
            else:
                success_count = 0

        return success_count >= 3

    def close(self) -> None:
        self._pixel_renderer.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pixel_obs(self) -> np.ndarray:
        """Render wrist cam at 84x84, return (84,84,3) uint8 HWC."""
        self._pixel_renderer.update_scene(self.data, camera=self._cam_id)
        return self._pixel_renderer.render().copy()

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        cart_action = action[:3]
        gripper_cmd = float(action[3])

        if self._target_ee_pos is None:
            self._target_ee_pos = self._ee_pos()
        self._target_ee_pos = self._target_ee_pos + cart_action * self.config.action_scale
        self._target_ee_pos = self._clip_target(self._target_ee_pos)
        self._command_cartesian_target(self._target_ee_pos)
        self._set_gripper(open_gripper=(gripper_cmd <= 0))

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        t = target.copy()
        t[0] = np.clip(t[0], self.config.table_x_min - 0.08, self.config.table_x_max + 0.08)
        t[1] = np.clip(t[1], self.config.table_y_min - 0.08, self.config.table_y_max + 0.08)
        t[2] = np.clip(t[2], self.config.object_z - 0.06, self.config.object_z + 0.25)
        return t

    def _command_cartesian_target(self, target_pos: np.ndarray) -> bool:
        current_q = self.data.qpos[self.qpos_idx].copy()
        free_joint_range = np.linspace(current_q[-1] - 0.8, current_q[-1] + 0.8, 31)
        solutions = self.kinematics.ik(
            target_pos, self._target_rot, free_joint_range=free_joint_range
        )
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
        step = self._gripper_ctrl_step
        next_ctrl = target if abs(delta) <= step else current + np.sign(delta) * step
        self.data.ctrl[self.gripper_actuator_id] = float(
            np.clip(next_ctrl, self.config.close_gripper_ctrl, self.config.open_gripper_ctrl)
        )

    def _within_joint_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.joint_limits[:, 0]) and np.all(q <= self.joint_limits[:, 1]))

    def _ee_pos(self) -> np.ndarray:
        left = self.data.xpos[self.left_finger_body_id]
        right = self.data.xpos[self.right_finger_body_id]
        return ((left + right) * 0.5).copy()

    def _finger_contact(self) -> tuple[bool, bool]:
        left_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in ("left_finger_collision", "left_fingertip_collision")
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0
        }
        right_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in ("right_finger_collision", "right_fingertip_collision")
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0
        }
        obj_geoms = set()
        obj_body = self.object_body_id
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] == obj_body:
                obj_geoms.add(gid)

        left_touch = right_touch = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            if pair & left_geoms and pair & obj_geoms:
                left_touch = True
            if pair & right_geoms and pair & obj_geoms:
                right_touch = True
        return left_touch, right_touch
