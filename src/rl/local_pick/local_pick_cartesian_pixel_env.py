from __future__ import annotations

import mujoco
import numpy as np
from gymnasium import spaces

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_direct_lift_env import LocalPickCartesianDirectLiftEnv

_PIXEL_H = 84
_PIXEL_W = 84


class LocalPickCartesianPixelEnv(LocalPickCartesianDirectLiftEnv):
    """Pixel-observation variant of the Option A env (DrQ-v2 style).

    Observation: 84×84×3 uint8 wrist-cam RGB.
    Reward:      identical to LocalPickCartesianDirectLiftEnv (progress + reached_box).

    Use with SB3 CnnPolicy + VecFrameStack(n_stack=3) for temporal context.
    The parent 120×160 renderer still runs for object detection (reward/info);
    this class adds a second 84×84 renderer dedicated to pixel observations.

    aug_pad > 0 enables random-shift augmentation (DrQ style): pad by aug_pad
    pixels on each side, then take a random crop back to 84×84. Set aug_pad=0
    at eval time to disable augmentation.
    """

    def __init__(
        self,
        config: LocalPickConfig | None = None,
        render_mode: str | None = None,
        aug_pad: int = 0,
    ):
        super().__init__(config=config, render_mode=render_mode)
        self.config.action_scale = 0.02
        self._aug_pad = aug_pad
        self._pixel_renderer = mujoco.Renderer(self.model, height=_PIXEL_H, width=_PIXEL_W)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(_PIXEL_H, _PIXEL_W, 3), dtype=np.uint8
        )

    def _get_obs(self) -> np.ndarray:
        self._pixel_renderer.update_scene(self.data, camera=self._cam_id)
        rgb = self._pixel_renderer.render().copy()  # (84, 84, 3) uint8, HWC
        if self._aug_pad > 0:
            rgb = self._random_shift(rgb)
        return rgb

    def _random_shift(self, img: np.ndarray) -> np.ndarray:
        pad = self._aug_pad
        h, w, c = img.shape
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        x = np.random.randint(0, 2 * pad + 1)
        y = np.random.randint(0, 2 * pad + 1)
        return padded[y : y + h, x : x + w, :]

    def close(self):
        if getattr(self, "_pixel_renderer", None) is not None:
            self._pixel_renderer.close()
            self._pixel_renderer = None
        super().close()
