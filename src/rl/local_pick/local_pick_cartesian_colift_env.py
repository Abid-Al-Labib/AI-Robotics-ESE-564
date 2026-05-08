from __future__ import annotations

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_cartesian_vertical_lift_env import LocalPickCartesianVerticalLiftEnv


class LocalPickCartesianCoLiftEnv(LocalPickCartesianVerticalLiftEnv):
    """Named co-lift experiment.

    This keeps the current vertical-lift mechanics under a distinct env name:
    125-step episodes, smaller Cartesian actions, gradual gripper control,
    no contact reward, z-action bridge reward, and co-lift shaping that favors
    the gripper and object moving upward together.
    """

    def __init__(self, config: LocalPickConfig | None = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
