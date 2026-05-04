"""Quick visual check of the approach pose before training."""
from __future__ import annotations
import sys, time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from rl.local_pick.config import LocalPickConfig
from rl.local_pick.local_pick_env import LocalPickEnv

config = LocalPickConfig(side_grasp=True)
env = LocalPickEnv(config=config, render_mode="human")
obs, info = env.reset()
joints = env._joint_positions()
ee_pos, ee_rot = env.kinematics.fk(joints)
print("Approach pose set. Object at:", info["object_pos"])
print("EE position:", ee_pos)
print("EE rotation:\n", ee_rot)
print("Close the viewer window to exit.")
env.render()
while env.viewer is not None and env.viewer.is_running():
    env.render()
    time.sleep(0.05)
env.close()
