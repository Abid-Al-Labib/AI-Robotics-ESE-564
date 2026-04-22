# src/debug_cam.py
import mujoco
import numpy as np
from pathlib import Path

xml_path = Path('.').resolve().parent / 'assets' / 'mujoco' / 'scene.xml'
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'perception_cam')
print('cam_pos:', data.cam_xpos[cam_id])
print('cam_mat:', data.cam_xmat[cam_id].reshape(3, 3))
print('fovy:', model.cam_fovy[cam_id])

extent = model.stat.extent
near = model.vis.map.znear * extent
far = model.vis.map.zfar * extent
print('near:', near, 'far:', far)