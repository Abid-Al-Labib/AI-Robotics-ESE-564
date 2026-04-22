import mujoco
import numpy as np
from pathlib import Path

project_root = Path('.').resolve().parent
xml_path = project_root / "assets" / "mujoco" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "perception_cam")

# RGB image
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera=cam_id)
rgb = renderer.render()

# Depth image
renderer.enable_depth_rendering()
renderer.update_scene(data, camera=cam_id)
depth = renderer.render()

print(f"RGB shape: {rgb.shape}")
print(f"Depth shape: {depth.shape}")
print(f"Depth range: {depth.min():.3f} to {depth.max():.3f}")

# Save to check the view
from PIL import Image
Image.fromarray(rgb).save("camera_test.png")
print("Saved camera_test.png")