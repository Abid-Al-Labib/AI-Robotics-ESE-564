# Save as src/debug_bodies.py
import mujoco
from pathlib import Path

xml_path = Path('.').resolve().parent / 'assets' / 'mujoco' / 'scene.xml'
model = mujoco.MjModel.from_xml_path(str(xml_path))

for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"Body {i}: {name}")