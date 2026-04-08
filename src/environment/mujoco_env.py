import mujoco
from pathlib import Path
class MujocoEnv:
    def __init__(self, xml_path: str):
        if not Path(xml_path).exists(): raise FileNotFoundError()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
    def step(self):
        mujoco.mj_step(self.model, self.data)
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
    def get_time(self):
        return self.data.time
