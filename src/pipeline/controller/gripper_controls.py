import mujoco
import numpy as np

CTRL_OPEN = 255.0   # fingers fully open (0.04m each)
CTRL_CLOSE = 0.0    # close command — contact forces stop fingers on object


class GripperController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.finger_qidx = model.jnt_qposadr[j1_id]
        self._arm_act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i+1}")
            for i in range(7)
        ]
        self._arm_qpos_idx = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")]
            for i in range(7)
        ]

    def _hold_arm(self):
        """Command arm actuators to hold their current joint positions."""
        for act_id, qidx in zip(self._arm_act_ids, self._arm_qpos_idx):
            self.data.ctrl[act_id] = float(self.data.qpos[qidx])

    def _step_until_settled(self, viewer, steps_per_frame=10, max_frames=200):
        prev = float(self.data.qpos[self.finger_qidx])
        for _ in range(max_frames):
            for _ in range(steps_per_frame):
                mujoco.mj_step(self.model, self.data)
            viewer.sync()
            curr = float(self.data.qpos[self.finger_qidx])
            if abs(curr - prev) < 1e-5:
                break
            prev = curr

    def open(self, viewer):
        """Open gripper fully using physics stepping."""
        self._hold_arm()
        self.data.ctrl[self.act_id] = CTRL_OPEN
        self._step_until_settled(viewer)

    def close(self, viewer):
        """Close gripper — physics contact stops fingers at the object, no target needed."""
        self._hold_arm()
        self.data.ctrl[self.act_id] = CTRL_CLOSE
        self._step_until_settled(viewer)
