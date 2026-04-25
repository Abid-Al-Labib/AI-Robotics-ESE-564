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

    def _hold_arm(self):
        """Snapshot current arm qpos so _step_until_settled can re-lock it each frame."""
        self._arm_lock = []
        for i in range(7):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            qidx = self.model.jnt_qposadr[jid]
            vidx = self.model.jnt_dofadr[jid]
            self._arm_lock.append((qidx, vidx, float(self.data.qpos[qidx])))

    def _relock_arm(self):
        """Pin arm back to snapshot — called before each physics frame to cancel gravity drift."""
        for qidx, vidx, target in self._arm_lock:
            self.data.qpos[qidx] = target
            self.data.qvel[vidx] = 0.0

    def _step_until_settled(self, viewer, steps_per_frame=10, max_frames=200):
        prev = float(self.data.qpos[self.finger_qidx])
        for _ in range(max_frames):
            for _ in range(steps_per_frame):
                mujoco.mj_step(self.model, self.data)
                self._relock_arm()      # snap arm back after every step, not just per frame
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
