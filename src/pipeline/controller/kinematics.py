# pipeline/controller/ikfast/kinematics.py
import numpy as np
from .ikfast import ikfast_panda_arm as ikfast


class Kinematics:
    def fk(self, joints: np.ndarray):
        result = ikfast.get_fk(list(joints))
        pos = np.array(result[0])
        rot = np.array(result[1])
        return pos, rot

    def ik(self, target_pos: np.ndarray, target_rot: np.ndarray, free_joint_samples: int = 100):
        rot_list = target_rot.tolist()
        pos_list = target_pos.tolist()
        
        solutions = []
        free_joint_range = np.linspace(-2.8, 2.8, free_joint_samples)
        for free_val in free_joint_range:
            result = ikfast.get_ik(rot_list, pos_list, [free_val])
            if result:
                for sol in result:
                    solutions.append(np.array(sol))
        return solutions