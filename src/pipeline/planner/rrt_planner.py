# pipeline/planner/rrt_planner.py
import numpy as np
import mujoco


class RRTPlanner:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            for i in range(7)
        ]
        self.joint_qpos_idx = model.jnt_qposadr[self.joint_ids]
        self.joint_limits = np.array([model.jnt_range[jid] for jid in self.joint_ids])

        # Allowed contact pairs (adjacent links that always slightly overlap)
        self.allowed_pairs = set()
        pairs_to_allow = [
            ("left_finger", "right_finger"),
            ("link5", "link7"),
            ("link5", "hand"),
        ]
        for name_a, name_b in pairs_to_allow:
            id_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name_a)
            id_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name_b)
            self.allowed_pairs.add((min(id_a, id_b), max(id_a, id_b)))

    def get_joint_positions(self):
        return self.data.qpos[self.joint_qpos_idx].copy()

    def set_joint_positions(self, q):
        self.data.qpos[self.joint_qpos_idx] = q
        mujoco.mj_forward(self.model, self.data)

    def _is_robot_collision(self, contact):
        robot_bodies = set(range(1, 12))
        body1 = self.model.geom_bodyid[contact.geom1]
        body2 = self.model.geom_bodyid[contact.geom2]

        if body1 not in robot_bodies and body2 not in robot_bodies:
            return False

        pair = (min(body1, body2), max(body1, body2))
        if pair in self.allowed_pairs:
            return False

        return True

    def is_config_valid(self, q, debug=False):
        q_orig = self.get_joint_positions()
        self.set_joint_positions(q)
        mujoco.mj_collision(self.model, self.data)

        colliding = False
        for i in range(self.data.ncon):
            if self.data.contact[i].dist < 0 and self._is_robot_collision(self.data.contact[i]):
                if debug:
                    body1 = self.model.geom_bodyid[self.data.contact[i].geom1]
                    body2 = self.model.geom_bodyid[self.data.contact[i].geom2]
                    name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1) or str(body1)
                    name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2) or str(body2)
                    print(f"  Penetration: {name1} <-> {name2}, dist={self.data.contact[i].dist:.6f}")
                colliding = True
                if not debug:
                    break

        self.set_joint_positions(q_orig)
        return not colliding

    def is_collision_free(self, q1, q2, max_step_size=0.1):
        q_orig = self.get_joint_positions()

        delta = q2 - q1
        dist = np.linalg.norm(delta)
        num_steps = int(np.ceil(dist / max_step_size))

        for step in range(num_steps + 1):
            q_interp = q1 + (step / num_steps) * delta
            self.set_joint_positions(q_interp)
            mujoco.mj_collision(self.model, self.data)
            for i in range(self.data.ncon):
                if self.data.contact[i].dist < 0 and self._is_robot_collision(self.data.contact[i]):
                    self.set_joint_positions(q_orig)
                    return False

        self.set_joint_positions(q_orig)
        return True

    def sample(self, q_goal, epsilon=0.1):
        if np.random.rand() < epsilon:
            return q_goal.copy()
        q = np.zeros(7)
        for i in range(7):
            q[i] = np.random.uniform(self.joint_limits[i, 0], self.joint_limits[i, 1])
        return q

    def nearest(self, nodes, q):
        dists = np.linalg.norm(nodes - q, axis=1)
        return nodes[np.argmin(dists)]

    def extend(self, q_near, q_sample, step_size):
        v = q_sample - q_near
        dist = np.linalg.norm(v)
        if dist <= step_size:
            return q_sample.copy()
        return q_near + (v / dist) * step_size

    def retrace(self, q_goal, tree):
        path = []
        curr = tuple(q_goal)
        while curr is not None:
            path.append(np.array(curr))
            curr = tree.get(curr, None)
        return path[::-1]

    def smooth(self, path, max_iters=1000):
        if path is None or len(path) <= 2:
            return path
        path = [np.array(q) for q in path]
        for _ in range(max_iters):
            if len(path) < 5:
                break
            i = np.random.randint(1, len(path) - 3)
            j = np.random.randint(i + 2, len(path) - 1)
            if self.is_collision_free(path[i], path[j]):
                path = path[:i+1] + path[j:]
        return path

    def plan(self, q_goal, q_init=None, threshold=0.01, step_size=0.1, max_iters=7000, smooth=True):
        if q_init is None:
            q_init = self.get_joint_positions()

        q_init = np.array(q_init, dtype=float)
        q_goal = np.array(q_goal, dtype=float)

        nodes = np.array([q_init])
        tree = {tuple(q_init): None}

        for iteration in range(max_iters):
            q_samp = self.sample(q_goal)
            q_near = self.nearest(nodes, q_samp)
            q_new = self.extend(q_near, q_samp, step_size)

            if not self.is_collision_free(q_near, q_new, max_step_size=step_size):
                continue

            nodes = np.vstack([nodes, q_new])
            tree[tuple(q_new)] = tuple(q_near)

            if iteration % 500 == 0:
                print(f"  RRT iteration {iteration}, nodes: {len(nodes)}")

            if np.linalg.norm(q_new - q_goal) <= threshold:
                path = self.retrace(q_new, tree)
                if smooth:
                    path = self.smooth(path)
                print(f"RRT found path with {len(path)} waypoints in {iteration} iterations")
                return path

        print(f"RRT failed after {max_iters} iterations ({len(nodes)} nodes explored)")
        return None