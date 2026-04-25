# pipeline/perception/perception.py
import numpy as np
import mujoco


class Perception:
    def __init__(self, model, data, cam_name="perception_cam", height=480, width=640):
        self.model = model
        self.data = data
        self.cam_name = cam_name
        self.height = height
        self.width = width
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam_id)
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def find_object_pixel(self, rgb, target_color, threshold=50):
            """Find object pixel: topmost y, midpoint x between leftmost and rightmost match."""
            diff = np.linalg.norm(rgb.astype(float) - np.array(target_color), axis=2)
            mask = diff < threshold
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return None
            mid_x = int((xs.min() + xs.max()) / 2)
            top_y = int(ys.min())
            return mid_x, top_y

    def pixel_to_world(self, pixel_x, pixel_y, depth_img):
            fovy = self.model.cam_fovy[self.cam_id]
            f = 0.5 * self.height / np.tan(np.radians(fovy / 2))

            # MuJoCo depth renderer returns linear depth directly
            linear_depth = depth_img[pixel_y, pixel_x]

            cx, cy = self.width / 2, self.height / 2
            x_cam = (pixel_x - cx) * linear_depth / f
            y_cam = -(pixel_y - cy) * linear_depth / f
            z_cam = -linear_depth

            cam_pos = self.data.cam_xpos[self.cam_id]
            cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3)

            point_cam = np.array([x_cam, y_cam, z_cam])
            point_world = cam_pos + cam_rot @ point_cam
            return point_world

    def get_object_positions(self):
        rgb, depth = self.render()
        pick_pos = self.get_pick_object_position(rgb, depth)
        place_pos = self.get_goal_position(rgb, depth)
        return pick_pos, place_pos

    def get_pick_object_position(self, rgb=None, depth=None):
        if rgb is None or depth is None:
            rgb, depth = self.render()
        red_pixel = self.find_object_pixel(rgb, [230, 38, 38])
        if red_pixel is None:
            return None
        return self.pixel_to_world(red_pixel[0], red_pixel[1], depth)

    def get_goal_position(self, rgb=None, depth=None):
        if rgb is None or depth is None:
            rgb, depth = self.render()
        green_pixel = self.find_object_pixel(rgb, [26, 217, 26])
        if green_pixel is None:
            return None
        return self.pixel_to_world(green_pixel[0], green_pixel[1], depth)