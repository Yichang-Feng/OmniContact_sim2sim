import time
import threading
import numpy as np
import zmq
import mujoco
from scipy.spatial.transform import Rotation

EMA_ALPHA = 0.2  # Smoothing factor for EMA


class VisionReceiver:
    def __init__(self, vision_port: int = 5556, sim_cam_port: int = 5555, publish_sim_camera: bool = False):
        self.vision_port = vision_port
        self.sim_cam_port = sim_cam_port
        self.publish_sim_camera = publish_sim_camera

        self.context = zmq.Context()
        self.pose_sub = self.context.socket(zmq.SUB)
        self.pose_sub.setsockopt(zmq.CONFLATE, 1)
        self.pose_sub.bind(f"tcp://127.0.0.1:{vision_port}")
        self.pose_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self.image_pub = None
        if self.publish_sim_camera:
            self.image_pub = self.context.socket(zmq.PUB)
            self.image_pub.bind(f"tcp://127.0.0.1:{sim_cam_port}")
            print(f"[VisionReceiver] Sim camera image publisher bound to tcp://127.0.0.1:{sim_cam_port}")

        self.obj_pose_cv = None
        self.last_good_pos = None
        self.last_good_quat = None
        self.ema_pos = None
        self.goal_pose_cv = None
        self.last_good_goal_pos = None
        self.ema_goal_pos = None
        self.running = True

        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        print(f"[VisionReceiver] Listening for AprilTag poses on tcp://127.0.0.1:{vision_port}")

    def _recv_loop(self):
        while self.running:
            try:
                msg = self.pose_sub.recv_json()
                if "box" in msg:
                    self.obj_pose_cv = msg["box"]
                if "goal" in msg:
                    self.goal_pose_cv = msg["goal"]
            except Exception:
                time.sleep(0.001)

    def publish_image(self, rgb_image: np.ndarray):
        if self.image_pub is None:
            return
        md = dict(
            dtype=str(rgb_image.dtype),
            shape=rgb_image.shape,
        )
        self.image_pub.send_json(md, zmq.SNDMORE)
        self.image_pub.send(rgb_image.tobytes())

    def get_validated_world_pose(self, model: mujoco.MjModel, data: mujoco.MjData, camera_name: str = "d435_camera_frame"):
        """
        Transforms camera-site pose (from vision_node) to robot world frame using MuJoCo data.site_xpos/xmat.
        Returns (pos, quat, is_valid). If no valid vision pose is available, returns (None, None, False).
        """
        if self.obj_pose_cv is None:
            if self.last_good_pos is not None:
                return self.last_good_pos.copy(), self.last_good_quat.copy(), True
            return None, None, False

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, camera_name)
        if site_id >= 0:
            P_cam_world = data.site_xpos[site_id]
            R_cam_world = data.site_xmat[site_id].reshape(3, 3)
        else:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")
            if cam_id < 0:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "d435_camera")
                if body_id >= 0:
                    P_cam_world = data.xpos[body_id]
                    R_cam_world = data.xmat[body_id].reshape(3, 3)
                else:
                    if self.last_good_pos is not None:
                        return self.last_good_pos.copy(), self.last_good_quat.copy(), True
                    return None, None, False
            else:
                P_cam_world = data.cam_xpos[cam_id]
                R_cam_world = data.cam_xmat[cam_id].reshape(3, 3)

        p_obj_m = np.array(self.obj_pose_cv["pos"], dtype=np.float32)
        q_obj_m = np.array(self.obj_pose_cv["quat"], dtype=np.float32) # wxyz
        R_obj_m = Rotation.from_quat([q_obj_m[1], q_obj_m[2], q_obj_m[3], q_obj_m[0]]).as_matrix()

        P_world = P_cam_world + R_cam_world @ p_obj_m
        R_world = R_cam_world @ R_obj_m

        if self.ema_pos is None:
            self.ema_pos = P_world.copy()
        else:
            self.ema_pos = EMA_ALPHA * P_world + (1 - EMA_ALPHA) * self.ema_pos
        self.last_good_pos = self.ema_pos.copy()

        q_world = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q_world, R_world.astype(np.float64).flatten())
        self.last_good_quat = q_world.astype(np.float32).copy()

        return self.last_good_pos.copy(), self.last_good_quat.copy(), True

    def get_validated_goal_pose(self, model: mujoco.MjModel, data: mujoco.MjData, camera_name: str = "d435_camera_frame"):
        if self.goal_pose_cv is None:
            if self.last_good_goal_pos is not None:
                return self.last_good_goal_pos.copy(), True
            return None, False

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, camera_name)
        if site_id >= 0:
            P_cam_world = data.site_xpos[site_id]
            R_cam_world = data.site_xmat[site_id].reshape(3, 3)
        else:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")
            if cam_id < 0:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "d435_camera")
                if body_id >= 0:
                    P_cam_world = data.xpos[body_id]
                    R_cam_world = data.xmat[body_id].reshape(3, 3)
                else:
                    if self.last_good_goal_pos is not None:
                        return self.last_good_goal_pos.copy(), True
                    return None, False
            else:
                P_cam_world = data.cam_xpos[cam_id]
                R_cam_world = data.cam_xmat[cam_id].reshape(3, 3)

        p_goal_m = np.array(self.goal_pose_cv["pos"], dtype=np.float32)
        P_world = P_cam_world + R_cam_world @ p_goal_m

        if self.ema_goal_pos is None:
            self.ema_goal_pos = P_world.copy()
        else:
            self.ema_goal_pos = EMA_ALPHA * P_world + (1 - EMA_ALPHA) * self.ema_goal_pos
        self.last_good_goal_pos = self.ema_goal_pos.copy()

        return self.last_good_goal_pos.copy(), True
