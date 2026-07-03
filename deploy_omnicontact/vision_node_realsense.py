#!/usr/bin/env python3
"""
SUGAR 多源视觉解算与实时预览节点 (Multi-Source Vision Node)
支持三种输入模式切换：
1. --sim : 接收 MuJoCo 仿真发出的虚拟深度相机画面 (ZMQ端口 5555)
2. --realsense : 读取真实机器人机载 Intel RealSense 物理相机
3. --usb (默认) : 读取标准 USB 摄像头 (指定ID如 --cam_id 0)

功能：检测画面中的 AprilTag 角点，实时弹窗展示带检测框的监控画面，
并将解算位姿通过 ZMQ PUB 发送给控制端 (端口 5556)。
"""

import zmq
import json
import cv2
import numpy as np
import argparse
import time
from scipy.spatial.transform import Rotation
from pupil_apriltags import Detector

try:
    from deploy_omnicontact.box_tag_config import BOX_HALF_DIMS as _CFG_DIMS, TAG_SIZE as _CFG_TAG, TAG_LAYOUT as _CFG_LAYOUT
except ImportError:
    try:
        from box_tag_config import BOX_HALF_DIMS as _CFG_DIMS, TAG_SIZE as _CFG_TAG, TAG_LAYOUT as _CFG_LAYOUT
    except ImportError:
        _CFG_DIMS, _CFG_TAG, _CFG_LAYOUT = (0.215, 0.185, 0.125), 0.1, "1tag"

class MultiSourceVisionNode:
    def __init__(self, mode="usb", cam_id=0, zmq_target="127.0.0.1", cam_ip="192.168.123.164", show_img=True, box_dims=_CFG_DIMS, tag_size=_CFG_TAG, fovy=58.76, tag_layout=_CFG_LAYOUT):
        self.mode = mode
        self.show_img = show_img
        self.tag_size = float(tag_size)
        self.fovy = float(fovy)
        self.tag_layout = tag_layout
        self.hx, self.hy, self.hz = float(box_dims[0]), float(box_dims[1]), float(box_dims[2])
        self.context = zmq.Context()

        # 发布解算位姿至控制脚本端 (PUB -> SUB 端口 5556)
        self.pose_pub = self.context.socket(zmq.PUB)
        self.pose_pub.connect(f"tcp://{zmq_target}:5556")
        print(f"[VisionNode] 解算位姿发布信道已就绪 -> tcp://{zmq_target}:5556")

        # 根据不同模式初始化相机源
        if self.mode in ["sim", "stream"]:
            cam_addr = f"{zmq_target}:5555" if self.mode == "sim" else f"{cam_ip}:5555"
            self.img_sub = self.context.socket(zmq.SUB)
            self.img_sub.setsockopt(zmq.RCVHWM, 2)
            self.img_sub.connect(f"tcp://{cam_addr}")
            self.img_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            mode_name = "MuJoCo仿真相机模式" if self.mode == "sim" else "实机网络推流模式"
            print(f"[VisionNode] 当前为【{mode_name}】，监听 tcp://{cam_addr} ...")
            if self.mode == "sim":
                f_sim = 240.0 / np.tan(np.deg2rad(self.fovy / 2.0))
                self.camera_matrix = np.array([
                    [f_sim, 0.0, 320.0],
                    [0.0, f_sim, 240.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            else: # stream 模式下填入实机实测 RealSense (1280x720) 内参
                self.camera_matrix = np.array([
                    [913.58, 0.0, 646.4236],
                    [0.0, 914.01, 382.7312],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        elif self.mode == "realsense":
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            self.pipeline.start(self.rs_config)
            print("[VisionNode] 当前为【Intel RealSense物理相机模式】，硬件驱动已就绪 (帧率: 60Hz)")
            
            profile = self.pipeline.get_active_profile()
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            self.camera_matrix = np.array([
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            self.dist_coeffs = np.array(intr.coeffs, dtype=np.float32)

        else: # usb
            self.cap = cv2.VideoCapture(cam_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            print(f"[VisionNode] 当前为【标准USB摄像头模式】 (设备ID={cam_id}, 帧率=60Hz)")
            self.camera_matrix = np.array([
                [610.0, 0.0, 320.0],
                [0.0, 610.0, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        self.detector = Detector(families='tag36h11')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.aruco_detector = None

    def rot2quat_wxyz(self, R):
        r = Rotation.from_matrix(R)
        q_xyzw = r.as_quat()
        return float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])

    def detect_tags(self, gray):
        if hasattr(self, 'detector'):
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def rvec_to_quat(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        r = Rotation.from_matrix(R)
        q_xyzw = r.as_quat()
        return q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2] # w, x, y, z

    def get_box_3d_points(self, tag_id):
        hx, hy, hz = self.hx, self.hy, self.hz
        s = self.tag_size / 2.0
        local_corners = np.array([[s,-s,0], [-s,-s,0], [-s,s,0], [s,s,0]], dtype=np.float32)
        
        if self.tag_layout == "4tag":
            dx, dy = hx - s, hy - s
            if tag_id == 1:
                pos = np.array([dx, dy, hz + 0.0005])
                R = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
            elif tag_id == 2:
                pos = np.array([-dx, dy, hz + 0.0005])
                R = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
            elif tag_id == 3:
                pos = np.array([-dx, -dy, hz + 0.0005])
                R = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
            elif tag_id == 4:
                pos = np.array([dx, -dy, hz + 0.0005])
                R = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
            else:
                return None
            return (R @ local_corners.T).T + pos
        else:
            if tag_id in [0, 582]:
                pos = np.array([0, 0, hz + 0.0005])
                R = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
            elif tag_id == 2:
                pos = np.array([hx + 0.0005, 0, 0])
                R = Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
            elif tag_id == 3:
                pos = np.array([-hx - 0.0005, 0, 0])
                R = Rotation.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix()
            elif tag_id == 4:
                pos = np.array([0, hy + 0.0005, 0])
                R = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
            elif tag_id == 5:
                pos = np.array([0, -hy - 0.0005, 0])
                R = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()
            else:
                return None
            return (R @ local_corners.T).T + pos

    def check_reprojection_error(self, obj_points, img_points, rvec, tvec):
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        errors = np.linalg.norm(projected.reshape(-1, 2) - img_points.reshape(-1, 2), axis=1)
        mean_err = np.mean(errors)
        return mean_err, mean_err < 25.0

    def get_frame(self):
        if self.mode in ["sim", "stream"]:
            latest_md = None
            latest_msg = None
            while True:
                try:
                    md = self.img_sub.recv_json(flags=zmq.NOBLOCK)
                    msg = self.img_sub.recv(flags=zmq.NOBLOCK)
                    latest_md = md
                    latest_msg = msg
                except zmq.Again:
                    break
            if latest_md is not None and latest_msg is not None:
                buf = np.frombuffer(latest_msg, dtype=latest_md['dtype'])
                return buf.reshape(latest_md['shape'])
            return None
        elif self.mode == "realsense":
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            return np.asanyarray(color_frame.get_data()) if color_frame else None
        else: # usb
            ret, frame = self.cap.read()
            return frame if ret else None

    def run(self):
        print("\n<<< 视觉解算循环启动 (目标频率 >= 50Hz) >>>")
        if self.show_img:
            cv2.namedWindow(f"SUGAR Vision View ({self.mode.upper()})", cv2.WINDOW_AUTOSIZE)

        fps = 0.0
        last_warn_time = 0
        while True:
            t0 = time.time()
            img = self.get_frame()
            if img is None:
                if time.time() - last_warn_time > 3.0:
                    hint = "请确保实机 stream_cam.py 已运行并推流" if self.mode == "stream" else "请确保已启动: python deploy_omnicontact/deploy_omnicontact.py --use-vision"
                    print(f"\r[VisionNode] 等待接收图像数据中 (源={self.mode}, 端口 5555)... {hint}   ", end="", flush=True)
                    last_warn_time = time.time()
                if self.show_img:
                    key = cv2.waitKey(10)
                    if key & 0xFF == ord('q'):
                        break
                time.sleep(0.01)
                continue

            # MuJoCo 仿真传来的是 RGB，转为 BGR 方便 OpenCV 处理与显示
            if self.mode == "sim":
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
                if self.mode == "stream" and not hasattr(self, '_stream_init_w'):
                    h, w = img_bgr.shape[:2]
                    scale_x, scale_y = w / 1280.0, h / 720.0
                    self.camera_matrix[0, 0] = 911.4385 * scale_x
                    self.camera_matrix[1, 1] = 912.9034 * scale_y
                    self.camera_matrix[0, 2] = 646.4236 * scale_x
                    self.camera_matrix[1, 2] = 382.7312 * scale_y
                    self._stream_init_w = w
                    print(f"\n[VisionNode] 成功接收 ZMQ 画面！当前推流分辨率: [{w}x{h}]，自适应精准内参: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}\n")


            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            cam_params = [float(self.camera_matrix[0,0]), float(self.camera_matrix[1,1]), float(self.camera_matrix[0,2]), float(self.camera_matrix[1,2])]
            eff_tag_size = self.tag_size * 0.8 if self.tag_layout == "4tag" else self.tag_size
            detections = list(self.detector.detect(gray, estimate_tag_pose=True, camera_params=cam_params, tag_size=eff_tag_size))

            # 兼容 ArUco (如 Original ArUco 生成的 ID 582) 检测与解算
            if hasattr(self, 'aruco_detector') and self.aruco_detector is not None:
                aruco_corners, aruco_ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                aruco_corners, aruco_ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if aruco_ids is not None and len(aruco_ids) > 0:
                from types import SimpleNamespace
                s = self.tag_size / 2.0
                obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)
                for i in range(len(aruco_ids)):
                    t_id = int(aruco_ids[i][0])
                    if any(det.tag_id == t_id for det in detections):
                        continue
                    pts = aruco_corners[i][0].astype(np.float32)
                    success, rvec, tvec = cv2.solvePnP(obj_pts, pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if success:
                        R, _ = cv2.Rodrigues(rvec)
                        detections.append(SimpleNamespace(tag_id=t_id, corners=pts, pose_t=tvec, pose_R=R))
            
            poses = {}
            best_box_pose = None
            img_annotated = img_bgr.copy()

            T_cv2mj = np.array([
                [0.0,  0.0,  1.0],
                [0.0, -1.0,  0.0],
                [1.0,  0.0,  0.0]
            ], dtype=np.float32)

            T_tag2obj = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]
            ], dtype=np.float32)

            box_pos_list = []
            box_rot_list = []

            if self.tag_layout == "4tag":
                dx = self.hx - self.tag_size / 2.0
                dy = self.hy - self.tag_size / 2.0
                dz = self.hz + 0.0005
                offsets_4tag = {
                    1: np.array([-dx, dy, dz], dtype=np.float32),
                    2: np.array([dx, dy, dz], dtype=np.float32),
                    3: np.array([dx, -dy, dz], dtype=np.float32),
                    4: np.array([-dx, -dy, dz], dtype=np.float32),
                }

            for det in detections:
                tag_id = det.tag_id
                if self.show_img:
                    pts_2d = det.corners.astype(int)
                    cv2.polylines(img_annotated, [pts_2d], True, (0, 255, 0), 2)
                    cv2.putText(img_annotated, f"AprilTag ID: {tag_id}", tuple(pts_2d[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if tag_id == 5:
                    pos_cv = det.pose_t.flatten()
                    rot_cv = det.pose_R
                    rot_mj = T_cv2mj @ rot_cv @ np.linalg.inv(T_tag2obj)
                    offset_goal = np.array([0.0, 0.0, -(self.hz - 0.0055)], dtype=np.float32)
                    pos_cv_goal = pos_cv + rot_cv @ offset_goal
                    goal_center_mj = T_cv2mj @ pos_cv_goal
                    w, x, y, z = self.rot2quat_wxyz(rot_mj)
                    poses["goal"] = {
                        "pos": [float(goal_center_mj[0]), float(goal_center_mj[1]), float(goal_center_mj[2])],
                        "quat": [w, x, y, z]
                    }
                elif self.tag_layout == "4tag":
                    if tag_id in [1, 2, 3, 4]:
                        pos_cv = det.pose_t.flatten()
                        rot_cv = det.pose_R
                        rot_mj = T_cv2mj @ rot_cv @ np.linalg.inv(T_tag2obj)
                        offset_tag = offsets_4tag[tag_id]
                        pos_cv_box = pos_cv + rot_cv @ offset_tag
                        box_center_mj = T_cv2mj @ pos_cv_box
                        box_pos_list.append(box_center_mj)
                        box_rot_list.append(rot_mj)
                else:
                    if tag_id in [0, 582, 2, 3, 4]:
                        pos_cv = det.pose_t.flatten()
                        rot_cv = det.pose_R
                        rot_mj = T_cv2mj @ rot_cv @ np.linalg.inv(T_tag2obj)
                        if tag_id in [0, 582]:
                            offset_tag = np.array([0, 0, -(self.hz + 0.0005)], dtype=np.float32)
                        elif tag_id in [2, 3]:
                            offset_tag = np.array([0, 0, -(self.hx + 0.0005)], dtype=np.float32)
                        else: # tag_id == 4
                            offset_tag = np.array([0, 0, -(self.hy + 0.0005)], dtype=np.float32)
                        pos_cv_box = pos_cv + rot_cv @ offset_tag
                        box_center_mj = T_cv2mj @ pos_cv_box
                        w, x, y, z = self.rot2quat_wxyz(rot_mj)
                        best_box_pose = {
                            "pos": [float(box_center_mj[0]), float(box_center_mj[1]), float(box_center_mj[2])],
                            "quat": [w, x, y, z]
                        }
                    elif tag_id == 1:
                        pos_cv = det.pose_t.flatten()
                        rot_cv = det.pose_R
                        pos_mj = T_cv2mj @ pos_cv
                        rot_mj = T_cv2mj @ rot_cv
                        w, x, y, z = self.rot2quat_wxyz(rot_mj)
                        poses["tag_1"] = {
                            "pos": [float(pos_mj[0]), float(pos_mj[1]), float(pos_mj[2])],
                            "quat": [w, x, y, z]
                        }

            if self.tag_layout == "4tag" and len(box_pos_list) > 0:
                avg_pos = np.mean(box_pos_list, axis=0)
                avg_rot = np.mean(box_rot_list, axis=0)
                U, _, Vt = np.linalg.svd(avg_rot)
                best_rot_matrix = U @ Vt
                if np.linalg.det(best_rot_matrix) < 0:
                    U[:, -1] *= -1
                    best_rot_matrix = U @ Vt
                w, x, y, z = self.rot2quat_wxyz(best_rot_matrix)
                best_box_pose = {
                    "pos": [float(avg_pos[0]), float(avg_pos[1]), float(avg_pos[2])],
                    "quat": [w, x, y, z]
                }

            if best_box_pose is not None:
                poses["box"] = best_box_pose

            dt = time.time() - t0
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = cur_fps if fps == 0 else 0.95 * fps + 0.05 * cur_fps

            if poses:
                self.pose_pub.send_json(poses)
                if best_box_pose is not None:
                    pos_str = [round(p, 4) for p in best_box_pose['pos']]
                    P_cam_ref = np.array([0.0684, 0.0175, 1.35])
                    R_cam_ref = np.array([[0.6743, 0.7385, 0.0], [0.0, 0.0, -1.0], [-0.7385, 0.6743, 0.0]])
                    w_pos = [round(v, 4) for v in (P_cam_ref + R_cam_ref @ np.array(best_box_pose['pos']))]
                    print(f"\r[VisionNode] ({fps:.5f}Hz) 相机位姿:{pos_str} | 相对机器人底座:{w_pos}   ", end="", flush=True)

            if self.show_img:
                cv2.imshow(f"SUGAR Vision View ({self.mode.upper()})", img_annotated)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="切换至 MuJoCo 仿真深度相机源")
    parser.add_argument("--realsense", action="store_true", help="切换至本机的 Intel RealSense 物理相机源")
    parser.add_argument("--stream", action="store_true", help="切换至网络推流接收模式 (通过 ZMQ 端口 5555 接收机器端发回的图像帧)")
    parser.add_argument("--cam_id", type=int, default=0, help="标准USB摄像头设备ID")
    parser.add_argument("--zmq_target", type=str, default="127.0.0.1", help="接收解算结果控制端PC的IP (默认 127.0.0.1)")
    parser.add_argument("--cam_ip", type=str, default="192.168.123.164", help="网络推流端机器人的IP地址 (默认 192.168.123.164)")
    parser.add_argument("--box_dims", type=float, nargs=3, default=list(_CFG_DIMS), help="箱子半尺寸 hx hy hz")
    parser.add_argument("--tag_size", type=float, default=_CFG_TAG, help="AprilTag 有效检测边界尺寸(米)")
    parser.add_argument("--fovy", type=float, default=58.76, help="MuJoCo 相机垂直视场角 fovy (度)")
    parser.add_argument("--tag_layout", type=str, default=_CFG_LAYOUT, choices=["1tag", "4tag"], help="标签布局方案：1tag为单标签方案，4tag为顶部四角4标签方案")
    parser.add_argument("--no_show", action="store_true", help="无头模式（关闭预览画面）")
    args = parser.parse_args()

    mode = "sim" if args.sim else ("stream" if args.stream else ("realsense" if args.realsense else "usb"))
    node = MultiSourceVisionNode(mode=mode, cam_id=args.cam_id, zmq_target=args.zmq_target, cam_ip=args.cam_ip, show_img=not args.no_show, box_dims=args.box_dims, tag_size=args.tag_size, fovy=args.fovy, tag_layout=args.tag_layout)
    node.run()
