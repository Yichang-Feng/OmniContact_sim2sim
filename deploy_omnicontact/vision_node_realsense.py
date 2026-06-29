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

class MultiSourceVisionNode:
    def __init__(self, mode="usb", cam_id=0, zmq_target="127.0.0.1", show_img=True, box_dims=(0.125, 0.185, 0.215), tag_size=0.122):
        self.mode = mode
        self.show_img = show_img
        self.tag_size = float(tag_size)
        self.hx, self.hy, self.hz = float(box_dims[0]), float(box_dims[1]), float(box_dims[2])
        self.context = zmq.Context()

        # 发布解算位姿至控制脚本端 (PUB -> SUB 端口 5556)
        self.pose_pub = self.context.socket(zmq.PUB)
        self.pose_pub.connect(f"tcp://{zmq_target}:5556")
        print(f"[VisionNode] 解算位姿发布信道已就绪 -> tcp://{zmq_target}:5556")

        # 根据不同模式初始化相机源
        if self.mode == "sim":
            self.img_sub = self.context.socket(zmq.SUB)
            self.img_sub.connect(f"tcp://{zmq_target}:5555")
            self.img_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            print(f"[VisionNode] 当前为【MuJoCo仿真相机模式】，监听 tcp://{zmq_target}:5555 ...")
            self.camera_matrix = np.array([
                [426.5, 0.0, 320.0],
                [0.0, 426.5, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        elif self.mode == "realsense":
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            self.pipeline.start(self.rs_config)
            print("[VisionNode] 当前为【Intel RealSense物理相机模式】，硬件驱动已就绪 (目标帧率: 60Hz)！")
            
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
            print(f"[VisionNode] 当前为【标准USB摄像头模式】 (设备ID={cam_id}, 目标帧率=60Hz)")
            self.camera_matrix = np.array([
                [610.0, 0.0, 320.0],
                [0.0, 610.0, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        try:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

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
        
        if tag_id == 0:
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
        if self.mode == "sim":
            try:
                md = self.img_sub.recv_json(flags=zmq.NOBLOCK)
                msg = self.img_sub.recv(flags=zmq.NOBLOCK)
                buf = np.frombuffer(msg, dtype=md['dtype'])
                return buf.reshape(md['shape'])
            except zmq.Again:
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
                    print(f"\r[VisionNode] 等待接收图像数据中 (端口 5555)... 请确保已在另一个终端启动: python deploy_omnicontact/deploy_omnicontact.py --task carrybox --use-vision   ", end="", flush=True)
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

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            corners, ids = self.detect_tags(gray)
            
            poses = {}
            min_reproj_error = float('inf')
            best_box_pose = None

            img_annotated = img_bgr.copy()

            if ids is not None:
                for i, tag_id in enumerate(ids.flatten()):
                    if self.show_img:
                        pts_2d = corners[i][0].astype(int)
                        cv2.polylines(img_annotated, [pts_2d], True, (0, 255, 0), 2)
                        cv2.putText(img_annotated, f"AprilTag ID: {tag_id}", tuple(pts_2d[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if tag_id in [0, 2, 3, 4, 5]:
                        pts_3d = self.get_box_3d_points(tag_id)
                        if pts_3d is not None:
                            success, rvec, tvec = cv2.solvePnP(pts_3d, corners[i][0], self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                            if success:
                                err, valid = self.check_reprojection_error(pts_3d, corners[i][0], rvec, tvec)
                                if valid and err < min_reproj_error:
                                    min_reproj_error = err
                                    w, x, y, z = self.rvec_to_quat(rvec)
                                    best_box_pose = {
                                        "pos": [float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])],
                                        "quat": [float(w), float(x), float(y), float(z)]
                                    }
                                    if self.show_img:
                                        cv2.drawFrameAxes(img_annotated, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                    elif tag_id == 1:
                        obj_pts_target = np.array([
                            [self.tag_size/2, -self.tag_size/2, 0], [-self.tag_size/2, -self.tag_size/2, 0], 
                            [-self.tag_size/2, self.tag_size/2, 0], [self.tag_size/2, self.tag_size/2, 0]
                        ], dtype=np.float32)
                        success, rvec, tvec = cv2.solvePnP(obj_pts_target, corners[i][0], self.camera_matrix, self.dist_coeffs)
                        if success:
                            err, valid = self.check_reprojection_error(obj_pts_target, corners[i][0], rvec, tvec)
                            if valid:
                                w, x, y, z = self.rvec_to_quat(rvec)
                                poses["tag_1"] = {"pos": [float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])], "quat": [float(w), float(x), float(y), float(z)]}

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
                    print(f"\r[VisionNode] ({fps:.1f}Hz) 检测到 AprilTag -> 发布位姿 pos={pos_str}   ", end="", flush=True)

            if self.show_img:
                cv2.imshow(f"SUGAR Vision View ({self.mode.upper()})", img_annotated)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="切换至 MuJoCo 仿真深度相机源")
    parser.add_argument("--realsense", action="store_true", help="切换至 Intel RealSense 物理相机源")
    parser.add_argument("--cam_id", type=int, default=0, help="标准USB摄像头设备ID")
    parser.add_argument("--zmq_target", type=str, default="127.0.0.1", help="部署端PC的IP")
    parser.add_argument("--box_dims", type=float, nargs=3, default=[0.125, 0.185, 0.215], help="箱子半尺寸 hx hy hz")
    parser.add_argument("--tag_size", type=float, default=0.122, help="AprilTag 有效检测边界尺寸(米)")
    parser.add_argument("--no_show", action="store_true", help="无头模式（关闭预览画面）")
    args = parser.parse_args()

    mode = "sim" if args.sim else ("realsense" if args.realsense else "usb")
    node = MultiSourceVisionNode(mode=mode, cam_id=args.cam_id, zmq_target=args.zmq_target, show_img=not args.no_show, box_dims=args.box_dims, tag_size=args.tag_size)
    node.run()
