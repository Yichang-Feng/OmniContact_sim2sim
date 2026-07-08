#!/usr/bin/python3
"""
OmniContact ROS2 <-> UDP Bridge
--------------------------------
在系统原生 ROS2 环境下运行（无需新建 Conda 环境！直接使用系统 Python）：
  source /opt/ros/humble/setup.bash
  /usr/bin/python3 ros2_bridge.py

本脚本会将 ROS2 的视觉 AprilTag 位姿与雷达里程计实时通过本地 UDP 发送给 Conda (omnicontact) 环境中的 deploy 脚本！
  - 视觉位姿转发至 UDP 端口: 9876
  - 雷达里程计转发至 UDP 端口: 9877
"""

import socket
import json
import time
import argparse
import sys
import os

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
except ImportError:
    print("=" * 70)
    print("[ROS2 Bridge] ❌ 无法导入 rclpy！")
    print(f"当前使用的解释器路径: {sys.executable} (Python {sys.version.split()[0]})")
    if "conda" in sys.executable.lower() or "miniconda" in sys.executable.lower() or "anaconda" in sys.executable.lower():
        print("\n⚠️  原因分析: 你当前运行在 Conda 环境中，而 Ubuntu 系统级 ROS2 是为原生 /usr/bin/python3 (Python 3.10) 编译的！")
        print("💡 解决方法: 请使用绝对路径显式调用系统原生 Python 运行本脚本：")
        print("\n    source /opt/ros/humble/setup.bash")
        print("    /usr/bin/python3 ros2_bridge.py\n")
    else:
        print("\n💡 请确保在运行前执行了: source /opt/ros/humble/setup.bash")
    print("=" * 70)
    sys.exit(1)


class ROS2UDPBridge(Node):
    def __init__(self, target_ip="127.0.0.1", vision_port=9876, odom_port=9877, odom_topic="/lio/odom"):
        super().__init__("omnicontact_ros2_udp_bridge")
        self.target_ip = target_ip
        self.vision_addr = (target_ip, vision_port)
        self.odom_addr = (target_ip, odom_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 订阅里程计
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        
        # 订阅视觉位姿
        self.create_subscription(PoseStamped, "/aruco/box_pose", lambda msg: self.pose_callback(msg, "cam"), 10)
        self.create_subscription(PoseStamped, "/aruco/box_pose_pelvis", lambda msg: self.pose_callback(msg, "pelvis"), 10)
        self.create_subscription(PoseStamped, "/aruco/box_pose_torso_link", lambda msg: self.pose_callback(msg, "torso"), 10)
        
        print("=" * 60)
        print(" 🚀 OmniContact ROS2 <-> UDP 桥接服务启动成功！")
        print("=" * 60)
        print(f" 📡 目标主机 IP: {target_ip}")
        print(f" 👁️ 视觉 AprilTag 转发端口 : {vision_port} (/aruco/box_pose_*)")
        print(f" 🧭 雷达里程计转发端口   : {odom_port} ({odom_topic})")
        print("=" * 60)
        print("现在请在另一个终端中激活 (omnicontact) Conda 环境运行实机部署脚本！")

    def send_udp(self, data, addr):
        try:
            payload = json.dumps(data).encode('utf-8')
            self.sock.sendto(payload, addr)
        except Exception:
            pass

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        vel = msg.twist.twist.linear
        data = {
            "type": "odom",
            "pos": [pos.x, pos.y, pos.z],
            "quat": [q.w, q.x, q.y, q.z], # [w, x, y, z]
            "lin_vel": [vel.x, vel.y, vel.z],
            "time": time.time()
        }
        self.send_udp(data, self.odom_addr)

    def pose_callback(self, msg, frame_type):
        pos = msg.pose.position
        q = msg.pose.orientation
        data = {
            "type": "aruco",
            "frame": frame_type,
            "pos": [pos.x, pos.y, pos.z],
            "quat": [q.w, q.x, q.y, q.z], # [w, x, y, z]
            "time": time.time()
        }
        self.send_udp(data, self.vision_addr)


def main():
    parser = argparse.ArgumentParser(description="OmniContact ROS2 to UDP Bridge")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Target UDP IP (default: 127.0.0.1)")
    parser.add_argument("--vision-port", type=int, default=9876, help="Target UDP Vision Port (default: 9876)")
    parser.add_argument("--odom-port", type=int, default=9877, help="Target UDP Odometry Port (default: 9877)")
    parser.add_argument("--odom-topic", type=str, default="/lio/odom", help="Odometry topic name")
    args = parser.parse_args()

    rclpy.init()
    bridge = ROS2UDPBridge(target_ip=args.ip, vision_port=args.vision_port, odom_port=args.odom_port, odom_topic=args.odom_topic)
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        print("\n[ROS2 Bridge] 正在退出桥接服务...")
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
