#!/usr/bin/env python3
import time, zmq, signal, sys
import numpy as np
import pyrealsense2 as rs

# --- 1. 注册优雅退出信号监听（防 Ctrl+C 或 kill 导致内核驱动不释放） ---
running = True
def signal_handler(sig, frame):
    global running
    print("\n[机器人推流] ⚠️ 接收到退出信号，正在优雅关停推流并释放相机...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print("[机器人推流启动] 正在初始化专享 RGB 彩色单流...")

# --- 2. 硬件级自愈防护：在启动前执行一次底层软复位，清空上一任进程卡死的独占锁 ---
try:
    ctx_rs = rs.context()
    devices = ctx_rs.query_devices()
    if len(devices) > 0:
        dev = devices[0]
        print(f"[硬件自愈] 检测到相机 [{dev.get_info(rs.camera_info.name)}]，正在执行底层硬件复位以清空独占锁...")
        dev.hardware_reset()
        time.sleep(1.5)  # 等待 USB 重新枚举上线
    else:
        print("[硬件自愈] 未查询到 RealSense 设备，尝试直接启动 pipeline...")
except Exception as e:
    print(f"[硬件自愈] 复位检测跳过 (可忽略): {e}")

# --- 3. 初始化 ZMQ 与 RealSense Pipeline ---
ctx_zmq = zmq.Context()
pub = ctx_zmq.socket(zmq.PUB)
pub.bind('tcp://0.0.0.0:5555')

pipeline = rs.pipeline()
config = rs.config()
# 只启用 color 彩色单流，屏蔽掉沉重的深度流同步
config.enable_stream(rs.stream.color)

try:
    profile = pipeline.start(config)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    print(f"✅ 成功锁定纯 RGB 镜头！分辨率: [{intr.width}x{intr.height}] @ {color_profile.fps()}Hz | 内参 fx={intr.fx:.2f}, fy={intr.fy:.2f}")

    while running:
        # 增加 timeout 防止因为瞬时 USB 干扰导致 wait_for_frames 永久死锁
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            continue
            
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        f = np.asanyarray(color_frame.get_data())
        md = {'dtype': str(f.dtype), 'shape': f.shape}
        pub.send_json(md, zmq.SNDMORE)
        pub.send(f.tobytes())

except Exception as e:
    print(f"❌ 推流运行发生异常: {e}")
finally:
    # --- 4. 终极释放保障：无论任何情况退出，保证 100% 释放内核驱动与端口 ---
    print("[释放资源] 正在调用 pipeline.stop() 释放 RealSense 内核驱动...")
    try:
        pipeline.stop()
    except Exception as e:
        pass
    print("[释放资源] 正在关闭 ZMQ 端口与上下文...")
    pub.close()
    ctx_zmq.term()
    print("✅ 相机与端口已完美释放！别人现在可以直接无缝连接，无需重启机器人！")
    sys.exit(0)
