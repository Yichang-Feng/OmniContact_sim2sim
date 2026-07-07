import ctypes
import time
import os
import numpy as np

def test_cpp_dds():
    print("======================================================================")
    print("🚀 [C++ DDS 测试] 正在验证基于 Sonic C++ 框架的底层 DDS 驱动...")
    print("======================================================================")

    so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "cpp", "build", "libreal_robot_interface_cpp.so"))
    if not os.path.exists(so_path):
        print(f"❌ 找不到动态链接库: {so_path}")
        return

    print(f"📦 [加载动态库]: {so_path}")
    lib = ctypes.CDLL(so_path)

    # 声明函数原型
    lib.init_real_robot_interface.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.init_real_robot_interface.restype = ctypes.c_void_p

    lib.wait_for_connection.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.wait_for_connection.restype = ctypes.c_bool

    lib.get_robot_state.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), # q
        ctypes.POINTER(ctypes.c_float), # dq
        ctypes.POINTER(ctypes.c_float), # quat
        ctypes.POINTER(ctypes.c_float), # gyro
        ctypes.POINTER(ctypes.c_float), # pos
        ctypes.POINTER(ctypes.c_float), # vel
        ctypes.POINTER(ctypes.c_uint32) # mode_machine
    ]
    lib.get_robot_state.restype = ctypes.c_bool

    lib.stop_robot_interface.argtypes = [ctypes.c_void_p]
    lib.stop_robot_interface.restype = None

    net_if = b"enx6c1ff724495a"
    num_joints = 29
    print(f"⚙️ [初始化通信端点]: 网卡={net_if.decode()}, 关节数={num_joints}")
    handle = lib.init_real_robot_interface(net_if, num_joints)
    if not handle:
        print("❌ 创建 C++ 通信句柄失败！")
        return

    try:
        print("⏳ [同步等待 rt/lowstate]: 正在监听等待机器人底包到来 (超时时间: 15秒)...")
        connected = lib.wait_for_connection(handle, 15.0)
        if not connected:
            print("❌ 未能在15秒内收到 rt/lowstate 数据包。请检查网络或机器人状态。")
            return

        print("✅ [连接成功] 成功闭环接收到 rt/lowstate！")
        print("📊 [数据采样采样中]: 连续读取3秒底层状态...")

        q_buf = (ctypes.c_float * num_joints)()
        dq_buf = (ctypes.c_float * num_joints)()
        quat_buf = (ctypes.c_float * 4)()
        gyro_buf = (ctypes.c_float * 3)()
        pos_buf = (ctypes.c_float * 3)()
        vel_buf = (ctypes.c_float * 3)()
        mode_machine_buf = ctypes.c_uint32()

        for i in range(3):
            time.sleep(1.0)
            success = lib.get_robot_state(
                handle, q_buf, dq_buf, quat_buf, gyro_buf, pos_buf, vel_buf, ctypes.byref(mode_machine_buf)
            )
            if success:
                q_arr = np.ctypeslib.as_array(q_buf)
                quat_arr = np.ctypeslib.as_array(quat_buf)
                print(f"   ⏱️ [第 {i+1} 秒] 机型代码(mode_machine): {mode_machine_buf.value} | "
                      f"关节0(左髋航向): {q_arr[0]:.4f} rad | "
                      f"IMU四元数[w,x,y,z]: [{quat_arr[0]:.3f}, {quat_arr[1]:.3f}, {quat_arr[2]:.3f}, {quat_arr[3]:.3f}]")
            else:
                print(f"   ⚠️ [第 {i+1} 秒] 读取状态失败")

    finally:
        print("🛑 [安全下线]: 正在停止 C++ DDS 引擎并释放资源...")
        lib.stop_robot_interface(handle)
        print("🏁 [测试完成] 所有相关资源已安全清理。")

if __name__ == "__main__":
    test_cpp_dds()
