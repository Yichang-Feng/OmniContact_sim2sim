import time
import threading
import os
import ctypes
import numpy as np

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, MotorCmd_
    from unitree_sdk2py.utils.crc import CRC
    SDK2_AVAILABLE = True
except ImportError:
    SDK2_AVAILABLE = False


class RealRobotInterfacePy:
    """
    Unitree G1/HG series hardware Ethernet DDS interface adapter.
    """
    def __init__(self, net_interface: str = "enx6c1ff724495a", num_joints: int = 29):
        if not SDK2_AVAILABLE:
            raise RuntimeError("unitree_sdk2py library not found. Please install unitree_sdk2py in conda environment.")
        
        self.num_joints = num_joints
        self.low_state = None
        self.odo_state = None
        self.sport_state = None
        self.last_state_time = 0.0
        self.crc_calculator = CRC()

        print(f"[RealRobotInterface] 初始化以太网通道网卡: {net_interface} ...")
        ChannelFactoryInitialize(0, net_interface)

        print("[RealRobotInterfacePy] ℹ️ 默认已通过手柄进入调试模式，直接启动底层 DDS 监听...")

        # ------------------------------------------------------------------------
        # 修复 Bug 2: 提前并在初始化阶段直接注册好 Subscriber 和 Publisher！
        # ------------------------------------------------------------------------
        print("[RealRobotInterface] 正在注册 DDS 底层通信双向端点 (rt/lowstate & rt/lowcmd)...")
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._state_handler, 10)

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        try:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import OdoState_
            self.odo_sub = ChannelSubscriber("rt/odostate", OdoState_)
            self.odo_sub.Init(self._odo_handler, 10)
        except Exception:
            pass
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            self.sport_sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
            self.sport_sub.Init(self._sport_handler, 10)
        except Exception:
            pass

        self.cmd_lock = threading.Lock()
        self.current_cmd = self._create_passive_cmd()
        self.cmd_initialized = False
        # 严格对齐 Sonic C++ (g1_deploy_onnx_ref.cpp) 与官方底层例程：
        # 在发送任何实际控制指令前，严禁往 rt/lowcmd 发送全零/空指令，仅静默监听底包
        self.running = True
        self.send_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.send_thread.start()

        print("[RealRobotInterface] DDS 底层双向通信端点与心跳线程初始化完成！正在同步等待接收底层反馈包...")

    def _create_passive_cmd(self):
        motor_cmds = []
        for i in range(35):
            mc = MotorCmd_(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=0)
            motor_cmds.append(mc)
        cmd = LowCmd_(
            mode_pr=0,
            mode_machine=0,
            motor_cmd=motor_cmds,
            reserve=[0, 0, 0, 0],
            crc=0
        )
        cmd.crc = self.crc_calculator.Crc(cmd)
        return cmd

    def _heartbeat_loop(self):
        while self.running:
            with self.cmd_lock:
                cmd_to_send = self.current_cmd
            if self.pub is not None and cmd_to_send is not None and self.cmd_initialized:
                # 必须动态同步当前的 mode_machine
                if self.low_state is not None:
                    cmd_to_send.mode_machine = self.low_state.mode_machine
                    cmd_to_send.crc = self.crc_calculator.Crc(cmd_to_send)
                self.pub.Write(cmd_to_send)
            time.sleep(0.005)

    def _state_handler(self, msg: LowState_):
        self.low_state = msg
        self.last_state_time = time.time()

    def _odo_handler(self, msg):
        self.odo_state = msg

    def _sport_handler(self, msg):
        self.sport_state = msg

    def wait_for_connection(self, timeout=60.0):
        start = time.time()
        last_print = 0.0
        while time.time() - start < timeout:
            if self.low_state is not None:
                print("[RealRobotInterface] ✅ 成功接收到机器人底层数据帧 (rt/lowstate)！底层通信闭环就绪！")
                return True
            if time.time() - last_print > 2.0:
                print("[RealRobotInterface] 正在同步监听等待机器人底层反馈 (rt/lowstate)...", flush=True)
                last_print = time.time()
            time.sleep(0.05)
        print("[RealRobotInterface] ⚠️ 等待超时，尚未收到底包。")
        return False

    def get_robot_state(self):
        """
        Returns:
            q: np.ndarray (num_joints,)
            dq: np.ndarray (num_joints,)
            quat: np.ndarray (4,) wxyz format
            gyro: np.ndarray (3,)
            base_pos: np.ndarray (3,) world position
            lin_vel: np.ndarray (3,) world linear velocity
        """
        if self.low_state is None:
            return None

        ls = self.low_state
        q = np.array([m.q for m in ls.motor_state[:self.num_joints]], dtype=np.float32)
        dq = np.array([m.dq for m in ls.motor_state[:self.num_joints]], dtype=np.float32)
        
        # Unitree IMU quaternion is usually [w, x, y, z]
        raw_q = ls.imu_state.quaternion
        quat = np.array(raw_q, dtype=np.float32)
        gyro = np.array(ls.imu_state.gyroscope, dtype=np.float32)

        if self.odo_state is not None:
            base_pos = np.array(self.odo_state.position[:3], dtype=np.float32)
            lin_vel = np.array(self.odo_state.linear_velocity[:3], dtype=np.float32)
        elif self.sport_state is not None:
            base_pos = np.array(self.sport_state.position[:3], dtype=np.float32)
            lin_vel = np.array(self.sport_state.velocity[:3], dtype=np.float32)
        else:
            base_pos = np.zeros(3, dtype=np.float32)
            lin_vel = np.zeros(3, dtype=np.float32)

        return q, dq, quat, gyro, base_pos, lin_vel

    def send_joint_commands(self, target_q, kps, kds, target_dq=None, tau_ff=None):
        if target_dq is None:
            target_dq = np.zeros_like(target_q)
        if tau_ff is None:
            tau_ff = np.zeros_like(target_q)

        motor_cmds = []
        for i in range(35):
            if i < self.num_joints:
                mc = MotorCmd_(
                    mode=1,
                    q=float(target_q[i]),
                    dq=float(target_dq[i]),
                    tau=float(tau_ff[i]),
                    kp=float(kps[i]),
                    kd=float(kds[i]),
                    reserve=0
                )
            else:
                mc = MotorCmd_(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=0)
            motor_cmds.append(mc)

        cmd = LowCmd_(
            mode_pr=0,
            mode_machine=0,
            motor_cmd=motor_cmds,
            reserve=[0, 0, 0, 0],
            crc=0
        )
        cmd.crc = self.crc_calculator.Crc(cmd)
        with self.cmd_lock:
            self.current_cmd = cmd
            self.cmd_initialized = True

    def stop(self):
        """
        Gracefully stop sending commands and send a damping-only command (safe shutdown pose),
        matching Sonic's Stop() and CreateDampingCommand() (kp=0, kd=8.0).
        """
        print("[RealRobotInterface] 正在发送安全阻尼停止指令 (参考 Sonic CreateDampingCommand, kp=0, kd=8.0)...")
        motor_cmds = []
        for i in range(35):
            if i < self.num_joints:
                mc = MotorCmd_(mode=1, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=8.0, reserve=0)
            else:
                mc = MotorCmd_(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=0)
            motor_cmds.append(mc)
        cmd = LowCmd_(
            mode_pr=0,
            mode_machine=0,
            motor_cmd=motor_cmds,
            reserve=[0, 0, 0, 0],
            crc=0
        )
        cmd.crc = self.crc_calculator.Crc(cmd)
        with self.cmd_lock:
            self.current_cmd = cmd
        time.sleep(0.5)  # 持续发送阻尼指令 0.5 秒以确保底层接收
        self.running = False


class RealRobotInterfaceCpp:
    """
    Unitree G1/HG series hardware Ethernet DDS interface adapter (C++ Backend).
    Directly based on Sonic Whole-Body Control C++ SDK implementation.
    """
    def __init__(self, net_interface: str = "enx6c1ff724495a", num_joints: int = 29):
        self.num_joints = num_joints
        self.net_interface = net_interface
        so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "cpp", "build", "libreal_robot_interface_cpp.so"))
        if not os.path.exists(so_path):
            raise RuntimeError(f"C++ DDS dynamic library not found: {so_path}")

        print(f"[RealRobotInterfaceCpp] 加载 C++ 底层 DDS 引擎: {so_path}")
        self.lib = ctypes.CDLL(so_path)

        self.lib.init_real_robot_interface.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.init_real_robot_interface.restype = ctypes.c_void_p

        self.lib.wait_for_connection.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.lib.wait_for_connection.restype = ctypes.c_bool

        self.lib.get_robot_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), # q
            ctypes.POINTER(ctypes.c_float), # dq
            ctypes.POINTER(ctypes.c_float), # quat
            ctypes.POINTER(ctypes.c_float), # gyro
            ctypes.POINTER(ctypes.c_float), # pos
            ctypes.POINTER(ctypes.c_float), # vel
            ctypes.POINTER(ctypes.c_uint32) # mode_machine
        ]
        self.lib.get_robot_state.restype = ctypes.c_bool

        self.lib.send_joint_commands.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), # target_q
            ctypes.POINTER(ctypes.c_float), # kps
            ctypes.POINTER(ctypes.c_float), # kds
            ctypes.POINTER(ctypes.c_float), # target_dq
            ctypes.POINTER(ctypes.c_float)  # tau_ff
        ]
        self.lib.send_joint_commands.restype = None

        self.lib.stop_robot_interface.argtypes = [ctypes.c_void_p]
        self.lib.stop_robot_interface.restype = None

        net_if_bytes = self.net_interface.encode('utf-8')
        self.handle = self.lib.init_real_robot_interface(net_if_bytes, self.num_joints)
        if not self.handle:
            raise RuntimeError("创建 C++ DDS 通信句柄失败！")

        self.q_buf = (ctypes.c_float * self.num_joints)()
        self.dq_buf = (ctypes.c_float * self.num_joints)()
        self.quat_buf = (ctypes.c_float * 4)()
        self.gyro_buf = (ctypes.c_float * 3)()
        self.pos_buf = (ctypes.c_float * 3)()
        self.vel_buf = (ctypes.c_float * 3)()
        self.mode_machine_buf = ctypes.c_uint32()

    def wait_for_connection(self, timeout=60.0):
        if not self.handle:
            return False
        return self.lib.wait_for_connection(self.handle, float(timeout))

    def get_robot_state(self):
        if not self.handle:
            return None
        success = self.lib.get_robot_state(
            self.handle, self.q_buf, self.dq_buf, self.quat_buf, self.gyro_buf, self.pos_buf, self.vel_buf, ctypes.byref(self.mode_machine_buf)
        )
        if not success:
            return None
        q = np.ctypeslib.as_array(self.q_buf).copy()
        dq = np.ctypeslib.as_array(self.dq_buf).copy()
        quat = np.ctypeslib.as_array(self.quat_buf).copy()
        gyro = np.ctypeslib.as_array(self.gyro_buf).copy()
        base_pos = np.ctypeslib.as_array(self.pos_buf).copy()
        lin_vel = np.ctypeslib.as_array(self.vel_buf).copy()
        return q, dq, quat, gyro, base_pos, lin_vel

    def send_joint_commands(self, target_q, kps, kds, target_dq=None, tau_ff=None):
        if not self.handle:
            return
        if target_dq is None:
            target_dq = np.zeros_like(target_q)
        if tau_ff is None:
            tau_ff = np.zeros_like(target_q)

        target_q_arr = np.ascontiguousarray(target_q, dtype=np.float32)
        kps_arr = np.ascontiguousarray(kps, dtype=np.float32)
        kds_arr = np.ascontiguousarray(kds, dtype=np.float32)
        target_dq_arr = np.ascontiguousarray(target_dq, dtype=np.float32)
        tau_ff_arr = np.ascontiguousarray(tau_ff, dtype=np.float32)

        self.lib.send_joint_commands(
            self.handle,
            target_q_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kps_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kds_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            target_dq_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            tau_ff_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

    def stop(self):
        if self.handle and self.lib:
            self.lib.stop_robot_interface(self.handle)
            self.handle = None


try:
    so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "cpp", "build", "libreal_robot_interface_cpp.so"))
    if os.path.exists(so_path):
        RealRobotInterface = RealRobotInterfaceCpp
        print("[real_robot_interface] ✅ 已自动选用 C++ 底层 DDS 高性能驱动端点！")
    else:
        RealRobotInterface = RealRobotInterfacePy
        print("[real_robot_interface] ℹ️ 未发现 C++ 驱动库，降级采用 Python DDS 驱动端点。")
except Exception as e:
    RealRobotInterface = RealRobotInterfacePy
    print(f"[real_robot_interface] ℹ️ C++ 驱动加载出现异常: {e}，降级采用 Python DDS 驱动端点。")

