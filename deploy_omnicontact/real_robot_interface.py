import time
import numpy as np

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, MotorCmd_
    from unitree_sdk2py.utils.crc import CRC
    SDK2_AVAILABLE = True
except ImportError:
    SDK2_AVAILABLE = False


class RealRobotInterface:
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
        
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._state_handler, 10)

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

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        print("[RealRobotInterface] DDS订阅与发布通道初始化完成！等待底层反馈包...")

    def _state_handler(self, msg: LowState_):
        self.low_state = msg
        self.last_state_time = time.time()

    def _odo_handler(self, msg):
        self.odo_state = msg

    def _sport_handler(self, msg):
        self.sport_state = msg

    def wait_for_connection(self, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            if self.low_state is not None:
                print("[RealRobotInterface] ✅ 成功接收到机器人底层数据帧！")
                return True
            time.sleep(0.05)
        print("[RealRobotInterface] ⚠️ 等待超时，尚未收到底包，请检查物理连线或网卡IP段配置。")
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
        self.pub.Write(cmd)
