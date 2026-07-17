import os
import sys
import time
import numpy as np
import mujoco

from common.path_config import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand
from FSM.FSMState import FSMStateName
from FSM.FSM import FSM

def pd_control(target_q, q, kp, target_dq, dq, kd, torque_limit_mj=None, torque_clip=True):
    tau = (target_q - q) * kp + (target_dq - dq) * kd
    if torque_clip and torque_limit_mj is not None:
        tau = np.clip(tau, -torque_limit_mj, torque_limit_mj)
    return tau

def run_comparison_test(enable_blend: bool):
    mode_name = "AFTER (有余弦缓动过渡与首帧静态广播)" if enable_blend else "BEFORE (无余弦过渡，原始阶跃突变与缓冲零填充)"
    print(f"\n=======================================================================================")
    print(f" 开始仿真运行：{mode_name}")
    print(f"=======================================================================================")

    # 初始化 MuJoCo 物理引擎
    xml_path = os.path.join(PROJECT_ROOT, "g1_description", "scene_29dof.xml")
    if not os.path.exists(xml_path):
        xml_path = os.path.join(PROJECT_ROOT, "g1_description", "g1_29dof.xml")
    
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    # 初始默认姿态站立
    state_cmd = StateAndCmd(29)
    policy_output = PolicyOutput(29)
    state_cmd.obj_pos = np.array([1.5, 0.0, 0.5], dtype=np.float32)
    state_cmd.obj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state_cmd.push_box_pos = np.array([1.5, 0.0, 0.5], dtype=np.float32)
    state_cmd.push_box_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state_cmd.carry_box_pos = np.array([1.5, 0.0, 0.5], dtype=np.float32)
    state_cmd.carry_box_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # 模拟传感器与状态反馈数据更新函数
    def update_state_cmd():
        state_cmd.q = d.qpos[7:7+29].astype(np.float32).copy()
        state_cmd.dq = d.qvel[6:6+29].astype(np.float32).copy()
        state_cmd.base_pos = d.qpos[0:3].astype(np.float32).copy()
        state_cmd.base_quat = d.qpos[3:7].astype(np.float32).copy()
        state_cmd.ang_vel = d.qvel[3:6].astype(np.float32).copy()
        state_cmd.gravity_ori = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # 初次同步数据
    update_state_cmd()

    # 初始化 FSM
    fsm = FSM(state_cmd, policy_output)
    fsm.omnicontact.enable_transition_blend = enable_blend
    torque_limit_mj = np.array([88, 88, 88, 139, 50, 50,
                                88, 88, 88, 139, 50, 50,
                                88, 50, 50,
                                25, 25, 25, 25, 5, 5, 5,
                                25, 25, 25, 25, 5, 5, 5], dtype=np.float32)

    # 如果有旧的对应日志文件，先清理确保干净重新输出
    log_file = os.path.join(PROJECT_ROOT, "transition_log_AFTER_with_blend.txt" if enable_blend else "transition_log_BEFORE_no_blend.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    # 初始化渲染器与跟踪相机 (渲染 GIF 动画序列供可视化验证)
    renderer = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera()
    if hasattr(mujoco, "mjv_defaultFreeCamera"):
        mujoco.mjv_defaultFreeCamera(m, cam)
    frames = []

    def capture_frame():
        try:
            cam.lookat[:] = d.qpos[0:3]
            cam.distance = 2.5
            cam.azimuth = 135.0
            cam.elevation = -15.0
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            from PIL import Image
            frames.append(Image.fromarray(img))
        except Exception:
            pass

    # Step 1: 先在 LocoMode 中运行 40 步 (0.8s)，模拟机器人正处于走路或站立的实际物理关节角
    fsm.cur_policy = fsm.loco_policy
    fsm.cur_policy.enter()
    for step in range(40):
        update_state_cmd()
        fsm.cur_policy.run()
        # 计算物理电机 PD 力矩并驱动仿真
        tau = pd_control(policy_output.actions, state_cmd.q, policy_output.kps, np.zeros_like(policy_output.kps), state_cmd.dq, policy_output.kds, torque_limit_mj=torque_limit_mj)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if step % 2 == 0:
            capture_frame()

    knee_at_switch = float(state_cmd.q[3])
    print(f"[{mode_name}] 经过 LocoMode 40步运行后，切入瞬间当前实际左膝角度 (q[3]) = {knee_at_switch:.4f} rad ({knee_at_switch*180/np.pi:.2f}°)")

    # Step 2: 触发切换到 OmniContact！
    fsm.cur_policy.exit()
    fsm.cur_policy = fsm.omnicontact
    fsm.cur_policy.enter()

    print(f"[{mode_name}] 成功切入 OmniContact，开启前 35 步逐帧记录与物理跟踪...")
    for step in range(35):
        update_state_cmd()
        fsm.cur_policy.run()
        # 计算物理电机 PD 力矩并驱动仿真
        tau = pd_control(policy_output.actions, state_cmd.q, policy_output.kps, np.zeros_like(policy_output.kps), state_cmd.dq, policy_output.kds, torque_limit_mj=torque_limit_mj)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        capture_frame()

    if len(frames) > 0:
        gif_path = os.path.join(PROJECT_ROOT, "transition_AFTER_with_blend.gif" if enable_blend else "transition_BEFORE_no_blend.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=25, loop=0)
        print(f"[{mode_name}] 可视化验证 GIF 视频已成功生成: {gif_path}")

    print(f"[{mode_name}] 仿真对比运行完成！详细记录已写入文件: {log_file}")

if __name__ == "__main__":
    print("正在执行 OmniContact 模式切换对比专项测试 (BEFORE vs AFTER)...")
    run_comparison_test(enable_blend=False)
    run_comparison_test(enable_blend=True)
    print("\n对比测试全部结束！可直接查看项目根目录下的 transition_log_BEFORE_no_blend.txt 与 transition_log_AFTER_with_blend.txt。")
