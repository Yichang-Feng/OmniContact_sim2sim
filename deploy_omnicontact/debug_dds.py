#!/usr/bin/env python3
"""
DDS 底层诊断与调试脚本 (debug_dds.py)
用于全面追踪为什么 Python 版本的 cyclonedds 无法与宇树 G1 机器人建立 RTPS/SEDP 数据通道绑定。
"""

import os
import sys
import time
import ctypes
import ctypes.util
import threading

print("=" * 70)
print("🔍 [DDS Debug] 启动 Python 底层 DDS 深度诊断与日志捕获...")
print(f"📌 [Python Executable]: {sys.executable}")
print(f"📌 [Current Working Dir]: {os.getcwd()}")
print(f"📌 [LD_LIBRARY_PATH]: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
print("=" * 70)

# 1. 诊断 cyclonedds 依赖库路径
try:
    import cyclonedds
    print(f"✅ [cyclonedds import]: 成功，路径为: {cyclonedds.__file__}")
    from cyclonedds.internal import dds_c_t
    print(f"✅ [cyclonedds c-lib]: 加载成功")
except Exception as e:
    print(f"❌ [cyclonedds import 失败]: {e}")
    sys.exit(1)

# 2. 强行注入 CycloneDDS 深度追踪配置 (Verbosity=trace)
import unitree_sdk2py.core.channel_config as cconfig

# 覆盖原有的 XML 配置，开启最底层的 Trace 追踪和日志输出
cconfig.ChannelConfigHasInterface = '''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
    <Domain Id="any">
        <General>
            <Interfaces>
                <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
            </Interfaces>
        </General>
        <Tracing>
            <Verbosity>finest</Verbosity>
            <OutputFile>/tmp/cdds_debug_trace.log</OutputFile>
        </Tracing>
    </Domain>
</CycloneDDS>'''

print("✅ [XML Config]: 已注入 CycloneDDS 深度追踪配置 (输出路径: /tmp/cdds_debug_trace.log)")

# 3. 初始化通道工厂
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_, MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

net_if = "enx6c1ff724495a"
print(f"🚀 [ChannelInit]: 正在初始化网卡 {net_if} ...")
init_res = ChannelFactoryInitialize(0, net_if)
print(f"📋 [ChannelInit Result]: {init_res}")

# 4. 注册订阅者与发布者
lowstate_received = 0
sportstate_received = 0

def lowstate_handler(msg: LowState_):
    global lowstate_received
    lowstate_received += 1
    if lowstate_received % 100 == 1:
        print(f"⚡ [收到 rt/lowstate #{lowstate_received}] mode_machine={msg.mode_machine}, 电压={msg.power_v:.2f}V")

def sportstate_handler(msg: SportModeState_):
    global sportstate_received
    sportstate_received += 1
    if sportstate_received % 50 == 1:
        print(f"🏃 [收到 rt/sportmodestate #{sportstate_received}] mode={msg.mode}")

print("📡 [Subscribe]: 注册订阅器 rt/lowstate 和 rt/sportmodestate...")
sub_low = ChannelSubscriber("rt/lowstate", LowState_)
sub_low.Init(lowstate_handler, 10)

sub_sport = ChannelSubscriber("rt/sportmodestate", SportModeState_)
sub_sport.Init(sportstate_handler, 10)

print("📡 [Publish]: 注册底层发包器 rt/lowcmd 并尝试发包...")
pub_low = ChannelPublisher("rt/lowcmd", LowCmd_)
pub_low.Init()

# 创建被动心跳包
motor_cmds = [MotorCmd_(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=0) for _ in range(35)]
from unitree_sdk2py.utils.crc import CRC
crc_calc = CRC()
cmd = LowCmd_(mode_pr=0, mode_machine=0, motor_cmd=motor_cmds, reserve=[0,0,0,0], crc=0)
cmd.crc = crc_calc.Crc(cmd)

print("=" * 70)
print("⏳ [Debug Loop]: 开始 10 秒监控循环... (请观察是否有数据包，以及发包器是否发生匹配)")
print("=" * 70)

start_time = time.time()
send_count = 0
write_success_count = 0

while time.time() - start_time < 10.0:
    send_count += 1
    # 尝试发包 (由于 Write 会在 matched_count == 0 时默认等待，我们用非阻塞或超时验证)
    res = pub_low.Write(cmd, timeout=0.01)
    if res:
        write_success_count += 1
    
    if send_count % 50 == 0:
        elapsed = time.time() - start_time
        print(f"[Time: {elapsed:.1f}s] 尝试发包次数: {send_count}, 写入成功: {write_success_count} | 收到底包: {lowstate_received}, 收到高层包: {sportstate_received}")
    time.sleep(0.02)

print("=" * 70)
print("🏁 [Debug 结束] 总结统计:")
print(f"   - 尝试发送低层包总数: {send_count}")
print(f"   - 发包成功 (意味着底层 endpoints 成功匹配!): {write_success_count}")
print(f"   - 收到 rt/lowstate 总数: {lowstate_received}")
print(f"   - 收到 rt/sportmodestate 总数: {sportstate_received}")
print("=" * 70)
print("📢 [重要提示]: 底层 CycloneDDS 的深度协议栈追踪已保存至 /tmp/cdds_debug_trace.log")
print("   请在终端中使用以下命令查看为什么 endpoints 没有匹配:")
print("   grep -i -E 'match|error|reject|ignore|spdp|sedp|guid' /tmp/cdds_debug_trace.log | head -n 50")
print("=" * 70)
