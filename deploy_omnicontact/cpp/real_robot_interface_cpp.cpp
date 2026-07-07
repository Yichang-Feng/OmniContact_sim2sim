#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include <array>
#include <vector>
#include <cstring>
#include <cmath>
#include <unistd.h>

// Unitree SDK2 C++
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string HG_CMD_TOPIC = "rt/lowcmd";

// CRC32 计算函数 (完全一致对照 Sonic C++ utils.hpp)
inline uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }
            if (data & xbit) CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

class RealRobotInterfaceCpp {
private:
    std::string net_if_;
    int num_joints_;
    uint8_t mode_machine_ = 0;
    
    std::shared_ptr<LowState_> latest_state_ = nullptr;
    std::mutex state_mutex_;

    LowCmd_ current_cmd_;
    std::mutex cmd_mutex_;

    ChannelSubscriberPtr<LowState_> sub_;
    ChannelPublisherPtr<LowCmd_> pub_;

    std::atomic<bool> running_{false};
    std::atomic<bool> cmd_initialized_{false};
    std::thread send_thread_;

public:
    RealRobotInterfaceCpp(const char* net_if, int num_joints) 
        : net_if_(net_if), num_joints_(num_joints) {
        
        std::cout << "[RealRobotInterfaceCpp] 初始化以太网通道网卡: " << net_if_ << " ..." << std::endl;
        ChannelFactory::Instance()->Init(0, net_if_);

        // 按照官方文档与用户实践：使用手柄 L2+R2 进入调试模式放权，无需以 B2 客户端强行释放
        std::cout << "[RealRobotInterfaceCpp] ℹ️ 默认已通过手柄进入调试模式，直接启动底层 DDS 监听..." << std::endl;

        // 初始化被动阻尼指令（安全兜底）
        {
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            current_cmd_.mode_pr() = 0;
            current_cmd_.mode_machine() = 0;
            for (int i = 0; i < 35; i++) {
                current_cmd_.motor_cmd()[i].mode() = 0;
                current_cmd_.motor_cmd()[i].q() = 0.0f;
                current_cmd_.motor_cmd()[i].dq() = 0.0f;
                current_cmd_.motor_cmd()[i].tau() = 0.0f;
                current_cmd_.motor_cmd()[i].kp() = 0.0f;
                current_cmd_.motor_cmd()[i].kd() = 0.0f;
            }
            current_cmd_.crc() = Crc32Core((uint32_t*)&current_cmd_, (sizeof(LowCmd_) >> 2) - 1);
        }

        std::cout << "[RealRobotInterfaceCpp] 正在初始化 C++ DDS 发布/订阅端点 (" << HG_STATE_TOPIC << " & " << HG_CMD_TOPIC << ")..." << std::endl;
        pub_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
        pub_->InitChannel();

        sub_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
        sub_->InitChannel(std::bind(&RealRobotInterfaceCpp::LowStateHandler, this, std::placeholders::_1), 1);

        // 启动 500Hz 实时心跳与发包线程
        running_ = true;
        send_thread_ = std::thread(&RealRobotInterfaceCpp::HeartbeatLoop, this);
        std::cout << "[RealRobotInterfaceCpp] C++ DDS 底层通信端点与 500Hz 心跳发包线程就绪！" << std::endl;
    }

    ~RealRobotInterfaceCpp() {
        stop();
    }

    void LowStateHandler(const void* message) {
        const LowState_* ls = (const LowState_*)message;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!latest_state_) {
                latest_state_ = std::make_shared<LowState_>(*ls);
            } else {
                *latest_state_ = *ls;
            }
        }
        if (mode_machine_ != ls->mode_machine()) {
            if (mode_machine_ == 0) {
                std::cout << "[RealRobotInterfaceCpp] ✅ 首次锁定机器人机型识别码 (mode_machine): " << unsigned(ls->mode_machine()) << std::endl;
            }
            mode_machine_ = ls->mode_machine();
        }
    }

    void HeartbeatLoop() {
        LowCmd_ cmd_to_send;
        while (running_) {
            bool can_send = false;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (latest_state_ != nullptr) {
                    can_send = true;
                }
            }
            if (can_send && cmd_initialized_ && pub_) {
                {
                    std::lock_guard<std::mutex> lock(cmd_mutex_);
                    cmd_to_send = current_cmd_;
                }
                if (mode_machine_ != 0) {
                    cmd_to_send.mode_machine() = mode_machine_;
                }
                cmd_to_send.crc() = Crc32Core((uint32_t*)&cmd_to_send, (sizeof(LowCmd_) >> 2) - 1);
                pub_->Write(cmd_to_send);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2)); // 500Hz
        }
    }

    bool wait_for_connection(float timeout_sec) {
        auto start_time = std::chrono::steady_clock::now();
        auto last_print_time = start_time;
        while (std::chrono::steady_clock::now() - start_time < std::chrono::duration<float>(timeout_sec)) {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (latest_state_ != nullptr) {
                    std::cout << "[RealRobotInterfaceCpp] ✅ 成功接收到 G1 底层数据帧 (rt/lowstate)！通信闭环完成！" << std::endl;
                    return true;
                }
            }
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count() >= 2) {
                std::cout << "[RealRobotInterfaceCpp] 正在同步监听等待 rt/lowstate 底层状态包到来..." << std::endl;
                last_print_time = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::cout << "[RealRobotInterfaceCpp] ⚠️ 等待底层 DDS 数据超时 (" << timeout_sec << "s)！" << std::endl;
        return false;
    }

    bool get_robot_state(float* q_out, float* dq_out, float* quat_out, float* gyro_out, float* pos_out, float* vel_out, uint32_t* mode_machine_out) {
        std::shared_ptr<LowState_> ls_copy = nullptr;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!latest_state_) return false;
            ls_copy = std::make_shared<LowState_>(*latest_state_);
        }

        for (int i = 0; i < num_joints_; i++) {
            q_out[i] = ls_copy->motor_state()[i].q();
            dq_out[i] = ls_copy->motor_state()[i].dq();
        }

        auto quat = ls_copy->imu_state().quaternion(); // [w, x, y, z]
        for (int i = 0; i < 4; i++) quat_out[i] = quat[i];

        auto gyro = ls_copy->imu_state().gyroscope(); // [x, y, z]
        for (int i = 0; i < 3; i++) gyro_out[i] = gyro[i];

        // 底层模式下默认物理世界基座坐标与线速度置为 0
        for (int i = 0; i < 3; i++) {
            pos_out[i] = 0.0f;
            vel_out[i] = 0.0f;
        }

        if (mode_machine_out) {
            *mode_machine_out = mode_machine_;
        }
        return true;
    }

    void send_joint_commands(const float* target_q, const float* kps, const float* kds, const float* target_dq, const float* tau_ff) {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        for (int i = 0; i < 35; i++) {
            if (i < num_joints_) {
                current_cmd_.motor_cmd()[i].mode() = 1; // Enable
                current_cmd_.motor_cmd()[i].q() = target_q[i];
                current_cmd_.motor_cmd()[i].dq() = target_dq ? target_dq[i] : 0.0f;
                current_cmd_.motor_cmd()[i].tau() = tau_ff ? tau_ff[i] : 0.0f;
                current_cmd_.motor_cmd()[i].kp() = kps[i];
                current_cmd_.motor_cmd()[i].kd() = kds[i];
            } else {
                current_cmd_.motor_cmd()[i].mode() = 0;
                current_cmd_.motor_cmd()[i].q() = 0.0f;
                current_cmd_.motor_cmd()[i].dq() = 0.0f;
                current_cmd_.motor_cmd()[i].tau() = 0.0f;
                current_cmd_.motor_cmd()[i].kp() = 0.0f;
                current_cmd_.motor_cmd()[i].kd() = 0.0f;
            }
        }
        cmd_initialized_ = true;
    }

    void stop() {
        if (!running_) return;
        if (cmd_initialized_) {
            std::cout << "[RealRobotInterfaceCpp] 正在发停机阻尼保护指令 (对照 Sonic CreateDampingCommand, kp=0, kd=8.0)..." << std::endl;
            {
                std::lock_guard<std::mutex> lock(cmd_mutex_);
                for (int i = 0; i < 35; i++) {
                    if (i < num_joints_) {
                        current_cmd_.motor_cmd()[i].mode() = 1;
                        current_cmd_.motor_cmd()[i].q() = 0.0f;
                        current_cmd_.motor_cmd()[i].dq() = 0.0f;
                        current_cmd_.motor_cmd()[i].tau() = 0.0f;
                        current_cmd_.motor_cmd()[i].kp() = 0.0f;
                        current_cmd_.motor_cmd()[i].kd() = 8.0f;
                    } else {
                        current_cmd_.motor_cmd()[i].mode() = 0;
                        current_cmd_.motor_cmd()[i].q() = 0.0f;
                        current_cmd_.motor_cmd()[i].dq() = 0.0f;
                        current_cmd_.motor_cmd()[i].tau() = 0.0f;
                        current_cmd_.motor_cmd()[i].kp() = 0.0f;
                        current_cmd_.motor_cmd()[i].kd() = 0.0f;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 持续 0.5s 阻尼
        } else {
            std::cout << "[RealRobotInterfaceCpp] ℹ️ 本次会话未发送过主动控制指令(纯读状态)，无需发送停机阻尼指令。" << std::endl;
        }
        running_ = false;
        if (send_thread_.joinable()) {
            send_thread_.join();
        }
        std::cout << "[RealRobotInterfaceCpp] DDS 心跳线程安全下线。" << std::endl;
    }
};

extern "C" {
    void* init_real_robot_interface(const char* net_interface, int num_joints) {
        return new RealRobotInterfaceCpp(net_interface, num_joints);
    }
    bool wait_for_connection(void* handle, float timeout_sec) {
        if (!handle) return false;
        return ((RealRobotInterfaceCpp*)handle)->wait_for_connection(timeout_sec);
    }
    bool get_robot_state(void* handle, float* q_out, float* dq_out, float* quat_out, float* gyro_out, float* pos_out, float* vel_out, uint32_t* mode_machine_out) {
        if (!handle) return false;
        return ((RealRobotInterfaceCpp*)handle)->get_robot_state(q_out, dq_out, quat_out, gyro_out, pos_out, vel_out, mode_machine_out);
    }
    void send_joint_commands(void* handle, const float* target_q, const float* kps, const float* kds, const float* target_dq, const float* tau_ff) {
        if (!handle) return;
        ((RealRobotInterfaceCpp*)handle)->send_joint_commands(target_q, kps, kds, target_dq, tau_ff);
    }
    void stop_robot_interface(void* handle) {
        if (!handle) return;
        ((RealRobotInterfaceCpp*)handle)->stop();
        delete (RealRobotInterfaceCpp*)handle;
    }
}
