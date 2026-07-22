import numpy as np
import yaml
import onnxruntime
import os
import copy
import threading
import time
from types import SimpleNamespace

from common.path_config import PROJECT_ROOT
from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.mujoco_kinematics import MujocoKinematics
from common.utils import (
    FSMCommand,
    matrix_from_quat,
    quat_conjugate,
    quat_rotate_inverse,
    subtract_frame_transforms,
    yaw_quat,
    quat_apply_batch,
    quat_mul_left_batch,
    quat_to_6d_batch,
)
from policy.omnicontact.CFgen_meta1_loco import DEFAULT_PELVIS_Z
from policy.omnicontact.CFgen_reference import (
    commit_cfgen_reference,
    initialize_cfgen_reference,
    init_cfgen_state,
    plan_cfgen_reference,
    set_active_object_profile,
)
from policy.omnicontact.NPZmotion_reference import load_tracking_npz_reference


class OmniContactStandPrepareTest(FSMState):
    _TRACKING_DIM_PER_FRAME = 49  # LW(9) + RW(9) + Torso(9) + LA(9) + RA(9) + Contact(4)
    _HISTORY_SLICES = (
        (0, 15),
        (15, 18),
        (18, 21),
        (21, 50),
        (50, 79),
        (79, 108),
        (108, 111),
        (111, 117),
        (117, 141),
    )

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput, onnx_path: str | None = None):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_OmniContact
        self.name_str = "omnicontact_stand_prepare_test"
        self.counter_step = 0
        self.stage0_counter = 0
        self.future_frames = [0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50]
        self.history_len = 5
        self.max_rel_norm = 4.0
        self.bbox_scale = 0.3

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "OmniContact.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.onnx_path = onnx_path or os.path.join(current_dir, "model", config["onnx_path"])
            self.kps_lab = np.array(config["kp_lab"], dtype=np.float32)
            self.kds_lab = np.array(config["kd_lab"], dtype=np.float32)
            self.default_angles_lab = np.array(config["default_angles_lab"], dtype=np.float32)
            self.mj2lab = np.array(config["mj2lab"], dtype=np.int32)
            self.lab2mj = np.array(config["lab2mj"], dtype=np.int32)
            self.joint_pos_lowerlimit_lab = np.array(config["joint_pos_lowerlimit_lab"], dtype=np.float32)
            self.joint_pos_upperlimit_lab = np.array(config["joint_pos_upperlimit_lab"], dtype=np.float32)
            self.action_scale_lab = np.array(config["action_scale_lab"], dtype=np.float32)
            self.enable_transition_blend = config.get("enable_transition_blend", True)

            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 1
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path, sess_options)
            self.input_names = [inpt.name for inpt in self.ort_session.get_inputs()]
            self.kinematics = MujocoKinematics(
                xml_path=os.path.join(PROJECT_ROOT, "g1_description", "g1_29dof.xml")
            )

        config_num_obs = int(config.get("num_obs", 0))
        onnx_input_shape = self.ort_session.get_inputs()[0].shape
        self.num_obs = int(onnx_input_shape[-1])
        if config_num_obs and config_num_obs != self.num_obs:
            print(
                f"[{self.name_str}] Warning: yaml num_obs={config_num_obs} != onnx num_obs={self.num_obs}. Using ONNX dim."
            )

        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(config["num_actions"], dtype=np.float32)

        tracking_total_dim = len(self.future_frames) * self._TRACKING_DIM_PER_FRAME
        self.single_obs_dim = (self.num_obs - tracking_total_dim) // self.history_len
        self.obs_history_buffer = np.zeros((self.history_len, self.single_obs_dim), dtype=np.float32)

        self.bbox_offsets = (
            np.array(
                [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]],
                dtype=np.float32,
            )
            * 0.5
        )
        self.bbox_offsets_scaled = self.bbox_offsets * self.bbox_scale
        self.box_dims = np.array([0.3, 0.3, 0.3])
        self.ball_dims = np.array([0.10, 0.10, 0.10], dtype=np.float32)
        self.push_box_dims = np.array([0.23, 0.25, 0.26], dtype=np.float32)
        self.carry_box_dims = np.array([0.15, 0.15, 0.15], dtype=np.float32)
        self.stack_box_names = ("stack_box_large", "stack_box_mid", "stack_box_small")
        self.stack_box_dims = np.array(
            [[0.20, 0.20, 0.15], [0.15, 0.15, 0.15], [0.10, 0.10, 0.10]],
            dtype=np.float32,
        )
        self.stack_box_goal_pos = np.array(
            [[1.0, 0.0, 0.15], [1.0, 0.0, 0.45], [1.0, 0.0, 0.70]],
            dtype=np.float32,
        )
        self.reference_source = "CFgen"  # 'CFgen' or 'NPZmotion'
        self.task = "carrybox"
        self.active_object_name = "box"
        init_cfgen_state(self, pad=30)
        self.npz_dir = ""
        self.tracking_start_frame = 0
        self.tracking_end_frame = -1
        self.target_yaw = 0.0
        self.async_stage_thread = None
        self.async_stage_pending = None
        self.async_stage_error = None
        self.async_stage_lock = threading.Lock()
        self.async_stage_generation = 0

    def _set_active_object_profile(self, object_name: str, dims: np.ndarray) -> None:
        set_active_object_profile(self, object_name, dims)

    def _plan_cfgen_reference(self, fk_info) -> None:
        plan_cfgen_reference(self, fk_info)

    def _snapshot_state_cmd(self):
        src = self.state_cmd
        return SimpleNamespace(
            q=src.q.copy(),
            dq=src.dq.copy(),
            base_pos=src.base_pos.copy(),
            base_quat=src.base_quat.copy(),
            obj_pos=src.obj_pos.copy(),
            obj_quat=src.obj_quat.copy(),
            ball_pos=src.ball_pos.copy(),
            ball_quat=src.ball_quat.copy(),
            push_box_pos=src.push_box_pos.copy(),
            push_box_quat=src.push_box_quat.copy(),
            carry_box_pos=src.carry_box_pos.copy(),
            carry_box_quat=src.carry_box_quat.copy(),
            stack_box_pos=src.stack_box_pos.copy(),
            stack_box_quat=src.stack_box_quat.copy(),
        )

    def _snapshot_cfgen_policy(self, state_cmd):
        snapshot = SimpleNamespace(
            state_cmd=state_cmd,
            bbox_offsets=self.bbox_offsets.copy(),
            task=self.task,
            active_object_name=str(getattr(self, "active_object_name", "box")),
            box_dims=np.asarray(self.box_dims, dtype=np.float32).copy(),
            ball_dims=np.asarray(self.ball_dims, dtype=np.float32).copy(),
            push_box_dims=np.asarray(self.push_box_dims, dtype=np.float32).copy(),
            carry_box_dims=np.asarray(self.carry_box_dims, dtype=np.float32).copy(),
            stack_box_names=tuple(self.stack_box_names),
            stack_box_dims=np.asarray(self.stack_box_dims, dtype=np.float32).copy(),
            stack_box_goal_pos=np.asarray(self.stack_box_goal_pos, dtype=np.float32).copy(),
            goal_pos=np.asarray(self.goal_pos, dtype=np.float32).copy(),
            push_carry_stage=copy.deepcopy(getattr(self, "push_carry_stage", "idle")),
            push_relocate_stage=copy.deepcopy(getattr(self, "push_relocate_stage", "idle")),
            stackbox_stage_idx=int(getattr(self, "stackbox_stage_idx", 0)),
            stackbox_stage_count=int(getattr(self, "stackbox_stage_count", 3)),
            reference_source=getattr(self, "reference_source", "CFgen"),
        )
        snapshot.bbox_scale = snapshot.box_dims * 2.0
        snapshot.bbox_offsets_scaled = snapshot.bbox_offsets * snapshot.bbox_scale.reshape(1, 3)
        init_cfgen_state(snapshot, pad=30)
        snapshot.push_carry_stage = copy.deepcopy(getattr(self, "push_carry_stage", "idle"))
        snapshot.push_relocate_stage = copy.deepcopy(getattr(self, "push_relocate_stage", "idle"))
        snapshot.stackbox_stage_idx = int(getattr(self, "stackbox_stage_idx", 0))
        snapshot.stackbox_stage_count = int(getattr(self, "stackbox_stage_count", 3))
        return snapshot

    @staticmethod
    def _collect_async_stage_result(policy, elapsed: float) -> dict:
        attrs = (
            "target_yaw",
            "goal_pos",
            "active_object_name",
            "box_dims",
            "bbox_scale",
            "bbox_offsets_scaled",
            "push_carry_stage",
            "push_relocate_stage",
            "stackbox_stage_idx",
            "stackbox_stage_count",
            "traj_generator",
            "ref_left_wrist_pos",
            "ref_left_wrist_quat",
            "ref_right_wrist_pos",
            "ref_right_wrist_quat",
            "ref_object_pos",
            "ref_object_quat",
            "ref_contact",
            "ref_phase",
            "ref_torso_future_pos",
            "ref_torso_future_quat",
            "ref_left_ankle_future_pos",
            "ref_left_ankle_future_quat",
            "ref_right_ankle_future_pos",
            "ref_right_ankle_future_quat",
            "dof_pos",
            "ref_base_pos",
            "ref_base_quat",
        )
        return {
            "attrs": {name: copy.deepcopy(getattr(policy, name)) for name in attrs if hasattr(policy, name)},
            "elapsed": float(elapsed),
        }

    def _async_stage_worker(self, generation: int, policy_snapshot, fk_info):
        start = time.perf_counter()
        try:
            plan_cfgen_reference(policy_snapshot, fk_info)
            result = self._collect_async_stage_result(policy_snapshot, time.perf_counter() - start)
            with self.async_stage_lock:
                if generation != self.async_stage_generation:
                    return
                self.async_stage_pending = result
                self.async_stage_error = None
        except Exception as exc:
            with self.async_stage_lock:
                if generation != self.async_stage_generation:
                    return
                self.async_stage_pending = None
                self.async_stage_error = exc

    def _start_async_stage_plan(self, next_stage_update) -> None:
        state_snapshot = self._snapshot_state_cmd()
        policy_snapshot = self._snapshot_cfgen_policy(state_snapshot)
        next_stage_update(policy_snapshot)
        fk_info = self.kinematics.forward(state_snapshot.q, state_snapshot.base_pos, state_snapshot.base_quat)
        with self.async_stage_lock:
            self.async_stage_generation += 1
            generation = self.async_stage_generation
            self.async_stage_pending = None
            self.async_stage_error = None
        thread = threading.Thread(
            target=self._async_stage_worker,
            args=(generation, policy_snapshot, fk_info),
            daemon=True,
        )
        with self.async_stage_lock:
            self.async_stage_thread = thread
        thread.start()
        print(f"[{self.name_str}] async stage plan requested.")

    def _async_stage_running(self) -> bool:
        thread = self.async_stage_thread
        return thread is not None and thread.is_alive()

    def _apply_async_stage_plan(self) -> bool:
        with self.async_stage_lock:
            result = self.async_stage_pending
            error = self.async_stage_error
            if result is None and error is None:
                return False
            self.async_stage_pending = None
            self.async_stage_error = None
            self.async_stage_thread = None

        if error is not None:
            print(f"[{self.name_str}] async stage plan failed: {error}")
            self.switch_to_loco = True
            self.policy_output.switch_to_loco = True
            self.success = "failure"
            self.policy_output.success = self.success
            return True

        for name, value in result["attrs"].items():
            setattr(self, name, value)
        self.counter_step = 0
        self.success = ""
        self.switch_to_loco = False
        self.policy_output.success = self.success
        self.policy_output.switch_to_loco = False
        print(f"[{self.name_str}] async stage plan applied | elapsed={result['elapsed'] * 1000.0:.1f} ms")
        return True

    def enter(self):
        self.counter_step = 0
        self.stage0_counter = 0
        self.success = ""
        self.switch_to_loco = False
        goal_override = getattr(self, "goal_pos_override", None)
        if goal_override is None:
            self.goal_pos = np.array([5.0, 0.0, 0.55], dtype=np.float32)
        else:
            self.goal_pos = np.asarray(goal_override, dtype=np.float32).reshape(3).copy()
        if self.task in {"pushbox-two", "pushbox-in", "slidebox", "slidebox-left", "slidebox-right", "kickball", "push-carry", "carry-push", "push-relocate"}:
            self.goal_pos[2] = float(self.box_dims[2])
        elif self.task == "loco":
            self.goal_pos[2] = float(DEFAULT_PELVIS_Z)
        elif self.task in {"stackbox", "carry-carry", "carry-carry-carry"}:
            self.stackbox_stage_idx = 0
            self.stackbox_stage_count = 2 if self.task == "carry-carry" else 3
            self.goal_pos = self.stack_box_goal_pos[0].copy()
        self.action = np.zeros(29, dtype=np.float32)
        self.obs_history_buffer.fill(0)
        self.enter_dof_pos = self.state_cmd.q.copy()

        # 【核心改造1：进入 Stage 0 时，严格以“理想默认站立姿态”创建初始参考轨迹与参考点】
        # 即使此时机器人从 Passive 倒地态或微歪态切入，也不以快倒下或飞起的姿态生成轨迹，
        # 确保所有可视化灰色参考点均处于完美的理想默认姿态（高度 z=0.79m），从根本上消除飞起或倾倒发散！
        # 彻底平滑继承当前的真实物理状态，防止强制切入绝对零点导致 RL 网络出现巨大观测误差而暴走抽搐
        ideal_q = self.state_cmd.q.copy()
        ideal_base_pos = self.state_cmd.base_pos.copy()
        ideal_base_quat = self.state_cmd.base_quat.copy().astype(np.float32)
        ideal_fk_info = self.kinematics.forward(ideal_q, ideal_base_pos, ideal_base_quat)
        self.entry_fk_info = ideal_fk_info  # 保存真实的初始运动学状态，用于平滑过渡

        if self.reference_source == "CFgen":
            initialize_cfgen_reference(self, ideal_fk_info)
        elif self.reference_source == "NPZmotion":
            load_tracking_npz_reference(self)
        else:
            raise ValueError(f"Unknown reference source: {self.reference_source}")

        # 【核心改造2：预备静止站立段 Stage 0 门控】
        self.manual_stage = 0
        self.manual_phase_control_enabled = True
        self.stage_max_allowed_step = {0: 0}
        total_steps = len(self.ref_left_wrist_pos) - 1 if hasattr(self, "ref_left_wrist_pos") and self.ref_left_wrist_pos is not None else 0

        if hasattr(self, "ref_phase") and self.ref_phase is not None and len(self.ref_phase) > 0:
            idx_13 = np.where(self.ref_phase == 13)[0]
            idx_11 = np.where(self.ref_phase == 11)[0]
            if len(idx_13) > 0:
                self.stage_max_allowed_step[1] = int(idx_13[-1])
            elif len(idx_11) > 0:
                self.stage_max_allowed_step[1] = int(idx_11[-1])
            else:
                self.stage_max_allowed_step[1] = total_steps

            idx_21 = np.where(self.ref_phase == 21)[0]
            if len(idx_21) > 0:
                self.stage_max_allowed_step[2] = int(idx_21[-1])
            else:
                idx_12 = np.where(self.ref_phase == 12)[0]
                self.stage_max_allowed_step[2] = int(idx_12[-1]) if len(idx_12) > 0 else total_steps
        else:
            self.stage_max_allowed_step[1] = total_steps
            self.stage_max_allowed_step[2] = total_steps

        self.stage_max_allowed_step[3] = total_steps
        print(f"\n========================================================================================================================")
        print(f"[OmniContact Stand Prepare Test] 🛰️ 已安全进入 Stage 0 (理想默认站立姿态保持段):")
        print(f"   - 1. 参考点重置：所有灰色参考点均已对齐至高度 0.79m 的完美理想默认站立姿态，不会出现倒下或飞起坐标！")
        print(f"   - 2. 平稳过渡：底层通过 PD 余弦缓动将关节平顺过渡到理想默认站立姿态并坚固稳立 (counter_step 锁死为 0)。")
        print(f"   - 3. 下发任务：当确认机器人站好且箱子摆放完毕后，按手柄 L1+B (或 Y / 键盘空格/回车/B键)，")
        print(f"                 系统将即时根据真实站立姿态和最新箱子位置【现场真正计算最佳轨迹规划】，然后开始搬箱任务！")
        print(f"========================================================================================================================\n")

    def trigger_next_manual_stage(self):
        old_stage = getattr(self, "manual_stage", 0)
        if old_stage == 0:
            if getattr(self, "_is_planning_stage1", False):
                print("[OmniContact Stand Prepare Test] 正在异步规划中，请稍候...")
                return 0
                
            print(f"\n========================================================================================================================")
            print(f"[OmniContact Stand Prepare Test] 🚀 收到手柄/按键任务触发指令！准备由 Stage 0 进入 Stage 1...")
            print(f"   - 1. 实时读取实机最新的真正物理姿态 (base_pos/quat, q)...")
            print(f"   - 2. 实时锁死当前最新的物体系/相机识别坐标...")
            print(f"   - 3. 正在【后台异步】执行 initialize_cfgen_reference()，防止阻塞底层通信导致掉电！")
            
            self._is_planning_stage1 = True
            
            # 保存快照传递给后台线程
            state_snapshot = self._snapshot_state_cmd()
            policy_snapshot = self._snapshot_cfgen_policy(state_snapshot)
            fk_info = self.kinematics.forward(state_snapshot.q, state_snapshot.base_pos, state_snapshot.base_quat)
            
            with self.async_stage_lock:
                self.async_stage_generation += 1
                generation = self.async_stage_generation
                self.async_stage_pending = None
                self.async_stage_error = None
                
            def initial_plan_worker():
                import time
                start_t = time.perf_counter()
                try:
                    if policy_snapshot.reference_source == "CFgen":
                        initialize_cfgen_reference(policy_snapshot, fk_info)
                    elif policy_snapshot.reference_source == "NPZmotion":
                        load_tracking_npz_reference(policy_snapshot)
                        
                    result = self._collect_async_stage_result(policy_snapshot, time.perf_counter() - start_t)
                    with self.async_stage_lock:
                        if generation == self.async_stage_generation:
                            self.async_stage_pending = result
                            self.async_stage_error = None
                except Exception as exc:
                    with self.async_stage_lock:
                        if generation == self.async_stage_generation:
                            self.async_stage_pending = None
                            self.async_stage_error = exc
                            
            thread = threading.Thread(target=initial_plan_worker, daemon=True)
            with self.async_stage_lock:
                self.async_stage_thread = thread
            thread.start()
            
            return self.manual_stage
        else:
            self.manual_stage = min(old_stage + 1, 3)
            stage_names = {
                0: "Stage 0: Hold Initial Pose (原地理想默认站立保持)",
                1: f"Stage 1: Approach Object (开始靠近并面对物体, Phase 11~13 -> 终点 Step {self.stage_max_allowed_step.get(1, 0)})",
                2: f"Stage 2: Crouch & Lift Box (蹲下抱起物体, Phase 12+21 -> 终点 Step {self.stage_max_allowed_step.get(2, 0)})",
                3: f"Stage 3: Walk & Place Goal (走向目标及放置, Phase 22+ -> 终点 Step {self.stage_max_allowed_step.get(3, 0)})",
            }
            print(f"\n[OmniContact Stand Prepare Test] 🎮 触发进入下一阶段: {stage_names.get(self.manual_stage, f'Stage {self.manual_stage}')}")
            return self.manual_stage

    def _clip_norm(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        safe_norm = np.maximum(norm, 1e-8)
        scale = np.minimum(1.0, self.max_rel_norm / safe_norm)
        return v * scale

    def _get_future_state(self):
        robot_heading = yaw_quat(self.torso_quat).astype(np.float32)
        robot_heading_conj = quat_conjugate(robot_heading).astype(np.float32)
        last_idx = len(self.ref_left_wrist_pos) - 1

        future_offsets = np.asarray(self.future_frames, dtype=np.int32)
        if getattr(self, "manual_phase_control_enabled", True) and hasattr(self, "stage_max_allowed_step"):
            cap_idx = self.stage_max_allowed_step.get(getattr(self, "manual_stage", 0), last_idx)
        else:
            cap_idx = last_idx
        idx = np.minimum(self.counter_step + future_offsets, min(cap_idx, last_idx))
        torso_pos = self.torso_pos.astype(np.float32)

        def build_rel(pos_ref: np.ndarray, quat_ref: np.ndarray):
            pos_sel = pos_ref[idx].astype(np.float32)
            quat_sel = quat_ref[idx].astype(np.float32)
            rel_pos = quat_apply_batch(robot_heading_conj, pos_sel - torso_pos[None, :])
            rel_quat = quat_mul_left_batch(robot_heading_conj, quat_sel)
            rel_6d = quat_to_6d_batch(rel_quat)
            return rel_pos, rel_6d

        future_l_pos, future_l_6d = build_rel(self.ref_left_wrist_pos, self.ref_left_wrist_quat)
        future_r_pos, future_r_6d = build_rel(self.ref_right_wrist_pos, self.ref_right_wrist_quat)
        future_t_pos, future_t_6d = build_rel(self.ref_torso_future_pos, self.ref_torso_future_quat)
        future_la_pos, future_la_6d = build_rel(self.ref_left_ankle_future_pos, self.ref_left_ankle_future_quat)
        future_ra_pos, future_ra_6d = build_rel(self.ref_right_ankle_future_pos, self.ref_right_ankle_future_quat)
        future_contact = self.ref_contact[idx].astype(np.float32)

        res = np.concatenate(
            [
                future_l_pos,
                future_l_6d,
                future_r_pos,
                future_r_6d,
                future_t_pos,
                future_t_6d,
                future_la_pos,
                future_la_6d,
                future_ra_pos,
                future_ra_6d,
            ],
            axis=-1,
        ).reshape(-1)
        res_contact = np.concatenate([res, future_contact.reshape(-1)], axis=-1)

        curr_idx = min(self.counter_step, last_idx)
        return (
            res_contact.astype(np.float32),
            self.ref_left_wrist_pos[curr_idx],
            self.ref_left_wrist_quat[curr_idx],
            self.ref_right_wrist_pos[curr_idx],
            self.ref_right_wrist_quat[curr_idx],
        )

    def _get_fk_info(self):
        fk_info = self.kinematics.forward(self.state_cmd.q, self.state_cmd.base_pos, self.state_cmd.base_quat)
        self.torso_pos, self.torso_quat = fk_info["torso_link"]["pos"], fk_info["torso_link"]["quat"]
        ee_names = ["left_palm_link", "right_palm_link", "left_ankle_pitch_link", "right_ankle_pitch_link", "mid360_link"]
        self.ee_pos = np.array([quat_rotate_inverse(self.torso_quat, fk_info[n]["pos"] - self.torso_pos) for n in ee_names]).flatten()
        self.ee_pos = self.ee_pos.astype(np.float32)
        return fk_info

    def _build_bbox_rel(self, robot_heading: np.ndarray) -> np.ndarray:
        offsets = self.bbox_offsets_scaled.astype(np.float32)
        if getattr(self.state_cmd, "use_direct_rel_poses", False) and hasattr(self.state_cmd, "rel_torso_pos") and self.state_cmd.rel_torso_pos is not None:
            obj_pos_rel = self.state_cmd.rel_torso_pos.astype(np.float32)
            obj_rot_rel = self.state_cmd.rel_torso_quat.astype(np.float32)
            bbox_rel = quat_apply_batch(obj_rot_rel, offsets) + obj_pos_rel[None, :]
            bbox_rel = self._clip_norm(bbox_rel).astype(np.float32)
            return bbox_rel.reshape(-1)
        obj_quat = self.state_cmd.obj_quat.astype(np.float32)
        obj_pos = self.state_cmd.obj_pos.astype(np.float32)
        torso_pos = self.torso_pos.astype(np.float32)
        heading_conj = quat_conjugate(robot_heading).astype(np.float32)

        bbox_world = quat_apply_batch(obj_quat, offsets) + obj_pos[None, :]
        bbox_rel = quat_apply_batch(heading_conj, bbox_world - torso_pos[None, :])
        bbox_rel = self._clip_norm(bbox_rel).astype(np.float32)
        return bbox_rel.reshape(-1)

    def _flatten_obs_history(self) -> np.ndarray:
        h = self.obs_history_buffer
        return np.concatenate([h[:, a:b].reshape(-1) for a, b in self._HISTORY_SLICES])

    def run(self):
        fk_info = self._get_fk_info()
        qj = (self.state_cmd.q[self.mj2lab] - self.default_angles_lab).astype(np.float32)
        dqj = self.state_cmd.dq[self.mj2lab].astype(np.float32)
        robot_heading = yaw_quat(self.torso_quat)

        if getattr(self.state_cmd, "use_direct_rel_poses", False) and hasattr(self.state_cmd, "rel_torso_pos") and self.state_cmd.rel_torso_pos is not None:
            obj_pos_rel = self.state_cmd.rel_torso_pos.copy()
            obj_rot_rel = self.state_cmd.rel_torso_quat.copy()
        else:
            obj_pos_rel, obj_rot_rel = subtract_frame_transforms(
                self.torso_pos, robot_heading, self.state_cmd.obj_pos, self.state_cmd.obj_quat
            )
        obj_pos_rel = self._clip_norm(obj_pos_rel)
        if self.task in {"relocateball", "kickball"} or self.active_object_name == "ball":
            obj_rot_rel = np.array([1, 0, 0, 0], dtype=np.float32)
        obj_rot_6d = matrix_from_quat(obj_rot_rel)[:, :2].reshape(-1).astype(np.float32)

        bbox_rel_flat = self._build_bbox_rel(robot_heading)

        tracking_obs, l_p, l_q, r_p, r_q = self._get_future_state()
        curr_contact = self.ref_contact[min(self.counter_step, len(self.ref_contact) - 1)].astype(np.float32)
        
        curr_obs_prop = np.concatenate(
            [
                self.ee_pos,
                self.state_cmd.ang_vel.astype(np.float32).reshape(-1),
                self.state_cmd.gravity_ori.astype(np.float32),
                qj,
                dqj,
                self.action,
                obj_pos_rel.astype(np.float32),
                obj_rot_6d,
                bbox_rel_flat,
            ]
        )
        self.obs_history_buffer = np.roll(self.obs_history_buffer, -1, axis=0)
        self.obs_history_buffer[-1] = curr_obs_prop

        obs_history_flatten = self._flatten_obs_history()
        full_obs = np.concatenate([tracking_obs, obs_history_flatten]).astype(np.float32)
        
        obs_dict = {self.input_names[0]: full_obs[None, ...], self.input_names[1]: np.array([[0.0]], dtype=np.float32)}
        self.action = self.ort_session.run(None, obs_dict)[0].squeeze()
        
        raw_actions = (self.action * self.action_scale_lab + self.default_angles_lab)[self.lab2mj]
        target_kps = self.kps_lab[self.lab2mj] * 1.25
        target_kds = self.kds_lab[self.lab2mj] * 1.25
        
        blend_steps = 25
        is_blend = getattr(self, "enable_transition_blend", True)
        
        # 【核心改造3：当处于 Stage 0 (预备站立态) 时，使用 ONNX 跟踪模型针对第 0 帧(z=0.79m 原地理想站立参考)输出 raw_actions 实时调姿站立，确保能把机器人平稳举起并牢牢站稳不动】
        if self.manual_stage == 0:
            self.stage0_counter += 1
            if is_blend and self.stage0_counter < blend_steps and hasattr(self, "enter_dof_pos") and self.enter_dof_pos is not None:
                alpha = float(self.stage0_counter) / float(blend_steps)
                alpha_smooth = 0.5 * (1.0 - np.cos(alpha * np.pi))
                self.policy_output.actions = (1.0 - alpha_smooth) * self.enter_dof_pos + alpha_smooth * raw_actions
                enter_kps = np.full_like(target_kps, 200.0)
                enter_kds = np.full_like(target_kds, 5.0)
                self.policy_output.kps = (1.0 - alpha_smooth) * enter_kps + alpha_smooth * target_kps
                self.policy_output.kds = (1.0 - alpha_smooth) * enter_kds + alpha_smooth * target_kds
            else:
                self.policy_output.actions = raw_actions
                self.policy_output.kps, self.policy_output.kds = target_kps, target_kds
        else:
            if is_blend and self.counter_step < blend_steps and hasattr(self, "enter_dof_pos") and self.enter_dof_pos is not None:
                alpha = float(self.counter_step) / float(blend_steps)
                alpha_smooth = 0.5 * (1.0 - np.cos(alpha * np.pi))
                self.policy_output.actions = (1.0 - alpha_smooth) * self.enter_dof_pos + alpha_smooth * raw_actions
                self.policy_output.kps, self.policy_output.kds = target_kps, target_kds
            else:
                self.policy_output.actions = raw_actions
                self.policy_output.kps, self.policy_output.kds = target_kps, target_kds

        # --- 新增：检查异步规划是否完成 ---
        if getattr(self, "_is_planning_stage1", False):
            if self._apply_async_stage_plan():
                self._is_planning_stage1 = False
                
                if self.success == "failure":
                    print(f"   - ❌ 后台轨迹计算失败！停留在 Stage 0...")
                    print(f"========================================================================================================================\n")
                else:
                    # 规划完成后，结算步数和状态
                    total_steps = len(self.ref_left_wrist_pos) - 1 if hasattr(self, "ref_left_wrist_pos") and self.ref_left_wrist_pos is not None else 0
                    if hasattr(self, "ref_phase") and self.ref_phase is not None and len(self.ref_phase) > 0:
                        idx_13 = np.where(self.ref_phase == 13)[0]
                        idx_11 = np.where(self.ref_phase == 11)[0]
                        if len(idx_13) > 0:
                            self.stage_max_allowed_step[1] = int(idx_13[-1])
                        elif len(idx_11) > 0:
                            self.stage_max_allowed_step[1] = int(idx_11[-1])
                        else:
                            self.stage_max_allowed_step[1] = total_steps
    
                        idx_21 = np.where(self.ref_phase == 21)[0]
                        if len(idx_21) > 0:
                            self.stage_max_allowed_step[2] = int(idx_21[-1])
                        else:
                            idx_12 = np.where(self.ref_phase == 12)[0]
                            self.stage_max_allowed_step[2] = int(idx_12[-1]) if len(idx_12) > 0 else total_steps
                    else:
                        self.stage_max_allowed_step[1] = total_steps
                        self.stage_max_allowed_step[2] = total_steps
    
                    self.stage_max_allowed_step[3] = total_steps
    
                    self.manual_stage = 1
                    self.counter_step = 0
                    self.enter_dof_pos = self.state_cmd.q.copy()
                    print(f"   - ✅ 后台轨迹计算完毕！已平滑切换至 Stage 1: Approach Object (靠近并面对物体, Phase 11~13 -> 终点 Step {self.stage_max_allowed_step.get(1, 0)})，正式出发！")
                    print(f"========================================================================================================================\n")
        # ---------------------------------

        if (self.manual_stage == 0 and self.stage0_counter <= 50) or (self.manual_stage == 1 and self.counter_step <= 50):
            self._log_dense_transition_step(self.policy_output.actions, self.policy_output.kps)

        self.policy_output.wrist_goal = np.concatenate([l_p, l_q, r_p, r_q], axis=-1)
        self.policy_output.contact = curr_contact

        curr_idx = min(self.counter_step, len(self.ref_torso_future_pos) - 1)
        self.policy_output.torso_goal = np.concatenate(
            [self.ref_torso_future_pos[curr_idx], self.ref_torso_future_quat[curr_idx]], axis=-1
        ).astype(np.float32)
        self.policy_output.l_ankle_goal = np.concatenate(
            [self.ref_left_ankle_future_pos[curr_idx], self.ref_left_ankle_future_quat[curr_idx]], axis=-1
        ).astype(np.float32)
        self.policy_output.r_ankle_goal = np.concatenate(
            [self.ref_right_ankle_future_pos[curr_idx], self.ref_right_ankle_future_quat[curr_idx]], axis=-1
        ).astype(np.float32)

        if (self.reference_source == "CFgen") and (
            self.counter_step == len(self.ref_left_wrist_pos) - 1
        ) and self.manual_stage > 0:
            if self._apply_async_stage_plan():
                return
            if self._async_stage_running():
                self.policy_output.success = self.success
                self.policy_output.switch_to_loco = False
                return
            if self.task == "push-carry" and self.push_carry_stage == self.push_carry_cfgen.PUSH_STAGE:
                self._start_async_stage_plan(
                    lambda policy: setattr(policy, "push_carry_stage", policy.push_carry_cfgen.CARRY_STAGE)
                )
                return
            if self.task == "carry-push" and self.push_carry_stage == self.carry_push_cfgen.CARRY_STAGE:
                self._start_async_stage_plan(
                    lambda policy: setattr(policy, "push_carry_stage", policy.carry_push_cfgen.PUSH_STAGE)
                )
                return
            if self.task == "push-relocate" and self.push_relocate_stage == self.push_relocate_cfgen.PUSH_STAGE:
                self._start_async_stage_plan(
                    lambda policy: setattr(policy, "push_relocate_stage", policy.push_relocate_cfgen.RELOCATE_STAGE)
                )
                return
            if self.task in {"stackbox", "carry-carry", "carry-carry-carry"} and self.stackbox_stage_idx < self.stackbox_stage_count - 1:
                self._start_async_stage_plan(
                    lambda policy: setattr(policy, "stackbox_stage_idx", policy.stackbox_stage_idx + 1)
                )
                return
            self.switch_to_loco = True
            goal_error = (
                np.linalg.norm(self.state_cmd.base_pos - self.goal_pos)
                if self.task == "loco"
                else np.linalg.norm(self.state_cmd.obj_pos - self.goal_pos)
            )
            if goal_error < 0.2:
                self.success = "success"
            else:
                self.success = "failure"

        self.policy_output.success = self.success
        self.policy_output.switch_to_loco = self.switch_to_loco
        is_stalled = getattr(self, "is_vision_stalled", False)
        if is_stalled and getattr(self, "manual_stage", 0) > 0:
            if not getattr(self, "_stalled_warned", False):
                print(f"\n[OmniContact Stand Prepare Test] ⏸️ 检测到视觉数据断流！暂停轨迹推进 (保持当前姿态) 直到视觉恢复...\n")
                self._stalled_warned = True
        else:
            if getattr(self, "_stalled_warned", False):
                print(f"\n[OmniContact Stand Prepare Test] ▶️ 视觉数据已恢复！继续推进轨迹...\n")
                self._stalled_warned = False
                
            if getattr(self, "manual_phase_control_enabled", True) and hasattr(self, "stage_max_allowed_step"):
                target_max_step = self.stage_max_allowed_step.get(getattr(self, "manual_stage", 0), len(self.ref_left_wrist_pos) - 1)
                if self.counter_step < target_max_step:
                    self.counter_step += 1
                    if self.counter_step == target_max_step and target_max_step < (len(self.ref_left_wrist_pos) - 1):
                        print(f"\n[OmniContact Stand Prepare Test] ⏸️ 机器人已到达 Stage {self.manual_stage} 终点 (Step {self.counter_step})，暂停自增并稳定保持当前姿态。请按手柄 L1+B 触发进入下一分段！\n")
            else:
                self.counter_step += 1

    def exit(self):
        print("OmniContactStandPrepareTest exited")

    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.SKILL_3):
            return FSMStateName.SKILL_COOLDOWN
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            return FSMStateName.PASSIVE
        elif(self.state_cmd.skill_cmd == FSMCommand.LOCO):
            return FSMStateName.LOCOMODE
        else:
            return FSMStateName.SKILL_OmniContact

    def _log_dense_transition_step(self, raw_actions, target_kps):
        try:
            is_blend = getattr(self, "enable_transition_blend", True)
            log_file = os.path.join(PROJECT_ROOT, "transition_log_stand_test.txt")
            mode_str = "w" if (self.counter_step == 0 and self.stage0_counter <= 1) else "a"
            with open(log_file, mode_str, encoding="utf-8") as f:
                if (self.counter_step == 0 and self.stage0_counter <= 1):
                    f.write("========================================================================================================================\n")
                    f.write("OmniContact 原地理想默认站立切入全过程密集调试日志 (Dense Step-by-Step Transition Log)\n")
                    f.write("========================================================================================================================\n")
                    f.write("Stage | Step | 时间(s) | 缓动权重(α) | 左膝角 -> 发令角 (阶跃差ΔQ) | 左髋角 -> 发令角 (阶跃差ΔQ) | 发令 Kp | 脚踝Z轴帧差速度\n")
                    f.write("------------------------------------------------------------------------------------------------------------------------\n")
                
                step_val = self.stage0_counter if self.manual_stage == 0 else self.counter_step
                step_t = step_val * 0.02
                blend_steps = 25
                if is_blend and step_val < blend_steps:
                    alpha = float(step_val) / float(blend_steps)
                    alpha_smooth = 0.5 * (1.0 - np.cos(alpha * np.pi))
                elif is_blend:
                    alpha_smooth = 1.0
                else:
                    alpha_smooth = 0.0

                knee_act = float(self.state_cmd.q[3])
                knee_cmd = float(self.policy_output.actions[3])
                knee_err_deg = (knee_cmd - knee_act) * 180.0 / np.pi

                hip_act = float(self.state_cmd.q[0])
                hip_cmd = float(self.policy_output.actions[0])
                hip_err_deg = (hip_cmd - hip_act) * 180.0 / np.pi

                kp_val = float(self.policy_output.kps[3])

                if step_val == 0:
                    ee_z_vel = 0.0
                else:
                    ee_z_vel = (float(self.obs_history_buffer[-1][2]) - float(self.obs_history_buffer[-2][2])) / 0.02

                f.write(f"S{self.manual_stage:4d} | {step_val:4d} | {step_t:6.2f}s | {alpha_smooth:11.3f} | {knee_act:6.3f} -> {knee_cmd:6.3f} (Δ={knee_err_deg:+6.2f}°) | {hip_act:6.3f} -> {hip_cmd:6.3f} (Δ={hip_err_deg:+6.2f}°) | {kp_val:7.2f} | {ee_z_vel:+8.2f} m/s\n")
        except Exception as e:
            pass
