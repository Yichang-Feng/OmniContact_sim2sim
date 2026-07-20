"""
标准部署脚本 (Standard Deployment Script)
此版本为包含视觉模块 (Vision) 的完整部署版本，通过 ZMQ/UDP 与独立视觉节点通信获取位姿。
"""
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

from dataclasses import dataclass, field
import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import PolicyOutput, StateAndCmd
from FSM.FSM import FSM
from common.utils import FSMCommand, FSMStateName, get_gravity_orientation, quat_mul, quat_conjugate, yaw_quat, quat_apply, subtract_frame_transforms
from common.joystick import JoyStick, JoystickButton
from omnicontact_vision import VisionReceiver

def pd_control(target_q, q, kp, target_dq, dq, kd, torque_limit_mj=None, torque_clip=True):
    """ Calculates torques from position commands """
    tau = (target_q - q) * kp + (target_dq - dq) * kd
    
    if torque_clip and torque_limit_mj is not None:
        tau = np.clip(tau, -torque_limit_mj, torque_limit_mj)
    return tau


def config_array(config, key, *, dtype=np.float32, shape=None):
    arr = np.asarray(config[key], dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def require_length(name, arr, length):
    if len(arr) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(arr)}.")


TASK_ALIASES = {
    "pushbox": "pushbox-in",
}


TASK_XML_PATHS = {
    "carrybox": "g1_description/omnicontact_carry_box.xml",
    "pushbox": "g1_description/omnicontact_push_box.xml",
    "pushbox-two": "g1_description/omnicontact_push_box.xml",
    "pushbox-in": "g1_description/omnicontact_push_box.xml",
    "slidebox": "g1_description/omnicontact_slide_box.xml",
    "slidebox-left": "g1_description/omnicontact_slide_box.xml",
    "slidebox-right": "g1_description/omnicontact_slide_box.xml",
    "relocateball": "g1_description/omnicontact_relocate_ball.xml",
    "kickball": "g1_description/omnicontact_kick_ball.xml",
    "kickbox": "g1_description/omnicontact_kick_ball.xml",
    "push-carry": "g1_description/omnicontact_pushcarry_box.xml",
    "carry-push": "g1_description/omnicontact_pushcarry_box.xml",
    "push-relocate": "g1_description/omnicontact_pushrelocate_ball.xml",
    "carry-carry": "g1_description/omnicontact_stack_2box.xml",
    "carry-carry-carry": "g1_description/omnicontact_stack_3box.xml",
    "carryheart": "g1_description/omnicontact_heart_10box.xml",
}

INIT_Z_AUTO_TASKS = {
    "pushbox-two",
    "pushbox-in",
    "slidebox",
    "slidebox-left",
    "slidebox-right",
    "relocateball",
    "kickball",
    "kickbox",
    "push-carry",
    "carry-push",
    "push-relocate",
}

GOAL_Z_AUTO_TASKS = {
    "pushbox-two",
    "pushbox-in",
    "slidebox",
    "slidebox-left",
    "slidebox-right",
    "kickball",
    "kickbox",
    "push-carry",
    "carry-push",
    "push-relocate",
}


def resolve_xml_path(task: str, xml_path_override: str, config: dict) -> str:
    xml_path = str(xml_path_override).strip()
    if not xml_path:
        xml_path = TASK_XML_PATHS.get(task, str(config.get("xml_path", "g1_description/omnicontact_carry_box.xml")))
    path = Path(xml_path).expanduser()
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    return str(path.resolve())


def geom_half_dims(model: mujoco.MjModel, *geom_names: str, default: tuple[float, float, float]) -> np.ndarray:
    for geom_name in geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            size = np.asarray(model.geom_size[geom_id, :3], dtype=np.float32).copy()
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                radius = float(size[0])
                return np.array([radius, radius, radius], dtype=np.float32)
            return size
    return np.asarray(default, dtype=np.float32).reshape(3).copy()


def infer_policy_dims_from_model(model: mujoco.MjModel, override_dims: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if override_dims is not None:
        dims = np.asarray(override_dims, dtype=np.float32).reshape(3)
        stack_dims = np.asarray(
            [[0.20, 0.20, 0.15], [0.15, 0.15, 0.15], [0.10, 0.10, 0.10]],
            dtype=np.float32,
        )
        return {
            "box_dims": dims.copy(),
            "ball_dims": dims.copy(),
            "push_box_dims": dims.copy(),
            "carry_box_dims": dims.copy(),
            "stack_box_dims": stack_dims,
        }

    stack_dims = []
    for geom_name, default in zip(
        ("stack_box_large_geom", "stack_box_mid_geom", "stack_box_small_geom"),
        ((0.20, 0.20, 0.15), (0.15, 0.15, 0.15), (0.10, 0.10, 0.10)),
    ):
        stack_dims.append(geom_half_dims(model, geom_name, default=default))

    return {
        "box_dims": geom_half_dims(
            model,
            "ghost_box_geom",
            "box_geom",
            "ball_geom",
            default=(0.15, 0.15, 0.15),
        ),
        "push_box_dims": geom_half_dims(
            model,
            "ghost_push_box_geom",
            "ghost_box_geom",
            "push_box_geom_top",
            "box_geom_top",
            default=(0.23, 0.25, 0.26),
        ),
        "ball_dims": geom_half_dims(
            model,
            "ghost_ball_geom",
            "ball_geom",
            default=(0.10, 0.10, 0.10),
        ),
        "carry_box_dims": geom_half_dims(
            model,
            "ghost_carry_box_geom",
            "carry_box_geom",
            "ghost_box_geom",
            "box_geom",
            default=(0.15, 0.15, 0.15),
        ),
        "stack_box_dims": np.asarray(stack_dims, dtype=np.float32).reshape(3, 3),
    }


def select_active_box_dims(task: str, dims_by_profile: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    if task in {"push-carry", "push-relocate"}:
        return "push_box", np.asarray(dims_by_profile["push_box_dims"], dtype=np.float32).copy()
    if task == "carry-push":
        return "carry_box", np.asarray(dims_by_profile["carry_box_dims"], dtype=np.float32).copy()
    if task in {"relocateball", "kickball", "kickbox"}:
        return "ball", np.asarray(dims_by_profile["ball_dims"], dtype=np.float32).copy()
    return "box", np.asarray(dims_by_profile["box_dims"], dtype=np.float32).copy()


def normalize_object_pos_for_task(pos: np.ndarray, task: str, half_z: float, *, is_goal: bool) -> np.ndarray:
    out = np.asarray(pos, dtype=np.float32).reshape(3).copy()
    auto_tasks = GOAL_Z_AUTO_TASKS if is_goal else INIT_Z_AUTO_TASKS
    if task in auto_tasks:
        out[2] = float(half_z)
    return out


def parse_object_pos_arg(value, default, half_z: float, task: str, *, is_goal: bool) -> np.ndarray:
    pos = np.asarray(value if value is not None else default, dtype=np.float32).reshape(-1)
    if len(pos) == 2:
        pos = np.array([pos[0], pos[1], half_z], dtype=np.float32)
    elif len(pos) == 3:
        pos = pos.astype(np.float32).copy()
    else:
        raise ValueError("--init-pos/--goal-pos must provide either X Y or X Y Z.")
    return normalize_object_pos_for_task(pos, task, half_z, is_goal=is_goal)


@dataclass
class ReplanState:
    identity_quat: np.ndarray
    has_held_object: bool = False
    detach_counter: int = 0
    cooldown_counter: int = 0
    replan_counter: int = 0
    waiting_for_static: bool = False
    static_counter: int = 0
    obj_quat_offset: np.ndarray = field(init=False)

    def __post_init__(self):
        self.obj_quat_offset = self.identity_quat.copy()

    def reset_detection(self, *, reset_held: bool):
        if reset_held:
            self.has_held_object = False
        self.detach_counter = 0
        self.waiting_for_static = False
        self.static_counter = 0

    def mark_held(self):
        self.has_held_object = True
        self.reset_detection(reset_held=False)

    def reset_session(self):
        self.reset_detection(reset_held=True)
        self.cooldown_counter = 0
        self.obj_quat_offset = self.identity_quat.copy()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config = config or {}
    task_choices = config.get("task_choices", ["carrybox", "pushbox-two", "pushbox-in", "relocateball", "kickball", "kickbox"])
    task_choices = list(task_choices)
    for alias in TASK_ALIASES:
        if alias not in task_choices:
            task_choices.append(alias)
    lab2mj = config_array(config, "lab2mj", dtype=np.int32)
    torque_limit_lab = config_array(config, "torque_limit_lab")
    default_joint_pos_mj = config_array(config, "default_joint_pos_mj")
    identity_quat = config_array(config, "identity_quat", shape=4)
    reference_yellow = config_array(config, "reference_no_contact_rgba", shape=4)
    reference_red = config_array(config, "reference_contact_rgba", shape=4)

    parser = argparse.ArgumentParser(description="Interactive Mujoco deploy script for carrybox and pushbox skills.")
    parser.add_argument(
        "--init-pos",
        type=float,
        nargs="+",
        default=config.get("init_pos", (1.0, 0.0, 0.55)),
        metavar="POS",
        help="Initial object position used by CFgen reset.",
    )
    parser.add_argument(
        "--box-half-dims",
        type=float,
        nargs=3,
        default=None,
        metavar=("HX", "HY", "HZ"),
        help="Optional override for object half dimensions. If omitted, dimensions are inferred from the selected XML.",
    )
    parser.add_argument(
        "--goal-pos",
        type=float,
        nargs="+",
        default=config.get("goal_pos", (3.0, 0.0, 0.26)),
        metavar="POS",
        help="Goal object position used by CFgen reset and table visualization.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=config.get("task", "carrybox"),
        choices=task_choices,
        help="Task preset used by WTAC CFgen planner.",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default="",
        help="Override Mujoco XML path. If omitted, the XML is selected from --task.",
    )
    parser.add_argument(
        "--replan",
        dest="replan",
        action="store_true",
        default=not bool(config.get("disable_replan", True)),
        help="Enable drop-triggered closed-loop replan for WTAC carrybox CFgen. Disabled by default.",
    )
    parser.add_argument(
        "--disable-replan",
        dest="replan",
        action="store_false",
        help="Disable drop-triggered closed-loop replan for WTAC carrybox CFgen.",
    )
    parser.add_argument(
        "--use-vision",
        action="store_true",
        help="Use AprilTag vision estimate for object pose instead of MuJoCo ground truth.",
    )
    parser.add_argument(
        "--vision-port",
        type=int,
        default=5556,
        help="ZMQ port for listening to AprilTag pose estimates.",
    )
    parser.add_argument(
        "--publish-sim-camera",
        action="store_true",
        help="Publish rendered sim camera images over ZMQ port 5555 for simulation vision testing.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Deploy to physical Unitree G1 robot over Ethernet DDS instead of simulation.",
    )
    parser.add_argument(
        "--net-if",
        type=str,
        default="enx6c1ff724495a",
        help="Network interface name connected to the robot (e.g., eth0, enp3s0).",
    )
    args = parser.parse_args()
    args.task = TASK_ALIASES.get(args.task, args.task)
    xml_path = resolve_xml_path(args.task, args.xml_path, config)
    print(f"[deploy] task={args.task} xml_path={xml_path}")

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    dims_by_profile = infer_policy_dims_from_model(
        m,
        None if args.box_half_dims is None else np.asarray(args.box_half_dims, dtype=np.float32).reshape(3),
    )
    active_object_name, box_half_dims = select_active_box_dims(args.task, dims_by_profile)
    init_pos = parse_object_pos_arg(
        args.init_pos,
        config.get("init_pos", (1.0, 0.0, 0.55)),
        float(box_half_dims[2]),
        args.task,
        is_goal=False,
    )
    goal_pos = parse_object_pos_arg(
        args.goal_pos,
        config.get("goal_pos", (3.0, 0.0, 0.26)),
        float(box_half_dims[2]),
        args.task,
        is_goal=True,
    )

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    vision_receiver = None
    sim_renderer = None
    if getattr(args, "use_vision", False) or getattr(args, "publish_sim_camera", False):
        publish_sim = getattr(args, "publish_sim_camera", False) or getattr(args, "use_vision", False)
        vision_receiver = VisionReceiver(
            vision_port=getattr(args, "vision_port", 5556),
            publish_sim_camera=publish_sim,
        )
        if publish_sim:
            sim_renderer = mujoco.Renderer(m, 480, 640)

    def mj_id(obj_type, name: str) -> int:
        return mujoco.mj_name2id(m, obj_type, name)

    def body_id(name: str) -> int:
        return mj_id(mujoco.mjtObj.mjOBJ_BODY, name)

    def geom_id(name: str) -> int:
        return mj_id(mujoco.mjtObj.mjOBJ_GEOM, name)

    def joint_id(name: str) -> int:
        return mj_id(mujoco.mjtObj.mjOBJ_JOINT, name)

    def mocap_id(name: str) -> int:
        bid = body_id(name)
        return int(m.body_mocapid[bid]) if bid >= 0 else -1

    # `ref_*` mocap bodies drive wrist, torso, ankle, and table visualizations.
    ref_mocap_ids = {
        "l_wrist": mocap_id("ref_l_wrist_frame"),
        "r_wrist": mocap_id("ref_r_wrist_frame"),
        "torso": mocap_id("ref_torso_frame"),
        "l_ankle": mocap_id("ref_l_ankle_frame"),
        "r_ankle": mocap_id("ref_r_ankle_frame"),
        "plane_1": mocap_id("plane_1_holder"),
        "plane_2": mocap_id("plane_2_holder"),
    }
    ref_contact_geom_ids = [
        geom_id("ref_l_ankle_mesh"),
        geom_id("ref_r_ankle_mesh"),
        geom_id("ref_l_rubber_hand"),
        geom_id("ref_r_rubber_hand"),
    ]
    print('Successfully load wrist/torso/anklereference visualization!', ref_mocap_ids["l_wrist"], ref_mocap_ids["r_wrist"])
    m.opt.timestep = simulation_dt
    num_joints = m.nu
    require_length("lab2mj", lab2mj, num_joints)
    require_length("torque_limit_lab", torque_limit_lab, num_joints)
    require_length("default_joint_pos_mj", default_joint_pos_mj, num_joints)
    if np.any(lab2mj < 0) or np.any(lab2mj >= num_joints):
        raise ValueError(f"lab2mj indices must be in [0, {num_joints - 1}].")
    torque_limit_mj = torque_limit_lab[lab2mj]
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    wrist_state = np.zeros(14, dtype=np.float32)
    contact_state = np.zeros(4, dtype=np.float32)
    torso_goal = np.zeros(7, dtype=np.float32)
    l_ankle_goal = np.zeros(7, dtype=np.float32)
    r_ankle_goal = np.zeros(7, dtype=np.float32)
    sim_counter = 0

    real_robot = None
    if getattr(args, "real", False):
        from real_robot_interface import RealRobotInterface
        real_robot = RealRobotInterface(net_interface=getattr(args, "net_if", "enx6c1ff724495a"), num_joints=num_joints)
        if not real_robot.wait_for_connection(timeout=60.0):
            print("[deploy] 错误: 未能连接到真实机器人底层 DDS 数据包，程序退出以确保安全。")
            sys.exit(1)

    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)
    contactflow_policy = FSM_controller.omnicontact
    contactflow_policy.task = args.task
    contactflow_policy.active_object_name = active_object_name
    contactflow_policy.ball_dims = np.asarray(dims_by_profile["ball_dims"], dtype=np.float32).copy()
    contactflow_policy.push_box_dims = np.asarray(dims_by_profile["push_box_dims"], dtype=np.float32).copy()
    contactflow_policy.carry_box_dims = np.asarray(dims_by_profile["carry_box_dims"], dtype=np.float32).copy()
    contactflow_policy.stack_box_dims = np.asarray(dims_by_profile["stack_box_dims"], dtype=np.float32).copy()
    contactflow_policy.box_dims = box_half_dims.copy()
    contactflow_policy.goal_pos_override = goal_pos.copy()
    contactflow_policy.bbox_scale = contactflow_policy.box_dims * 2.0
    contactflow_policy.bbox_offsets_scaled = contactflow_policy.bbox_offsets * contactflow_policy.bbox_scale.reshape(1, 3)
    contactflow_policy.replan_active = False
    print(f"[deploy] active_object={active_object_name} half_dims={box_half_dims.tolist()}")

    ghost_robot_joint_id = joint_id("ghost_floating_base_joint")
    ghost_robot_qpos_adr = int(m.jnt_qposadr[ghost_robot_joint_id]) if ghost_robot_joint_id >= 0 else -1
    ghost_robot_qvel_adr = int(m.jnt_dofadr[ghost_robot_joint_id]) if ghost_robot_joint_id >= 0 else -1

    def ghost_joint_qpos_adr(name: str) -> int:
        jid = joint_id(f"ghost_{name}")
        return int(m.jnt_qposadr[jid]) if jid >= 0 else -1

    ghost_robot_joint_qpos_adrs = np.array(
        [ghost_joint_qpos_adr(name) for name in contactflow_policy.kinematics.joint_names],
        dtype=np.int32,
    )
    
    joystick = JoyStick()
    Running = True

    box_body_id = body_id("box")
    box_geom_id = geom_id("box_geom")
    box_joint_id = joint_id("box")
    box_qvel_adr = int(m.jnt_dofadr[box_joint_id]) if box_joint_id >= 0 else -1
    left_palm_body_id = body_id("left_palm_link")
    right_palm_body_id = body_id("right_palm_link")
    left_wrist_yaw_body_id = body_id("left_wrist_yaw_link")
    right_wrist_yaw_body_id = body_id("right_wrist_yaw_link")
    mid360_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mid360_link")
    mid360_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mid360_link") if mid360_site_id < 0 else -1

    def body_geom_ids(body_id: int) -> list[int]:
        if body_id < 0:
            return []
        geom_num = int(m.body_geomnum[body_id])
        geom_adr = int(m.body_geomadr[body_id])
        return [geom_adr + i for i in range(geom_num)]

    hand_collision_geom_set = set(body_geom_ids(left_wrist_yaw_body_id) + body_geom_ids(right_wrist_yaw_body_id))

    replan_config = config.get("closed_loop_replan", {})
    REPLAN_DETACH_DIST_THRESHOLD = float(replan_config.get("detach_dist_threshold", 0.32))
    REPLAN_DETACH_TRIGGER_TICKS = int(replan_config.get("detach_trigger_ticks", 8))
    REPLAN_COOLDOWN_TICKS = int(replan_config.get("cooldown_ticks", 80))
    REPLAN_GOAL_TOLERANCE = float(replan_config.get("goal_tolerance", 0.2))
    REPLAN_OBJ_SPEED_THRESHOLD = float(replan_config.get("obj_speed_threshold", 0.05))
    REPLAN_OBJ_ANG_SPEED_THRESHOLD = float(replan_config.get("obj_ang_speed_threshold", 0.15))
    REPLAN_STATIC_TICKS = int(replan_config.get("static_ticks", 5))

    replan_state = ReplanState(identity_quat=identity_quat)

    def reset_replan_session():
        replan_state.reset_session()
        contactflow_policy.replan_active = False

    def wtac_carrybox_replan_enabled() -> bool:
        return (
            bool(getattr(args, "replan", False))
            and FSM_controller.cur_policy is contactflow_policy
            and contactflow_policy.reference_source == "CFgen"
            and contactflow_policy.task != "kickball"
        )

    def object_contact_with_hand_geoms() -> bool:
        if box_geom_id < 0 or not hand_collision_geom_set:
            return False
        for i in range(int(d.ncon)):
            contact = d.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == box_geom_id and geom2 in hand_collision_geom_set:
                return True
            if geom2 == box_geom_id and geom1 in hand_collision_geom_set:
                return True
        return False

    def object_palm_min_dist() -> float:
        if box_body_id < 0:
            return np.inf
        obj_pos = d.xpos[box_body_id]
        dists = []
        if left_palm_body_id >= 0:
            dists.append(float(np.linalg.norm(obj_pos - d.xpos[left_palm_body_id])))
        if right_palm_body_id >= 0:
            dists.append(float(np.linalg.norm(obj_pos - d.xpos[right_palm_body_id])))
        if not dists:
            return np.inf
        return min(dists)

    def is_object_held() -> tuple[bool, float, bool]:
        min_dist = object_palm_min_dist()
        in_contact = object_contact_with_hand_geoms()
        return in_contact or (min_dist <= REPLAN_DETACH_DIST_THRESHOLD), min_dist, in_contact

    def current_goal_pos() -> np.ndarray:
        if hasattr(contactflow_policy, "goal_pos"):
            return np.asarray(contactflow_policy.goal_pos, dtype=np.float32).reshape(3).copy()
        return goal_pos.copy()

    def is_object_near_goal() -> bool:
        if box_body_id < 0:
            return False
        return float(np.linalg.norm(d.xpos[box_body_id] - current_goal_pos())) <= REPLAN_GOAL_TOLERANCE

    def current_obj_linear_speed() -> float:
        if box_qvel_adr < 0 or box_qvel_adr + 3 > d.qvel.shape[0]:
            return 0.0
        return float(np.linalg.norm(d.qvel[box_qvel_adr : box_qvel_adr + 3]))

    def current_obj_angular_speed() -> float:
        start = box_qvel_adr + 3
        if box_qvel_adr < 0 or start + 3 > d.qvel.shape[0]:
            return 0.0
        return float(np.linalg.norm(d.qvel[start : start + 3]))

    def compute_replan_object_quat_offset() -> np.ndarray:
        if box_body_id < 0:
            return identity_quat.copy()
        raw_obj_quat = d.xquat[box_body_id].copy().astype(np.float32)
        target_obj_quat = yaw_quat(raw_obj_quat).astype(np.float32)
        quat_offset = quat_mul(target_obj_quat, quat_conjugate(raw_obj_quat)).astype(np.float32)
        quat_offset /= max(float(np.linalg.norm(quat_offset)), 1e-8)
        return quat_offset

    def should_apply_replan_object_quat_offset() -> bool:
        return wtac_carrybox_replan_enabled() and bool(getattr(contactflow_policy, "replan_active", False))

    def sync_object_state():
        if box_body_id < 0:
            return
        needs_odom_calibration = False
        use_vis = getattr(args, "use_vision", False)
        if use_vis and vision_receiver is not None:
            v_pos, v_quat, valid = vision_receiver.get_validated_world_pose(m, d)
            g_pos, valid_goal = vision_receiver.get_validated_goal_pose(m, d)
            if valid_goal and g_pos is not None:
                goal_pos[:] = g_pos
                if hasattr(contactflow_policy, "goal_pos"):
                    contactflow_policy.goal_pos[:] = g_pos
                if hasattr(contactflow_policy, "goal_pos_override") and contactflow_policy.goal_pos_override is not None:
                    contactflow_policy.goal_pos_override[:] = g_pos
                table_z_offset = float(contactflow_policy.box_dims[2]) + 0.005
                table_offset = np.array([0.0, 0.0, table_z_offset], dtype=np.float32)
                set_table_positions(init_pos - table_offset, goal_pos - table_offset)

            gt_pos = d.xpos[box_body_id].copy()
            if valid and v_pos is not None:
                vision_cache["last_pos"] = v_pos.copy()
                vision_cache["last_quat"] = v_quat.copy()
                state_cmd.obj_pos = v_pos
                state_cmd.obj_quat = v_quat
                err = float(np.linalg.norm(v_pos - gt_pos))
                if sim_counter % 40 == 0:
                    v_yaw = float(np.degrees(np.arctan2(2*(v_quat[0]*v_quat[3] + v_quat[1]*v_quat[2]), 1 - 2*(v_quat[2]**2 + v_quat[3]**2))))
                    gt_quat = d.xquat[box_body_id]
                    gt_yaw = float(np.degrees(np.arctan2(2*(gt_quat[0]*gt_quat[3] + gt_quat[1]*gt_quat[2]), 1 - 2*(gt_quat[2]**2 + gt_quat[3]**2))))
                    print(f"\r[Vision Compare] GT: [{gt_pos[0]:.2f}, {gt_pos[1]:.2f}, {gt_pos[2]:.2f}] | Est: [{v_pos[0]:.2f}, {v_pos[1]:.2f}, {v_pos[2]:.2f}] | EstYaw: {v_yaw:.1f}° (GT: {gt_yaw:.1f}°) | 误差: {err*100:.1f} cm   ", end="", flush=True)
            else:
                if real_robot is not None and vision_cache["last_pos"] is not None:
                    state_cmd.obj_pos = vision_cache["last_pos"].copy()
                    state_cmd.obj_quat = vision_cache["last_quat"].copy()
                    if sim_counter % 40 == 0:
                        print(f"\r[Vision Compare]  视觉丢帧/遮挡！真机已自动保持上次有效位姿", end="", flush=True)
                else:
                    state_cmd.obj_pos = gt_pos
                    state_cmd.obj_quat = d.xquat[box_body_id].copy()
                    needs_odom_calibration = True
                    if sim_counter % 40 == 0:
                        print(f"\r[Vision Compare] 等待视觉 AprilTag 位姿解算输入 (暂用GT)   ", end="", flush=True)
        else:
            if real_robot is not None and vision_cache["last_pos"] is not None:
                state_cmd.obj_pos = vision_cache["last_pos"].copy()
                state_cmd.obj_quat = vision_cache["last_quat"].copy()
            else:
                state_cmd.obj_pos = d.xpos[box_body_id].copy()
                state_cmd.obj_quat = d.xquat[box_body_id].copy()
                needs_odom_calibration = True
        if should_apply_replan_object_quat_offset():
            # Re-anchor a dropped box to a yaw-only baseline while keeping later relative in-hand tilt cues.
            state_cmd.obj_quat = quat_mul(
                replan_state.obj_quat_offset,
                state_cmd.obj_quat.astype(np.float32),
            ).astype(np.float32)
            state_cmd.obj_quat /= max(float(np.linalg.norm(state_cmd.obj_quat)), 1e-8)

        # 确保物体坐标与已校准的里程计局部坐标完全统一
        if needs_odom_calibration:
            if odom_calibration["initial_pos_xy"] is not None:
                state_cmd.obj_pos[0] -= odom_calibration["initial_pos_xy"][0]
                state_cmd.obj_pos[1] -= odom_calibration["initial_pos_xy"][1]
            if odom_calibration.get("initial_pos_z") is not None:
                state_cmd.obj_pos[2] = (state_cmd.obj_pos[2] - odom_calibration["initial_pos_z"]) + m.qpos0[2]
            if odom_calibration["initial_yaw_quat"] is not None:
                rel_vec = state_cmd.obj_pos - state_cmd.base_pos
                rel_vec_rot = quat_apply(quat_conjugate(odom_calibration["initial_yaw_quat"]), rel_vec)
                state_cmd.obj_pos = state_cmd.base_pos + rel_vec_rot
                state_cmd.obj_quat = quat_mul(quat_conjugate(odom_calibration["initial_yaw_quat"]), state_cmd.obj_quat).astype(np.float32)
                state_cmd.obj_quat /= max(float(np.linalg.norm(state_cmd.obj_quat)), 1e-8)

    def replan_from_current_state(min_dist: float, in_contact: bool, obj_lin_speed: float, obj_ang_speed: float):
        if not wtac_carrybox_replan_enabled():
            return
        replan_state.obj_quat_offset = compute_replan_object_quat_offset()
        contactflow_policy.replan_active = True
        sync_object_state()
        contactflow_policy.enter()
        sync_object_state()
        replan_state.replan_counter += 1
        replan_state.reset_detection(reset_held=True)
        replan_state.cooldown_counter = REPLAN_COOLDOWN_TICKS
        policy_output.switch_to_loco = False
        policy_output.success = ""
        print(
            f"[closed_loop] replan#{replan_state.replan_counter} | reason=drop_box_wait_static, "
            f"min_palm_dist={min_dist:.3f}, hand_contact={int(in_contact)}, "
            f"obj_lin_speed={obj_lin_speed:.3f} m/s, obj_ang_speed={obj_ang_speed:.3f} rad/s"
        )

    def maybe_closed_loop_replan():
        if not wtac_carrybox_replan_enabled():
            contactflow_policy.replan_active = False
            return

        held, min_dist, in_contact = is_object_held()
        near_goal = is_object_near_goal()

        if held:
            replan_state.mark_held()
            return

        if replan_state.cooldown_counter > 0:
            replan_state.cooldown_counter -= 1
            return

        if (not replan_state.has_held_object) or near_goal:
            replan_state.reset_detection(reset_held=False)
            return

        obj_lin_speed = current_obj_linear_speed()
        obj_ang_speed = current_obj_angular_speed()

        if replan_state.waiting_for_static:
            if obj_lin_speed <= REPLAN_OBJ_SPEED_THRESHOLD and obj_ang_speed <= REPLAN_OBJ_ANG_SPEED_THRESHOLD:
                replan_state.static_counter += 1
            else:
                replan_state.static_counter = 0

            if replan_state.static_counter >= REPLAN_STATIC_TICKS:
                replan_from_current_state(min_dist, in_contact, obj_lin_speed, obj_ang_speed)
            return

        replan_state.detach_counter += 1
        if replan_state.detach_counter >= REPLAN_DETACH_TRIGGER_TICKS:
            replan_state.waiting_for_static = True
            replan_state.static_counter = 0
            print(
                f"[closed_loop] drop detected, waiting for object to settle | "
                f"lin={obj_lin_speed:.3f}/{REPLAN_OBJ_SPEED_THRESHOLD:.3f} m/s, "
                f"ang={obj_ang_speed:.3f}/{REPLAN_OBJ_ANG_SPEED_THRESHOLD:.3f} rad/s"
            )

    def reset_robot_pose():
        d.qpos[7 : 7 + num_joints] = default_joint_pos_mj
        d.qvel[6 : 6 + num_joints] = 0.0
        d.qvel[0:6] = 0.0

    def set_object_pose(pos, quat=None):
        if quat is None:
            quat = identity_quat
        box_qpos_adr = int(m.jnt_qposadr[box_joint_id])
        d.qpos[box_qpos_adr : box_qpos_adr + 3] = pos
        d.qpos[box_qpos_adr + 3 : box_qpos_adr + 7] = quat
        if box_qvel_adr >= 0:
            d.qvel[box_qvel_adr : box_qvel_adr + 6] = 0.0

    def set_table_positions(table_1_pos, table_2_pos):
        d.mocap_pos[ref_mocap_ids["plane_1"]] = table_1_pos
        d.mocap_pos[ref_mocap_ids["plane_2"]] = table_2_pos

    def reset_env_ref(op, oq, tp_1, tp_2):
        reset_robot_pose()
        set_object_pose(op, oq)
        set_table_positions(tp_1, tp_2)
        mujoco.mj_step(m, d)
    
    def reset_env():
        reset_replan_session()
        reset_robot_pose()
        set_object_pose(init_pos)
        table_z_offset = float(contactflow_policy.box_dims[2]) + 0.005
        table_offset = np.array([0.0, 0.0, table_z_offset], dtype=np.float32)
        set_table_positions(init_pos - table_offset, goal_pos - table_offset)
        mujoco.mj_step(m, d)

    def request_skill(command, reset_fn=None) -> bool:
        state_cmd.skill_cmd = command
        if reset_fn is not None:
            reset_fn()
        return True

    def handle_joystick() -> tuple[bool, bool]:
        if joystick.is_button_pressed(JoystickButton.SELECT):
            return False, False

        reset_counter = False
        joystick.update()
        if joystick.is_button_released(JoystickButton.L3):
            state_cmd.skill_cmd = FSMCommand.PASSIVE
        if joystick.is_button_released(JoystickButton.START):
            state_cmd.skill_cmd = FSMCommand.POS_RESET
        if joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.R1):
            state_cmd.skill_cmd = FSMCommand.LOCO
        if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
            state_cmd.skill_cmd = FSMCommand.SKILL_3
        if joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.L1):
            reset_replan_session()
            reset_fn_by_source = {
                "CFgen": reset_env,
            }
            reset_counter = request_skill(
                FSMCommand.SKILL_OmniContact,
                reset_fn_by_source.get(contactflow_policy.reference_source),
            )

        if FSM_controller.cur_policy.name == FSMStateName.LOCOMODE or state_cmd.skill_cmd == FSMCommand.LOCO:
            max_lin_vel = 0.5
            max_ang_vel = 1.0
            deadzone = 0.05
            ax_x = -joystick.get_axis_value(1)
            ax_y = -joystick.get_axis_value(0)
            ax_z = -joystick.get_axis_value(3)
            ax_x = ax_x if abs(ax_x) > deadzone else 0.0
            ax_y = ax_y if abs(ax_y) > deadzone else 0.0
            ax_z = ax_z if abs(ax_z) > deadzone else 0.0

            target_vx = float(np.clip(ax_x * max_lin_vel, -0.4, max_lin_vel))
            target_vy = float(np.clip(ax_y * max_lin_vel, -0.4, 0.4))
            target_wz = float(np.clip(ax_z * max_ang_vel, -1.57, 1.57))

            state_cmd.vel_cmd[0] = 2.0 * (target_vx - (-0.4)) / (0.7 - (-0.4)) - 1.0
            state_cmd.vel_cmd[1] = 2.0 * (target_vy - (-0.4)) / (0.4 - (-0.4)) - 1.0
            state_cmd.vel_cmd[2] = 2.0 * (target_wz - (-1.57)) / (1.57 - (-1.57)) - 1.0

        return True, reset_counter

    odom_calibration = {"initial_pos_xy": None, "initial_pos_z": None, "initial_yaw_quat": None}
    vision_cache = {"last_pos": None, "last_quat": None}

    def sync_robot_state():
        if real_robot is not None:
            res = real_robot.get_robot_state()
            if res is not None:
                q, dq, quat, gyro, base_pos, lin_vel = res

                # 修复真机无SLAM时的坐标漂移与沉地问题：
                # 1. 若底层未给出物理绝对位姿(全0)，默认采用物理站立高度(m.qpos0[:3])；
                # 若接收到了实时里程计，则锁定开机第一帧的XY与Z作为坐标原点，保留后续XY位移并自动校准绝对Z站立高度。
                if np.allclose(base_pos, 0.0):
                    base_pos = m.qpos0[:3].copy()
                else:
                    if odom_calibration["initial_pos_xy"] is None or odom_calibration.get("initial_pos_z") is None:
                        odom_calibration["initial_pos_xy"] = base_pos[:2].copy()
                        odom_calibration["initial_pos_z"] = float(base_pos[2])
                        print(f"\n[Odom Calibration] 🛰️ 锁定初始位移锚点 XY: [{odom_calibration['initial_pos_xy'][0]:.3f}, {odom_calibration['initial_pos_xy'][1]:.3f}], Z: {odom_calibration['initial_pos_z']:.3f}m")
                    base_pos[0] -= odom_calibration["initial_pos_xy"][0]
                    base_pos[1] -= odom_calibration["initial_pos_xy"][1]
                    if odom_calibration.get("initial_pos_z") is not None:
                        base_pos[2] = (base_pos[2] - odom_calibration["initial_pos_z"]) + m.qpos0[2]

                # 2. 仅在首次开机时锁定并移除初始全局偏航角(Yaw Offset)，但在后续运动过程中保留真实的转弯角！
                # 这样既能对齐物理工作空间，又能确保当机器人在行进中发生转向或受扰动时，RL 策略能实时感知自身朝向变化，解决向右跑偏且不回正的问题！
                if odom_calibration["initial_yaw_quat"] is None:
                    odom_calibration["initial_yaw_quat"] = yaw_quat(quat).astype(np.float32)
                    print(f"[Odom Calibration] 🧭 锁定开机第一帧朝向四元数: [{odom_calibration['initial_yaw_quat'][0]:.3f}, {odom_calibration['initial_yaw_quat'][1]:.3f}, {odom_calibration['initial_yaw_quat'][2]:.3f}, {odom_calibration['initial_yaw_quat'][3]:.3f}]")
                quat_aligned = quat_mul(quat_conjugate(odom_calibration["initial_yaw_quat"]), quat).astype(np.float32)
                quat_aligned /= max(float(np.linalg.norm(quat_aligned)), 1e-8)

                state_cmd.q = q.copy()
                state_cmd.dq = dq.copy()
                state_cmd.gravity_ori = get_gravity_orientation(quat).copy()
                state_cmd.base_pos = base_pos.copy()
                state_cmd.base_quat = quat_aligned.copy()
                state_cmd.ang_vel = gyro.copy()
                state_cmd.lin_vel = lin_vel.copy()

                d.qpos[:3] = base_pos
                d.qpos[3:7] = quat_aligned
                d.qpos[7 : 7 + num_joints] = q
                d.qvel[:3] = lin_vel
                d.qvel[3:6] = gyro
                d.qvel[6 : 6 + num_joints] = dq
                mujoco.mj_forward(m, d)
        else:
            # 【完全对齐真机硬件 IMU 与里程计分工】
            # 1. 机器人身体姿态 quat 和角速度始终来自物理机身 IMU (d.qpos[3:7] / gyro)，切勿使用带 2.3° 机械倾角的雷达安装框 (mid360_quat) 覆盖，否则会导致重心误判与往后倒退！
            quat = d.qpos[3:7].copy()
            # 2. 里程计位置 base_pos 读取当前物理位置 (d.qpos[:3])，并执行对齐校准
            # 彻底排除由于高位雷达在躯干俯仰(拥抱纸箱)时引起的胸晃位移和竖直下沉杠杆臂干扰！
            raw_odom_pos = d.qpos[:3].copy().astype(np.float32)

            base_pos = raw_odom_pos.copy()
            if odom_calibration["initial_pos_xy"] is None or odom_calibration.get("initial_pos_z") is None:
                odom_calibration["initial_pos_xy"] = base_pos[:2].copy()
                odom_calibration["initial_pos_z"] = float(base_pos[2])
                print(f"\n[Sim Odom Calibration] 🛰️ 仿真模式锁定雷达里程计锚点 XY: [{odom_calibration['initial_pos_xy'][0]:.3f}, {odom_calibration['initial_pos_xy'][1]:.3f}], 原始雷达Z: {odom_calibration['initial_pos_z']:.3f}m -> 映射对齐至目标站立高: {m.qpos0[2]:.3f}m")
            base_pos[0] -= odom_calibration["initial_pos_xy"][0]
            base_pos[1] -= odom_calibration["initial_pos_xy"][1]
            if odom_calibration.get("initial_pos_z") is not None:
                # 无论雷达/传感器起点在 1.26m、-0.45m 还是 0.0m，都通过 (当前Z - 初始Z) + m.qpos0[2] 映射到物理目标高度 0.77m
                base_pos[2] = (base_pos[2] - odom_calibration["initial_pos_z"]) + m.qpos0[2]

            if odom_calibration["initial_yaw_quat"] is None:
                odom_calibration["initial_yaw_quat"] = yaw_quat(quat).astype(np.float32)
                print(f"[Sim Odom Calibration] 🧭 仿真模式锁定开机第一帧朝向四元数: [{odom_calibration['initial_yaw_quat'][0]:.3f}, {odom_calibration['initial_yaw_quat'][1]:.3f}, {odom_calibration['initial_yaw_quat'][2]:.3f}, {odom_calibration['initial_yaw_quat'][3]:.3f}]")
            quat_aligned = quat_mul(quat_conjugate(odom_calibration["initial_yaw_quat"]), quat).astype(np.float32)
            quat_aligned /= max(float(np.linalg.norm(quat_aligned)), 1e-8)

            state_cmd.q = d.qpos[7 : 7 + num_joints].copy()
            state_cmd.dq = d.qvel[6 : 6 + num_joints].copy()
            state_cmd.gravity_ori = get_gravity_orientation(quat).copy()
            state_cmd.base_pos = base_pos.copy()
            state_cmd.base_quat = quat_aligned.copy()
            state_cmd.ang_vel = d.qvel[3:6].copy()
            state_cmd.lin_vel = d.qvel[0:3].copy()
        sync_object_state()

    def set_mocap_pose(mocap_name: str, pose):
        d.mocap_pos[ref_mocap_ids[mocap_name]] = pose[0:3]
        d.mocap_quat[ref_mocap_ids[mocap_name]] = pose[3:7]

    def set_freejoint_pose(qpos_adr: int, qvel_adr: int, pose7):
        if qpos_adr < 0:
            return
        pose7 = np.asarray(pose7, dtype=np.float32).reshape(7)
        d.qpos[qpos_adr : qpos_adr + 7] = pose7
        if qvel_adr >= 0:
            d.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def update_ghost_robot_visualization():
        if ghost_robot_qpos_adr < 0:
            return

        ref_joint_pos = getattr(contactflow_policy, "ref_joint_pos", None)
        if (
            ref_joint_pos is not None
            and hasattr(contactflow_policy, "ref_base_pos")
            and hasattr(contactflow_policy, "ref_base_quat")
        ):
            dof_pos = np.asarray(ref_joint_pos, dtype=np.float32)
            if dof_pos.ndim != 2 or len(dof_pos) == 0:
                return
            lab2mj_policy = getattr(contactflow_policy, "lab2mj", None)
            if lab2mj_policy is not None and dof_pos.shape[1] == len(lab2mj_policy):
                dof_pos = dof_pos[:, lab2mj_policy]
            curr_idx = min(contactflow_policy.counter_step, len(dof_pos) - 1)
            ghost_base_pose = np.concatenate(
                [
                    contactflow_policy.ref_base_pos[curr_idx],
                    contactflow_policy.ref_base_quat[curr_idx],
                ],
                axis=0,
            )
            ghost_q = dof_pos[curr_idx]
        else:
            ghost_base_pose = np.concatenate([state_cmd.base_pos, state_cmd.base_quat], axis=0)
            ghost_q = state_cmd.q

        set_freejoint_pose(ghost_robot_qpos_adr, ghost_robot_qvel_adr, ghost_base_pose)
        ghost_q = np.asarray(ghost_q, dtype=np.float32).reshape(-1)
        count = min(len(ghost_robot_joint_qpos_adrs), ghost_q.shape[0])
        if count <= 0:
            return
        valid = ghost_robot_joint_qpos_adrs[:count] >= 0
        d.qpos[ghost_robot_joint_qpos_adrs[:count][valid]] = ghost_q[:count][valid]

    def update_reference_visualization():
        set_mocap_pose("l_wrist", wrist_state[0:7])
        set_mocap_pose("r_wrist", wrist_state[7:14])
        set_mocap_pose("torso", torso_goal)
        set_mocap_pose("l_ankle", l_ankle_goal)
        set_mocap_pose("r_ankle", r_ankle_goal)
        update_ghost_robot_visualization()

        for geom, contact in zip(ref_contact_geom_ids, contact_state):
            m.geom_rgba[geom] = reference_red if contact >= 0.5 else reference_yellow

    log_file_path = os.path.join(PROJECT_ROOT, "object_pose_logging.txt")
    log_step_counter = 0
    has_entered_loco = False
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("# OmniContact 物体位姿追踪日志 (每20帧记录一次)\n")
        f.write("# ------------------------------------------------------------------\n")
        f.write("# 核查说明与对应连杆关系：\n")
        f.write("# 1. [视觉原始接收] 相对连杆/坐标系：【d435_camera_frame】(头部相机光学中心)\n")
        f.write("# 2. [上层 CFgen 输入] 相对连杆/坐标系：【pelvis】(Unitree G1 根连杆/骨盆浮动基座)\n")
        f.write("# 3. [底层 RL 策略输入] 相对连杆/坐标系：【torso_link】(胸口连杆，已剔除俯仰与翻滚保留水平yaw朝向)\n")
        f.write("# ------------------------------------------------------------------\n\n")

    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and Running:
                try:
                    Running, should_reset_counter = handle_joystick()
                    if should_reset_counter:
                        sim_counter = 0

                    step_start = time.time()
                    if sim_counter % control_decimation == 0:
                        sync_robot_state()
                        prev_policy = FSM_controller.cur_policy

                        # 【核心防突变机制】在 FSM_controller.run() 真正执行策略切换（内部将自动调用 contactflow_policy.enter()）前，
                        # 提前检测前一个策略状态是否准备切换到 omnicontact (SKILL_OmniContact)。
                        # 若检测到准备进入 omnicontact，则在 enter() 之前先清零校准里程计并完成状态同步。
                        # 这样当随后 FSM_controller.run() 内部执行 OmniContact.enter() 时，
                        # 其获取到的 state_cmd.base_pos 将完全准确对准当前真实归零坐标 [0.0, 0.0, 0.77]，
                        # 同时在进入策略后的首帧 run() (counter_step == 0) 中，将首帧静态特征 curr_obs_prop 广播复制填满整个 5 步 obs_history_buffer，
                        # 既抹除所有切模式前的旧里程计与动作记录，又保证时序帧差速度差完全为 0 (消除了前4帧全0导致的37.5m/s虚假速度脉冲)，
                        # 彻底解决了实机切模式瞬间神经网络产生假想超高速碰撞、往前猛踹一脚导致失稳倒地的问题！
                        is_switching_to_omni = (
                            prev_policy is not contactflow_policy
                            and state_cmd.skill_cmd == FSMCommand.SKILL_OmniContact
                        )
                        if is_switching_to_omni:
                            print("\n[Odom Calibration] 检测到准备进入 omnicontact 模式！在调用策略 enter() 前完成里程计校准锁零...")
                            if real_robot is not None and hasattr(real_robot, "subscribe_odom"):
                                real_robot.subscribe_odom("/lio/odom")
                            odom_calibration["initial_pos_xy"] = None
                            odom_calibration["initial_pos_z"] = None
                            odom_calibration["initial_yaw_quat"] = None
                            vision_cache["last_pos"] = None
                            vision_cache["last_quat"] = None
                            sync_robot_state()
                            forward_dist = float(args.goal_pos[0]) if hasattr(args, "goal_pos") and args.goal_pos is not None else 2.0
                            lateral_dist = float(args.goal_pos[1]) if hasattr(args, "goal_pos") and args.goal_pos is not None else 0.0
                            goal_pos[0] = forward_dist
                            goal_pos[1] = lateral_dist
                            if hasattr(contactflow_policy, "box_dims"):
                                goal_pos[2] = float(contactflow_policy.box_dims[2])
                            print(f"[Odom Calibration] 切换前已完成里程计锚点归零锁定，正前方目标点设置: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
                            if hasattr(contactflow_policy, "goal_pos"):
                                contactflow_policy.goal_pos[:] = goal_pos
                            if hasattr(contactflow_policy, "goal_pos_override") and contactflow_policy.goal_pos_override is not None:
                                contactflow_policy.goal_pos_override[:] = goal_pos

                        FSM_controller.run()

                        if FSM_controller.cur_policy.name in [FSMStateName.LOCOMODE, FSMStateName.SKILL_OmniContact] or state_cmd.skill_cmd == FSMCommand.LOCO:
                            if not has_entered_loco:
                                has_entered_loco = True
                                if real_robot is not None and hasattr(real_robot, "subscribe_odom"):
                                    print("\n[Odom] 机器人已切换至 LOCO 站立/工作模式，开始订阅 ROS2 /lio/odom 里程计并重置校准锚点！")
                                    real_robot.subscribe_odom("/lio/odom")
                                    odom_calibration["initial_pos_xy"] = None
                                    odom_calibration["initial_pos_z"] = None
                                    odom_calibration["initial_yaw_quat"] = None

                        # 保底兜底：若前序非 checkChange 触发而是直接手动改变 cur_policy 切换至 omnicontact，则确保同样重新调用 enter() 与归零，绝不重复叠加 run()
                        if prev_policy is not contactflow_policy and FSM_controller.cur_policy is contactflow_policy and not is_switching_to_omni:
                            print("\n[Odom Calibration] 兜底检测到已直接切换至 omnicontact，进行重置校准并执行 enter() 归零缓冲...")
                            if real_robot is not None and hasattr(real_robot, "subscribe_odom"):
                                real_robot.subscribe_odom("/lio/odom")
                            odom_calibration["initial_pos_xy"] = None
                            odom_calibration["initial_pos_z"] = None
                            odom_calibration["initial_yaw_quat"] = None
                            vision_cache["last_pos"] = None
                            vision_cache["last_quat"] = None
                            sync_robot_state()
                            forward_dist = float(args.goal_pos[0]) if hasattr(args, "goal_pos") and args.goal_pos is not None else 2.0
                            lateral_dist = float(args.goal_pos[1]) if hasattr(args, "goal_pos") and args.goal_pos is not None else 0.0
                            goal_pos[0] = forward_dist
                            goal_pos[1] = lateral_dist
                            if hasattr(contactflow_policy, "box_dims"):
                                goal_pos[2] = float(contactflow_policy.box_dims[2])
                            if hasattr(contactflow_policy, "goal_pos"):
                                contactflow_policy.goal_pos[:] = goal_pos
                            if hasattr(contactflow_policy, "goal_pos_override") and contactflow_policy.goal_pos_override is not None:
                                contactflow_policy.goal_pos_override[:] = goal_pos
                            contactflow_policy.enter()
                            contactflow_policy.run()

                        maybe_closed_loop_replan()

                        policy_output_action = policy_output.actions.copy()
                        kps = policy_output.kps.copy()
                        kds = policy_output.kds.copy()
                        wrist_state = policy_output.wrist_goal.copy()
                        contact_state = policy_output.contact.copy()
                        torso_goal = policy_output.torso_goal.copy()
                        l_ankle_goal = policy_output.l_ankle_goal.copy()
                        r_ankle_goal = policy_output.r_ankle_goal.copy()

                        log_step_counter += 1
                        is_omni = (FSM_controller.cur_policy is contactflow_policy)
                        log_freq = 20 if is_omni else 100
                        if log_step_counter % log_freq == 0:
                            cam_pos_str = "None"
                            cam_quat_str = "None"
                            if vision_receiver is not None and vision_receiver.obj_pose_cv is not None:
                                c_pos = vision_receiver.obj_pose_cv.get("pos", None)
                                c_quat = vision_receiver.obj_pose_cv.get("quat", None)
                                if c_pos is not None:
                                    cam_pos_str = f"[{c_pos[0]:.4f}, {c_pos[1]:.4f}, {c_pos[2]:.4f}]"
                                if c_quat is not None:
                                    cam_quat_str = f"[{c_quat[0]:.4f}, {c_quat[1]:.4f}, {c_quat[2]:.4f}, {c_quat[3]:.4f}]"
                            if cam_pos_str == "None":
                                site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "d435_camera_frame")
                                if site_id >= 0:
                                    p_cam_w = d.site_xpos[site_id]
                                    R_cam_w = d.site_xmat[site_id].reshape(3, 3)
                                    p_cam_rel = R_cam_w.T @ (state_cmd.obj_pos - p_cam_w)
                                    cam_pos_str = f"[{p_cam_rel[0]:.4f}, {p_cam_rel[1]:.4f}, {p_cam_rel[2]:.4f}] (仿真GT)"
                                    q_cam_w = np.zeros(4, dtype=np.float64)
                                    mujoco.mju_mat2Quat(q_cam_w, R_cam_w.astype(np.float64).flatten())
                                    inv_q_cam = quat_conjugate(q_cam_w.astype(np.float32))
                                    q_cam_rel = quat_mul(inv_q_cam, state_cmd.obj_quat)
                                    cam_quat_str = f"[{q_cam_rel[0]:.4f}, {q_cam_rel[1]:.4f}, {q_cam_rel[2]:.4f}, {q_cam_rel[3]:.4f}] (仿真GT)"

                            inv_base = quat_conjugate(state_cmd.base_quat)
                            upper_pos_rel = quat_apply(inv_base, state_cmd.obj_pos - state_cmd.base_pos)
                            upper_quat_rel = quat_mul(inv_base, state_cmd.obj_quat)

                            if hasattr(contactflow_policy, "torso_pos") and hasattr(contactflow_policy, "torso_quat"):
                                torso_heading = yaw_quat(contactflow_policy.torso_quat)
                                lower_pos_rel, lower_quat_rel = subtract_frame_transforms(
                                    contactflow_policy.torso_pos, torso_heading, state_cmd.obj_pos, state_cmd.obj_quat
                                )
                            else:
                                torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
                                t_pos = d.xpos[torso_id] if torso_id >= 0 else state_cmd.base_pos
                                t_quat = d.xquat[torso_id] if torso_id >= 0 else state_cmd.base_quat
                                lower_pos_rel, lower_quat_rel = subtract_frame_transforms(
                                    t_pos, yaw_quat(t_quat), state_cmd.obj_pos, state_cmd.obj_quat
                                )

                            log_msg = (
                                f"=== Step: {log_step_counter} (Sim Counter: {sim_counter}, Time: {time.time()-step_start:.3f}s) ===\n"
                                f"[1. 视觉原始接收] 相对连杆：【d435_camera_frame】(头部相机光学系)\n"
                                f"   - Pos: {cam_pos_str}\n"
                                f"   - Quat: {cam_quat_str}\n"
                                f"[2. 上层 CFgen 输入] 相对连杆：【pelvis】(G1骨盆浮动基座，第0号连杆)\n"
                                f"   - Pos: [{upper_pos_rel[0]:.4f}, {upper_pos_rel[1]:.4f}, {upper_pos_rel[2]:.4f}] (m)\n"
                                f"   - Quat: [{upper_quat_rel[0]:.4f}, {upper_quat_rel[1]:.4f}, {upper_quat_rel[2]:.4f}, {upper_quat_rel[3]:.4f}]\n"
                                f"[3. 底层 RL 策略输入] 相对连杆：【torso_link】(胸口连杆，剔除俯仰与翻滚后的水平Yaw朝向系)\n"
                                f"   - Pos: [{lower_pos_rel[0]:.4f}, {lower_pos_rel[1]:.4f}, {lower_pos_rel[2]:.4f}] (m)\n"
                                f"   - Quat: [{lower_quat_rel[0]:.4f}, {lower_quat_rel[1]:.4f}, {lower_quat_rel[2]:.4f}, {lower_quat_rel[3]:.4f}]\n"
                                f"------------------------------------------------------------------\n"
                            )
                            with open(log_file_path, "a", encoding="utf-8") as f:
                                f.write(log_msg)

                    if real_robot is not None:
                        real_robot.send_joint_commands(
                            policy_output_action,
                            kps,
                            kds,
                        )
                        robot_qpos = state_cmd.q
                        robot_qvel = state_cmd.dq
                        tau = pd_control(policy_output_action, robot_qpos, kps, np.zeros_like(kps), robot_qvel, kds, torque_limit_mj=torque_limit_mj)
                        d.ctrl[:] = tau
                        mujoco.mj_forward(m, d)
                    else:
                        robot_qpos = d.qpos[7 : 7 + num_joints]
                        robot_qvel = d.qvel[6 : 6 + num_joints]
                        tau = pd_control(
                            policy_output_action,
                            robot_qpos,
                            kps,
                            np.zeros_like(kps),
                            robot_qvel,
                            kds,
                            torque_limit_mj=torque_limit_mj,
                        )
                        d.ctrl[:] = tau
                        mujoco.mj_step(m, d)
                    if sim_renderer is not None and vision_receiver is not None:
                        sim_renderer.update_scene(d, camera="depth_camera")
                        rgb_img = sim_renderer.render()
                        vision_receiver.publish_image(rgb_img)
                    
                    update_reference_visualization()

                    sim_counter += 1
                except ValueError as e:
                    print(f"ValueError detected: {str(e)}")
                    print(f"Debug Info: num_joints={num_joints}, qpos.shape={d.qpos.shape}, qvel.shape={d.qvel.shape}")
                    break
                
                viewer.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        if real_robot is not None:
            print("[deploy] 正在退出程序，启动安全阻尼保护 (参考 Sonic CreateDampingCommand, kp=0, kd=8.0)...")
            real_robot.stop()
        
