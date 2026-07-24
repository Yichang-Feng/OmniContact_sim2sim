"""
RobotStateProvider: Single Source of Truth for Robot State and Forward Kinematics Poses.

Strict Constraints:
- Perception, FK, and Policy inputs must consume the exact same `RobotState`.
- Does NOT allow perception pipeline to read MuJoCo `data.xpos/xquat` directly as authoritative coordinate transformation sources.
- Uses a unified filtering strategy for base pose (e.g. filtered `state_cmd.base_pos` and `base_quat`).
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from .frame_definitions import (
    Pose,
    FRAME_PELVIS_LINK,
    FRAME_TORSO_LINK,
    FRAME_TORSO_YAW,
    FRAME_CAMERA_LINK,
    yaw_quat,
)


@dataclass
class RobotState:
    timestamp: float
    q: np.ndarray                 # joint positions (29,)
    dq: np.ndarray                # joint velocities (29,)
    base_pos: np.ndarray          # policy world base position (3,)
    base_quat: np.ndarray         # policy world base quaternion [w, x, y, z] (4,)
    link_poses: Dict[str, Pose]   # link frame poses in policy_world


class RobotStateProvider:
    """
    Computes and maintains unified robot states and link forward kinematics poses.
    """
    def __init__(self, kinematics):
        self.kinematics = kinematics
        self.last_state: Optional[RobotState] = None

    def update_from_state_cmd(self, state_cmd, timestamp: float = 0.0) -> RobotState:
        """
        Update RobotState using filtered base pose (state_cmd.base_pos, state_cmd.base_quat)
        and joint states (state_cmd.q). Computes forward kinematics for all relevant links.
        """
        base_pos = np.asarray(state_cmd.base_pos, dtype=np.float32).copy()
        base_quat = np.asarray(state_cmd.base_quat, dtype=np.float32).copy()
        q = np.asarray(state_cmd.q, dtype=np.float32).copy()
        dq = np.asarray(getattr(state_cmd, "dq", np.zeros(29)), dtype=np.float32).copy()

        # Execute FK via MujocoKinematics using unified base_pos and base_quat
        fk_info = self.kinematics.forward(q, base_pos, base_quat)

        pelvis_pos = fk_info["pelvis"]["pos"].astype(np.float32)
        pelvis_quat = fk_info["pelvis"]["quat"].astype(np.float32)
        pelvis_pose = Pose(pos=pelvis_pos, quat=pelvis_quat)

        torso_pos = fk_info["torso_link"]["pos"].astype(np.float32)
        torso_quat = fk_info["torso_link"]["quat"].astype(np.float32)
        torso_pose = Pose(pos=torso_pos, quat=torso_quat)

        # torso_yaw: origin co-located with torso_link, orientation has yaw only (pitch/roll zeroed out)
        torso_yaw_q = yaw_quat(torso_quat).astype(np.float32)
        torso_yaw_pose = Pose(pos=torso_pos.copy(), quat=torso_yaw_q)

        # Camera link pose from FK
        if "d435_camera" in fk_info:
            cam_pos = fk_info["d435_camera"]["pos"].astype(np.float32)
            cam_quat = fk_info["d435_camera"]["quat"].astype(np.float32)
        else:
            cam_pos = torso_pos.copy()
            cam_quat = torso_quat.copy()
        camera_pose = Pose(pos=cam_pos, quat=cam_quat)

        link_poses = {
            FRAME_PELVIS_LINK: pelvis_pose,
            FRAME_TORSO_LINK: torso_pose,
            FRAME_TORSO_YAW: torso_yaw_pose,
            FRAME_CAMERA_LINK: camera_pose,
        }

        self.last_state = RobotState(
            timestamp=timestamp,
            q=q,
            dq=dq,
            base_pos=base_pos,
            base_quat=base_quat,
            link_poses=link_poses,
        )
        return self.last_state

    def get_link_pose(self, frame_id: str) -> Optional[Pose]:
        if self.last_state is None or frame_id not in self.last_state.link_poses:
            return None
        return self.last_state.link_poses[frame_id].copy()
