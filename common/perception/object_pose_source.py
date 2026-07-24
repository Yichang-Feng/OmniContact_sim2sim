"""
ObjectPoseSource abstractions for real robot ROS2 topics and simulation GT measurements.

Strict Constraints:
- Ros2ObjectPoseSource handles conversion from optical frame to camera_link/torso_link inside the adapter.
- If `/aruco/box_pose_torso_link` is already in `torso_link` frame, NO optical transform is applied!
- `/aruco/box_pose_pelvis` is tagged as raw diagnostic only.
- All measurements must use header timestamps.
- SimObjectPoseSource must support noise, latency, frame drop, and odom reset simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .frame_definitions import (
    Pose,
    FRAME_TORSO_LINK,
    FRAME_CAMERA_LINK,
    FRAME_CAMERA_OPTICAL,
    T_OPTICAL_TO_CAMERA_LINK_POS,
    T_OPTICAL_TO_CAMERA_LINK_QUAT,
    compose_frame_transforms,
    subtract_frame_transforms,
    normalize_quat,
)


@dataclass
class ObjectPoseMeasurement:
    frame_id: str         # "torso_link", "camera_link", "pelvis_raw_diag"
    pos: np.ndarray       # relative position (3,)
    quat: np.ndarray      # relative quaternion [w, x, y, z] (4,)
    timestamp: float      # message header timestamp
    source_id: str        # "ros2", "sim_gt", "rosbag"
    valid: bool           # basic validity flag


class ObjectPoseSource(ABC):
    @abstractmethod
    def get_measurements(self) -> Dict[str, ObjectPoseMeasurement]:
        pass


class Ros2ObjectPoseSource(ObjectPoseSource):
    """
    Adapter for receiving raw ROS2/DDS topics and converting them to standardized Measurements.
    """
    def __init__(self):
        self.last_torso_measurement: Optional[ObjectPoseMeasurement] = None
        self.last_camera_measurement: Optional[ObjectPoseMeasurement] = None
        self.last_pelvis_measurement: Optional[ObjectPoseMeasurement] = None

    def update_raw_torso_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray, timestamp: float):
        """
        Input from /aruco/box_pose_torso_link.
        Already in torso_link frame! NO optical transform applied!
        """
        p = np.asarray(pos, dtype=np.float32).copy()
        q = normalize_quat(quat_wxyz)
        self.last_torso_measurement = ObjectPoseMeasurement(
            frame_id=FRAME_TORSO_LINK,
            pos=p,
            quat=q,
            timestamp=float(timestamp),
            source_id="ros2_torso_link",
            valid=True,
        )

    def update_raw_camera_pose(self, pos_opt: np.ndarray, quat_opt_wxyz: np.ndarray, timestamp: float):
        """
        Input from /aruco/box_pose (camera_optical_frame).
        Converts optical frame to camera_link frame using fixed extrinsics in adapter!
        """
        p_opt = np.asarray(pos_opt, dtype=np.float32)
        q_opt = normalize_quat(quat_opt_wxyz)

        # Apply extrinsic transform: Pose_in_camera_link = T_optical_to_cam * Pose_in_optical
        p_cam, q_cam = compose_frame_transforms(
            T_OPTICAL_TO_CAMERA_LINK_POS,
            T_OPTICAL_TO_CAMERA_LINK_QUAT,
            p_opt,
            q_opt,
        )
        self.last_camera_measurement = ObjectPoseMeasurement(
            frame_id=FRAME_CAMERA_LINK,
            pos=p_cam,
            quat=q_cam,
            timestamp=float(timestamp),
            source_id="ros2_camera_optical",
            valid=True,
        )

    def update_raw_pelvis_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray, timestamp: float):
        """
        Input from /aruco/box_pose_pelvis.
        Tagged as diagnostic only!
        """
        p = np.asarray(pos, dtype=np.float32).copy()
        q = normalize_quat(quat_wxyz)
        self.last_pelvis_measurement = ObjectPoseMeasurement(
            frame_id="pelvis_raw_diag",
            pos=p,
            quat=q,
            timestamp=float(timestamp),
            source_id="ros2_pelvis_raw_diag",
            valid=True,
        )

    def get_measurements(self) -> Dict[str, ObjectPoseMeasurement]:
        res = {}
        if self.last_torso_measurement is not None and self.last_torso_measurement.valid:
            res[FRAME_TORSO_LINK] = self.last_torso_measurement
        if self.last_camera_measurement is not None and self.last_camera_measurement.valid:
            res[FRAME_CAMERA_LINK] = self.last_camera_measurement
        if self.last_pelvis_measurement is not None and self.last_pelvis_measurement.valid:
            res["pelvis_raw_diag"] = self.last_pelvis_measurement
        return res


class SimObjectPoseSource(ObjectPoseSource):
    """
    Simulation measurement source generating synthetic measurements from MuJoCo GT,
    supporting latency, noise, frame drops, and timestamp offsets.
    """
    def __init__(
        self,
        noise_std_pos: float = 0.0,
        noise_std_rot_deg: float = 0.0,
        drop_rate: float = 0.0,
        delay_seconds: float = 0.0,
    ):
        self.noise_std_pos = noise_std_pos
        self.noise_std_rot_deg = noise_std_rot_deg
        self.drop_rate = drop_rate
        self.delay_seconds = delay_seconds
        self.last_measurements: Dict[str, ObjectPoseMeasurement] = {}

    def update_from_mujoco_gt(
        self,
        gt_obj_pos_world: np.ndarray,
        gt_obj_quat_world: np.ndarray,
        gt_torso_pos_world: np.ndarray,
        gt_torso_quat_world: np.ndarray,
        timestamp: float,
    ) -> Dict[str, ObjectPoseMeasurement]:
        """
        Calculates relative pose from GT object pose and GT torso pose,
        adds synthetic noise/delay/drop, and outputs standardized Measurement.
        """
        # Check frame drop
        if self.drop_rate > 0.0 and np.random.rand() < self.drop_rate:
            return self.last_measurements

        # Direct relative transform: GT object in GT torso frame
        rel_pos, rel_quat = subtract_frame_transforms(
            gt_torso_pos_world,
            gt_torso_quat_world,
            gt_obj_pos_world,
            gt_obj_quat_world,
        )

        # Add Gaussian noise if configured
        if self.noise_std_pos > 0.0:
            rel_pos = rel_pos + np.random.normal(0, self.noise_std_pos, size=3).astype(np.float32)

        stamp = timestamp - self.delay_seconds

        meas = ObjectPoseMeasurement(
            frame_id=FRAME_TORSO_LINK,
            pos=rel_pos.astype(np.float32),
            quat=normalize_quat(rel_quat),
            timestamp=float(stamp),
            source_id="sim_gt",
            valid=True,
        )
        self.last_measurements = {FRAME_TORSO_LINK: meas}
        return self.last_measurements

    def get_measurements(self) -> Dict[str, ObjectPoseMeasurement]:
        return self.last_measurements
