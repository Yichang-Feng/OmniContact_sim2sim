"""
Perception and coordinate transformation package for OmniContact.
"""
from .frame_definitions import (
    Pose,
    FRAME_WORLD,
    FRAME_PELVIS_LINK,
    FRAME_TORSO_LINK,
    FRAME_TORSO_YAW,
    FRAME_CAMERA_LINK,
    FRAME_CAMERA_OPTICAL,
    T_OPTICAL_TO_CAMERA_LINK_POS,
    T_OPTICAL_TO_CAMERA_LINK_QUAT,
    compose_frame_transforms,
)
from .robot_state_provider import RobotState, RobotStateProvider
from .object_pose_source import (
    ObjectPoseMeasurement,
    ObjectPoseSource,
    Ros2ObjectPoseSource,
    SimObjectPoseSource,
)
from .validators import ValidationConfig, PoseValidator
from .object_pose_pipeline import (
    PerceptionContext,
    ObjectPoseState,
    ObjectPosePipeline,
)

__all__ = [
    "Pose",
    "FRAME_WORLD",
    "FRAME_PELVIS_LINK",
    "FRAME_TORSO_LINK",
    "FRAME_TORSO_YAW",
    "FRAME_CAMERA_LINK",
    "FRAME_CAMERA_OPTICAL",
    "T_OPTICAL_TO_CAMERA_LINK_POS",
    "T_OPTICAL_TO_CAMERA_LINK_QUAT",
    "compose_frame_transforms",
    "RobotState",
    "RobotStateProvider",
    "ObjectPoseMeasurement",
    "ObjectPoseSource",
    "Ros2ObjectPoseSource",
    "SimObjectPoseSource",
    "ValidationConfig",
    "PoseValidator",
    "PerceptionContext",
    "ObjectPoseState",
    "ObjectPosePipeline",
]
