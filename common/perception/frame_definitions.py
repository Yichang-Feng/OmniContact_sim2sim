"""
Frame definitions, Pose data structure, and coordinate transformation utilities for OmniContact perception pipeline.

Strict Constraint:
- Re-exports core quaternion math from `common.utils`.
- Does NOT reimplement quaternion math.
"""

from dataclasses import dataclass
import numpy as np

# Re-export core math from common.utils (Single Source of Truth for Quaternion Math)
from common.utils import (
    quat_mul,
    quat_apply,
    quat_conjugate,
    yaw_quat,
    subtract_frame_transforms,
    matrix_from_quat,
    normalize_quat,
)

# Standard Frame Identifiers
FRAME_WORLD = "policy_world"
FRAME_PELVIS_LINK = "pelvis_link"
FRAME_TORSO_LINK = "torso_link"
FRAME_TORSO_YAW = "torso_yaw"
FRAME_CAMERA_LINK = "camera_link"
FRAME_CAMERA_OPTICAL = "camera_optical_frame"

# Extrinsic transform from OpenCV optical frame (x-right, y-down, z-forward)
# to MuJoCo camera link frame (x-forward, y-left, z-up)
T_OPTICAL_TO_CAMERA_LINK_POS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
T_OPTICAL_TO_CAMERA_LINK_QUAT = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)  # [w, x, y, z]


@dataclass
class Pose:
    pos: np.ndarray   # shape (3,), float32
    quat: np.ndarray  # shape (4,), float32, format: [w, x, y, z]

    def copy(self) -> "Pose":
        return Pose(
            pos=self.pos.copy().astype(np.float32),
            quat=self.quat.copy().astype(np.float32),
        )


def compose_frame_transforms(
    t01: np.ndarray,
    q01: np.ndarray,
    t12: np.ndarray,
    q12: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compose two coordinate frame transformations: T_02 = T_01 * T_12

    Args:
        t01: Position of frame 1 in frame 0 (3,)
        q01: Quaternion orientation of frame 1 in frame 0 [w, x, y, z] (4,)
        t12: Position of frame 2 in frame 1 (3,)
        q12: Quaternion orientation of frame 2 in frame 1 [w, x, y, z] (4,)

    Returns:
        (t02, q02): Position and quaternion of frame 2 in frame 0.
    """
    q02 = normalize_quat(quat_mul(q01, q12))
    t02 = t01 + quat_apply(q01, t12)
    return t02.astype(np.float32), q02.astype(np.float32)
