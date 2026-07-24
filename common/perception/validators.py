"""
Validation and diagnostic functions for perception measurements, quaternions, timestamps, and residuals.

Strict Constraints:
- Configurable thresholds for stamp_delta and world residuals.
- /aruco/box_pose_pelvis is raw diagnostic ONLY and does NOT trigger validation failure.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .frame_definitions import Pose, normalize_quat, quat_mul, quat_conjugate


@dataclass
class ValidationConfig:
    max_stamp_delta: float = 0.2             # 200ms
    max_world_residual_pos: float = 0.05     # 5 cm
    max_world_residual_rot_deg: float = 15.0 # 15 degrees
    max_z_offset_drift: float = 0.08      # 8 cm


class PoseValidator:
    def __init__(self, config: ValidationConfig = ValidationConfig()):
        self.config = config

    def validate_quaternion(self, quat: np.ndarray) -> Tuple[bool, str]:
        if quat is None or not np.all(np.isfinite(quat)) or len(quat) != 4:
            return False, "quaternion_non_finite_or_invalid_shape"
        norm = float(np.linalg.norm(quat))
        if abs(norm - 1.0) > 0.1 or norm < 1e-4:
            return False, f"quaternion_norm_invalid_{norm:.4f}"
        return True, "valid_quaternion"

    def validate_timestamps(
        self,
        measurement_timestamp: float,
        robot_state_timestamp: float,
    ) -> Tuple[bool, float, str]:
        if measurement_timestamp <= 0.0 or robot_state_timestamp <= 0.0:
            # If timestamps are not provided, pass with warning
            return True, 0.0, "timestamp_not_provided"
        stamp_delta = abs(measurement_timestamp - robot_state_timestamp)
        if stamp_delta > self.config.max_stamp_delta:
            return False, stamp_delta, f"stamp_delta_exceeds_threshold_{stamp_delta:.3f}s"
        return True, stamp_delta, "timestamp_valid"

    def compute_world_residuals(
        self,
        primary_world_pose: Pose,
        secondary_world_pose: Optional[Pose],
    ) -> Tuple[bool, float, float, str]:
        """
        Computes position residual (L2 norm) and angle residual (degrees) between
        primary world pose (e.g. torso-derived) and secondary world pose (e.g. camera-derived).
        """
        if secondary_world_pose is None:
            return True, 0.0, 0.0, "secondary_pose_none"

        pos_diff = float(np.linalg.norm(primary_world_pose.pos - secondary_world_pose.pos))

        # Relative quaternion: q_diff = q_primary * q_secondary^{-1}
        q_rel = quat_mul(primary_world_pose.quat, quat_conjugate(secondary_world_pose.quat))
        q_rel = normalize_quat(q_rel)
        w = float(np.clip(abs(q_rel[0]), 0.0, 1.0))
        angle_deg = float(np.degrees(2.0 * np.arccos(w)))

        valid = True
        reason = "residuals_within_threshold"
        if pos_diff > self.config.max_world_residual_pos:
            valid = False
            reason = f"pos_residual_exceeds_{pos_diff:.4f}m"
        elif angle_deg > self.config.max_world_residual_rot_deg:
            valid = False
            reason = f"rot_residual_exceeds_{angle_deg:.1f}deg"

        return valid, pos_diff, angle_deg, reason
