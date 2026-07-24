"""
ObjectPosePipeline: Core pipeline for unified object pose perception, coordinate transformation,
consistency validation, and safety strategy execution.

Strict Constraints:
- Primary measurement source is `torso_link` (or `camera_link`).
- Reconstructs `object_pose_world` in policy_world frame.
- Derives `obj_pos_pelvis` (true pelvis_link frame) and `obj_pos_torso_yaw` (torso_yaw frame).
- All relative poses originate from the EXACT SAME unified `object_pose_world`.
- Internally maintains `last_valid_state`, `held_T_pelvis_obj`, `held_T_torso_yaw_obj`, `held_object_locked`.
- Provides `reset()` method to clear cached states.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np

from .frame_definitions import (
    Pose,
    FRAME_WORLD,
    FRAME_PELVIS_LINK,
    FRAME_TORSO_LINK,
    FRAME_TORSO_YAW,
    FRAME_CAMERA_LINK,
    compose_frame_transforms,
    subtract_frame_transforms,
    normalize_quat,
)
from .robot_state_provider import RobotState
from .object_pose_source import ObjectPoseMeasurement
from .validators import ValidationConfig, PoseValidator


@dataclass
class PerceptionContext:
    manual_stage: int = 0
    current_phase: int = 11
    is_holding_object: bool = False
    allow_held_object_prior: bool = False
    box_half_dims: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.2], dtype=np.float32))


@dataclass
class ObjectPoseState:
    valid: bool
    source_id: str
    frame_id: str
    timestamp: float
    measurement_timestamp: float
    robot_state_timestamp: float
    stamp_delta: float
    valid_reason: str

    obj_pos_world: np.ndarray        # policy_world object position (3,)
    obj_quat_world: np.ndarray       # policy_world object quaternion [w, x, y, z] (4,)

    obj_pos_pelvis: np.ndarray       # relative to true pelvis_link frame (3,) -> for CF-Gen
    obj_quat_pelvis: np.ndarray      # relative quaternion in true pelvis_link frame (4,)

    obj_pos_torso_yaw: np.ndarray     # relative to torso_yaw frame (3,) -> for CF-Tracker
    obj_quat_torso_yaw: np.ndarray    # relative quaternion in torso_yaw frame (4,)

    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ObjectPosePipeline:
    def __init__(self, validation_config: ValidationConfig = ValidationConfig()):
        self.validator = PoseValidator(validation_config)
        self.last_valid_world_pose: Optional[Pose] = None
        self.last_valid_state: Optional[ObjectPoseState] = None

        # Internal Held-Object Prior attributes
        self.held_T_pelvis_obj: Optional[Pose] = None
        self.held_T_torso_yaw_obj: Optional[Pose] = None
        self.held_object_locked: bool = False

    def reset(self):
        """Reset pipeline state, cached world pose, and locked held-object prior."""
        self.last_valid_world_pose = None
        self.last_valid_state = None
        self.held_T_pelvis_obj = None
        self.held_T_torso_yaw_obj = None
        self.held_object_locked = False

    def update(
        self,
        measurements: Dict[str, ObjectPoseMeasurement],
        robot_state: RobotState,
        context: PerceptionContext,
    ) -> ObjectPoseState:
        """
        Process incoming measurements and robot state to generate unified ObjectPoseState.
        """
        diagnostics: Dict[str, Any] = {}
        robot_stamp = robot_state.timestamp

        # 1. Select Primary Measurement (Prefer torso_link, fallback to camera_link)
        primary_meas: Optional[ObjectPoseMeasurement] = None
        if FRAME_TORSO_LINK in measurements and measurements[FRAME_TORSO_LINK].valid:
            primary_meas = measurements[FRAME_TORSO_LINK]
        elif FRAME_CAMERA_LINK in measurements and measurements[FRAME_CAMERA_LINK].valid:
            primary_meas = measurements[FRAME_CAMERA_LINK]

        # 2. Check Raw Pelvis Diagnostic Topic if present
        if "pelvis_raw_diag" in measurements:
            diag_meas = measurements["pelvis_raw_diag"]
            diagnostics["pelvis_raw_pos"] = diag_meas.pos.tolist()
            diagnostics["pelvis_raw_quat"] = diag_meas.quat.tolist()

        # Phase 22+ / Carrying check
        in_carry_phase = (context.current_phase >= 22 or context.manual_stage >= 3 or context.is_holding_object)
        allow_held = context.allow_held_object_prior or in_carry_phase

        has_valid_primary = (primary_meas is not None and primary_meas.valid)

        if not has_valid_primary:
            # Check if locked Held-Object Prior can be used
            if allow_held and self.held_object_locked and self.held_T_pelvis_obj is not None:
                diagnostics["used_held_object_prior"] = True
                pelvis_link_pose = robot_state.link_poses[FRAME_PELVIS_LINK]

                # Reconstruct obj_pos_world from current pelvis_link pose + locked held_T_pelvis_obj
                obj_pos_world, obj_quat_world = compose_frame_transforms(
                    pelvis_link_pose.pos,
                    pelvis_link_pose.quat,
                    self.held_T_pelvis_obj.pos,
                    self.held_T_pelvis_obj.quat,
                )
                object_world_pose = Pose(pos=obj_pos_world, quat=obj_quat_world)

                # Relative to pelvis_link
                obj_pos_pelvis = self.held_T_pelvis_obj.pos.copy()
                obj_quat_pelvis = self.held_T_pelvis_obj.quat.copy()

                # Relative to torso_yaw
                torso_yaw_pose = robot_state.link_poses[FRAME_TORSO_YAW]
                obj_pos_torso_yaw, obj_quat_torso_yaw = subtract_frame_transforms(
                    torso_yaw_pose.pos,
                    torso_yaw_pose.quat,
                    object_world_pose.pos,
                    object_world_pose.quat,
                )

                stamp_delta = 0.0
                meas_stamp = robot_stamp
                source_id = "held_object_prior_locked"
                frame_id = FRAME_PELVIS_LINK
                valid_reason = "held_object_prior_locked"

                diagnostics["measurement_timestamp"] = meas_stamp
                diagnostics["robot_state_timestamp"] = robot_stamp
                diagnostics["stamp_delta"] = stamp_delta
                diagnostics["timestamp_source"] = "held_prior"

                state = ObjectPoseState(
                    valid=True,
                    source_id=source_id,
                    frame_id=frame_id,
                    timestamp=robot_stamp,
                    measurement_timestamp=meas_stamp,
                    robot_state_timestamp=robot_stamp,
                    stamp_delta=stamp_delta,
                    valid_reason=valid_reason,
                    obj_pos_world=object_world_pose.pos.copy(),
                    obj_quat_world=object_world_pose.quat.copy(),
                    obj_pos_pelvis=obj_pos_pelvis.astype(np.float32),
                    obj_quat_pelvis=normalize_quat(obj_quat_pelvis),
                    obj_pos_torso_yaw=obj_pos_torso_yaw.astype(np.float32),
                    obj_quat_torso_yaw=normalize_quat(obj_quat_torso_yaw),
                    diagnostics=diagnostics,
                )
                self.last_valid_state = state
                return state

            # No valid measurement and no locked held prior
            return self._handle_invalid_fallback(
                robot_state,
                context,
                reason="no_valid_measurement_source",
                diagnostics=diagnostics,
            )

        # 3. Standard Measurement Path: Reconstruct Object World Pose from Primary Measurement
        meas_stamp = primary_meas.timestamp
        source_id = primary_meas.source_id
        frame_id = primary_meas.frame_id

        # Validate Quaternion
        q_valid, q_reason = self.validator.validate_quaternion(primary_meas.quat)
        if not q_valid:
            diagnostics["quaternion_error"] = q_reason
            return self._handle_invalid_fallback(
                robot_state, context, reason=q_reason, diagnostics=diagnostics
            )

        # Validate Timestamps
        t_valid, stamp_delta, t_reason = self.validator.validate_timestamps(meas_stamp, robot_stamp)
        diagnostics["measurement_timestamp"] = meas_stamp
        diagnostics["robot_state_timestamp"] = robot_stamp
        diagnostics["stamp_delta"] = stamp_delta
        diagnostics["timestamp_source"] = "header_stamp"
        if not t_valid:
            diagnostics["timestamp_error"] = t_reason
            return self._handle_invalid_fallback(
                robot_state, context, reason=t_reason, diagnostics=diagnostics
            )

        # Reconstruct object_pose_world = compose(link_world_pose, primary_measurement)
        ref_link_frame = primary_meas.frame_id
        if ref_link_frame not in robot_state.link_poses:
            err_msg = f"link_frame_{ref_link_frame}_not_in_robot_state"
            return self._handle_invalid_fallback(
                robot_state, context, reason=err_msg, diagnostics=diagnostics
            )

        link_world_pose = robot_state.link_poses[ref_link_frame]
        obj_pos_world, obj_quat_world = compose_frame_transforms(
            link_world_pose.pos,
            link_world_pose.quat,
            primary_meas.pos,
            primary_meas.quat,
        )
        object_world_pose = Pose(pos=obj_pos_world, quat=obj_quat_world)

        # 4. Secondary Source Residual Check (if camera_link measurement is also present)
        if (
            ref_link_frame == FRAME_TORSO_LINK
            and FRAME_CAMERA_LINK in measurements
            and measurements[FRAME_CAMERA_LINK].valid
        ):
            cam_meas = measurements[FRAME_CAMERA_LINK]
            cam_link_pose = robot_state.link_poses[FRAME_CAMERA_LINK]
            cam_obj_world_pos, cam_obj_world_quat = compose_frame_transforms(
                cam_link_pose.pos, cam_link_pose.quat, cam_meas.pos, cam_meas.quat
            )
            secondary_world_pose = Pose(pos=cam_obj_world_pos, quat=cam_obj_world_quat)

            res_valid, pos_res, rot_res_deg, res_reason = self.validator.compute_world_residuals(
                object_world_pose, secondary_world_pose
            )
            diagnostics["camera_world_pos_residual"] = pos_res
            diagnostics["camera_world_rot_residual_deg"] = rot_res_deg
            if not res_valid:
                diagnostics["residual_warning"] = res_reason

        is_valid = True
        valid_reason = "primary_measurement_valid"
        self.last_valid_world_pose = object_world_pose.copy()

        # 5. Derive Policy Relative Poses from Unified object_world_pose
        # A. Derive object_pose_in_pelvis_link (for CF-Gen)
        pelvis_link_pose = robot_state.link_poses[FRAME_PELVIS_LINK]
        obj_pos_pelvis, obj_quat_pelvis = subtract_frame_transforms(
            pelvis_link_pose.pos,
            pelvis_link_pose.quat,
            object_world_pose.pos,
            object_world_pose.quat,
        )

        # B. Derive object_pose_in_torso_yaw (for CF-Tracker)
        torso_yaw_pose = robot_state.link_poses[FRAME_TORSO_YAW]
        obj_pos_torso_yaw, obj_quat_torso_yaw = subtract_frame_transforms(
            torso_yaw_pose.pos,
            torso_yaw_pose.quat,
            object_world_pose.pos,
            object_world_pose.quat,
        )

        # 6. Lock Held-Object Relative Poses if allow_held_object_prior is True
        if allow_held:
            self.held_T_pelvis_obj = Pose(pos=obj_pos_pelvis.copy(), quat=normalize_quat(obj_quat_pelvis))
            self.held_T_torso_yaw_obj = Pose(pos=obj_pos_torso_yaw.copy(), quat=normalize_quat(obj_quat_torso_yaw))
            self.held_object_locked = True
            diagnostics["held_object_locked"] = True

        # Diagnostics: Z offset diff between pelvis and torso
        z_diff = float(obj_pos_torso_yaw[2] - obj_pos_pelvis[2])
        diagnostics["pelvis_torso_z_diff"] = z_diff
        diagnostics["obj_pos_world"] = object_world_pose.pos.tolist()

        state = ObjectPoseState(
            valid=is_valid,
            source_id=source_id,
            frame_id=frame_id,
            timestamp=robot_stamp,
            measurement_timestamp=meas_stamp,
            robot_state_timestamp=robot_stamp,
            stamp_delta=stamp_delta,
            valid_reason=valid_reason,
            obj_pos_world=object_world_pose.pos.copy(),
            obj_quat_world=object_world_pose.quat.copy(),
            obj_pos_pelvis=obj_pos_pelvis.astype(np.float32),
            obj_quat_pelvis=normalize_quat(obj_quat_pelvis),
            obj_pos_torso_yaw=obj_pos_torso_yaw.astype(np.float32),
            obj_quat_torso_yaw=normalize_quat(obj_quat_torso_yaw),
            diagnostics=diagnostics,
        )

        self.last_valid_state = state
        return state

    def _handle_invalid_fallback(
        self,
        robot_state: RobotState,
        context: PerceptionContext,
        reason: str,
        diagnostics: Dict[str, Any],
    ) -> ObjectPoseState:
        """
        Safety fallback policy execution when vision is invalid or stale.
        Holds last valid world pose if available, or flags invalid state.
        """
        diagnostics["fallback_triggered"] = True
        diagnostics["fallback_reason"] = reason
        diagnostics["measurement_timestamp"] = 0.0
        diagnostics["robot_state_timestamp"] = robot_state.timestamp
        diagnostics["stamp_delta"] = 0.0
        diagnostics["timestamp_source"] = "none"

        if self.last_valid_world_pose is not None:
            world_pose = self.last_valid_world_pose.copy()
            fallback_reason = f"hold_last_valid ({reason})"
            is_valid = (context.manual_stage == 0) # Allow standing in Stage 0
            source_id = "fallback_hold_last_valid"
        else:
            # Zero / Identity fallback
            world_pose = Pose(
                pos=np.array([0.5, 0.0, 0.2], dtype=np.float32),
                quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            fallback_reason = f"default_fallback ({reason})"
            is_valid = (context.manual_stage == 0) # Allow standing in Stage 0
            source_id = "fallback_default"

        # Re-derive relative poses for robot state
        pelvis_link_pose = robot_state.link_poses[FRAME_PELVIS_LINK]
        obj_pos_pelvis, obj_quat_pelvis = subtract_frame_transforms(
            pelvis_link_pose.pos,
            pelvis_link_pose.quat,
            world_pose.pos,
            world_pose.quat,
        )

        torso_yaw_pose = robot_state.link_poses[FRAME_TORSO_YAW]
        obj_pos_torso_yaw, obj_quat_torso_yaw = subtract_frame_transforms(
            torso_yaw_pose.pos,
            torso_yaw_pose.quat,
            world_pose.pos,
            world_pose.quat,
        )

        return ObjectPoseState(
            valid=is_valid,
            source_id=source_id,
            frame_id="world",
            timestamp=robot_state.timestamp,
            measurement_timestamp=0.0,
            robot_state_timestamp=robot_state.timestamp,
            stamp_delta=0.0,
            valid_reason=fallback_reason,
            obj_pos_world=world_pose.pos.copy(),
            obj_quat_world=world_pose.quat.copy(),
            obj_pos_pelvis=obj_pos_pelvis.astype(np.float32),
            obj_quat_pelvis=normalize_quat(obj_quat_pelvis),
            obj_pos_torso_yaw=obj_pos_torso_yaw.astype(np.float32),
            obj_quat_torso_yaw=normalize_quat(obj_quat_torso_yaw),
            diagnostics=diagnostics,
        )
