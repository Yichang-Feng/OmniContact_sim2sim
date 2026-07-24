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
from common.utils import yaw_quat
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
class StaticAnchorConfig:
    buffer_size: int = 5
    pos_std_threshold: float = 0.02
    residual_threshold: float = 0.08
    alpha: float = 0.2
    max_anchor_age: float = 3.0


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
    def __init__(
        self,
        validation_config: ValidationConfig = ValidationConfig(),
        static_anchor_config: StaticAnchorConfig = StaticAnchorConfig()
    ):
        self.validator = PoseValidator(validation_config)
        self.static_anchor_config = static_anchor_config
        self.last_valid_world_pose: Optional[Pose] = None
        self.last_valid_state: Optional[ObjectPoseState] = None

        # Internal Held-Object Prior attributes
        self.held_T_pelvis_obj: Optional[Pose] = None
        self.held_T_torso_yaw_obj: Optional[Pose] = None
        self.held_object_locked: bool = False

        # Static Anchor state
        self.static_anchor_pose: Optional[Pose] = None
        self.static_anchor_locked: bool = False
        self.static_anchor_buffer = []
        self.static_anchor_timestamp: float = 0.0
        self.static_anchor_source_id: str = "none"
        self.static_anchor_outlier_counter: int = 0

    def reset(self):
        """Reset pipeline state, cached world pose, and locked priors."""
        self.last_valid_world_pose = None
        self.last_valid_state = None
        self.held_T_pelvis_obj = None
        self.held_T_torso_yaw_obj = None
        self.held_object_locked = False
        self.static_anchor_pose = None
        self.static_anchor_locked = False
        self.static_anchor_buffer = []
        self.static_anchor_timestamp = 0.0
        self.static_anchor_source_id = "none"
        self.static_anchor_outlier_counter = 0

    def _update_static_anchor(self, measured_world_pose: Pose, timestamp: float, context: PerceptionContext):
        if context.current_phase >= 22:
            return
        if measured_world_pose is None:
            return
            
        self.static_anchor_buffer.append(measured_world_pose.pos.copy())
        if len(self.static_anchor_buffer) > self.static_anchor_config.buffer_size:
            self.static_anchor_buffer.pop(0)
            
        if len(self.static_anchor_buffer) < self.static_anchor_config.buffer_size:
            return
            
        positions = np.asarray(self.static_anchor_buffer, dtype=np.float32)
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        
        if float(np.linalg.norm(std_pos)) < self.static_anchor_config.pos_std_threshold:
            self.static_anchor_pose = Pose(
                pos=mean_pos.astype(np.float32),
                quat=yaw_quat(measured_world_pose.quat).astype(np.float32)
            )
            self.static_anchor_locked = True
            self.static_anchor_timestamp = timestamp
            self.static_anchor_source_id = "ros2_torso_link"
            self.static_anchor_outlier_counter = 0

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

        measured_world_pose = None
        measurement_usable = False
        stamp_delta = 0.0
        reject_reason = None

        if primary_meas is not None:
            q_valid, q_reason = self.validator.validate_quaternion(primary_meas.quat)
            t_valid, stamp_delta, t_reason = self.validator.validate_timestamps(
                primary_meas.timestamp,
                robot_state.timestamp,
            )

            diagnostics["measurement_timestamp"] = primary_meas.timestamp
            diagnostics["robot_state_timestamp"] = robot_state.timestamp
            diagnostics["stamp_delta"] = stamp_delta

            if not q_valid:
                reject_reason = q_reason
            elif not t_valid:
                reject_reason = t_reason
            elif primary_meas.frame_id not in robot_state.link_poses:
                reject_reason = f"missing_link_frame_{primary_meas.frame_id}"
            else:
                link_pose = robot_state.link_poses[primary_meas.frame_id]
                obj_pos_world, obj_quat_world = compose_frame_transforms(
                    link_pose.pos,
                    link_pose.quat,
                    primary_meas.pos,
                    primary_meas.quat,
                )
                measured_world_pose = Pose(pos=obj_pos_world, quat=obj_quat_world)
                measurement_usable = True
        else:
            reject_reason = "no_valid_primary_measurement"

        # 3. Phase 25+ unlock held prior
        if context.current_phase >= 25:
            self.held_object_locked = False
            self.held_T_pelvis_obj = None
            self.held_T_torso_yaw_obj = None

        in_carry_phase = (
            (22 <= context.current_phase < 25)
            or context.manual_stage >= 3
            or context.is_holding_object
        )
        allow_held = (context.allow_held_object_prior or in_carry_phase) and context.current_phase < 25

        # 4. Select Final World Pose
        object_world_pose = None
        source_id = "none"
        valid_reason = "none"
        valid = False

        # 4.1 Vision Usable
        if measurement_usable:
            object_world_pose = measured_world_pose
            source_id = primary_meas.source_id
            valid_reason = "primary_measurement_valid"
            valid = True

            if context.manual_stage == 0 or context.current_phase < 22:
                if self.static_anchor_locked and self.static_anchor_pose is not None:
                    residual = float(np.linalg.norm(
                        object_world_pose.pos - self.static_anchor_pose.pos
                    ))
                    diagnostics["static_anchor_residual"] = residual

                    if residual < self.static_anchor_config.residual_threshold:
                        self.static_anchor_outlier_counter = 0
                        alpha = self.static_anchor_config.alpha
                        self.static_anchor_pose.pos = (
                            (1 - alpha) * self.static_anchor_pose.pos
                            + alpha * object_world_pose.pos
                        )
                        self.static_anchor_pose.quat = object_world_pose.quat.copy()
                        self.static_anchor_timestamp = robot_state.timestamp
                    else:
                        self.static_anchor_outlier_counter += 1
                        diagnostics["static_anchor_outlier"] = True
                        diagnostics["static_anchor_outlier_counter"] = self.static_anchor_outlier_counter

                        if self.static_anchor_outlier_counter > 10:
                            # Continuous outliers -> anchor might be invalid, re-lock
                            self.static_anchor_locked = False
                            self.static_anchor_buffer = []
                            self.static_anchor_outlier_counter = 0
                            self._update_static_anchor(object_world_pose, robot_state.timestamp, context)
                            source_id = "ros2_torso_link_anchor_relock"
                            valid_reason = "static_anchor_relocked_after_outliers"
                        else:
                            object_world_pose = Pose(pos=self.static_anchor_pose.pos.copy(), quat=self.static_anchor_pose.quat.copy())
                            source_id = "static_world_anchor_outlier_rejection"
                            valid_reason = "static_anchor_locked_outlier_rejected"
                else:
                    self._update_static_anchor(object_world_pose, robot_state.timestamp, context)

            # Secondary Source Residual Check
            if (
                primary_meas.frame_id == FRAME_TORSO_LINK
                and FRAME_CAMERA_LINK in measurements
                and measurements[FRAME_CAMERA_LINK].valid
            ):
                cam_meas = measurements[FRAME_CAMERA_LINK]
                if cam_meas.frame_id in robot_state.link_poses:
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

        # 4.2 Vision Unusable, use Held Prior if possible
        elif allow_held and self.held_object_locked and self.held_T_pelvis_obj is not None:
            pelvis_pose = robot_state.link_poses[FRAME_PELVIS_LINK]
            obj_pos_world, obj_quat_world = compose_frame_transforms(
                pelvis_pose.pos,
                pelvis_pose.quat,
                self.held_T_pelvis_obj.pos,
                self.held_T_pelvis_obj.quat,
            )
            object_world_pose = Pose(pos=obj_pos_world, quat=obj_quat_world)
            source_id = "held_object_prior_locked"
            valid_reason = "held_object_prior_locked"
            valid = True
            diagnostics["used_held_object_prior"] = True

        # 4.3 Vision Unusable, use Static Anchor if possible
        elif (
            context.current_phase < 22
            and self.static_anchor_locked
            and self.static_anchor_pose is not None
        ):
            anchor_age = robot_state.timestamp - self.static_anchor_timestamp
            diagnostics["static_anchor_age"] = anchor_age

            if anchor_age < self.static_anchor_config.max_anchor_age:
                object_world_pose = Pose(pos=self.static_anchor_pose.pos.copy(), quat=self.static_anchor_pose.quat.copy())
                source_id = "static_world_anchor"
                valid_reason = "static_anchor_locked"
                valid = True
                diagnostics["used_static_anchor"] = True
            else:
                self.static_anchor_locked = False
                self.static_anchor_pose = None
                self.static_anchor_buffer = []
                self.static_anchor_outlier_counter = 0
                reject_reason = "static_anchor_expired"

        # 4.4 Fallback if no valid source
        if object_world_pose is None:
            return self._handle_invalid_fallback(
                robot_state,
                context,
                reason=reject_reason or "no_usable_source",
                diagnostics=diagnostics,
            )

        # 5. Save last valid world
        self.last_valid_world_pose = object_world_pose.copy()

        # 6. Derive Policy Relative Poses from object_world_pose
        pelvis_pose = robot_state.link_poses[FRAME_PELVIS_LINK]
        obj_pos_pelvis, obj_quat_pelvis = subtract_frame_transforms(
            pelvis_pose.pos,
            pelvis_pose.quat,
            object_world_pose.pos,
            object_world_pose.quat,
        )

        torso_yaw_pose = robot_state.link_poses[FRAME_TORSO_YAW]
        obj_pos_torso_yaw, obj_quat_torso_yaw = subtract_frame_transforms(
            torso_yaw_pose.pos,
            torso_yaw_pose.quat,
            object_world_pose.pos,
            object_world_pose.quat,
        )

        # 7. Lock Held Prior if we are allowed to hold it
        if allow_held:
            if measurement_usable:
                self.held_T_pelvis_obj = Pose(
                    pos=obj_pos_pelvis.copy(),
                    quat=normalize_quat(obj_quat_pelvis),
                )
                self.held_T_torso_yaw_obj = Pose(
                    pos=obj_pos_torso_yaw.copy(),
                    quat=normalize_quat(obj_quat_torso_yaw),
                )
                self.held_object_locked = True
                diagnostics["held_object_locked"] = True
            elif not self.held_object_locked and object_world_pose is not None:
                self.held_T_pelvis_obj = Pose(
                    pos=obj_pos_pelvis.copy(),
                    quat=normalize_quat(obj_quat_pelvis),
                )
                self.held_T_torso_yaw_obj = Pose(
                    pos=obj_pos_torso_yaw.copy(),
                    quat=normalize_quat(obj_quat_torso_yaw),
                )
                self.held_object_locked = True
                diagnostics["held_object_locked_from_fallback"] = True

        diagnostics["pelvis_torso_z_diff"] = float(obj_pos_torso_yaw[2] - obj_pos_pelvis[2])
        diagnostics["obj_pos_world"] = object_world_pose.pos.tolist()

        state = ObjectPoseState(
            valid=valid,
            source_id=source_id,
            frame_id=getattr(primary_meas, "frame_id", "world") if primary_meas else "world",
            timestamp=robot_state.timestamp,
            measurement_timestamp=getattr(primary_meas, "timestamp", robot_state.timestamp) if primary_meas else robot_state.timestamp,
            robot_state_timestamp=robot_state.timestamp,
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
            is_valid = True # Always valid if we have a last known pose
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
