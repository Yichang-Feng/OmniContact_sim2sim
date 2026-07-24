"""
Comprehensive Unit and Integration Tests for Perception Pipeline and Coordinate Transformations.

Quantitative Acceptance Criteria:
1. `compose_frame_transforms` and `subtract_frame_transforms` round-trip error < 1e-5.
2. Reconstructed `object_pose_world` is invariant (< 1e-4) to robot pitch, roll, and yaw rotation.
3. Policy inputs for CF-Gen (pelvis_link) and CF-Tracker (torso_yaw) reconstruct back to the EXACT SAME world pose (< 1e-4).
4. `pelvis_p + quat_apply(pelvis_q, rel_pelvis_pos)` strictly equals `object_pose_world`.
5. Held-Object Prior locks pose during Phase 22+ valid vision, and restores pose relative to moving robot when vision is lost.
6. Pipeline reset() clears cached world pose and locked held pose.
7. Stage 0 fallback cannot be used as a valid source for Stage 1 planning.
8. Deploy script deploy_omnicontact_stand_prepare_test.py contains NO hardcoded approx_rel_pelvis_pos.
"""

import os
import sys
import unittest
import numpy as np

# Add repository root to path
sys.path.append("/home/feng/OmniContact_sim2sim")

from common.utils import (
    quat_mul,
    quat_apply,
    quat_conjugate,
    yaw_quat,
    yaw_to_quat,
    subtract_frame_transforms,
    normalize_quat,
)
from common.perception.frame_definitions import (
    Pose,
    FRAME_WORLD,
    FRAME_PELVIS_LINK,
    FRAME_TORSO_LINK,
    FRAME_TORSO_YAW,
    FRAME_CAMERA_LINK,
    compose_frame_transforms,
)
from common.perception.robot_state_provider import RobotState, RobotStateProvider
from common.perception.object_pose_source import (
    ObjectPoseMeasurement,
    Ros2ObjectPoseSource,
    SimObjectPoseSource,
)
from common.perception.validators import ValidationConfig, PoseValidator
from common.perception.object_pose_pipeline import (
    PerceptionContext,
    ObjectPoseState,
    ObjectPosePipeline,
)


class MockKinematics:
    """
    Mock kinematics provider for unit testing.
    Calculates rigid offset link poses based on base_pos and base_quat.
    """
    def __init__(self):
        # Local link offsets in base frame
        self.pelvis_offset_base = np.array([0.0, 0.0, -0.05], dtype=np.float32)
        self.torso_offset_base = np.array([0.0, 0.0, 0.15], dtype=np.float32)
        self.cam_offset_base = np.array([0.10, 0.0, 0.35], dtype=np.float32)

    def forward(self, q, base_pos, base_quat):
        # Apply base_quat to link offsets
        p_pelvis = base_pos + quat_apply(base_quat, self.pelvis_offset_base)
        p_torso = base_pos + quat_apply(base_quat, self.torso_offset_base)
        p_cam = base_pos + quat_apply(base_quat, self.cam_offset_base)

        return {
            "pelvis": {"pos": p_pelvis, "quat": base_quat},
            "torso_link": {"pos": p_torso, "quat": base_quat},
            "d435_camera": {"pos": p_cam, "quat": base_quat},
        }


class TestObjectPosePipeline(unittest.TestCase):

    def setUp(self):
        self.kinematics = MockKinematics()
        self.state_provider = RobotStateProvider(self.kinematics)
        self.pipeline = ObjectPosePipeline()

    def test_compose_and_subtract_transforms_round_trip(self):
        """Test round-trip composition and subtraction of frame transforms."""
        t01 = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        q01 = yaw_to_quat(0.5) # ~28.6 deg yaw
        t12 = np.array([0.3, -0.2, 0.1], dtype=np.float32)
        q12 = normalize_quat(np.array([0.92388, 0.0, 0.38268, 0.0], dtype=np.float32)) # pitch

        # Compose: T_02 = T_01 * T_12
        t02, q02 = compose_frame_transforms(t01, q01, t12, q12)

        # Subtract: T_12_reconstructed = T_01^{-1} * T_02
        t12_rec, q12_rec = subtract_frame_transforms(t01, q01, t02, q02)

        np.testing.assert_allclose(t12_rec, t12, atol=1e-5)
        # Quaternions q and -q represent same rotation
        dot = abs(np.dot(normalize_quat(q12_rec), normalize_quat(q12)))
        self.assertAlmostEqual(dot, 1.0, places=5)

    def test_robot_pitch_roll_yaw_invariance(self):
        """
        Test that object_pose_world reconstructed from torso measurement
        remains invariant when robot base roll, pitch, and yaw change.
        """
        gt_obj_pos_world = np.array([1.5, 0.2, 0.4], dtype=np.float32)
        gt_obj_quat_world = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Test across multiple robot base poses (different roll, pitch, yaw)
        base_quats = [
            yaw_to_quat(0.0),                          # neutral
            yaw_to_quat(0.4),                          # yaw rotation
            normalize_quat(np.array([0.9659, 0.2588, 0.0, 0.0])), # roll 30 deg
            normalize_quat(np.array([0.9659, 0.0, 0.2588, 0.0])), # pitch 30 deg
            normalize_quat(np.array([0.9238, 0.2209, 0.2209, 0.2209])), # combined roll/pitch/yaw
        ]

        reconstructed_world_positions = []

        class SimpleStateCmd:
            def __init__(self, pos, quat):
                self.base_pos = pos
                self.base_quat = quat
                self.q = np.zeros(29, dtype=np.float32)

        for b_quat in base_quats:
            state_cmd = SimpleStateCmd(pos=np.array([0.5, 0.1, 0.8], dtype=np.float32), quat=b_quat)
            robot_state = self.state_provider.update_from_state_cmd(state_cmd, timestamp=1.0)

            # Torso link pose in world
            torso_pose = robot_state.link_poses[FRAME_TORSO_LINK]

            # Compute relative measurement of object in torso frame
            rel_pos, rel_quat = subtract_frame_transforms(
                torso_pose.pos, torso_pose.quat, gt_obj_pos_world, gt_obj_quat_world
            )

            source = Ros2ObjectPoseSource()
            source.update_raw_torso_pose(rel_pos, rel_quat, timestamp=1.0)

            ctx = PerceptionContext(manual_stage=1, current_phase=11)
            state = self.pipeline.update(source.get_measurements(), robot_state, ctx)

            self.assertTrue(state.valid)
            reconstructed_world_positions.append(state.obj_pos_world)

        # All reconstructed world positions must be identical to ground truth (< 1e-4)
        for reconstructed_pos in reconstructed_world_positions:
            np.testing.assert_allclose(reconstructed_pos, gt_obj_pos_world, atol=1e-4)

    def test_cfgen_and_cftracker_world_restoration(self):
        """
        Verify that obj_pos_pelvis (for CF-Gen) and obj_pos_torso_yaw (for CF-Tracker)
        both restore back to the EXACT SAME object world pose.
        """
        gt_obj_pos_world = np.array([2.0, -0.3, 0.35], dtype=np.float32)
        gt_obj_quat_world = yaw_to_quat(0.2)

        class SimpleStateCmd:
            base_pos = np.array([0.8, -0.1, 0.75], dtype=np.float32)
            base_quat = normalize_quat(np.array([0.9238, 0.1, 0.2, 0.33]))
            q = np.zeros(29, dtype=np.float32)

        robot_state = self.state_provider.update_from_state_cmd(SimpleStateCmd(), timestamp=1.0)
        torso_pose = robot_state.link_poses[FRAME_TORSO_LINK]

        rel_pos, rel_quat = subtract_frame_transforms(
            torso_pose.pos, torso_pose.quat, gt_obj_pos_world, gt_obj_quat_world
        )

        source = Ros2ObjectPoseSource()
        source.update_raw_torso_pose(rel_pos, rel_quat, timestamp=1.0)

        ctx = PerceptionContext(manual_stage=1, current_phase=11)
        state = self.pipeline.update(source.get_measurements(), robot_state, ctx)

        # 1. Restore from CF-Gen pelvis pose: obj_p = pelvis_p + quat_apply(pelvis_q, rel_pelvis_pos)
        pelvis_pose = robot_state.link_poses[FRAME_PELVIS_LINK]
        obj_p_from_cfgen = pelvis_pose.pos + quat_apply(pelvis_pose.quat, state.obj_pos_pelvis)

        # 2. Restore from CF-Tracker torso_yaw pose: obj_p = torso_yaw_p + quat_apply(torso_yaw_q, rel_torso_yaw_pos)
        torso_yaw_pose = robot_state.link_poses[FRAME_TORSO_YAW]
        obj_p_from_cftracker = torso_yaw_pose.pos + quat_apply(torso_yaw_pose.quat, state.obj_pos_torso_yaw)

        # Check equality with pipeline obj_pos_world and ground truth
        np.testing.assert_allclose(state.obj_pos_world, gt_obj_pos_world, atol=1e-4)
        np.testing.assert_allclose(obj_p_from_cfgen, state.obj_pos_world, atol=1e-4)
        np.testing.assert_allclose(obj_p_from_cftracker, state.obj_pos_world, atol=1e-4)

    def test_sim_object_pose_source(self):
        """Test SimObjectPoseSource with noise and delay."""
        sim_source = SimObjectPoseSource(noise_std_pos=0.01, delay_seconds=0.05)

        gt_obj_pos = np.array([1.2, 0.0, 0.5], dtype=np.float32)
        gt_obj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gt_torso_pos = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        gt_torso_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        measurements = sim_source.update_from_mujoco_gt(
            gt_obj_pos, gt_obj_quat, gt_torso_pos, gt_torso_quat, timestamp=10.0
        )

        self.assertIn(FRAME_TORSO_LINK, measurements)
        meas = measurements[FRAME_TORSO_LINK]
        self.assertAlmostEqual(meas.timestamp, 9.95, places=5)
        # Position error within noise bound
        diff = np.linalg.norm(meas.pos - (gt_obj_pos - gt_torso_pos))
        self.assertLess(diff, 0.05)

    def test_held_object_prior_locking_and_movement(self):
        """Test Phase 22+ held object locking, vision loss recovery, and movement following robot."""
        class SimpleStateCmd:
            def __init__(self, pos, quat):
                self.base_pos = pos
                self.base_quat = quat
                self.q = np.zeros(29, dtype=np.float32)

        # 1. Robot at initial position, Phase 22 with valid vision
        state_cmd1 = SimpleStateCmd(pos=np.array([0.0, 0.0, 0.8], dtype=np.float32), quat=yaw_to_quat(0.0))
        robot_state1 = self.state_provider.update_from_state_cmd(state_cmd1, timestamp=1.0)
        gt_obj_pos_world1 = np.array([0.3, 0.0, 0.85], dtype=np.float32)
        gt_obj_quat_world1 = yaw_to_quat(0.0)

        torso_pose1 = robot_state1.link_poses[FRAME_TORSO_LINK]
        rel_pos1, rel_quat1 = subtract_frame_transforms(
            torso_pose1.pos, torso_pose1.quat, gt_obj_pos_world1, gt_obj_quat_world1
        )

        source = Ros2ObjectPoseSource()
        source.update_raw_torso_pose(rel_pos1, rel_quat1, timestamp=1.0)

        ctx_phase22 = PerceptionContext(manual_stage=3, current_phase=22, is_holding_object=True)
        state1 = self.pipeline.update(source.get_measurements(), robot_state1, ctx_phase22)

        self.assertTrue(state1.valid)
        self.assertTrue(self.pipeline.held_object_locked)
        self.assertIsNotNone(self.pipeline.held_T_pelvis_obj)

        # 2. Vision lost during Phase 22, robot moves forward 1.0m and rotates 45 deg yaw
        state_cmd2 = SimpleStateCmd(pos=np.array([1.0, 0.5, 0.8], dtype=np.float32), quat=yaw_to_quat(0.785398))
        robot_state2 = self.state_provider.update_from_state_cmd(state_cmd2, timestamp=2.0)

        # Update pipeline with empty measurements (vision lost)
        state2 = self.pipeline.update({}, robot_state2, ctx_phase22)

        self.assertTrue(state2.valid)
        self.assertEqual(state2.source_id, "held_object_prior_locked")
        self.assertTrue(state2.diagnostics.get("used_held_object_prior", False))

        # Reconstructed obj_pos_world should move along with pelvis
        pelvis_pose2 = robot_state2.link_poses[FRAME_PELVIS_LINK]
        expected_obj_pos2 = pelvis_pose2.pos + quat_apply(pelvis_pose2.quat, self.pipeline.held_T_pelvis_obj.pos)
        np.testing.assert_allclose(state2.obj_pos_world, expected_obj_pos2, atol=1e-4)

        # 3. Test pipeline reset clears locked held pose
        self.pipeline.reset()
        self.assertFalse(self.pipeline.held_object_locked)
        self.assertIsNone(self.pipeline.held_T_pelvis_obj)
        self.assertIsNone(self.pipeline.last_valid_world_pose)

    def test_stage0_fallback_cannot_be_stage1_source(self):
        """Verify Stage 0 fallback produces source_id starting with 'fallback', blocking Stage 1 entry."""
        class SimpleStateCmd:
            base_pos = np.array([0.0, 0.0, 0.8], dtype=np.float32)
            base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            q = np.zeros(29, dtype=np.float32)

        robot_state = self.state_provider.update_from_state_cmd(SimpleStateCmd(), timestamp=1.0)
        ctx_stage0 = PerceptionContext(manual_stage=0, current_phase=11)
        state_stage0 = self.pipeline.update({}, robot_state, ctx_stage0)

        # Valid for standing in Stage 0, but source_id is fallback
        self.assertTrue(state_stage0.valid)
        self.assertTrue(state_stage0.source_id.startswith("fallback"))

    def test_no_approx_rel_in_deploy_scripts(self):
        """Assert deploy_omnicontact_stand_prepare_test.py contains NO hardcoded approx_rel_pelvis_pos."""
        script_path = "/home/feng/OmniContact_sim2sim/deploy_omnicontact/deploy_omnicontact_stand_prepare_test.py"
        self.assertTrue(os.path.exists(script_path))
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("approx_rel_pelvis_pos", content)
        self.assertNotIn("approx_rel_torso_pos", content)


if __name__ == "__main__":
    unittest.main()
