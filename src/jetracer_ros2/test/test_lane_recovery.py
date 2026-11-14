"""Metric path propagation and independent fail-safe stop limits."""

import math

from jetracer_ros2.lane_recovery import (
    Pose2D, RecoveryTarget, snapshot_path, recovery_target,
    right_boundary_clearance, brief_hold_allowed,
)


def test_camera_gap_fallback_is_bounded_without_odom():
    assert brief_hold_allowed(True, 0.20, 0.50)
    assert brief_hold_allowed(True, 0.50, 0.50)
    assert not brief_hold_allowed(True, 0.51, 0.50)
    assert not brief_hold_allowed(True, -0.01, 0.50)
    assert not brief_hold_allowed(False, 0.20, 0.50)
    assert not brief_hold_allowed(True, float('nan'), 0.50)


def test_odom_motion_moves_stored_target_back_in_vehicle_frame():
    snapshot = snapshot_path(((0.0, 0.20), (0.08, 0.80)), Pose2D(0, 0, 0), 0, 0)
    target = recovery_target(snapshot, Pose2D(0.04, 0, 0), .04,
                             100_000_000, .50, .06, .40, .15)
    assert target is not None
    assert math.isclose(target.lateral_m, .08, abs_tol=1e-6)
    assert math.isclose(target.forward_m, .76, abs_tol=1e-6)


def test_vehicle_yaw_reprojects_path_instead_of_freezing_old_steer():
    snapshot = snapshot_path(((0.0, 0.20), (0.0, 0.80)), Pose2D(0, 0, 0), 0, 0)
    target = recovery_target(snapshot, Pose2D(0, 0, .10), .01,
                             100_000_000, .50, .06, .40, .15)
    assert target is not None
    assert target.lateral_m > 0  # A left yaw leaves the path to the right.


def test_half_of_last_valid_path_and_absolute_caps_both_stop():
    snapshot = snapshot_path(((0.0, .20), (0.0, .80)), Pose2D(0, 0, 0), 0, 0)
    assert recovery_target(snapshot, Pose2D(0, 0, 0), .399,
                           100_000_000, .50, 1.0, .40, .15)
    assert recovery_target(snapshot, Pose2D(0, 0, 0), .400,
                           100_000_000, .50, 1.0, .40, .15) is None
    assert recovery_target(snapshot, Pose2D(0, 0, 0), .06,
                           100_000_000, .50, .06, .40, .15) is None
    assert recovery_target(snapshot, Pose2D(0, 0, 0), .01,
                           500_000_000, .50, .06, .40, .15) is None


def test_invalid_path_and_path_behind_vehicle_cannot_recover():
    assert snapshot_path(((0, .3), (0, .2)), Pose2D(0, 0, 0), 0, 0) is None
    snapshot = snapshot_path(((0, .2), (0, .8)), Pose2D(0, 0, 0), 0, 0)
    assert recovery_target(snapshot, Pose2D(1, 0, 0), .01,
                           100_000_000, .50, .06, .40, .15) is None


def test_inner_curve_needs_measured_boundary_and_clearance():
    target = RecoveryTarget(.10, .40, .01, .05)
    boundary = ((.24, .20), (.24, .60))
    assert right_boundary_clearance(target, boundary, .13)
    assert not right_boundary_clearance(target, boundary, .15)
    assert not right_boundary_clearance(target, ((.24, .20),), .13)
    assert not right_boundary_clearance(target, ((.24, .50), (.24, .80)), .13)
