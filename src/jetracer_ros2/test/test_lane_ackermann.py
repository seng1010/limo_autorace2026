import math

import pytest

from jetracer_ros2.lane_ackermann import target_to_inner_angle
from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega


def test_metric_target_uses_rear_axle_curvature_and_ros_direction():
    for lateral in (-0.05, 0.05):
        angle = target_to_inner_angle(lateral, 0.4)
        omega = inner_angle_to_omega(angle, 0.3)
        expected_curvature = -2.0 * lateral / (0.4 ** 2 + lateral ** 2)
        assert omega / 0.3 == pytest.approx(expected_curvature)
        assert math.copysign(1, angle) == math.copysign(1, -lateral)
    assert target_to_inner_angle(0, 0.4) == 0.0


def test_inner_angle_clamped_and_bad_calibration_rejected():
    assert target_to_inner_angle(0.4, 0.1) == pytest.approx(-0.42)
    for lateral, forward in ((float('nan'), .4), (.1, 0), (.1, -.2)):
        with pytest.raises(ValueError):
            target_to_inner_angle(lateral, forward)


def test_inverse_conversion_uses_same_geometry_as_target_controller():
    angle = target_to_inner_angle(.04, .35, wheelbase_m=.2, track_m=.14)
    omega = inner_angle_to_omega(angle, .2, wheelbase=.2, track=.14)
    assert omega / .2 == pytest.approx(-2 * .04 / (.35 ** 2 + .04 ** 2))
