"""A missing metric target must not silently fall back to pixel steering."""

from jetracer_ros2.lane_ackermann import metric_target_ready, target_to_inner_angle


def test_metric_target_gate_rejects_missing_and_unphysical_points():
    for lateral, forward in ((None, .4), (.1, None), (float('nan'), .4),
                             (.1, float('inf')), (.1, 0), (.1, -.2)):
        assert not metric_target_ready(lateral, forward)


def test_metric_target_gate_accepts_ground_point_for_ackermann():
    assert metric_target_ready(.1, .4)
    assert target_to_inner_angle(.1, .4) < 0
