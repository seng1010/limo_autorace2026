import pytest
from jetracer_ros2.lane_steering import right_curve_metric_target, left_curve_preview_target

@pytest.mark.parametrize('target', [-.7, 0., .3])
@pytest.mark.parametrize('confirmed', [True, False])
def test_floor_disabled_preserves_pure_pursuit_direction(target, confirmed):
    assert right_curve_metric_target(target, .4, .471, .1, .2, confirmed,
                                     enabled=False) == target

def test_preview_switch_is_independent():
    assert left_curve_preview_target(.3, .2, .471, .5, .16, enabled=False) == .3
    assert left_curve_preview_target(.3, .2, .471, .5, .16) < .3
    assert right_curve_metric_target(.3, .4, .471, .1, .2, True) < 0
