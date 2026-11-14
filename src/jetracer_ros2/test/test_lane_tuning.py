import pytest
from jetracer_ros2.lane_tuning import validate_changes

CURRENT = dict(yellow_hue_min=15, yellow_hue_max=35, windows=10, bottom_tracking_windows=3)

def test_valid_and_atomic_validation():
    assert validate_changes({'windows': 20}, CURRENT, False) == {'windows': 20}
    assert CURRENT['windows'] == 10

@pytest.mark.parametrize('change', [
    {'windows': 0}, {'windows': 5.5}, {'windows': True},
    {'normal_speed': float('nan')}, {'max_steer_angle': 1.0},
    {'publish_commands': True}, {'yellow_hue_min': 50},
    {'windows': 5, 'bottom_tracking_windows': 6}, {'detect_yellow': 1}])
def test_reject_invalid(change):
    with pytest.raises(ValueError):
        validate_changes(change, CURRENT, False)

def test_command_enabled_is_always_locked():
    with pytest.raises(ValueError, match='locked'):
        validate_changes({'windows': 20}, CURRENT, True)
