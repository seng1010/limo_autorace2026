"""Physical-lane confirmation must precede each fork transition."""

from jetracer_ros2.fork_policy import ForkPolicy


def test_missing_direction_or_lane_position_stops():
    policy = ForkPolicy()
    policy.event('FORK_ENTER')
    assert policy.mode is None
    policy.event('FORK_LEFT')
    assert policy.mode is None
    policy.event('FORK_POSITION_RIGHT')
    assert policy.mode == 'center'


def test_right_to_left_requires_confirmed_center_stage():
    policy = ForkPolicy()
    policy.event('FORK_POSITION_RIGHT')
    policy.event('FORK_LEFT')
    policy.event('FORK_ENTER')
    assert policy.mode == 'center'
    assert not policy.event('FORK_RIGHT_REACHED')
    assert policy.mode == 'center'
    assert policy.event('FORK_CENTER_REACHED')
    assert policy.mode == 'left'
    policy.event('FORK_EXIT')
    assert policy.mode is None and policy.position is None


def test_test_default_still_needs_position_confirmation():
    policy = ForkPolicy('right')
    policy.event('FORK_ENTER')
    assert policy.mode is None
    policy.event('FORK_POSITION_CENTER')
    assert policy.mode == 'right'
