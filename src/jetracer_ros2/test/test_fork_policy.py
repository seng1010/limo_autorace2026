import pytest
from jetracer_ros2.fork_policy import ForkPolicy


def test_default_stop_and_explicit_mission_override():
    policy = ForkPolicy()
    policy.event('FORK_ENTER')
    assert policy.active and policy.direction is None
    policy.event('FORK_RIGHT')
    assert policy.direction == 'right'


def test_test_default_is_overridden_and_does_not_leak_next_mission():
    policy = ForkPolicy('left')
    policy.event('FORK_ENTER')
    assert policy.direction == 'left'
    policy.event('FORK_RIGHT')
    assert policy.direction == 'right'
    policy.event('FORK_EXIT')
    assert not policy.active and policy.requested is None
    policy.event('FORK_ENTER')
    assert policy.direction == 'left'


def test_invalid_policy_fails_closed_and_unrelated_events_are_ignored():
    with pytest.raises(ValueError):
        ForkPolicy('random')
    policy = ForkPolicy()
    assert not policy.event('ESTOP_CLEAR')
    assert not policy.active
