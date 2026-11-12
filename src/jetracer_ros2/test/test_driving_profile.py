from types import SimpleNamespace

from jetracer_ros2.driving_profile import (
    ZigzagTracker, select_speed, yellow_zigzag_evidence,
)


def test_speed_priority_and_requested_values():
    assert select_speed(False, False, .70, .30, .18) == .70
    assert select_speed(False, True, .70, .30, .18) == .30
    assert select_speed(True, True, .70, .30, .18) == .18


def test_yellow_zigzag_required_not_white_dash_or_invalid_geometry():
    result = SimpleNamespace(valid=True, zigzag=True,
                             left_type='yellow/solid/zigzag',
                             right_type='white/dashed/zigzag')
    assert yellow_zigzag_evidence(result)
    result.left_type = 'unobserved'
    assert not yellow_zigzag_evidence(result)
    result.left_type = 'yellow/dashed/zigzag'
    assert not yellow_zigzag_evidence(result)
    result.left_type = 'yellow/solid/zigzag'
    result.valid = False
    assert not yellow_zigzag_evidence(result)


def test_zigzag_slows_immediately_and_short_gaps_do_not_reset_dwell():
    tracker = ZigzagTracker()
    assert tracker.update(True, 0.0)
    assert tracker.update(False, 0.2)
    assert tracker.update(True, 0.3)
    assert tracker.update(True, 2.05)
    # A gap >0.35 resets unconfirmed evidence, so the above is a new episode.
    assert not tracker.confirmed
    assert tracker.update(False, 2.36)
    assert not tracker.update(False, 2.41)


def test_confirmed_zigzag_holds_for_one_and_half_seconds():
    tracker = ZigzagTracker()
    for step in range(11):
        assert tracker.update(True, step * .2)
    assert tracker.confirmed
    assert tracker.update(False, 3.49)
    assert not tracker.update(False, 3.51)


def test_new_episode_and_clock_reset_clear_old_confirmation():
    tracker = ZigzagTracker()
    tracker.update(True, 1.0)
    tracker.update(True, 3.0)  # A long gap begins a new episode.
    assert not tracker.confirmed
    tracker.update(True, 2.0)
    assert tracker.first_seen == 2.0
    assert not tracker.confirmed
