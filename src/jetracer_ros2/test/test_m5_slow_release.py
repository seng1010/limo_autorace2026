"""M5 section end releases the sign latch; no changes to driving geometry."""
from types import SimpleNamespace as NS
import pytest

from jetracer_ros2.section_logic import SectionRule, SectionTracker, LinePassDetector
from test_mission_manager_transitions import modules, owner, lane, step, event


def test_m5_real_section_end_releases_sign_latch_without_changing_curvature(modules):
    n = owner(modules, 'M5', yolo_from_mission='M5')
    section = SectionTracker({'M5': SectionRule(slow_from_m=1.7, slow_to_m=5.)},
                             LinePassDetector())
    section.update(0., 'M5', None)
    n._yolo_callback(NS(data='slow_sign'))
    for distance in (1., 1.7, 3., 5., 5.01, 6.):
        out = section.update(distance, 'M5', None)
        n._speed_limit_callback(NS(data=.15 if out.slow else -1.))
        # Repeated detection of the same sign cannot relatch a finished zone.
        n._yolo_callback(NS(data='slow_sign'))
        step(n)
        cmd = n.cmd_pub.messages[-1]
        assert n._active == 'M5'
        assert cmd.linear.x == pytest.approx(.12 if distance <= 5. else .25)
        assert cmd.angular.z/cmd.linear.x == pytest.approx(.10/.25)
    assert n._m5_section_slow_done and not n._slow


def test_initial_no_limit_and_timeout_do_not_fake_m5_completion(modules):
    n = owner(modules, 'M5', yolo_from_mission='M5')
    n._yolo_callback(NS(data='slow_sign'))
    n._speed_limit_callback(NS(data=-1.))
    step(n)
    assert n._slow and not n._m5_section_slow_done
    n._speed_limit_callback(NS(data=.15))
    step(n, dt=2.)  # No explicit section end; retain old sign limit.
    assert n._slow and n.cmd_pub.messages[-1].linear.x == .12
    n._speed_limit_callback(NS(data=float('nan')))
    assert not n._m5_section_slow_done
    n._speed_limit_callback(NS(data=-1.))
    step(n)
    assert n.cmd_pub.messages[-1].linear.x == .25


def test_m5_release_does_not_clear_estop_or_override_lane_stop(modules):
    n = owner(modules, 'M5', yolo_from_mission='M5')
    n._yolo_callback(NS(data='slow_sign'))
    n._speed_limit_callback(NS(data=.15))
    event(n, 'ESTOP')
    n._speed_limit_callback(NS(data=-1.))
    step(n)
    assert n._operator_stop and n.cmd_pub.messages[-1].linear.x == 0.
    event(n, 'ESTOP_CLEAR')
    lane(n, 0., 0.)
    n._control_tick()
    assert n.cmd_pub.messages[-1].linear.x == 0.


@pytest.mark.parametrize('mission', ['M6', 'M3', 'M7', 'M2', 'M1', 'M4'])
def test_other_missions_do_not_use_m5_sign_release(modules, mission):
    n = owner(modules, mission)
    n._speed_limit_callback(NS(data=.15))
    n._speed_limit_callback(NS(data=-1.))
    assert not n._m5_section_slow_seen and not n._m5_section_slow_done
    assert n._active == mission


def test_next_mission_releases_m5_lifecycle_state(modules):
    n = owner(modules, 'M5', yolo_from_mission='M5')
    n._speed_limit_callback(NS(data=.15))
    n._speed_limit_callback(NS(data=-1.))
    assert n._m5_section_slow_done
    event(n, 'M5_DONE')
    assert n._active == 'M6'
    assert not n._m5_section_slow_done and not n._m5_section_slow_seen


def test_yolo_is_ignored_before_m7_by_default(modules):
    # 기본(yolo_from_mission=M7): M5 서행 표지판·M3 보행자 표지판 이벤트는 무시
    n = owner(modules, 'M5')
    n._yolo_callback(NS(data='slow_sign'))
    n._yolo_callback(NS(data='pedestrian'))
    assert not n._slow and n._last_pedestrian_s is None
    step(n)
    assert n.cmd_pub.messages[-1].linear.x == pytest.approx(.25)


def test_yolo_is_used_from_m7(modules):
    n = owner(modules, 'M7')
    n._yolo_callback(NS(data='pedestrian'))
    assert n._last_pedestrian_s is not None


def test_bad_yolo_from_mission_is_rejected(modules):
    with pytest.raises(ValueError):
        owner(modules, 'M5', yolo_from_mission='M9')
