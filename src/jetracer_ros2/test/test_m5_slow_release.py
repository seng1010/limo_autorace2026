"""M5 slow zone is distance-only (section_node); no changes to driving geometry."""
from types import SimpleNamespace as NS
import pytest

from jetracer_ros2.section_logic import SectionRule, SectionTracker, LinePassDetector
from test_mission_manager_transitions import modules, owner, lane, step, event


def test_m5_slow_only_between_1_45_and_3_0_m_ignoring_sign(modules):
    """서행은 출발선 기준 1.45~3.0 m 거리로만. YOLO 서행 표지판은 무시한다."""
    n = owner(modules, 'M5')
    section = SectionTracker({'M5': SectionRule(slow_from_m=1.45, slow_to_m=3.0)},
                             LinePassDetector())
    section.update(0., 'M5', None)
    for distance in (0.5, 1.0, 1.44, 1.45, 2.0, 3.0, 3.01, 4.0):
        n._yolo_callback(NS(data='slow_sign'))      # 표지판을 봐도 서행을 켜지 않음
        out = section.update(distance, 'M5', 0.3 if distance < 1.0 else None)
        n._speed_limit_callback(NS(data=.15 if out.slow else -1.))
        step(n)
        cmd = n.cmd_pub.messages[-1]
        assert n._active == 'M5'
        assert cmd.linear.x == pytest.approx(.15 if 1.45 <= distance <= 3.0 else .25)
        assert cmd.angular.z/cmd.linear.x == pytest.approx(.10/.25)
    assert n._m5_section_slow_done and not n._slow


def test_initial_no_limit_and_timeout_do_not_fake_m5_completion(modules):
    n = owner(modules, 'M5')
    n._yolo_callback(NS(data='slow_sign'))
    n._speed_limit_callback(NS(data=-1.))
    step(n)
    assert not n._slow and not n._m5_section_slow_done
    assert n.cmd_pub.messages[-1].linear.x == .25
    n._speed_limit_callback(NS(data=float('nan')))
    assert not n._m5_section_slow_done


def test_m5_release_does_not_clear_estop_or_override_lane_stop(modules):
    n = owner(modules, 'M5')
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
    n = owner(modules, 'M5')
    n._speed_limit_callback(NS(data=.15))
    n._speed_limit_callback(NS(data=-1.))
    assert n._m5_section_slow_done
    event(n, 'M5_DONE')
    assert n._active == 'M6'
    assert not n._m5_section_slow_done and not n._m5_section_slow_seen


def test_m5_start_boost_until_slow_zone_then_normal(modules):
    """M5 출발 ~ 1.45 m 만 0.5 (곡률 유지), 서행 구간은 서행, 그 뒤는 원래 속도."""
    n = owner(modules, 'M5', m5_start_speed=.5, m5_start_until_m=1.45)
    section = SectionTracker({'M5': SectionRule(slow_from_m=1.45, slow_to_m=3.0)},
                             LinePassDetector())
    for distance in (0., 0.5, 1.4, 1.45, 2.0, 3.01, 4.0):
        n._odom.s = distance
        out = section.update(distance, 'M5', None)
        n._speed_limit_callback(NS(data=.2 if out.slow else -1.))
        step(n)
        cmd = n.cmd_pub.messages[-1]
        expected = .5 if distance < 1.45 else .2 if out.slow else .25
        assert cmd.linear.x == pytest.approx(expected)
        assert cmd.angular.z/cmd.linear.x == pytest.approx(.10/.25)


@pytest.mark.parametrize('mission', ['M6', 'M3', 'M2', 'M1'])
def test_m5_start_boost_not_used_in_other_missions(modules, mission):
    n = owner(modules, mission, m5_start_speed=.5)
    step(n)
    assert n.cmd_pub.messages[-1].linear.x <= .25
