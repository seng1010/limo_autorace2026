import pytest

from jetracer_ros2.m3_pedestrian import M3Config, PedestrianStopMission
from jetracer_ros2.stop_approach import (
    FullStopDetector, StopApproachConfig, StopApproachController)


def _mission(**kwargs):
    approach = StopApproachController(StopApproachConfig(stop_offset_m=0.01))
    return PedestrianStopMission(M3Config(**kwargs), approach, FullStopDetector(0.01, 0.3))


# ---- 감속 프로파일 / 완전정지 -------------------------------------------------- #

def test_speed_profile():
    ctrl = StopApproachController(StopApproachConfig(
        approach_start_m=0.5, gain=0.8, creep_speed=0.06, max_speed=0.25, stop_offset_m=0.01))
    assert ctrl.speed_limit(None) is None
    assert ctrl.speed_limit(0.8) is None
    assert ctrl.speed_limit(0.3) == pytest.approx(0.8 * 0.29)
    assert ctrl.speed_limit(0.05) == pytest.approx(0.06)        # 최저 속도
    assert ctrl.speed_limit(0.01) == 0.0
    assert ctrl.speed_limit(0.0) == 0.0


def test_full_stop_needs_hold_time():
    det = FullStopDetector(0.01, 0.3)
    assert det.update(0.0, 0.0) is None
    assert det.update(0.2, 0.005) is None
    assert det.update(0.35, 0.0) == pytest.approx(0.3)          # 확인 시각 = 0.0 + 0.3
    det.reset()
    assert det.update(1.0, 0.05) is None                        # 움직이면 다시 처음부터


# ---- 차량 시뮬레이션으로 전체 흐름 ------------------------------------------------ #

def _simulate(mission, start_distance=1.2, sign_s=0.0, dt=0.05, lag=0.15,
              line_visible=lambda d: True, until=60.0):
    """1차 지연 차량 모델. 거리 = 앞바퀴 -> 정지선, s = 출발 후 이동거리.
    sign_s: 보행자 표지판을 본 위치 (None = 못 봄, 음수 = M3 시작 전에 봄)."""
    t, v, s = 0.0, 0.0, 0.0
    log = []
    while t < until:
        d = start_distance - s
        visible = d >= -0.10 and line_visible(d)
        decision = mission.update(t, d if visible else None, v, s, last_sign_s=sign_s)
        target = 0.25 if decision.speed_limit is None else min(0.25, decision.speed_limit)
        v += (target - v) * min(1.0, dt / lag)
        if abs(v) < 1e-4:
            v = 0.0
        s += v * dt
        log.append((t, decision.state, d, v))
        if decision.done:
            break
        t += dt
    return log


def test_full_sequence_stops_on_line_and_waits():
    log = _simulate(_mission())
    states = [state for _, state, _, _ in log]
    assert states[0] == 'approach' and states[-1] == 'done'
    hold = [(t, d) for t, state, d, _ in log if state == 'hold']
    assert hold, 'hold 상태가 없음'
    assert abs(hold[0][1]) <= 0.05                       # 규정 ±5cm
    go_t = next(t for t, state, _, _ in log if state == 'go')
    first_stop_t = next(t for t, _, _, v in log if t > 0.5 and v == 0.0)
    assert go_t - first_stop_t >= 3.0                    # 규정 3초 이상
    assert go_t - first_stop_t < 15.0                    # 규정 15초 안


def test_never_moves_during_hold():
    log = _simulate(_mission())
    hold_v = [v for _, state, _, v in log if state == 'hold']
    assert all(v <= 0.01 for v in hold_v)                # 완전정지 기준 이하
    assert all(b <= a for a, b in zip(hold_v, hold_v[1:]))  # 다시 가속하지 않음


def test_default_needs_sign_and_skips_without_it():
    """표지판을 못 보면 정지선이 있어도 서지 않고, 일정 거리 뒤 M3를 건너뜀."""
    log = _simulate(_mission(), sign_s=None)
    states = {state for _, state, _, _ in log}
    assert 'stopping' not in states and 'hold' not in states
    assert log[-1][1] == 'skipped'


def test_sign_seen_before_m3_still_counts():
    """M6 구간(M3 시작 전)에서 본 표지판도 인정."""
    log = _simulate(_mission(), sign_s=-0.5)
    assert 'hold' in {state for _, state, _, _ in log}


def test_old_sign_is_ignored():
    """표지판을 너무 오래 전에 봤으면 그 뒤의 정지선은 무시."""
    log = _simulate(_mission(), sign_s=-2.0)
    assert 'hold' not in {state for _, state, _, _ in log}


def test_line_only_mode_for_testing_without_yolo():
    log = _simulate(_mission(trigger='sign_or_line'), sign_s=None)
    assert log[-1][1] == 'done'
    assert 'hold' in {state for _, state, _, _ in log}


def test_blind_zone_line_lost_means_stop():
    """정지선 값이 중간에 끊기면(추적 실패) 그 자리에 선다."""
    log = _simulate(_mission(), line_visible=lambda d: not (0.2 < d < 0.3))
    stopping = [(d, v) for _, state, d, v in log if state in ('stopping', 'hold')]
    assert min(d for d, _ in stopping) > 0.1             # 선 앞에서 멈춤 (지나가지 않음)


def test_ignores_sign_after_go():
    mission = _mission()
    log = _simulate(mission)
    assert log[-1][1] == 'done'
    decision = mission.update(100.0, 0.3, 0.2, 10.0, last_sign_s=9.9)
    assert decision.done                                  # 다시 서지 않음


def test_invalid_trigger():
    with pytest.raises(ValueError):
        _mission(trigger='both')
