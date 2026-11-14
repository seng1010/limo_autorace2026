import numpy as np
import pytest

from jetracer_ros2.m1_traffic_light import M1Config, TrafficLightMission
from jetracer_ros2.stop_approach import (
    FullStopDetector, StopApproachConfig, StopApproachController)
from jetracer_ros2.traffic_light_logic import (
    TrafficLightConfig, TrafficLightFilter, frame_color, lit_color_ratios)

BGR = {'red': (0, 0, 255), 'yellow': (0, 220, 255), 'green': (0, 200, 0)}


# ---- 신호 색 판단 ------------------------------------------------------------- #

def _light_image(color, box=(300, 100, 340, 200)):
    image = np.full((480, 640, 3), 40, dtype=np.uint8)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = (30, 30, 30)                      # 신호등 몸체
    cy = {'red': y1 + 15, 'yellow': (y1 + y2) // 2, 'green': y2 - 15}[color]
    image[cy - 10:cy + 10, x1 + 10:x2 - 10] = BGR[color]     # 켜진 불
    return image


def _det(label, box=(300, 100, 340, 200), conf=0.9):
    return {'label': label, 'confidence': conf, 'bbox': list(box)}


def test_hsv_ratio_finds_lit_color():
    for color in ('red', 'yellow', 'green'):
        ratios = lit_color_ratios(_light_image(color), (300, 100, 340, 200), TrafficLightConfig())
        assert max(ratios, key=ratios.get) == color


def test_frame_color_uses_label_and_hsv():
    cfg = TrafficLightConfig()
    image = _light_image('red')
    assert frame_color([_det('red')], image, image.shape, cfg) == 'red'
    assert frame_color([_det('traffic_red')], image, image.shape, cfg) == 'red'
    # 라벨은 초록인데 실제 불빛은 빨강 → 버림
    assert frame_color([_det('green')], image, image.shape, cfg) is None


def test_frame_color_ignores_small_low_conf_and_other_labels():
    cfg = TrafficLightConfig(min_area_ratio=0.05)
    image = _light_image('green')
    assert frame_color([_det('green')], image, image.shape, cfg) is None            # 너무 작음
    cfg = TrafficLightConfig()
    assert frame_color([_det('green', conf=0.2)], image, image.shape, cfg) is None  # 신뢰도 낮음
    assert frame_color([_det('pedestrian')], image, image.shape, cfg) is None


# ---- (B) 몸체 클래스 ---------------------------------------------------------- #

def test_housing_label_reads_color_from_lamp_position():
    cfg = TrafficLightConfig()
    for color in ('red', 'yellow', 'green'):
        image = _light_image(color)
        assert frame_color([_det('traffic_light')], image, image.shape, cfg) == color


def test_housing_rejects_when_nothing_lit():
    cfg = TrafficLightConfig()
    image = np.full((480, 640, 3), 40, dtype=np.uint8)
    image[100:200, 300:340] = (30, 30, 30)              # 함체만, 불 꺼짐
    assert frame_color([_det('traffic_light')], image, image.shape, cfg) is None


def test_housing_rejects_when_lamp_position_and_color_disagree():
    """위 칸(=적색 자리)인데 초록불 → 신호등이 아니거나 오검출로 보고 버림."""
    image = np.full((480, 640, 3), 40, dtype=np.uint8)
    image[100:200, 300:340] = (30, 30, 30)
    image[115:135, 310:330] = BGR['green']              # 위 칸에 초록
    assert frame_color([_det('traffic_light')], image, image.shape, TrafficLightConfig()) is None


def test_housing_rejects_bright_screen_without_single_lamp():
    """광고 모니터처럼 박스 전체가 고르게 밝으면 칸 구분이 안 돼 버림."""
    image = np.full((480, 640, 3), 40, dtype=np.uint8)
    image[100:200, 300:340] = BGR['red']                # 전체가 빨강
    assert frame_color([_det('traffic_light')], image, image.shape, TrafficLightConfig()) is None


def test_frame_color_picks_largest_light():
    cfg = TrafficLightConfig(verify_hsv=False)
    dets = [_det('green', (10, 10, 20, 30)), _det('red', (300, 100, 340, 200))]
    assert frame_color(dets, None, (480, 640), cfg) == 'red'


def test_filter_confirms_after_n_frames_and_times_out():
    f = TrafficLightFilter(TrafficLightConfig(confirm_frames=3, lost_timeout_sec=1.0))
    assert f.update(0.0, 'red') == 'unknown'
    assert f.update(0.1, 'red') == 'unknown'
    assert f.update(0.2, 'red') == 'red'
    assert f.update(0.3, 'green') == 'red'          # 한 번 튄 건 무시
    assert f.update(0.4, 'red') == 'red'
    assert f.update(0.5, None) == 'red'             # 잠깐 놓쳐도 유지
    assert f.tick(2.0) == 'unknown'                 # 오래 안 보이면 unknown


# ---- M1 판단 (차량 시뮬레이션) ------------------------------------------------------ #

def _mission(**kwargs):
    approach = StopApproachController(StopApproachConfig(stop_offset_m=0.01))
    return TrafficLightMission(M1Config(**kwargs), approach, FullStopDetector(0.01, 0.3))


def _simulate(mission, light_at, start_distance=1.2, light_s=0.0, dt=0.05, lag=0.15,
              until=60.0):
    """light_at(t) -> 신호 색. light_s: 신호등을 본 위치 (None = 못 봄)."""
    t, v, s = 0.0, 0.0, 0.0
    log = []
    while t < until:
        d = start_distance - s
        decision = mission.update(t, d if d >= -0.1 else None, v, s, light_at(t),
                                  last_light_s=light_s)
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


def _states(log):
    return [state for _, state, _, _ in log]


def test_red_then_green_stops_on_line_and_goes():
    log = _simulate(_mission(), lambda t: 'red' if t < 15.0 else 'green')
    states = _states(log)
    assert 'hold' in states and states[-1] == 'done'
    hold = [(t, d) for t, state, d, _ in log if state == 'hold']
    assert abs(hold[0][1]) <= 0.05                       # 정지선 ±5cm
    go_t = next(t for t, state, _, _ in log if state == 'go')
    assert 15.0 <= go_t < 15.0 + 0.2                     # 녹색 확정 즉시 출발 (15초 규정)
    assert all(v <= 0.01 for _, state, _, v in log if state == 'hold')


def test_yellow_and_unknown_also_stop():
    for light in ('yellow', 'unknown'):
        log = _simulate(_mission(), lambda t, c=light: c if t < 15.0 else 'green')
        assert 'hold' in _states(log), light


def test_green_on_arrival_passes_without_stopping():
    log = _simulate(_mission(), lambda t: 'green')
    states = _states(log)
    assert 'passing' in states and 'hold' not in states and states[-1] == 'done'
    assert min(v for t, _, _, v in log if t > 2.0) > 0.1  # 멈추지 않음


def test_green_turns_red_while_passing_stops():
    log = _simulate(_mission(), lambda t: 'green' if t < 3.4 else 'red', until=10.0)
    states = _states(log)
    assert 'passing' in states and 'stopping' in states
    hold = [d for _, state, d, _ in log if state == 'hold']
    assert hold and abs(hold[0]) <= 0.05


def test_stop_on_green_arrival_option():
    log = _simulate(_mission(stop_on_green_arrival=True), lambda t: 'green')
    states = _states(log)
    assert 'stopping' in states and 'passing' not in states
    assert 'go' in states and states[-1] == 'done'
    go_i = states.index('go')
    assert log[go_i - 1][3] <= 0.01                      # 완전정지 확인 후 바로 출발


def test_max_wait_goes_without_green():
    log = _simulate(_mission(max_wait_sec=10.0), lambda t: 'red')
    hold_t = [t for t, state, _, _ in log if state == 'hold']
    go_t = next(t for t, state, _, _ in log if state == 'go')
    assert go_t - hold_t[0] == pytest.approx(10.0, abs=0.2)
    assert _states(log)[-1] == 'done'


def test_requires_light_by_default_and_skips():
    log = _simulate(_mission(), lambda t: 'unknown', light_s=None)
    states = set(_states(log))
    assert 'stopping' not in states and 'hold' not in states
    assert _states(log)[-1] == 'skipped'


def test_line_only_mode_for_testing_without_yolo():
    log = _simulate(_mission(trigger='light_or_line'), lambda t: 'red' if t < 15 else 'green',
                    light_s=None)
    assert 'hold' in _states(log) and _states(log)[-1] == 'done'


def test_invalid_trigger():
    with pytest.raises(ValueError):
        _mission(trigger='sign')
