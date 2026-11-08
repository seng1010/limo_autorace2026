"""M6 서행 유도선(지그재그) 모듈 테스트.

트랙 도면(2026Track) 치수를 BEV 픽셀로 옮겨 만든 가짜 영상을 쓴다.
  정상 차선 550mm = 360px  →  0.655 px/mm
  지그재그: 톱니 진폭 약 ±43mm(±28px), 주기 약 185mm(121px), 양쪽 같은 방향
"""

import numpy as np
import cv2

from jetracer_ros2.m6_zigzag import ACTIVE, ARMED, IDLE, M6Config, M6ZigzagGuide

H, W = 480, 640
LANE = 360.0
AMP, PERIOD = 28.0, 121.0


def _saw(y, phase=0.0):
    u = ((y + phase) / PERIOD) % 1.0
    return AMP * (4.0 * np.abs(u - 0.5) - 1.0)


def _draw(image, xs, ys, thickness=14):
    points = np.int32(np.column_stack((xs, ys))).reshape(-1, 1, 2)
    cv2.polylines(image, [points], False, 255, thickness)


def zigzag_frame(offset=0.0, slope=0.0, phase=0.0, mirrored=False, zz_from=-1.0, zz_to=1e9,
                 text=False, boundary=False):
    """offset: 차선 중심이 영상 중심에서 오른쪽으로 몇 px 인지, slope: dx/dy.
    zz_from/zz_to: 이 y 범위(영상 좌표)에만 지그재그, 나머지는 직선."""
    image = np.zeros((H, W), np.uint8)
    ys = np.arange(-10, H + 10, 1.0)
    center = W / 2 + offset + slope * (ys - H)
    zz = (ys >= zz_from) & (ys <= zz_to)
    for side in (-1, 1):
        saw = _saw(ys, phase) * (side if mirrored else 1)
        xs = center + side * 0.52 * LANE + np.where(zz, saw, 0.0)
        _draw(image, xs, ys)
    if text:   # 차선 가운데 마름모·글자 (가운데 무시 영역 안)
        cv2.rectangle(image, (W // 2 - 70, 120), (W // 2 + 70, 135), 255, -1)
        cv2.rectangle(image, (W // 2 - 8, 120), (W // 2 + 8, 230), 255, -1)
        cv2.polylines(image, [np.int32([[W // 2, 250], [W // 2 + 60, 330], [W // 2, 410],
                                        [W // 2 - 60, 330], [W // 2, 250]])], False, 255, 9)
    if boundary:  # 오른쪽 선 바깥의 흰 경계선
        _draw(image, center + 0.52 * LANE + 95.0, ys, 12)
    return image


def straight_frame(offset=0.0, slope=0.0, curve=0.0, s_amp=0.0, s_period=600.0, dashed=False):
    image = np.zeros((H, W), np.uint8)
    ys = np.arange(-10, H + 10, 1.0)
    center = (W / 2 + offset + slope * (ys - H) + curve * (H - ys) ** 2
              + s_amp * np.sin(2 * np.pi * ys / s_period))
    for side in (-1, 1):
        _draw(image, center + side * 0.5 * LANE, ys)
    if dashed:
        for y0 in range(0, H, 90):
            cv2.rectangle(image, (W // 2 - 6, y0), (W // 2 + 6, y0 + 45), 255, -1)
    return image


def run(guide, frames, t0=0.0, dt=1 / 15, stop=None):
    outs = []
    for i, frame in enumerate(frames):
        distance = stop[i] if stop is not None else None
        outs.append(guide.update(frame, t0 + i * dt, stop_line_distance=distance))
    return outs


def moving_zigzag(n, **kw):
    """차가 앞으로 가는 것처럼 톱니 위상을 조금씩 옮긴 프레임들."""
    return [zigzag_frame(phase=-6.0 * i, **kw) for i in range(n)]


# ---------------------------------------------------------------- 감지

def test_detects_in_phase_zigzag_and_enters_after_three_frames():
    guide = M6ZigzagGuide(M6Config())
    outs = run(guide, moving_zigzag(4))
    assert [o.zigzag for o in outs[:3]] == [True, True, True]
    assert outs[1].state == ARMED
    assert outs[2].state == ACTIVE and outs[2].just_entered


def test_detects_mirrored_zigzag_too():
    """양쪽이 거울처럼 반대로 꺾여 가운데가 안 흔들려도 감지돼야 한다."""
    guide = M6ZigzagGuide(M6Config())
    outs = run(guide, moving_zigzag(3, mirrored=True))
    assert outs[-1].state == ACTIVE


def test_waits_until_zigzag_is_close():
    """지그재그가 멀리(화면 위쪽)만 보이면 아직 켜지지 않는다."""
    guide = M6ZigzagGuide(M6Config())
    outs = run(guide, [zigzag_frame(zz_to=230.0) for _ in range(5)])
    assert outs[0].left.flips >= 3              # 멀리 있는 지그재그는 보이지만
    assert all(o.state == ARMED for o in outs)  # 아직 켜지지 않는다


def test_no_false_detection_on_normal_roads():
    scenes = {
        'straight': [straight_frame(offset=o) for o in (-40, 0, 40)],
        'angled': [straight_frame(slope=s) for s in (-0.2, 0.2)],
        'curve': [straight_frame(curve=c) for c in (-0.0012, 0.0008, 0.0012)],
        's_bend': [straight_frame(s_amp=a, s_period=p) for a, p in ((40, 700), (25, 500), (60, 900))],
        'dashed': [straight_frame(dashed=True)],
    }
    for name, frames in scenes.items():
        guide = M6ZigzagGuide(M6Config())
        outs = run(guide, frames * 4)
        assert not any(o.zigzag for o in outs), name
        assert all(o.state == ARMED for o in outs), name


# ---------------------------------------------------------------- 주행

def _active_guide(**kw):
    guide = M6ZigzagGuide(M6Config(blend_sec=0.0, **kw))
    run(guide, moving_zigzag(3))
    assert guide.state == ACTIVE
    return guide


def _settle(guide, frames, t0=1.0):
    return run(guide, frames, t0=t0)[-1]


def test_centered_with_text_and_diamond_goes_straight():
    guide = _active_guide()
    out = _settle(guide, moving_zigzag(30, text=True, boundary=True))
    assert abs(out.steering) < 0.05
    assert abs(out.lateral_error) < 0.03


def test_steers_back_toward_lane_center():
    """차선 중심이 오른쪽에 있으면(차가 왼쪽으로 치우침) 오른쪽(음수)으로 조향."""
    guide = _active_guide()
    out = _settle(guide, [zigzag_frame(offset=60.0, phase=-6.0 * i) for i in range(30)])
    assert out.lateral_error > 0.1
    assert out.steering < -0.1
    guide = _active_guide()
    out = _settle(guide, [zigzag_frame(offset=-60.0, phase=-6.0 * i) for i in range(30)])
    assert out.steering > 0.1


def test_steering_is_limited_and_smooth():
    guide = _active_guide()
    outs = run(guide, [zigzag_frame(offset=150.0, phase=-6.0 * i) for i in range(40)], t0=1.0)
    steer = np.array([o.steering for o in outs])
    assert np.all(np.abs(steer) <= guide.config.max_steer + 1e-9)
    assert np.all(np.abs(np.diff(steer)) <= guide.config.max_steer_rate + 1e-9)


def test_does_not_follow_the_teeth():
    """차가 똑바로 가운데 있으면 톱니 위상이 바뀌어도 조향이 거의 흔들리지 않는다."""
    guide = _active_guide()
    outs = run(guide, moving_zigzag(60), t0=1.0)
    steer = np.array([o.steering for o in outs[20:]])
    assert np.ptp(steer) < 0.08


def test_one_side_only_still_guides():
    guide = _active_guide()
    frames = []
    for i in range(20):
        frame = zigzag_frame(phase=-6.0 * i)
        frame[:, W // 2:] = 0          # 오른쪽 선이 안 보임
        frames.append(frame)
    out = _settle(guide, frames)
    assert out.state == ACTIVE
    assert abs(out.steering) < 0.1


# ---------------------------------------------------------------- 끝내기

def test_exits_when_zigzag_ends_and_sends_done_once():
    guide = _active_guide()
    outs = run(guide, moving_zigzag(20) + [straight_frame() for _ in range(10)], t0=0.5)
    exits = [o for o in outs if o.just_exited]
    assert len(exits) == 1 and exits[0].reason == 'zigzag ended'
    assert sum(o.send_done for o in outs) == 1
    assert outs[-1].state == IDLE
    # 쿨다운 동안 다시 지그재그가 보여도 켜지지 않는다
    later = run(guide, moving_zigzag(6), t0=5.0)
    assert all(o.state == IDLE for o in later)


def test_respects_minimum_active_time():
    guide = _active_guide()
    # 켜진 직후(1초 이내) 지그재그가 잠깐 안 보여도 끝내지 않는다
    outs = run(guide, [straight_frame() for _ in range(8)], t0=0.2, dt=0.05)
    assert all(o.state == ACTIVE for o in outs)


def test_ignores_start_line_then_uses_stop_line():
    """켜질 때 보이는 시작 가로선은 무시, 지나친 뒤 나오는 정지선에서 M6_DONE (조향은 계속)."""
    guide = _active_guide()
    stop = [0.4] * 5 + [-1.0] * 8 + [1.5, 1.2, 0.7, 0.6, 0.5]
    outs = run(guide, moving_zigzag(len(stop)), t0=0.2, stop=stop)
    done = [i for i, o in enumerate(outs) if o.send_done]
    assert done == [len(stop) - 1]
    assert outs[done[0]].reason == 'stop line ahead'
    assert outs[-1].state == ACTIVE
    # 이후 지그재그가 끝나도 M6_DONE 은 다시 보내지 않는다
    tail = run(guide, [straight_frame() for _ in range(10)], t0=5.0)
    assert not any(o.send_done for o in tail)


def test_lost_lines_returns_to_armed_without_done():
    guide = _active_guide()
    outs = run(guide, [np.zeros((H, W), np.uint8) for _ in range(25)], t0=0.5)
    assert outs[-1].state == ARMED
    assert not any(o.send_done for o in outs)


def test_timeout_finishes():
    guide = _active_guide(max_active_sec=2.0)
    outs = run(guide, moving_zigzag(40), t0=0.1, dt=0.1)
    assert any(o.just_exited and o.reason == 'timeout' for o in outs)
    assert sum(o.send_done for o in outs) == 1


def test_mission_gate():
    guide = M6ZigzagGuide(M6Config())
    guide.set_enabled(False)
    assert all(o.state == IDLE for o in run(guide, moving_zigzag(5)))
    guide.set_enabled(True)
    assert run(guide, moving_zigzag(4), t0=1.0)[-1].state == ACTIVE
    guide.set_enabled(False)
    assert guide.state != ACTIVE


def test_debug_draw_runs():
    guide = _active_guide()
    out = run(guide, moving_zigzag(2), t0=1.0)[-1]
    canvas = guide.draw(zigzag_frame(), out)
    assert canvas.shape == (H, W, 3)
