"""lane3_gap: 같은 자리에서 또 막히면 후진으로 차 앞을 열린 쪽으로 돌린다 (REV_AIM_*). 첫 후진은 전과 같다."""
import math

import numpy as np

from jetracer_ros2 import lane3_gap as G

N, AMIN = 250, -math.radians(110.0)
INC = math.radians(220.0) / (N - 1)


def _scan(pose, wall_x=None, wall_y_max=None):
    """세계 좌표 x = wall_x 에 선 벽(y < wall_y_max 인 부분만, None 이면 전부)을 차 자세에서 본 라이다."""
    r = np.full(N, np.inf, np.float32)
    if wall_x is not None:
        x, y, yaw = pose
        lx, ly = x + G.CAM * math.cos(yaw), y + G.CAM * math.sin(yaw)      # 라이다 = 카메라 바닥 투영점(코드 가정)
        for i in range(N):
            a = yaw + AMIN + i * INC
            if math.cos(a) > 1e-3:
                t = (wall_x - lx) / math.cos(a)
                if t > 0 and (wall_y_max is None or ly + t * math.sin(a) < wall_y_max):
                    r[i] = t
    return (r, AMIN, INC, 0.05)


def _drive(core, pose, wall_x, wall_y_max, steps, t0=0.0, dt=0.1):
    """core 의 명령대로 자전거 모델(뒤 차축 기준)을 굴린다. 반환: 자세, [(상태, v, steer, psi, 그때 x, 그때 yaw)]."""
    x, y, yaw = pose
    log = []
    for i in range(steps):
        out = core.step_obs(None, (x, y, yaw), _scan((x, y, yaw), wall_x, wall_y_max), None, t0 + i * dt)
        log.append((out['note'].split()[1], out['v'], out['steer'], core.psi, x, yaw))
        ds = out['v'] * dt
        yaw += ds * math.tan(out['steer']) / G.WB
        x, y = x + ds * math.cos(yaw), y + ds * math.sin(yaw)
    return (x, y, yaw), log


def _core_reversing(aim, psi=0.0):
    core = G.GapCore()
    core.start(psi, (0.0, 0.0, psi))
    core.back_room = 1.0
    core.rev_left = core.rev_start = G.REVERSE_M
    core.rev_count, core.aim = 2, aim
    return core


def test_first_kind_of_reverse_is_unchanged():
    core = _core_reversing(0, psi=0.2)                       # aim 0 = 전과 같은 후진: 도로와 나란히 펴는 조향
    out = core.step_obs(None, (0.0, 0.0, 0.2), _scan((0.0, 0.0, 0.2)), None, 0.0)
    assert out['v'] == -G.V_REVERSE
    assert abs(out['steer'] - min(G.REV_STEER_GAIN * 0.2, G.REV_STEER_MAX)) < 1e-6


def test_aimed_reverse_goes_straight_first_then_turns_the_nose_to_the_open_side():
    for aim in (1, -1):
        core = _core_reversing(aim)
        pose, log = _drive(core, (0.0, 0.0, 0.0), wall_x=0.75, wall_y_max=None, steps=70)     # 앞이 통째로 막힘: 길이 안 보인다
        rev = [row for row in log if row[0] == 'REVERSE']
        assert all(abs(row[2]) < 1e-6 for row in rev[:15])                                     # 0.15 m 까지는 곧게
        assert any(row[2] * aim < -math.radians(5) for row in rev)                             # 왼쪽이 열렸으면 오른쪽으로 꺾고 후진
        end_psi, end_x, end_yaw = rev[-1][3], rev[-1][4], rev[-1][5]
        assert math.radians(8) < end_psi * aim <= math.radians(G.REV_AIM_DEG) + 1e-3           # 차 앞이 열린 쪽으로, 목표 각을 넘지 않는다
        assert abs(end_yaw - end_psi) < math.radians(3)                                        # 실제 방향도 그렇게 돌았다
        assert -G.REVERSE_M - 0.03 < end_x < -G.REVERSE_M + 0.12                                # 물러난 거리는 전과 같은 한도


def test_aimed_reverse_leaves_as_soon_as_the_open_side_shows_a_way():
    core = _core_reversing(1)
    # 가운데·오른쪽(y < 0.20)만 막힌 벽: 조금 물러나면 왼쪽으로 지나갈 길이 나온다
    pose, log = _drive(core, (0.0, 0.0, 0.0), wall_x=0.75, wall_y_max=0.20, steps=60)
    states = [row[0] for row in log]
    n_rev = states.index(next(s for s in states if s != 'REVERSE'))
    assert 0.15 <= n_rev * 0.1 * G.V_REVERSE < G.REVERSE_M - 0.05          # 한도(0.5 m) 전에 끝낸다
    assert all(s != 'REVERSE' for s in states[n_rev:n_rev + 5])
    assert core.tgt is not None and core.tgt - core.latP > 0.2 or pose[1] > 0.05    # 왼쪽 줄을 골라 간다


def _blocked_again(rev_count_before, left, right):
    """정면이 막혀 선 차: 같은 자리에서 이미 rev_count_before 번 후진했고, 막히기 전 좌/우 갈 수 있는 거리가 left/right."""
    core = G.GapCore()
    core.start(0.0, (0.0, 0.0, 0.0))
    core.back_room = 1.0
    core.rev_count, core.s_stuck = rev_count_before, core.sP
    for k in range(12):
        core.side_hist.append((core.sP - 0.60 + 0.04 * k, left, right))
    for i in range(30):
        core.step_obs(None, (0.0, 0.0, 0.0), _scan((0.0, 0.0, 0.0), wall_x=0.45), None, i * 0.1)
        if core.rev_left > 0.0:
            break
    return core


def test_second_block_at_the_same_place_aims_to_the_side_that_was_open():
    assert _blocked_again(1, 2.4, 0.5).aim == 1
    core = _blocked_again(1, 0.5, 2.4)
    assert core.aim == -1 and '오른쪽' in core.message


def test_first_block_reverses_as_before():
    core = _blocked_again(0, 2.4, 0.5)
    assert core.rev_count == 1 and core.aim == 0 and '열린 쪽' not in core.message


def test_no_clear_open_side_reverses_as_before():
    assert _blocked_again(1, 1.2, 1.0).aim == 0


def test_switch_off_reverses_as_before(monkeypatch):
    monkeypatch.setattr(G, 'REV_AIM_DEG', 0.0)
    assert _blocked_again(1, 2.4, 0.5).aim == 0
