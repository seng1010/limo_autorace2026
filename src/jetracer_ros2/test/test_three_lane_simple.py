import math
import os
import sys

import numpy as np

from jetracer_ros2 import three_lane_simple as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_three_lane_simple as SIM  # noqa: E402

FAR = S.NO_OBSTACLE


def scan_with(points):
    """points: 라이다 기준 (앞, 왼쪽) m -> 250점 (-110~110도)."""
    n, a0, inc = 250, -1.92, 3.84 / 249
    r = np.full(n, np.inf, np.float32)
    for x, y in points:
        for dy in (-0.01, 0.0, 0.01):
            i = int(round((math.atan2(y + dy, x) - a0) / inc))
            if 0 <= i < n:
                r[i] = math.hypot(x, y + dy)
    return r, a0, inc, 0.15


def box(x, y):
    """정차 LIMO 뒷면 (라이다에서 보이는 면) 점들."""
    return [(x, y + d) for d in np.linspace(-0.10, 0.10, 6)]


def core_at(c=0.0, th=0.0):
    k = S.SimpleThreeLane()
    k.start(c, th, (0.0, 0.0, 0.0), 0.0)
    return k


def test_odom_moves_lateral_position_and_camera_swings_with_heading():
    k = core_at(0.0, 0.0)
    # 왼쪽으로 30도 돌면서(제자리 회전 흉내) 카메라 투영점은 뒤 차축보다 옆으로 더 움직인다
    k._advance((0.0, 0.0, math.radians(30)))
    assert abs(k.c - S.CAM_FROM_ODOM * math.sin(math.radians(30))) < 1e-6
    # 그 방향으로 0.5 m
    k._advance((0.5 * math.cos(math.radians(30)), 0.5 * math.sin(math.radians(30)), math.radians(30)))
    assert abs(k.c - (S.CAM_FROM_ODOM + 0.5) * math.sin(math.radians(30))) < 1e-6
    assert abs(k.th + math.radians(30)) < 1e-9


def test_blocked_lane_moves_to_free_neighbour_and_prefers_goal_on_tie():
    for goal, want in ((None, S.DEFAULT_SIDE), (0, 0), (2, 2)):
        k = core_at()
        out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(box(1.0, 0.0)), goal, now=0.0)
        assert k.changing and k.lane == want, (goal, k.lane, out)


def test_picks_the_lane_that_stays_free_longer():
    k = core_at()
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(box(1.0, 0.0) + box(1.9, -0.535)), 2, now=0.0)
    assert k.lane == 0                  # 표지판은 B 지만 오른쪽은 곧 막힌다


def test_obstacle_beside_with_gap_does_not_block_change_but_one_ahead_does():
    k = core_at()
    # 옆 차로 장애물이 거의 다 지나감 (라이다 바로 옆) -> 옮겨도 된다
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(box(0.0, 0.535) + box(1.0, 0.0)), 0, now=0.0)
    assert k.changing and k.lane == 0
    k = core_at()
    # 옆 차로 장애물이 앞 0.5 m -> 그쪽으로는 못 간다
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(box(0.5, 0.535) + box(1.0, 0.0)), 0, now=0.0)
    assert k.lane == 2


def test_no_free_lane_slows_then_stops_then_reverses():
    k = core_at()
    pts = box(1.0, 0.0) + box(1.0, 0.535) + box(1.0, -0.535)
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=0.0)
    assert out['state'] == 'BLOCKED' and 0.0 < out['v'] <= S.V_SLOW
    k = core_at()
    pts = box(0.30, 0.0) + box(0.6, 0.535) + box(0.6, -0.535)
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=0.0)
    assert out['v'] == 0.0
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=S.STUCK_SEC + 0.1)
    assert out['state'] == 'REVERSE' and out['v'] < 0.0 and out['steer'] == 0.0


def test_center_lane_leaves_before_island_to_sign_side():
    for goal in (0, 2):
        k = core_at()
        k.s = S.START_TO_ISLAND - S.LEAVE_DIST + 0.05
        k.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), goal, now=0.0)
        assert k.changing and k.lane == goal


def test_center_lane_slows_to_read_sign_when_unknown():
    k = core_at()
    k.s = S.START_TO_ISLAND - 1.8
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), None, now=0.0)
    assert out['v'] == S.V_READ and not k.changing


def test_outer_yellow_line_corrects_position_only_when_close():
    k = core_at(0.50)                  # 실제는 왼쪽 차로 가운데 0.535, 바깥 노란 선은 차에서 왼쪽 0.265
    obs = dict(th=0.0, lines=[(0.265, 150, 1.0)], n=1000, ystart={0.265: 0.42}, short=set())
    for _ in range(20):
        k._camera_edge(obs)
    assert abs(k.c - 0.535) < 0.01
    k = core_at(0.0)                   # 노란 선이 맞을 위치(0.535)가 지금 위치와 너무 다르면 안 쓴다
    k._camera_edge(obs)
    assert k.c == 0.0


def test_supervisor_hands_back_after_island_in_outer_lane():
    sup = S.Lane3SimpleSupervisor(forced=True)
    sup.mode = 'on'
    sup.core.start(-0.535, 0.0, (0.0, 0.0, 0.0))
    sup.core.s = S.START_TO_ISLAND + 0.3
    outs = [sup.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), False, now=0.0) for _ in range(S.EXIT_FRAMES)]
    assert outs[-1] is None and sup.mode == 'done'


def test_supervisor_only_in_m7_and_ignores_side_lane_signs():
    sup = S.Lane3SimpleSupervisor()
    sup.set_mission('M3')
    assert sup.mode == 'off'
    sup.set_mission('M7')
    assert sup.mode == 'armed'
    sup.mode = 'on'
    sup.core.start(-0.535, 0.0, (0.0, 0.0, 0.0))
    det = [{'label': 'a', 'confidence': 0.9, 'bbox': [0, 0, 40, 50]}]
    for _ in range(5):
        sup.see_signs(det)
    assert sup.sign_goal is None
    sup.core.c = 0.0
    for _ in range(5):
        sup.see_signs(det)
    assert sup.sign_goal == 0


def test_closed_loop_simulation_passes_without_contact():
    for seed in range(6):
        r = SIM.run(seed)
        assert r['contact'] == 0 and r['out'] == 0 and r['island'] == 0, r
        assert r['exit_lane'] in (0, 2), r


def test_entry_position_from_dash_start_seen_ahead():
    """흰 점선이 차 앞 1.0 m 에서 시작하는 이미지 -> 시작선 기준 위치 0.30 - 1.0 = -0.70 m."""
    import cv2
    from jetracer_ros2 import three_lane as T
    fitter = T.LineFitter()
    bev = np.zeros((fitter.h, fitter.w), np.uint8)
    for u in (0.27, -0.27):                     # 점선 (왼쪽 +), 1.0 m 부터
        px = int((-u + T.XW) / T.RES)
        for f0 in np.arange(1.0, 1.9, 0.30):
            y0, y1 = int((T.F1 - f0 - 0.15) / T.RES), int((T.F1 - f0) / T.RES)
            cv2.rectangle(bev, (px - 2, y0), (px + 2, y1), 255, -1)
    fitter.white = bev
    obs = dict(th=0.0, lines=[(-0.27, 100, 0.0), (0.27, 100, 0.0)])
    d = S.dash_start_ahead(fitter, obs)
    assert d is not None and abs(d - 1.0) < 0.05
    bev[:] = 0
    px = int((0.27 + T.XW) / T.RES)
    cv2.rectangle(bev, (px - 2, int((T.F1 - 1.5) / T.RES)), (px + 2, fitter.h - 1), 255, -1)
    fitter.white = bev
    assert S.dash_start_ahead(fitter, dict(th=0.0, lines=[(-0.27, 100, 0.0)])) is None   # 화면 아래부터 보임
