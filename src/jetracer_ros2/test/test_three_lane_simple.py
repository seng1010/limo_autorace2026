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
    k.start(c, th, (0.0, 0.0, 0.0), 1.0)     # 3차선 안 (도로가 다 벌어진 곳)
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


def test_lines_correct_position_only_when_close():
    k = core_at(0.50)                  # 실제는 왼쪽 차로 가운데 0.535, 바깥 노란 선은 차에서 왼쪽 0.265
    obs = dict(th=0.0, lines=[(0.265, 150, 1.0)], n=1000, ystart={0.265: 0.42}, short=set())
    for _ in range(20):
        k._camera_lines(obs)
    assert abs(k.c - 0.535) < 0.01
    k = core_at(0.0)                   # 노란 선이 맞을 위치(0.535)가 지금 위치와 너무 다르면 안 쓴다
    k._camera_lines(obs)
    assert k.c == 0.0


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


def test_no_lane_change_before_the_road_widens():
    """시작선 앞(1차선)에서는 표지판 방향이 정해져 있어도, 앞이 막혀 있어도 옮기지 않는다."""
    k = S.SimpleThreeLane()
    k.start(0.0, 0.0, (0.0, 0.0, 0.0), -0.4)
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), 2, now=0.0)
    assert not k.changing and k.lane == 1
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(box(1.2, 0.0)), None, now=0.0)
    assert not k.changing and k.lane == 1
    k.s = S.CHANGE_FROM_S + 0.01
    for _ in range(S.PREF_FRAMES):
        k.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), 2, now=0.0)
    assert k.changing and k.lane == 2


def test_entry_clamps_bad_camera_heading_and_offset(monkeypatch):
    """입구에서 카메라가 c -0.52, 방향 +40도로 잘못 봐도 가운데 차로·곧은 방향 근처로 시작한다."""
    sup = S.Lane3SimpleSupervisor(forced=True)
    monkeypatch.setattr(sup._look, 'look', lambda image, tune=None: ((-0.52, math.radians(40)), True))
    monkeypatch.setattr(sup._look.fitter, 'fit', lambda image, prior=None, tune=None: None)
    for i in range(S.ENTRY_FRAMES):
        sup._try_enter(None, (0.0, 0.0, 0.0), None)
    assert sup.mode == 'on'
    assert abs(sup.core.c) <= S.ENTRY_C_MAX + 1e-9 and abs(sup.core.th) <= S.ENTRY_TH_MAX + 1e-9


def test_white_dashes_pull_position_back_after_odom_drift():
    """실제로는 가운데 차로 가운데(c 0)인데 odom 이 -0.15 로 밀렸다: 양쪽 흰 점선(차에서 +-0.27)으로 되돌린다."""
    k = core_at(-0.15)
    obs = dict(th=0.0, lines=[(-0.27, 120, 0.0), (0.27, 120, 0.0), (0.80, 150, 1.0)], n=1000, ystart={}, short=set())
    for _ in range(25):
        k._camera_lines(obs)
    assert abs(k.c) < 0.01


def test_merges_to_center_after_island_then_hands_back():
    sup = S.Lane3SimpleSupervisor(forced=True)
    sup.mode = 'on'
    sup.core.start(-0.535, 0.0, (0.0, 0.0, 0.0), S.START_TO_ISLAND + 1.0)
    sup.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), False, now=0.0)
    assert sup.core.lane == 2 and sup.side_taken == 2           # 섬 옆: 아직 B
    sup.core.s = S.START_TO_ISLAND + S.MERGE_TURN + 0.01
    sup.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), False, now=0.0)
    assert sup.core.lane == 1 and sup.core.changing              # 합류 차로로
    sup.core.s, sup.core.c, sup.core.th = S.START_TO_ISLAND + S.MERGE_END + 0.3, 0.0, 0.0
    outs = [sup.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), False, now=0.0) for _ in range(S.EXIT_FRAMES)]
    assert outs[-1] is None and sup.mode == 'done' and 'B' in sup.message


def test_sign_post_on_island_does_not_block_side_lane():
    """섬 앞끝의 표지판 받침대(차로 칸 가장자리까지 걸침)는 옆 차로를 막지 않는다."""
    k = core_at(-0.535)
    k.lane = 2
    k.s = S.START_TO_ISLAND - 0.6
    post = [(0.6, 0.535 - 0.30 + d) for d in (-0.03, 0.0, 0.03)]     # 차에서 왼쪽 0.235 m = 도로 -0.30
    for _ in range(3):
        out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(post), 2, now=0.0)
    assert k.ahead[2] >= S.NO_OBSTACLE - 1e-6 and out['v'] > 0.0


def test_neighbour_lane_with_obstacle_almost_passed_counts_as_free_when_blocked():
    """A 차로 앞 0.6 m 막힘, 가운데 LIMO 는 옆에 (앞끝이 카메라 투영점 바로 앞): 가운데로 옮긴다 (11/16 실차에서는 섰다)."""
    k = core_at(0.535)
    k.lane = 0
    side = [(x, 0.535 - 0.10) for x in (-0.15, -0.05, 0.03)]       # 가운데 LIMO 의 왼쪽 옆면 (차에서 오른쪽 0.43 m)
    for _ in range(3):
        k.step_obs(None, (0.0, 0.0, 0.0), scan_with(box(0.6, 0.0) + side), None, now=0.0)
    assert k.changing and k.lane == 1


def test_entry_is_forced_when_lines_never_confirm(monkeypatch):
    sup = S.Lane3SimpleSupervisor(forced=True)
    monkeypatch.setattr(sup._look, 'look', lambda image, tune=None: (None, False))
    monkeypatch.setattr(sup.fitter, 'fit', lambda image, prior=None, tune=None: None)
    assert sup.step(None, (0.0, 0.0, 0.0), None, False, now=0.0) is None and sup.mode == 'armed'
    out = sup.step(None, (0.0, 0.0, 0.0), None, False, now=S.ENTRY_WAIT_SEC + 0.1)
    assert sup.mode == 'on' and out is not None and out['v'] > 0.0


def test_island_is_not_seen_near_the_start_line():
    """1차선 구간에서 1차선 노란 선을 섬으로 보지 않는다 (11/16 실차: 섬을 -0.6 m 로 잡고 출발 못 함)."""
    k = core_at(0.0)
    k.s = -1.0
    xy = np.array([[x, f] for x in (-0.26, 0.26) for f in np.arange(0.6, 1.8, 0.02)])
    for _ in range(5):
        k._camera_island(dict(yellow_xy=xy))
    assert k.island_s is None and abs(k.island_at - S.START_TO_ISLAND) < 1e-9
