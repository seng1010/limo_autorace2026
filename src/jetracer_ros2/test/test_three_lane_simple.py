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


def test_no_free_lane_slows_then_stops_then_reverses(monkeypatch):
    monkeypatch.setattr(S, 'TURN_ON', False)      # 예전 방식 (곧은 후진)
    k = core_at()
    pts = box(1.0, 0.0) + box(1.0, 0.535) + box(1.0, -0.535)
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=0.0)
    assert out['state'] == 'BLOCKED' and 0.0 < out['v'] <= S.V_SLOW
    # 너무 가까워서 지금은 못 꺾지만, 물러나면 옆 차로로 갈 수 있다 -> 후진
    k = core_at()
    pts = box(0.30, 0.0)
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=0.0)
    assert out['v'] == 0.0
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=S.STUCK_SEC + 0.1)
    assert out['state'] == 'REVERSE' and out['v'] < 0.0 and out['steer'] == 0.0
    # 세 차로가 모두 막힌 막다른 길: 후진해도 소용없으니 그냥 선다 (같은 길 되풀이 안 함)
    k = core_at()
    pts = box(0.30, 0.0) + box(0.6, 0.535) + box(0.6, -0.535)
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=0.0)
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=S.STUCK_SEC + 0.1)
    assert out['v'] == 0.0 and out['state'] == 'STOP'


def test_reverse_stops_before_remembered_obstacle_behind(monkeypatch):
    """뒤에 지나온 장애물이 있으면 그 앞까지만 후진한다."""
    monkeypatch.setattr(S, 'TURN_ON', False)
    k = core_at()
    k.map.add(np.full(6, k.s - S.CAR_REAR - 0.25), np.linspace(-0.08, 0.08, 6))
    k.map.add(np.full(6, k.s - S.CAR_REAR - 0.25), np.linspace(-0.08, 0.08, 6))
    pts = box(0.30, 0.0)
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=0.0)
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=S.STUCK_SEC + 0.1)
    assert not k.reversing or k.reverse_left <= 0.25 - 0.10 + 1e-6


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


def test_center_lane_stops_to_read_then_falls_back_to_guess():
    k = core_at()
    k.s = S.START_TO_ISLAND - S.READ_STOP + 0.02
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), None, now=0.0)
    assert out['v'] == 0.0 and k.state == 'READ' and not k.changing
    # 서서 읽는 동안은 후진하지 않는다
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), None, now=S.STUCK_SEC + 0.5)
    assert out['v'] == 0.0 and not k.reversing
    # 시간이 지나면 많이 읽힌 쪽(guess)으로
    k.guess = 0
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), None, now=S.READ_WAIT_SEC + 0.1)
    k.s = S.START_TO_ISLAND - S.LEAVE_DIST + 0.05
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with([]), None, now=S.READ_WAIT_SEC + 0.2)
    assert k.changing and k.lane == 0


def test_fork_does_not_take_wrong_side_when_sign_side_briefly_blocked():
    k = core_at()
    k.s = S.START_TO_ISLAND - S.LEAVE_DIST + 0.05
    # 표지판은 A(왼쪽)인데 왼쪽 차로 바로 앞에 장애물, 오른쪽은 비어 있음
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(box(0.6, 0.535)), 0, now=0.0)
    assert k.lane != 2


def test_supervisor_guess_uses_close_reads_and_side_reads():
    sup = S.Lane3SimpleSupervisor()
    sup.set_mission('M7')
    sup.mode = 'on'
    sup.core.start(0.0, 0.0, (0.0, 0.0, 0.0))
    far_a = [{'label': 'a', 'confidence': 0.9, 'bbox': [0, 0, 20, 20]}]       # 멀리서 A (모델이 잘 틀림) -> 안 셈
    for _ in range(10):
        sup.see_signs(far_a)
    assert sup.guess is None
    close_b = [{'label': 'b', 'confidence': 0.9, 'bbox': [0, 0, 40, 40]}]
    sup.see_signs(close_b)
    sup.see_signs(close_b)
    assert sup.guess == 2 and sup.sign_goal is None
    sup2 = S.Lane3SimpleSupervisor()
    sup2.set_mission('M7')
    sup2.mode = 'on'
    sup2.core.start(-0.535, 0.0, (0.0, 0.0, 0.0))                            # 옆 차로
    for _ in range(3):
        sup2.see_signs([{'label': 'a', 'confidence': 0.9, 'bbox': [0, 0, 40, 40]}])
    assert sup2.sign_goal is None and sup2.guess == 0


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


def test_plan_avoids_lane_that_is_blocked_further_ahead():
    """가운데 1.0 m 막힘. 오른쪽은 1.9 m 에서 막히고 왼쪽은 끝까지 빔: 표지판이 B 여도 왼쪽으로 (지금 더 빈 쪽이 아니라 더 멀리 가는 쪽)."""
    k = core_at()
    pts = box(1.0, 0.0) + box(1.9, -0.535)
    for _ in range(3):
        k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), 2, now=0.0)
    assert k.changing and k.lane == 0


def test_island_position_from_its_rear_end_when_passing():
    """섬 옆을 지날 때 섬 뒤끝이 앞 1.0 m 에 보이면 섬 앞끝 = s + 1.0 - 섬 길이 (진입 위치 가정이 틀려도 합류 시점이 맞게)."""
    k = core_at(-0.535)
    k.s = S.START_TO_ISLAND + 0.30          # 실제 섬 앞끝은 0.4 m 뒤 (s 가 0.4 m 앞서 있음)
    true_front = k.s - 0.4
    end_ahead = true_front + S.ISLAND_LEN - k.s
    f = np.arange(0.45, end_ahead, 0.02)
    xy = np.stack([np.full(f.size, -0.265), f], 1)     # 가까운 섬 테두리(도로 -0.27): 차(-0.535)에서 왼쪽 0.265 m (오른쪽 + 좌표)
    for _ in range(4):
        k._camera_island(dict(yellow_xy=xy))
    assert k.island_s is not None and abs(k.island_at - true_front) < 0.05


def test_change_stuck_short_of_target_replans_when_target_is_blocked():
    """B 로 옮기다 노란 선 벽 때문에 c -0.41 에서 멈춤 + B 앞 0.6 m 막힘 + 가운데 빔: 다시 계획해 가운데로 (11/16 run19)."""
    k = core_at(-0.41)
    k.lane, k.changing, k.change_from = 2, True, 1
    pts = box(0.6, -0.535 + 0.41)            # B 차로 장애물 (차에서 오른쪽 0.125 m)
    for _ in range(3):
        k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=0.0)
    assert k.lane == 1


# ---------------------------------------------------------------------------
# 막혔을 때 방향 틀기 (TURN_ON)
# ---------------------------------------------------------------------------
def _drive(k, cars, x, y, psi, seconds, dt=0.05):
    """뒤 차축 (x, y, psi) 에서 k 를 닫힌 고리로 돌린다 (odom = 참값, 조향 1차 지연).
    반환: (기록 [(상태, v, steer, psi)], 접촉 여부, 마지막 (x, y, psi))."""
    polys = [SIM.rect(cx, cy, 0.0, SIM.CAR_L, SIM.CAR_W) for cx, cy in cars]
    steer_act, t, log, contact = 0.0, 0.0, [], False
    while t < seconds:
        cx, cy = x + SIM.CAM * math.cos(psi), y + SIM.CAM * math.sin(psi)
        out = k.step_obs(None, (x, y, psi), SIM.raycast(cx, cy, psi, polys), None, now=t)
        v, cmd = out['v'], max(-SIM.T.MAX_STEER, min(SIM.T.MAX_STEER, out['steer']))
        steer_act += (cmd - steer_act) * min(1.0, dt / 0.15)
        ds = v * dt
        dpsi = 0.0
        if abs(steer_act) > 1e-4:
            dpsi = (math.copysign(abs(ds) / (SIM.T.WHEELBASE / math.tan(abs(steer_act)) + SIM.T.TRACK / 2.0), steer_act)
                    * (1.0 if ds >= 0 else -1.0))
        x += ds * math.cos(psi + dpsi / 2)
        y += ds * math.sin(psi + dpsi / 2)
        psi += dpsi
        body = SIM.rect(x + SIM.BODY_C * math.cos(psi), y + SIM.BODY_C * math.sin(psi), psi, SIM.CAR_L, SIM.CAR_W)
        contact = contact or any(SIM.overlap(body, p) for p in polys)
        log.append((out['state'], v, out['steer'], psi))
        t += dt
    return log, contact, (x, y, psi)


def _stuck_core(cars, cam_s=1.5, c=0.0):
    k = S.SimpleThreeLane()
    x = cam_s - SIM.CAM
    k.start(c, 0.0, (x, c, 0.0), cam_s)
    return k, x


def test_turn_when_stuck_points_nose_to_free_lane_and_goes_there():
    """가운데 차로 코앞(0.3 m)이 막히고 왼쪽 차로도 앞이 막힘, 오른쪽은 빔:
    짧게 꺾으며 후진(차 앞이 오른쪽으로) -> 오른쪽 차로로 옮긴다. 접촉 없음."""
    cars = [(1.5 + 0.30 + SIM.CAR_L / 2, 0.0), (2.1, 0.535)]
    k, x = _stuck_core(cars)
    log, contact, (x1, y1, psi1) = _drive(k, cars, x, 0.0, 0.0, 12.0)
    states = [s for s, *_ in log]
    assert 'TURN' in states
    first_back = next(i for i, (s, v, *_) in enumerate(log) if s == 'TURN' and v < 0)
    assert log[first_back][2] > 0.0                  # 오른쪽 차로로 가려면 후진은 왼쪽으로 꺾는다 (차 앞이 오른쪽으로 돈다)
    assert min(p for *_, p in log) < -math.radians(5)
    assert not contact
    assert k.lane == 2 and y1 < -0.30                # 오른쪽 차로로 들어감


def test_turn_not_used_when_every_lane_is_blocked():
    """막다른 길(세 차로 모두 막힘): 돌릴 차로가 없으니 방향 틀기 안 함."""
    k = core_at()
    pts = box(0.30, 0.0) + box(0.6, 0.535) + box(0.6, -0.535)
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=0.0)
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=S.STUCK_SEC + 0.1)
    assert out['state'] != 'TURN' and k.turn is None


def test_turn_rev_sign_flips_only_the_reverse_command(monkeypatch):
    """TURN_REV_SIGN = -1 이면 후진할 때만 조향 명령 부호가 뒤집힌다 (실차 확인용 스위치)."""
    cars = [(1.5 + 0.30 + SIM.CAR_L / 2, 0.0), (2.1, 0.535)]
    k, x = _stuck_core(cars)
    k.step_obs(None, (x, 0.0, 0.0), SIM.raycast(1.5, 0.0, 0.0, [SIM.rect(cx, cy, 0, SIM.CAR_L, SIM.CAR_W) for cx, cy in cars]),
               None, now=0.0)
    monkeypatch.setattr(S, 'TURN_REV_SIGN', -1.0)
    out = k.step_obs(None, (x, 0.0, 0.0), SIM.raycast(1.5, 0.0, 0.0, [SIM.rect(cx, cy, 0, SIM.CAR_L, SIM.CAR_W) for cx, cy in cars]),
                     None, now=S.STUCK_SEC + 0.1)
    assert out['state'] == 'TURN' and out['v'] < 0.0 and out['steer'] < 0.0


def _yellow_obs(u, start=0.0):
    """카메라가 차 옆 u m(왼쪽 +)에 노란 실선을 본 것처럼 만든 선 검출 결과."""
    return dict(th=0.0, lines=[(u, 200, 1.0)], n=400, short=set(), ystart={u: start}, yellow_xy=None)


def test_turn_reverse_keeps_off_yellow_line_seen_by_camera():
    """오른쪽 차로에서 차 앞을 왼쪽으로 돌리며 후진하면 차 뒤가 오른쪽(바깥 노란 선)으로 휜다.
    카메라로 본 노란 선이 가까우면 그만큼 덜 움직인다 (odom 옆 위치가 틀려도 카메라 기준 거리로 막는다)."""
    k = core_at(c=-0.535)
    k.map.refresh()
    ps, pl = k._turn_points()
    free_run = k._sweep(-S.TURN_STEER, -S.TURN_BACK_M, ps, pl)
    k._camera_walls(_yellow_obs(-0.15))            # 노란 선이 카메라 오른쪽 15 cm (차 옆면에서 4 cm)
    near_run = k._sweep(-S.TURN_STEER, -S.TURN_BACK_M, ps, pl)
    assert free_run >= S.TURN_BACK_M - 0.03 and near_run < free_run
    # 멈춘 자리에서 차체 모서리는 선 안쪽 (여유 TURN_LINE_MARGIN)
    psi, x, y = 0.0, k.s - S.CAM_FROM_ODOM, k.c
    r = SIM.T.WHEELBASE / math.tan(S.TURN_STEER) + SIM.T.TRACK / 2.0
    d = 0.0
    while d + 0.02 <= near_run + 1e-9:
        dpsi = 0.02 / r                                # 오른쪽 조향 후진 -> 차 앞이 왼쪽으로
        x -= 0.02 * math.cos(psi + dpsi / 2)
        y -= 0.02 * math.sin(psi + dpsi / 2)
        psi += dpsi
        d += 0.02
    line = k.c - 0.15
    for fx in (SIM.T.CAR_BODY_CENTER - SIM.T.CAR_BODY_HALF_LEN, SIM.T.CAR_BODY_CENTER + SIM.T.CAR_BODY_HALF_LEN):
        for fy in (-SIM.T.CAR_HALF_WIDTH, SIM.T.CAR_HALF_WIDTH):
            assert y + fx * math.sin(psi) + fy * math.cos(psi) >= line + S.TURN_LINE_MARGIN - 1e-6


def test_turn_walls_remembered_while_turning_and_reset_after():
    k = core_at()
    k._camera_walls(_yellow_obs(-0.80))
    k.turn = dict(target=0, side=1.0, phase='back', t0=0.0, cycles=0, idle=0)
    k._camera_walls(None)                          # 차가 비스듬해서 선이 안 보여도 기억
    assert len(k._walls) == 1
    k.turn = None
    k._camera_walls(None)
    assert k._walls == []


def test_no_turn_near_start_line():
    """시작선 가까이(뒤가 1차선 구간)에서는 방향 틀기를 하지 않는다."""
    k = S.SimpleThreeLane()
    k.start(0.0, 0.0, (0.0, 0.0, 0.0), S.TURN_FROM_S - 0.2)
    pts = box(0.30, 0.0)
    k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=0.0)
    out = k.step_obs(None, (0.0, 0.0, 0.0), scan_with(pts), None, now=S.STUCK_SEC + 0.1)
    assert out['state'] != 'TURN' and k.turn is None


def test_entry_position_fixed_when_three_lane_yellow_seen():
    k = S.SimpleThreeLane()
    k.start(0.0, 0.0, (0.0, 0.0, 0.0), -1.31)            # 첫 점선을 놓쳐 시작선 1.3 m 전이라고 잘못 잡음
    narrow = dict(th=0.0, lines=[(0.26, 150, 1.0), (-0.26, 150, 1.0)], n=1000,
                  ystart={0.26: 0.42, -0.26: 0.42}, short=set())
    for _ in range(10):
        k._camera_wide(narrow)                          # 1차선 노란 선(+-0.26): 그대로
    assert k.s == -1.31
    u = 0.80
    wide = dict(th=0.0, lines=[(u, 150, 1.0), (-u, 150, 1.0)], n=1000, ystart={u: 0.42, -u: 0.42}, short=set())
    for _ in range(S.WIDE_FRAMES):
        k._camera_wide(wide)
    assert k.s == S.WIDE_S and '위치 고침' in k.message


def test_wall_ignores_yellow_line_where_no_yellow_should_be():
    # 왼쪽 차로(c +0.55)에서 오른쪽 0.28 m 에 '노란 선'이 보임 (그 자리는 흰 점선) -> 그걸 피해 왼쪽으로 꺾으면 안 된다
    k = core_at(0.55)
    k.lane = 0
    obs = dict(th=0.0, lines=[(-0.28, 150, 1.0)], n=1000, ystart={-0.28: 0.42}, short=set(), edges=[])
    out = k.step_obs(obs, (0.0, 0.0, 0.0), scan_with([]), 0, now=0.0)
    assert out['wall_text'] == '' and out['steer'] <= 0.05
    # 진짜 왼쪽 바깥 노란 선(0.20 m 왼쪽, 도로 +0.75)에 다가가면 오른쪽으로 꺾는다
    k = core_at(0.55, th=-0.35)
    k.lane = 0
    obs = dict(th=-0.35, lines=[(0.20, 150, 1.0)], n=1000, ystart={0.20: 0.42}, short=set(), edges=[])
    out = k.step_obs(obs, (0.0, 0.0, 0.0), scan_with([]), 0, now=0.0)
    assert out['wall_text'] != '' and out['steer'] < 0.0
