import math

import numpy as np

from jetracer_ros2 import three_lane as T


def obs(c, th=0.0):
    """도로 가운데에서 c 만큼 왼쪽에 있는 차가 보는 선들."""
    lines = []
    for pos, yellow in ((0.80, 1.0), (0.27, 0.0), (-0.27, 0.0), (-0.80, 1.0)):
        u = pos - c
        if abs(u) > T.FAR_U:
            u = math.copysign(T.FAR_U + (abs(u) - T.FAR_U) * T.FAR_SHRINK, u)
        if abs(u) < 1.0:
            lines.append((u, 150, yellow))
    return dict(th=th, lines=sorted(lines), score=1.0, n=1000)


def test_road_state_finds_each_lane_from_line_colors():
    for lane, center in enumerate(T.LANE_CENTER):
        road = T.RoadState()
        road.update(obs(center + 0.03))
        assert road.lane == lane
        assert abs(road.c - (center + 0.03)) < 0.04


def test_road_state_dead_reckons_when_lines_disappear():
    road = T.RoadState()
    road.update(obs(0.0))
    road.th = -math.radians(30)            # 차가 도로보다 왼쪽을 30도 보고 있음
    for _ in range(20):
        road.predict(0.02, 0.0)
        road.update(None)
    assert abs(road.c - 0.4 * math.sin(math.radians(30))) < 0.02
    assert abs(road.blind_m - 0.4) < 1e-6


def scan_with(points):
    """points: (앞, 왼쪽) m 목록 -> 라이다 250점 (-110~110도)."""
    n, a0, inc = 250, -1.92, 3.84 / 249
    r = np.full(n, np.inf, np.float32)
    for x, y in points:
        for dx in (-0.01, 0.0, 0.01):
            i = int(round((math.atan2(y + dx, x) - a0) / inc))
            r[i] = math.hypot(x, y)
    return r, a0, inc, 0.15


def test_lane_free_assigns_obstacles_to_lanes():
    ranges, a0, inc, rmin = scan_with([(1.5, 0.0), (2.0, 0.53)])
    ahead, side = T.lane_free(ranges, a0, inc, rmin, c=0.0, th=0.0)
    assert abs(ahead[1] - 1.5) < 0.05 and abs(ahead[0] - 2.0) < 0.08 and ahead[2] == T.LIDAR_MAX
    # 같은 장면을 오른쪽 차로에서 보면 가운데 장애물은 왼쪽 앞에 있다
    ranges, a0, inc, rmin = scan_with([(1.5, 0.535)])
    ahead, _ = T.lane_free(ranges, a0, inc, rmin, c=-0.535, th=0.0)
    assert abs(ahead[1] - 1.5) < 0.05 and ahead[2] == T.LIDAR_MAX


def test_points_outside_road_are_ignored():
    ranges, a0, inc, rmin = scan_with([(1.0, 1.2), (1.2, -1.0)])
    ahead, _ = T.lane_free(ranges, a0, inc, rmin, c=0.0, th=0.0)
    assert ahead == [T.LIDAR_MAX] * 3


def test_planner_moves_to_free_lane_and_prefers_goal_side():
    far = T.LIDAR_MAX
    plan = T.Planner().update(1, 0.0, [far, 1.2, far], [far, 1.2, far], goal=0)
    assert plan.state == 'CHANGE' and plan.target == 0
    plan = T.Planner().update(1, 0.0, [far, 1.2, far], [far, 1.2, far], goal=2)
    assert plan.target == 2
    plan = T.Planner().update(1, 0.0, [1.3, 1.2, far], [1.3, 1.2, far], goal=0)
    assert plan.target == 2                      # 왼쪽도 막혀 있으면 표지판 방향이 아니어도 빈 쪽으로


def test_planner_does_not_cut_in_when_side_lane_is_occupied_or_too_late():
    far = T.LIDAR_MAX
    plan = T.Planner().update(1, 0.0, [far, 1.2, far], [0.3, 1.2, 0.5], goal=None)
    assert plan.state == 'BLOCKED' and plan.target == 1
    plan = T.Planner().update(1, 0.0, [far, 0.5, far], [far, 0.5, far], goal=None)
    assert plan.target == 1                      # 0.5 m 앞에서는 꺾지 않는다
    plan = T.Planner().update(1, 0.0, [far, 0.3, far], [far, 0.3, far], goal=None)
    assert plan.state == 'STOP' and plan.speed == 0.0


def test_planner_goes_to_goal_lane_only_when_it_is_empty():
    far = T.LIDAR_MAX
    planner = T.Planner()
    for _ in range(20):                  # 왼쪽 2.0 m 앞에 장애물이 보이는 동안은 옮기지 않는다
        assert planner.update(1, 0.0, [2.0, far, far], [2.0, far, far], goal=0).target == 1
    for k in range(T.GOAL_CLEAR_FRAMES):  # 장애물이 안 보이는 상태가 이어져야 옮긴다 (한두 프레임으로는 안 옮김)
        plan = planner.update(1, 0.0, [far, far, far], [far, far, far], goal=0)
        assert plan.target == (0 if k == T.GOAL_CLEAR_FRAMES - 1 else 1)


def test_two_lane_move_through_empty_middle_lane():
    far = T.LIDAR_MAX
    plan = T.Planner().update(2, T.LANE_CENTER[2], [far, 1.5, 1.2], [far, 1.5, 1.2], goal=None)
    assert plan.target == 0
    # 가운데 차로가 1.3 m 밖에 안 비었으면 가로지르지 않는다 (가로지르는 데 약 1.4 m 필요)
    plan = T.Planner().update(2, T.LANE_CENTER[2], [far, 1.3, 1.2], [far, 1.3, 1.2], goal=None)
    assert plan.target == 2


def test_steering_points_toward_target_lane():
    assert T.steer_to_lane(0.0, 0.0, 0) > math.radians(10)       # 왼쪽 차로 -> 왼쪽(+)
    assert T.steer_to_lane(0.0, 0.0, 2) < -math.radians(10)
    assert abs(T.steer_to_lane(0.0, 0.0, 1)) < 1e-6
    assert T.steer_to_lane(0.0, math.radians(15), 1) > 0.0       # 차가 오른쪽을 보고 있으면 왼쪽으로


def test_supervisor_arms_only_in_m7_and_takes_sign():
    sup = T.Lane3Supervisor()
    assert sup.mode == 'off' and not sup.latch_off
    sup.set_mission('M3')
    assert sup.mode == 'off'
    sup.set_mission('M7')
    assert sup.mode == 'armed' and not sup.latch_off           # 아직 코너 구간
    sup.arm_travel = T.ARM_AFTER_M + 0.1
    assert sup.latch_off
    sup.set_mission('M2')
    assert sup.mode == 'armed'
    sup.set_sign('sign_a')
    assert sup.sign_goal == 0
    sup.set_mission('M8')
    assert sup.mode == 'off'


def test_island_lines_do_not_pull_position_to_the_other_side():
    # 가운데 차로에서 섬 테두리(노란 선 2개, 0.5 m 간격)가 보여도 오른쪽/왼쪽 차로로 착각하지 않는다
    road = T.RoadState()
    road.update(obs(0.0))
    island = dict(th=0.0, lines=[(-0.25, 230, 1.0), (0.26, 250, 1.0)], score=1.0, n=1000)
    for _ in range(40):
        road.predict(0.01, 0.0)
        road.update(island)
    assert abs(road.c) < 0.05 and road.lane == 1


def test_island_distance_and_middle_lane_blocked_by_island():
    f = np.linspace(1.2, 1.7, 40)
    xy = np.concatenate([np.stack([np.full(40, -0.25), f], 1), np.stack([np.full(40, 0.25), f], 1)])
    d = T.island_distance(xy, c=0.0, th=0.0)
    assert d is not None and abs(d - 1.25) < 0.08
    assert T.island_distance(xy, c=0.0, th=0.0) > T.island_distance(xy - [0.0, 0.3], c=0.0, th=0.0)
    # 바깥 차로의 노란 실선(도로 가운데에서 0.8 m)은 섬이 아니다
    outer = np.stack([np.full(60, 0.27), np.linspace(0.5, 1.8, 60)], 1)      # 차 오른쪽 0.27 m (x 는 오른쪽 +)
    assert T.island_distance(outer, c=-0.535, th=0.0) is None
    far = T.LIDAR_MAX
    plan = T.Planner().update(1, 0.0, [far, d, far], [far, d, far], goal=0)
    assert plan.state == 'CHANGE' and plan.target == 0


def test_yellow_wall_turns_away_before_crossing():
    # 왼쪽 30 cm 에 노란 선, 그쪽으로 35도 향함 (th = -35도) -> 오른쪽으로 최대 조향
    steer, _ = T.yellow_wall([(0.30, 300, 1.0)], -math.radians(35))
    assert steer == -T.MAX_STEER
    steer, _ = T.yellow_wall([(-0.30, 300, 1.0)], math.radians(35))
    assert steer == T.MAX_STEER


def test_yellow_wall_is_quiet_in_lane_center_and_for_white_lines():
    assert T.yellow_wall([(0.27, 300, 1.0), (-0.27, 150, 0.0)], 0.0) is None       # 바깥 차로 가운데, 나란함
    assert T.yellow_wall([(0.27, 300, 1.0)], -math.radians(15)) is None           # 조금 비스듬해도 아직 여유
    assert T.yellow_wall([(0.05, 300, 0.0)], -math.radians(35)) is None           # 흰 점선은 넘어도 된다
    assert T.yellow_wall([(0.30, 300, 1.0)], math.radians(35)) is None            # 멀어지는 중


def test_yellow_wall_overrides_steering_even_when_position_is_wrong():
    tl = T.ThreeLane()
    tl.step_obs(obs(T.LANE_CENTER[0]), (0.0, 0.0, 0.0), None)
    tl.road.c = 0.0                      # 실제는 왼쪽 차로인데 가운데라고 착각
    tl.planner.target = 0                # -> 왼쪽으로 더 가려고 함
    seen = dict(th=-math.radians(30), lines=[(0.24, 300, 1.0)], score=1.0, n=1000)
    out = tl.step_obs(seen, (0.01, 0.0, 0.0), None)
    assert out.wall is not None and out.steer == -T.MAX_STEER


def test_outer_solid_line_keeps_position_from_sliding_one_lane():
    # 11/15 04:21 실차: 가운데 차로에서 왼쪽을 볼 때 왼쪽 점선(+0.41)과 바깥 흰 실선(+1.03)이 보였다.
    # 실선을 점선으로 치면 '오른쪽 차로'(-0.63)가 가장 잘 맞아 위치가 한 차로 밀렸다
    seen = dict(th=-math.radians(24), lines=[(0.41, 120, 0.0)], edges=[(1.03, 290)], score=1.0, n=1000)
    road = T.RoadState()
    road.update(obs(-0.13))
    for _ in range(30):
        road.predict(0.0, 0.0)
        road.update(seen)
    assert abs(road.c - (-0.13)) < 0.08 and road.lane == 1
    # 위치를 전혀 모를 때도 실선과 맞는 쪽을 고른다
    fresh = T.RoadState()
    fresh.update(seen)
    assert fresh.c is not None and fresh.lane == 1


def test_short_yellow_piece_is_not_used_as_outer_line():
    # 섬 꼭짓점의 짧은 노란 조각(-0.19)을 오른쪽 바깥 선으로 쓰면 오른쪽 차로로 착각한다
    seen = dict(th=-math.radians(24), lines=[(-0.19, 53, 1.0), (0.41, 120, 0.0)], short={-0.19},
                score=1.0, n=1000)
    road = T.RoadState()
    road.update(obs(-0.13))
    for _ in range(30):
        road.predict(0.0, 0.0)
        road.update(seen)
    assert road.lane == 1


def test_heading_is_limited():
    left = T.limit_heading(math.radians(20), -math.radians(50))     # 왼쪽으로 50도 틀어진 채 더 왼쪽으로 꺾으려 함
    assert left < -math.radians(10)
    assert T.limit_heading(math.radians(20), -math.radians(20)) == math.radians(20)
    assert T.limit_heading(-math.radians(20), math.radians(50)) > math.radians(10)


def test_goal_side_is_preferred_only_when_free_distance_is_similar():
    far = T.LIDAR_MAX
    # 왼쪽은 장애물이 없고 갈림길 방향(오른쪽)은 2.4 m 앞에 있으면 왼쪽 (11/15 05:02 실차에서 막힌 배치)
    plan = T.Planner().update(1, 0.0, [far, 1.2, 2.4], [far, 1.2, 2.4], goal=2)
    assert plan.target == 0
    # 양쪽 다 장애물이 있고 거리가 비슷하면 갈림길 방향
    plan = T.Planner().update(1, 0.0, [2.2, 1.2, 2.0], [2.2, 1.2, 2.0], goal=2)
    assert plan.target == 2
    # 양쪽 다 장애물이 있는데 한쪽이 훨씬 가까우면 먼 쪽
    plan = T.Planner().update(1, 0.0, [2.3, 1.2, 1.7], [2.3, 1.2, 1.7], goal=2)
    assert plan.target == 0
    # 양쪽 다 장애물이 없으면 갈림길 방향
    plan = T.Planner().update(1, 0.0, [far, 1.2, far], [far, 1.2, far], goal=2)
    assert plan.target == 2


def test_reverse_recovery_backs_up_then_changes_lane():
    tl = T.ThreeLane()
    right = T.LANE_CENTER[2]
    x = 0.0
    n, a0, inc = 250, -1.92, 3.84 / 249

    def scan(dist):                       # 오른쪽 차로(내 차로) 정면 dist m 에 장애물
        r = np.full(n, np.inf, np.float32)
        for dy in (-0.05, 0.0, 0.05):
            r[int(round((math.atan2(dy, dist) - a0) / inc))] = math.hypot(dist, dy)
        return r, a0, inc, 0.15

    states = []
    dist = 0.30                           # 너무 가까워서 옮기지 못하고 멈춘 상태
    for _ in range(200):
        out = tl.step_obs(obs(right), (x, 0.0, 0.0), scan(dist))
        states.append(out.state)
        step = out.speed / 22.0
        x += abs(step)                    # odom 위치는 후진해도 '이동 거리'만 준다
        dist -= step
        if out.state == 'CHANGE':
            break
    assert 'STOP' in states and 'REVERSE' in states and states[-1] == 'CHANGE'
    assert out.target == 1 and dist >= T.MIN_START - 0.02      # 마지막 프레임에서 한 걸음 전진한 만큼
    assert tl.reverse_done <= T.REVERSE_MAX + 0.02


def test_no_lane_decision_without_lidar():
    far = T.LIDAR_MAX
    planner = T.Planner()
    for _ in range(30):                   # 라이다가 아직 없으면 '전부 비었음'으로 보고 옮기지 않는다
        plan = planner.update(1, 0.0, [far] * 3, [far] * 3, goal=2, have_scan=False)
        assert plan.target == 1 and plan.speed == T.V_SLOW


def test_dead_end_is_not_entered_again_after_reversing():
    # 내 차로(오른쪽) 앞이 막혔고 가운데는 지나친 장애물이 계속 옆에 있다 -> 후진해도 옮길 곳이 없다
    tl = T.ThreeLane()
    right = T.LANE_CENTER[2]
    n, a0, inc = 250, -1.92, 3.84 / 249
    x, dist = 0.0, 0.30

    def scan(d):
        r = np.full(n, np.inf, np.float32)
        # 가운데 차로의 장애물은 옆으로 길게 이어진다 (점 몇 개짜리면 그 앞뒤로 돌아가는 경로가 나온다)
        side = [(px, py) for px in np.arange(0.10, 1.25, 0.05) for py in (0.48, 0.55)]
        for px, py in [(d, -0.05), (d, 0.0), (d, 0.05)] + side:
            r[int(round((math.atan2(py, px) - a0) / inc))] = math.hypot(px, py)
        return r, a0, inc, 0.15

    states = []
    for _ in range(500):
        out = tl.step_obs(obs(right), (x, 0.0, 0.0), scan(dist))
        states.append(out.state)
        step = out.speed / 22.0
        x += abs(step)
        dist -= step
    assert 'REVERSE' in states
    last_reverse = len(states) - 1 - states[::-1].index('REVERSE')
    assert set(states[last_reverse + 1:]) == {'STOP'}           # 후진 뒤에는 다시 전진하지 않는다
    assert states.count('REVERSE') < 250                        # 후진은 한 번만 (전진·후진을 되풀이하지 않는다)


def test_outer_solid_line_never_moves_position_by_a_lane_in_one_frame():
    # 11/15 06:01 실차: 오른쪽 차로(-0.53)에서 섬 옆을 지날 때 흰 실선 한 줄 때문에 한 프레임에 왼쪽 차로로 튐
    road = T.RoadState()
    road.update(obs(-0.53))
    seen = obs(-0.53)
    seen['edges'] = [(0.63, 104)]                 # 지금 위치와 맞지 않는 흰 실선
    for _ in range(5):
        road.predict(0.007, 0.0)
        road.update(seen)
    assert abs(road.c + 0.53) < 0.05


def test_tracking_correction_is_limited_per_frame():
    road = T.RoadState()
    road.update(obs(0.0))
    road.predict(0.007, 0.0)
    road.update(obs(0.14, th=math.radians(15)))   # 한 프레임 만에 14 cm, 15도 옮겨 보임
    assert abs(road.c) <= T.TRACK_C_STEP + 0.01
    assert abs(road.th) <= T.TRACK_TH_STEP + 1e-6
    for _ in range(20):                           # 계속 그렇게 보이면 따라간다
        road.predict(0.0, 0.0)
        road.update(obs(0.14, th=math.radians(15)))
    assert abs(road.c - 0.14) < 0.03


def test_acquire_needs_a_clear_lane():
    assert T.acquire(None) is None
    c, th = T.acquire(obs(0.05, th=0.1))
    assert abs(c - 0.05) < 0.04 and abs(th - 0.1) < 1e-6
    # 흰 선 한 줄뿐: 가운데 차로인지 옆 차로인지 가릴 수 없다
    assert T.acquire(dict(th=0.0, lines=[(0.25, 250, 0.0)], score=1.0, n=500)) is None
    # 반사광: 22 cm 떨어진 흰 줄 두 개 (11/15 12:38)
    assert T.acquire(dict(th=0.5, lines=[(-0.31, 285, 0.0), (-0.09, 708, 0.0)], score=1.0, n=1000)) is None


def test_island_borders_hold_position_in_front_of_island():
    # 가운데 차로에서 섬을 마주 봄: 노란 테두리 두 줄이 +-0.27 자리에, 갈라지는 흰 선은 제자리가 아님 (bag_3lane2 #388)
    road = T.RoadState()
    road.update(obs(0.02))
    seen = dict(th=0.0, score=1.0, n=1000,
                lines=[(-0.57, 115, 0.0), (-0.33, 190, 1.0), (-0.07, 91, 0.0), (0.24, 274, 1.0), (0.55, 92, 0.0)])
    for _ in range(15):
        road.predict(0.007, 0.0)
        road.update(seen)
    assert abs(road.c - 0.045) < 0.05


def test_front_gap_ignores_things_beside_the_car():
    ranges, a0, inc, rmin = scan_with([(0.25, 0.0)])
    assert abs(T.front_gap(ranges, a0, inc, rmin) - 0.25) < 0.02
    ranges, a0, inc, rmin = scan_with([(0.25, 0.30), (0.60, -0.25)])
    assert T.front_gap(ranges, a0, inc, rmin) == T.LIDAR_MAX


def test_stops_for_object_straight_ahead_even_if_lane_estimate_says_free():
    tl = T.ThreeLane()
    tl.step_obs(obs(0.27), (0.0, 0.0, 0.0), None)
    # 차로 경계(흰 점선 위)에 있어 어느 차로 계산에도 애매하게 걸린 물체가 바로 앞 0.25 m 에 있다
    tl.road.c = 0.535                             # 실제와 다르게 왼쪽 차로 가운데라고 착각
    out = tl.step_obs(None, (0.0, 0.0, 0.0), scan_with([(0.25, 0.0), (0.25, 0.05), (0.25, -0.05)]))
    assert out.state == 'STOP' and out.speed == 0.0 and '정면' in out.reason


def test_supervisor_starts_only_after_consistent_frames(monkeypatch):
    sup = T.Lane3Supervisor(forced=True)
    answers = iter([((-0.16, 0.54), True), ((0.06, -0.30), True), ((0.06, -0.30), True), ((0.07, -0.30), True)])
    monkeypatch.setattr(sup.tl, 'look', lambda image, tune=None: next(answers))
    monkeypatch.setattr(sup.tl, 'step', lambda *a, **k: T.ThreeLaneOutput())
    pose = (0.0, 0.0, 0.0)
    for expected in ('armed', 'armed', 'armed', 'on'):
        sup.step(None, pose, None, False)
        assert sup.mode == expected
    assert abs(sup.tl.road.c - 0.07) < 1e-6 and abs(sup.tl.road.th + 0.30) < 1e-6


def test_sign_vote_ignores_small_boxes_and_needs_four_of_five():
    vote = T.SignVote()
    for _ in range(10):                                   # 멀리서(작게) 읽은 A 는 몇 번이 와도 쓰지 않는다
        assert vote.add('a', 16.0) is None
    assert vote.goal is None
    for _ in range(6):                                    # 11/15 실차: B 표지판을 29 px 에서 A 로 읽음 -> 이 크기도 버린다
        assert vote.add('a', 29.0) is None
    big = T.SIGN_MIN_H + 2.0
    assert vote.add('b', big) is None and vote.add('a', big) is None
    assert vote.add('b', big) is None and vote.add('b', big) is None
    assert vote.add('b', big) == 2                        # 최근 5개 중 4개가 B
    assert vote.add('a', 60.0) is None and vote.goal == 2     # 한 번 튄 값으로는 안 바뀐다


def test_supervisor_decides_sign_from_detections_only_inside_three_lane():
    det = lambda label, h: [{'label': label, 'confidence': 0.8, 'bbox': [100.0, 100.0, 120.0, 100.0 + h]}]  # noqa: E731
    sup = T.Lane3Supervisor()
    for _ in range(6):
        sup.see_signs(det('b', 40))                       # 아직 3차선 구간이 아님
    assert sup.sign_goal is None
    sup.set_mission('M7')
    for _ in range(4):
        sup.see_signs(det('a', 15))                       # 멀리서 읽은 A
    assert sup.sign_goal is None
    for _ in range(4):
        sup.see_signs(det('b', T.SIGN_MIN_H + 5))
    assert sup.sign_goal == 2 and 'B' in sup.message
    fixed = T.Lane3Supervisor(forced=True, goal=0)        # goal_a 로 정해 두면 표지판을 보지 않는다
    for _ in range(6):
        fixed.see_signs(det('b', 40))
    assert fixed.sign_goal == 0


def clear():
    return [T.LIDAR_MAX] * 3, [T.LIDAR_MAX] * 3


def test_waits_in_center_lane_until_sign_is_decided():
    planner = T.Planner()
    ahead, side = clear()
    plan = None
    for _ in range(T.GOAL_CLEAR_FRAMES):
        plan = planner.update(2, T.LANE_CENTER[2], ahead, side, goal=None, may_center=True)
    assert plan.state == 'CHANGE' and plan.target == 1
    # 어느 차로든 앞에 뭔가 보이면 돌아가지 않는다 (돌아간 뒤 빠져나갈 길이 막힐 수 있다)
    planner = T.Planner()
    ahead, side = clear()
    ahead[0] = side[0] = 1.8
    for _ in range(30):
        plan = planner.update(2, T.LANE_CENTER[2], ahead, side, goal=None, may_center=True)
    assert plan.state == 'KEEP' and plan.target == 2


def test_crosses_two_lanes_to_sign_side_only_when_island_is_far():
    planner = T.Planner()
    ahead, side = clear()
    plan = None
    for k in range(T.CROSS_FRAMES):
        plan = planner.update(2, T.LANE_CENTER[2], ahead, side, goal=0)
        assert (plan.state == 'CHANGE') == (k == T.CROSS_FRAMES - 1)
    assert plan.target == 0
    # 섬(가운데 차로 끝)이 2.0 m 앞에 보이면 건너지 않는다 (11/15 13:33 실차 배치)
    planner = T.Planner()
    ahead, side = clear()
    ahead[1] = side[1] = 2.0
    for _ in range(60):
        plan = planner.update(2, T.LANE_CENTER[2], ahead, side, goal=0)
    assert plan.state == 'KEEP' and plan.target == 2


def test_obstacle_avoidance_comes_before_sign_side():
    planner = T.Planner()
    ahead, side = clear()
    ahead[1] = side[1] = 1.0                              # 가운데 차로(지금 차로) 앞이 막힘, 표지판은 A
    ahead[0] = side[0] = 0.6                              # A 쪽 차로도 막힘
    plan = planner.update(1, 0.0, ahead, side, goal=0)
    assert plan.state == 'CHANGE' and plan.target == 2    # 표지판과 반대여도 빈 쪽으로 피한다


def test_undecided_sign_at_island_takes_default_side():
    planner = T.Planner()
    ahead, side = clear()
    ahead[1] = side[1] = 1.2                              # 섬
    plan = planner.update(1, 0.0, ahead, side, goal=None)
    assert plan.state == 'CHANGE' and plan.target == T.DEFAULT_SIDE


def test_yellow_line_that_starts_ahead_is_not_a_wall_until_it_is_close():
    lines = [(0.19, 280, 1.0)]                            # 왼쪽 19 cm 노란 선, 20도로 다가가는 중 (11/15 03:50 #361: 섬 왼쪽 테두리)
    th = -math.radians(20)
    assert T.yellow_wall(lines, th) is not None                       # 시작 지점을 모르면 벽
    assert T.yellow_wall(lines, th, {0.19: 1.05}) is None             # 1.05 m 앞에서 시작 -> 그 전에 지나간다
    assert T.yellow_wall(lines, th, {0.19: 0.45}) is not None         # 바로 앞부터 있는 선 (바깥 노란 선)
    beside = [(0.14, 280, 1.0)]                           # 나란히 가는 중
    assert T.yellow_wall(beside, 0.0, {0.14: 1.5}) is None
    assert T.yellow_wall(beside, 0.0, {0.14: 0.6}) is not None


def test_yellow_wall_stays_on_once_triggered():
    tl = T.ThreeLane()
    near = dict(th=-math.radians(30), lines=[(0.28, 200, 1.0)], score=1.0, n=1000, ystart={0.28: 0.45})
    tl.step_obs(obs(0.3), (0.0, 0.0, 0.0), None)
    out = tl.step_obs(near, (0.0, 0.0, 0.0), None)
    assert out.wall is not None and out.wall < 0                      # 왼쪽 선 -> 오른쪽으로
    # 코앞에서는 카메라 사각 때문에 선이 '앞에서 시작'하는 것처럼 보인다. 이미 걸린 쪽은 계속 피한다
    blind = dict(th=-math.radians(25), lines=[(0.06, 200, 1.0)], score=1.0, n=1000, ystart={0.06: 0.47})
    assert T.yellow_wall(blind['lines'], blind['th'], blind['ystart']) is None
    out = tl.step_obs(blind, (0.0, 0.0, 0.0), None)
    assert out.wall is not None and out.wall < 0


def test_side_lane_counts_as_clear_once_obstacle_is_almost_passed_and_well_aside():
    # 11/15 실차: 가운데 차로 장애물을 다 지나칠 때까지 기다리는 사이 왼쪽 차로 앞 장애물이 0.8 m 안으로 들어와 못 옮김
    c = T.LANE_CENTER[0]
    tail = [(0.05, -0.35), (0.00, -0.35), (-0.05, -0.35)]              # 옆 장애물의 끝이 라이다 바로 옆, 차 옆면에서 24 cm
    ranges, a0, inc, rmin = scan_with(tail)
    _, side = T.lane_free(ranges, a0, inc, rmin, c=c, th=0.0)
    assert side[1] == T.LIDAR_MAX
    close = [(0.05, -0.22), (0.00, -0.22), (-0.05, -0.22)]             # 차가 가운데 차로 쪽으로 붙어 있어 옆면에서 11 cm
    ranges, a0, inc, rmin = scan_with(close)
    _, side = T.lane_free(ranges, a0, inc, rmin, c=0.40, th=0.0)       # -> 아직 못 꺾는다
    assert side[1] < 0.2
    ahead_of_car = [(0.30, -0.35), (0.35, -0.35)]                      # 아직 앞에 있는 장애물은 그대로 막힘
    ranges, a0, inc, rmin = scan_with(ahead_of_car)
    _, side = T.lane_free(ranges, a0, inc, rmin, c=c, th=0.0)
    assert side[1] < 0.5


def test_escape_starts_closer_than_before_and_steers_harder():
    planner = T.Planner()
    ahead, side = clear()
    ahead[0] = side[0] = 0.70                             # 예전 기준(0.8 m)보다 가깝다
    plan = planner.update(0, T.LANE_CENTER[0], ahead, side)
    assert plan.state == 'CHANGE' and plan.target == 1 and planner.tight
    planner = T.Planner()
    ahead[0] = side[0] = T.MIN_START - 0.05               # 이보다 가까우면 여전히 시작하지 않는다
    assert planner.update(0, T.LANE_CENTER[0], ahead, side).state == 'BLOCKED'
    planner = T.Planner()
    ahead[0] = side[0] = 1.2                              # 여유 있을 때는 평소대로
    planner.update(0, T.LANE_CENTER[0], ahead, side)
    assert planner.changing and not planner.tight
    tight = T.steer_to_lane(T.LANE_CENTER[0], 0.0, 1, T.LOOKAHEAD_TIGHT)
    normal = T.steer_to_lane(T.LANE_CENTER[0], 0.0, 1, T.LOOKAHEAD_CHANGE)
    assert abs(tight) >= abs(normal)


def drive(tl, scans_by_step, steps, x0=0.0):
    """obs(왼쪽 차로 가운데) 로 steps 번 호출. scans_by_step(k, x) -> 라이다. 반환: 상태 목록과 마지막 x."""
    x, states = x0, []
    for k in range(steps):
        out = tl.step_obs(obs(T.LANE_CENTER[0]), (x, 0.0, 0.0), scans_by_step(k, x))
        states.append(out.state)
        x += out.speed / 22.0                             # odom: 후진이면 x 가 줄어든다
    return states, x


def test_reverse_keeps_backing_until_there_is_room_then_goes():
    # 앞 장애물 0.36 m, 옆(가운데) 차로는 비어 있음: 후진을 시작하면 곧 옮길 수 있지만 REVERSE_GO_AHEAD 까지는 더 물러난다
    tl = T.ThreeLane()
    face = 0.36
    def scan(k, x):
        d = face - x
        return scan_with([(d, 0.0), (d, 0.05), (d, -0.05)])
    states, x = drive(tl, scan, 400)
    assert 'REVERSE' in states and 'CHANGE' in states
    first_change = states.index('CHANGE', states.index('REVERSE'))
    assert all(s != 'CHANGE' for s in states[:first_change])
    backed = -min(0.0, min(_x for _x in [x]))             # 끝 위치로는 알 수 없으니 후진 프레임 수로 확인
    reverse_frames = sum(1 for s in states[:first_change] if s == 'REVERSE')
    assert reverse_frames * T.V_REVERSE / 22.0 >= T.REVERSE_GO_AHEAD - face - 0.08


def test_no_straight_reverse_right_after_a_lane_change():
    # 옮기는 중에 지나온 길은 곧지 않다 -> 똑바로 후진하면 방금 피한 장애물에 닿는다 (시뮬레이션에서 2건)
    tl = T.ThreeLane()
    tl.step_obs(obs(0.0), (0.0, 0.0, 0.0), None)
    tl.lane_run = 2.0
    tl.planner.target, tl.planner.changing = 0, True      # 왼쪽 차로로 옮기는 중
    tl.step_obs(obs(0.2), (0.0, 0.0, 0.0), scan_with([(2.0, 0.3)]))
    assert tl.lane_run == 0.0


# ---------------------------------------------------------------------------
# 경로를 미리 굴려 보는 판단 (11/15: 거리 문턱값 대신 '실제로 지나갈 수 있는가')
# ---------------------------------------------------------------------------
def test_path_clearance_open_road_and_blocked_path():
    # 빈 도로: 가운데 -> 왼쪽 차로는 넉넉히 간다
    assert T.path_clearance(None, None, 0.0, 0.0, 0, T.LOOKAHEAD_CHANGE, T.HEADING_MAX) >= T.PATH_GOOD
    # 가는 길 한가운데(대각선 위)에 점이 있으면 여유가 안 나온다
    along = np.array([0.45, 0.50, 0.55]); lateral = np.array([0.25, 0.27, 0.29])
    assert T.path_clearance(along, lateral, 0.0, 0.0, 0, T.LOOKAHEAD_CHANGE, T.HEADING_MAX) < T.PATH_MARGIN
    # 같은 점이라도 반대쪽 차로로 가는 데는 지장이 없다
    assert T.path_clearance(along, lateral, 0.0, 0.0, 2, T.LOOKAHEAD_CHANGE, T.HEADING_MAX) >= T.PATH_GOOD


def test_path_clearance_refuses_leaving_the_road():
    # 왼쪽 차로에서 왼쪽으로 크게 틀어진 채 출발: 어떤 차로로 가든 차체가 바깥 선을 넘는다
    assert T.path_clearance(None, None, 0.70, -math.radians(40.0), 0, T.LOOKAHEAD_CHANGE, T.HEADING_MAX) == -1.0


def test_reverse_room_stops_before_remembered_points():
    # 뒤가 비어 있으면 끝까지, 차 뒤 0.3 m 에 점이 있으면 그 앞까지만
    assert T.reverse_room(np.zeros(0), np.zeros(0), 0.0, 0.0, 0.6) >= 0.55
    tail = -T.CAMERA_FORWARD_OFFSET + T.CAR_BODY_CENTER - T.CAR_BODY_HALF_LEN      # 차 뒤끝 (카메라 기준)
    along = np.array([tail - 0.30, tail - 0.30]); lateral = np.array([-0.05, 0.05])
    room = T.reverse_room(along, lateral, 0.0, 0.0, 0.6)
    assert 0.15 <= room <= 0.30 - T.REVERSE_MARGIN + 1e-6
    # 앞에 있는 점은 후진을 막지 않는다
    assert T.reverse_room(np.array([0.3, 0.3]), np.array([0.0, 0.05]), 0.0, 0.0, 0.6) >= 0.55


def test_island_front_is_remembered_when_it_goes_under_the_camera():
    # 섬이 1.0 m 앞에 보이다가, 0.6 m 나아간 뒤에는 옆 테두리만(늘 0.45 m 로) 보인다 -> 앞끝은 0.4 m 앞으로 남아야 한다
    tl = T.ThreeLane()
    left = T.LANE_CENTER[0]

    def frame(x, isl):
        o = obs(left)
        f = np.linspace(isl, isl + 0.6, 40)
        o['yellow_xy'] = np.stack([np.full(40, left - 0.27), f], 1)       # 섬 왼쪽 테두리 (카메라 오른쪽 +)
        return tl.step_obs(o, (x, 0.0, 0.0), None)

    out = frame(0.0, 1.0)
    assert abs(out.island - 1.0) < 0.08
    for x in np.arange(0.05, 0.61, 0.05):
        out = frame(float(x), max(0.45, 1.0 - float(x)))
    assert abs(out.island - 0.40) < 0.10


def test_chosen_lane_is_kept_for_a_moment():
    # 막혀서 차로 0 을 골랐으면, 바로 다음 프레임에 차로 2 가 조금 더 좋아 보여도 (경로가 나오는 한) 바꾸지 않는다
    p = T.Planner()
    reach = lambda j: ('normal', 0.30)
    plan = p.update(1, 0.0, [2.5, 0.8, 2.0], [2.5, 0.8, 2.0], reach=reach)
    assert (plan.state, plan.target) == ('CHANGE', 0)
    plan = p.update(1, 0.05, [0.9, 0.7, 2.5], [0.9, 0.7, 2.5], reach=reach)
    assert (plan.state, plan.target) == ('CHANGE', 0)
