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
    assert planner.update(1, 0.0, [2.0, far, far], [2.0, far, far], goal=0).target == 1
    assert planner.update(1, 0.0, [far, far, far], [far, far, far], goal=0).target == 0


def test_two_lane_move_through_empty_middle_lane():
    far = T.LIDAR_MAX
    plan = T.Planner().update(2, T.LANE_CENTER[2], [far, 1.6, 1.0], [far, 1.6, 1.0], goal=None)
    assert plan.target == 0
    # 가운데 차로가 1.3 m 밖에 안 비었으면 가로지르지 않는다 (가로지르는 데 약 1.4 m 필요)
    plan = T.Planner().update(2, T.LANE_CENTER[2], [far, 1.3, 1.0], [far, 1.3, 1.0], goal=None)
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
    outer = np.stack([np.full(60, -0.27), np.linspace(0.5, 1.8, 60)], 1)
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
    # 갈림길 방향(오른쪽)이 0.6 m 일찍 막히면 더 멀리 빈 왼쪽으로 간다 (11/15 실차에서 멈춰 선 배치)
    plan = T.Planner().update(1, 0.0, [far, 1.2, 1.9], [far, 1.2, 1.9], goal=2)
    assert plan.target == 0
    # 빈 거리가 비슷하면 갈림길 방향
    plan = T.Planner().update(1, 0.0, [far, 1.2, 2.3], [far, 1.2, 2.3], goal=2)
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
