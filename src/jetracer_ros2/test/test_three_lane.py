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
    plan = T.Planner().update(2, T.LANE_CENTER[2], [far, 1.3, 1.2], [far, 1.3, 1.2], goal=None)
    assert plan.target == 0


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
