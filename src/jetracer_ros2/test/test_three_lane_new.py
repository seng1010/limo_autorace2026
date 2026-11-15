"""새 판단 (PLANNER = 'new', PathPlanner): 옆 위치 후보마다 경로를 굴려 가장 멀리 가는 쪽을 고른다."""
import math

import numpy as np
import pytest

from jetracer_ros2 import three_lane as T


@pytest.fixture(autouse=True)
def _new(monkeypatch):
    monkeypatch.setattr(T, 'PLANNER', 'new')


def box(x0, lane, length=0.32, width=0.22, dy=0.0):
    """정차 LIMO 한 대의 둘레 점 (카메라 기준 앞 거리, 도로 기준 옆 위치)."""
    y = T.LANE_CENTER[lane] + dy
    xs = np.arange(x0, x0 + length + 1e-6, 0.02)
    ys = np.arange(y - width / 2, y + width / 2 + 1e-6, 0.02)
    a = np.concatenate([xs, xs, np.full(ys.size, x0), np.full(ys.size, x0 + length)])
    l = np.concatenate([np.full(xs.size, y - width / 2), np.full(xs.size, y + width / 2), ys, ys])
    return a, l


def points(*boxes):
    if not boxes:
        return np.zeros(0), np.zeros(0)
    return np.concatenate([b[0] for b in boxes]), np.concatenate([b[1] for b in boxes])


def settle(p, c, a, l, island=None, goal=None, n=4, **kw):
    for _ in range(n):
        plan = p.update(c, 0.0, a, l, island, goal, **kw)
    return plan


def test_rollout_matches_scalar_steering_rule():
    X, Y, PSI = T.np_rollouts(0.0, 0.0, np.array([0.535]), np.array([T.LOOKAHEAD_CHANGE]), np.array([T.HEADING_MAX]))
    assert Y[0, -1] + T.CAMERA_FORWARD_OFFSET * math.sin(PSI[0, -1]) == pytest.approx(0.535, abs=0.05)
    assert abs(PSI[0]).max() <= T.HEADING_MAX + math.radians(6.0)
    assert 0.8 < X[0, np.argmax(Y[0] > 0.45)] + T.CAMERA_FORWARD_OFFSET < 1.4      # 한 칸에 앞으로 약 1 m


def test_free_distance_matches_obstacle_distance():
    a, l = points(box(1.0, 1))
    X, Y, PSI = T.np_rollouts(0.0, 0.0, T.NP_YS, np.full(13, T.LOOKAHEAD_CHANGE), np.full(13, T.HEADING_MAX))
    F, C = T.np_free(X, Y, PSI, T.np_grid(a, l), -T.CAMERA_FORWARD_OFFSET)
    assert 0.75 < F[6] < 1.0                     # 가운데: 장애물 앞까지
    assert F[0] == T.NP_FREE_CAP and F[12] == T.NP_FREE_CAP
    assert C[12] > 0.10


def test_keeps_lane_on_empty_road_and_returns_to_center_when_sign_unknown():
    p = T.PathPlanner()
    plan = settle(p, 0.0, *points())
    assert plan.y == pytest.approx(0.0) and plan.state == 'KEEP' and plan.speed == T.V_KEEP
    p = T.PathPlanner()
    plan = settle(p, 0.535, *points())
    assert plan.y == pytest.approx(0.0) and plan.state == 'CHANGE'


def test_moves_aside_for_obstacle_and_not_into_blocked_side():
    p = T.PathPlanner()
    a, l = points(box(1.0, 1), box(0.9, 0))      # 가운데와 왼쪽이 막힘 -> 오른쪽
    plan = settle(p, 0.0, a, l)
    assert plan.y < -0.3 and plan.speed > 0.0


def test_far_obstacle_does_not_cause_early_swerve():
    p = T.PathPlanner()
    plan = settle(p, 0.0, *points(box(1.9, 1)))
    assert plan.y == pytest.approx(0.0)          # NP_STAY 보다 멀면 그대로 간다


def test_goal_side_only_if_it_goes_as_far():
    a, l = points(box(1.2, 1), box(1.0, 0))      # 표지판 쪽(왼쪽)이 곧 막힌다 -> 오른쪽 (장애물 회피가 먼저)
    plan = settle(T.PathPlanner(), 0.0, a, l, goal=0)
    assert plan.y < -0.3
    plan = settle(T.PathPlanner(), 0.0, *points(), goal=0)
    assert plan.y == pytest.approx(0.535)


def test_threads_between_staggered_cars_with_partial_shift():
    # 왼쪽 차로 차 바로 뒤에 가운데 차로 차: 차로 가운데끼리는 못 옮기지만 비스듬히는 지나간다
    a, l = points(box(0.3, 1), box(0.95, 0), box(0.95, 2))
    plan = settle(T.PathPlanner(), 0.535, a, l)
    assert plan.speed > 0.0 and plan.free > 0.6


def test_waits_in_center_for_sign_then_commits_at_last_moment():
    p = T.PathPlanner()
    plan = settle(p, 0.0, *points(), island=1.6)
    assert plan.y == pytest.approx(0.0) and '읽는 중' in plan.reason and plan.speed > 0.0
    plan = settle(p, 0.0, *points(), island=0.85)
    assert abs(plan.y) == pytest.approx(0.535) and p.side in (0, 2)


def test_one_side_blocked_far_from_island_keeps_waiting():
    # 11/15 실차: 왼쪽 옆에 장애물이 있다고 섬 1.8 m 앞에서 바로 오른쪽으로 정해 버렸다
    p = T.PathPlanner()
    plan = settle(p, 0.0, *points(box(0.0, 0)), island=1.8)
    assert plan.y == pytest.approx(0.0) and p.side is None


def test_sign_known_in_center_turns_at_once():
    plan = settle(T.PathPlanner(), 0.0, *points(), island=1.5, goal=0)
    assert plan.y == pytest.approx(0.535)


def test_sign_side_blocked_at_island():
    a, l = points(box(-0.3, 0, length=1.3))      # 표지판 쪽(왼쪽)이 섬 앞까지 막혀 있다
    p = T.PathPlanner()
    plan = settle(p, 0.0, a, l, island=1.3, goal=0)
    assert plan.y == pytest.approx(0.0) and plan.speed > 0.0       # 반대쪽으로 빠질 여유가 있는 동안은 기다린다
    plan = settle(p, 0.0, a, l, island=0.75, goal=0)
    assert plan.y == pytest.approx(-0.535)       # 물러나도 소용없다 -> 반대쪽 (서지 않는다)
    p = T.PathPlanner()
    plan = settle(p, 0.0, *points(), island=0.45, goal=0)
    assert plan.state == 'STOP'                  # 너무 가까이서 읽음: 물러나면 갈 수 있다 -> 서서 후진


def test_stops_when_nothing_is_passable_and_no_scan_means_slow():
    p = T.PathPlanner()
    a, l = points(box(0.25, 0), box(0.25, 1), box(0.25, 2), box(0.25, 1, dy=0.27), box(0.25, 1, dy=-0.27))
    plan = settle(p, 0.0, a, l)
    assert plan.state == 'STOP' and plan.speed == 0.0
    plan = T.PathPlanner().update(0.0, 0.0, None, None, have_scan=False)
    assert plan.speed == T.V_SLOW


def test_front_gap_follows_the_arc_when_steering():
    n = 250
    a0, inc = -1.92, 3.84 / (n - 1)
    r = np.full(n, np.inf, np.float32)
    for i in range(n):                            # 정면 0.25 m, 폭 0.10 m 물체
        ang = a0 + i * inc
        if abs(ang) < 0.2:
            r[i] = 0.25 / math.cos(ang)
    assert T.front_gap(r, a0, inc, 0.15) < T.GUARD_DIST
    assert T.front_gap(r, a0, inc, 0.15, math.radians(25.0)) < T.GUARD_DIST      # 넓은 물체는 꺾어도 걸린다
    r2 = np.full(n, np.inf, np.float32)
    for i in range(n):                            # 오른쪽 앞 모서리에 살짝 걸친 물체
        ang = a0 + i * inc
        x, y = 0.25, 0.25 * math.tan(ang)
        if -0.20 < y < -0.10:
            r2[i] = math.hypot(x, y)
    assert T.front_gap(r2, a0, inc, 0.15) < T.GUARD_DIST
    assert T.front_gap(r2, a0, inc, 0.15, math.radians(20.0)) >= T.GUARD_DIST    # 왼쪽으로 꺾는 중이면 지나간다


def test_three_lane_uses_path_planner_and_steers_to_lateral_target():
    tl = T.ThreeLane()
    assert isinstance(tl.planner, T.PathPlanner)
    tl.start(0.0, 0.0, (0.0, 0.0, 0.0))
    obs = dict(th=0.0, lines=[(-0.80, 300, 1.0), (-0.27, 150, 0.0), (0.27, 150, 0.0), (0.80, 300, 1.0)], score=1.0, n=1000,
               yellow_xy=np.zeros((0, 2)), ystart={})
    n = 250
    a0, inc = -1.92, 3.84 / (n - 1)
    r = np.full(n, np.inf, np.float32)
    for i in range(n):                            # 가운데 차로 1.0 m 앞에 차
        ang = a0 + i * inc
        if abs(1.0 * math.tan(ang)) < 0.11 and abs(ang) < 0.5:
            r[i] = 1.0 / math.cos(ang)
    for _ in range(4):
        out = tl.step_obs(obs, (0.0, 0.0, 0.0), (r, a0, inc, 0.15), None)
    assert out.ok and out.state == 'CHANGE' and abs(out.steer) > math.radians(5.0)
