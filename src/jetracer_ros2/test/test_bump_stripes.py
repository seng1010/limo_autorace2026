"""과속방지턱 노랑·검정 사선 줄무늬를 차선으로 잡지 않기 (simple_lane.LaneEngine).
11/17 경기장 기록: 차선 폭 1.5~4.5 cm, 방지턱 줄무늬 11~15 cm. 줄무늬를 선으로 잡아 사선을 코너로 보고 오른쪽 25도 고정 조향."""
import cv2
import numpy as np

from jetracer_ros2 import simple_lane as SL


def _line(eng, bev, x0, f0, x1, f1, width_m):
    g = eng.geo
    a = g.m_to_px(x0, f0); b = g.m_to_px(x1, f1)
    cv2.line(bev, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, max(1, int(round(width_m / SL.RES))))


def test_steep_corner_line_is_still_a_line():
    # 급한 코너: 바깥 선(폭 3 cm)이 차 앞을 70도로 가로지름. 가로 줄 폭으로 재면 9 cm 가 넘어 줄무늬로 잘못 보던 경우
    eng = SL.LaneEngine()
    eng.bump_enabled = True
    g = eng.geo
    bev = np.zeros((g.bev_h, g.bev_w), np.uint8)
    _line(eng, bev, -0.20, 0.55, 0.20, 0.70, 0.03)
    comps = eng._components(bev)
    assert len(comps) == 1 and not eng.thick


def test_filter_off_outside_m5():
    eng = SL.LaneEngine()                       # 기본: 꺼짐 (M5 가 아닐 때 기본 주행은 그대로)
    g = eng.geo
    bev = np.zeros((g.bev_h, g.bev_w), np.uint8)
    a = g.m_to_px(0.00, 0.50); b = g.m_to_px(0.12, 0.50); c = g.m_to_px(0.42, 0.90); d = g.m_to_px(0.30, 0.90)
    cv2.fillPoly(bev, [np.int32([a, b, c, d])], 255)
    eng._components(bev)
    assert not eng.thick


def test_stripe_is_not_a_lane_line():
    eng = SL.LaneEngine()
    eng.bump_enabled = True
    g = eng.geo
    bev = np.zeros((g.bev_h, g.bev_w), np.uint8)
    # 차 왼쪽 20 cm, 폭 2.5 cm 곧은 차선 (앞 0.45 ~ 1.2 m)
    x0, y0 = g.m_to_px(-0.20, 1.2)
    x1, y1 = g.m_to_px(-0.175, 0.45)
    bev[int(y0):int(y1), int(x0):int(x1)] = 255
    # 폭 12 cm, 오른쪽으로 기운 사선 줄무늬
    for dx in (0.0, 0.22):                      # 줄무늬 두 줄
        a = g.m_to_px(dx, 0.50); b = g.m_to_px(dx + 0.12, 0.50); c = g.m_to_px(dx + 0.42, 0.90); d = g.m_to_px(dx + 0.30, 0.90)
        cv2.fillPoly(bev, [np.int32([a, b, c, d])], 255)
    comps = eng._components(bev)
    assert len(comps) == 1 and comps[0]['near_x'] < -0.1
    assert len(eng.thick) == 2


def test_single_thick_blob_or_far_line_is_kept():
    # 두꺼운 덩어리 하나뿐(갈라지는 차선이 붙음)이거나, 먼 곳(1.2 m)을 가로지르는 코너 선은 줄무늬로 보지 않는다
    eng = SL.LaneEngine()
    eng.bump_enabled = True
    g = eng.geo
    bev = np.zeros((g.bev_h, g.bev_w), np.uint8)
    a = g.m_to_px(0.00, 0.50); b = g.m_to_px(0.12, 0.50); c = g.m_to_px(0.42, 0.90); d = g.m_to_px(0.30, 0.90)
    cv2.fillPoly(bev, [np.int32([a, b, c, d])], 255)
    _line(eng, bev, -0.40, 1.20, 0.40, 1.25, 0.09)
    eng._components(bev)
    assert not eng.thick


def test_m5_corner_lock_uses_imu_until_85_deg():
    import math
    eng = SL.LaneEngine()
    eng.turn_deg_override = SL.M5_TURN_DEG
    eng.lock_t, eng.lock_yaw, eng.lock_imu_yaw = 0.0, 0.0, 0.0
    eng.yaw = math.radians(63)          # odom 은 63도 돌았다고 하지만
    eng.imu_yaw = -math.radians(50)     # 실제(IMU)는 50도 -> 아직 안 끝남
    assert not eng._lock_done(3.3, math.radians(60))
    eng.imu_yaw = -math.radians(86)
    assert eng._lock_done(4.5, math.radians(60))
    eng.imu_yaw = -math.radians(60)     # 그래도 7초가 넘으면 끝
    assert eng._lock_done(7.1, math.radians(60))


def test_other_missions_keep_odom_turn():
    import math
    eng = SL.LaneEngine()               # turn_deg_override 없음 -> 예전처럼 odom 60도
    eng.lock_t, eng.lock_yaw, eng.lock_imu_yaw = 0.0, 0.0, 0.0
    eng.yaw, eng.imu_yaw = math.radians(61), -math.radians(40)
    assert eng._lock_done(3.0, math.radians(60))
