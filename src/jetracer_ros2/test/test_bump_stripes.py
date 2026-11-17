"""과속방지턱 노랑·검정 사선 줄무늬를 차선으로 잡지 않기 (simple_lane.LaneEngine).
11/17 경기장 기록: 차선 폭 1.5~4.5 cm, 방지턱 줄무늬 11~15 cm. 줄무늬를 선으로 잡아 사선을 코너로 보고 오른쪽 25도 고정 조향."""
import cv2
import numpy as np

from jetracer_ros2 import simple_lane as SL


def test_row_width():
    py = np.repeat(np.arange(50), 5)
    xm = np.tile(np.arange(5) * SL.RES, 50)
    assert abs(SL._row_width(py, xm) - 5 * SL.RES) < 1e-9


def test_stripe_is_not_a_lane_line():
    eng = SL.LaneEngine()
    g = eng.geo
    bev = np.zeros((g.bev_h, g.bev_w), np.uint8)
    # 차 왼쪽 20 cm, 폭 2.5 cm 곧은 차선 (앞 0.45 ~ 1.2 m)
    x0, y0 = g.m_to_px(-0.20, 1.2)
    x1, y1 = g.m_to_px(-0.175, 0.45)
    bev[int(y0):int(y1), int(x0):int(x1)] = 255
    # 폭 12 cm, 오른쪽으로 기운 사선 줄무늬
    a = g.m_to_px(0.00, 0.50); b = g.m_to_px(0.12, 0.50); c = g.m_to_px(0.42, 0.90); d = g.m_to_px(0.30, 0.90)
    cv2.fillPoly(bev, [np.int32([a, b, c, d])], 255)
    comps = eng._components(bev)
    assert len(comps) == 1 and comps[0]['near_x'] < -0.1
    assert len(eng.thick) == 1
