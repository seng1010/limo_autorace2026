"""lane3_gap 의 흰 점선 기준(WHITE_V_MIN)과 라이다로 거르기(WHITE_LIDAR_MARGIN)."""
import cv2
import numpy as np

from jetracer_ros2 import lane3_gap as G

T = G.T


def _ground_image(dashes, value):
    """바닥(BEV) 위 (옆 x[m, 오른쪽 +], 앞 f0~f1[m]) 자리에 폭 4 cm 흰 선을 그린 카메라 영상."""
    fit = G.make_fitter()
    bev = np.zeros((fit.h, fit.w, 3), np.uint8)
    bev[:] = (18, 18, 18)
    for x, f0, f1 in dashes:
        px = int((x + T.XW) / T.RES)
        bev[int((T.F1 - f1) / T.RES):int((T.F1 - f0) / T.RES), px - 2:px + 2] = (value, value, value)
    return cv2.warpPerspective(bev, np.linalg.inv(fit.i2b), (T.IMG_W, T.IMG_H), flags=cv2.INTER_NEAREST)


def _scan(obstacle_at=None):
    """250줄 ±110도. obstacle_at = (각 [도, 왼쪽 +], 거리 [m]) 둘레 ±15도에 물체."""
    n = 250
    amin, inc = -np.radians(110.0), np.radians(220.0) / (n - 1)
    r = np.full(n, np.inf, np.float32)
    if obstacle_at is not None:
        a = amin + np.arange(n) * inc
        r[np.abs(a - np.radians(obstacle_at[0])) < np.radians(15.0)] = obstacle_at[1]
    return (r, amin, inc, 0.05)


def test_same_as_three_lane_when_threshold_is_150_and_no_lidar(monkeypatch):
    monkeypatch.setattr(G, 'WHITE_V_MIN', 150)
    image = _ground_image([(-0.27, 0.6, 0.9), (0.27, 1.1, 1.4)], 200)
    cv2.rectangle(image, (60, 300), (110, 470), (0, 200, 230), -1)          # 노란 선
    fit = G.make_fitter()
    yellow, white = fit.masks(image, {'v_min': 50})
    ref = G.make_fitter()
    ry, rw = T.LineFitter.masks(ref, image, {'v_min': 50})
    assert yellow.any() and white.any()
    assert np.array_equal(yellow, ry) and np.array_equal(white, rw)


def test_dim_dashes_are_seen_with_100_but_not_with_150(monkeypatch):
    image = _ground_image([(-0.27, 0.6, 0.9), (-0.27, 1.2, 1.5), (0.27, 0.6, 0.9), (0.27, 1.2, 1.5)], 125)
    fit = G.make_fitter()
    fit.masks(image)
    assert G.WHITE_V_MIN == 100
    assert G.white_heading(fit.white) is not None and abs(G.white_heading(fit.white)) < np.radians(3.0)
    monkeypatch.setattr(G, 'WHITE_V_MIN', 150)
    fit.masks(image)
    assert not fit.white.any()


def test_white_behind_a_lidar_object_is_dropped_and_floor_in_front_is_kept():
    image = _ground_image([(0.0, 0.6, 0.9), (0.0, 1.5, 1.8)], 200)
    fit = G.make_fitter()
    fit.masks(image)
    assert fit.white[int((T.F1 - 1.65) / T.RES), fit.w // 2 - 2:fit.w // 2 + 2].any()      # 라이다 없음: 둘 다 보임
    fit.scan = _scan((0.0, 1.3))                                                            # 정면 1.3 m 에 물체
    fit.masks(image)
    assert fit.white[int((T.F1 - 0.75) / T.RES), fit.w // 2 - 2:fit.w // 2 + 2].any()      # 물체 앞 바닥은 남는다
    assert not fit.white[:int((T.F1 - 1.15) / T.RES)].any()                                 # 물체 0.15 m 앞부터 뒤는 지운다


def test_lidar_object_on_the_other_side_does_not_remove_the_dash():
    image = _ground_image([(-0.5, 0.8, 1.1)], 200)                                          # 왼쪽 0.5 m
    fit = G.make_fitter()
    fit.scan = _scan((-40.0, 0.5))                                                          # 오른쪽 40도, 0.5 m 에 물체
    fit.masks(image)
    assert fit.white.any()


def test_broken_scan_falls_back_to_unfiltered_white():
    image = _ground_image([(0.0, 0.6, 0.9), (0.0, 1.5, 1.8)], 200)
    fit = G.make_fitter()
    fit.masks(image)
    plain = fit.white.copy()
    fit.scan = (np.zeros(0, np.float32), -1.9, 0.0154, 0.05)                                # 빈 라이다 값 -> 계산 실패
    fit.masks(image)
    assert np.array_equal(fit.white, plain)


def test_supervisor_hands_the_scan_to_the_fitter():
    sup = G.GapSupervisor(forced=True)
    scan = _scan((0.0, 1.3))
    sup.step(_ground_image([(0.0, 0.6, 0.9)], 200), (0.0, 0.0, 0.0), scan, False, {})
    assert sup.fitter.scan is scan
