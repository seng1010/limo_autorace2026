import numpy as np
import pytest

from jetracer_ros2.stop_line_detector import (
    BlindZoneTracker, OdomPathIntegrator, RowDistanceModel, StopLineConfig,
    detect_stop_line, fit_row_distance_table)

WHITE = (255, 255, 255)
YELLOW = (0, 220, 240)          # BGR


def _image_with_band(top, bottom, color=WHITE, left=60, right=580):
    image = np.full((480, 640, 3), 60, dtype=np.uint8)
    image[top:bottom, left:right] = color
    return image


# ---- 검출 ---------------------------------------------------------------- #

def test_detects_white_line_bottom_edge():
    detection = detect_stop_line(_image_with_band(380, 400), StopLineConfig())
    assert detection is not None
    assert detection.row_ratio == pytest.approx(400 / 480, abs=2 / 480)


def test_picks_nearest_of_two_lines():
    image = _image_with_band(300, 318)
    image[420:440, 60:580] = WHITE
    detection = detect_stop_line(image, StopLineConfig())
    assert detection.row_ratio == pytest.approx(440 / 480, abs=2 / 480)


def test_yellow_line_needs_yellow_enabled():
    image = _image_with_band(380, 400, YELLOW)
    assert detect_stop_line(image, StopLineConfig()) is None
    config = StopLineConfig(detect_white=False, detect_yellow=True)
    assert detect_stop_line(image, config) is not None


def test_vertical_lane_lines_are_not_stop_lines():
    image = np.full((480, 640, 3), 60, dtype=np.uint8)
    image[:, 150:165] = WHITE
    image[:, 480:495] = WHITE
    assert detect_stop_line(image, StopLineConfig()) is None


def test_line_above_roi_is_ignored():
    assert detect_stop_line(_image_with_band(100, 120), StopLineConfig()) is None


# ---- 행 -> 거리 ------------------------------------------------------------ #

def test_uncalibrated_model_matches_legacy_formula():
    model = RowDistanceModel([0.0], [0.0], legacy_scale_m=1.6, roi_top_ratio=0.58)
    assert not model.calibrated
    assert model.distance(1.0) == pytest.approx(0.0)
    assert model.distance(0.58) == pytest.approx(1.6)
    assert model.distance(0.79) == pytest.approx(0.8)


def test_lut_interpolates_and_extrapolates():
    model = RowDistanceModel([0.6, 0.8, 0.95], [0.50, 0.25, 0.12])
    assert model.calibrated
    assert model.distance(0.7) == pytest.approx(0.375)
    assert model.distance(1.0) == pytest.approx(0.12 - 0.13 / 0.15 * 0.05)
    assert model.distance(0.55) > 0.50


def test_lut_rejects_non_monotonic():
    with pytest.raises(ValueError):
        RowDistanceModel([0.6, 0.8], [0.2, 0.4])


# ---- 사각지대 추적 ---------------------------------------------------------- #

def _tracker():
    tracker = BlindZoneTracker(max_track_m=0.6, pass_margin_m=0.10, jump_tolerance_m=0.15)
    tracker.update_odom(0.0)
    return tracker


def test_tracks_with_odom_after_line_disappears():
    tracker = _tracker()
    assert tracker.update(0.30) == pytest.approx(0.30)
    tracker.update_odom(0.12)
    assert tracker.update(None) == pytest.approx(0.18)
    assert tracker.source == 'odom'


def test_stationary_car_keeps_distance():
    """빨간불 대기: 선이 안 보여도 거리 유지 -> 재출발하지 않음."""
    tracker = _tracker()
    tracker.update(0.20)
    tracker.update_odom(0.19)
    for _ in range(100):
        assert tracker.update(None) == pytest.approx(0.01)


def test_tracking_ends_after_passing_line():
    tracker = _tracker()
    tracker.update(0.20)
    tracker.update_odom(0.25)
    assert tracker.update(None) == pytest.approx(0.0)       # 살짝 지나침 -> 0
    tracker.update_odom(0.31)
    assert tracker.update(None) is None


def test_far_dropout_is_not_tracked():
    tracker = _tracker()
    tracker.update(1.0)
    assert tracker.update(None) is None


def test_farther_line_ignored_while_tracking():
    tracker = _tracker()
    tracker.update(0.25)
    tracker.update_odom(0.10)
    assert tracker.update(0.70) == pytest.approx(0.15)       # 횡단보도 반대편 선
    assert tracker.update(0.16) == pytest.approx(0.16)       # 같은 선이면 다시 화면 값


def test_without_odom_behaves_like_before():
    tracker = BlindZoneTracker()
    assert tracker.update(0.3) == pytest.approx(0.3)
    assert tracker.update(None) is None


def test_odom_integrator_signed_and_ignores_jumps():
    odom = OdomPathIntegrator(max_step_m=0.5)
    odom.update(0.0, 0.0, 0.1)
    odom.update(0.3, 0.4, 0.1)
    assert odom.s == pytest.approx(0.5)
    odom.update(0.3, 0.3, -0.1)
    assert odom.s == pytest.approx(0.4)
    odom.update(5.0, 5.0, 0.1)                                # odom 리셋
    assert odom.s == pytest.approx(0.4)


# ---- 보정표 ---------------------------------------------------------------- #

def test_fit_recovers_monotonic_table():
    rng = np.random.default_rng(0)
    distances = np.linspace(0.12, 0.60, 200)
    rows = 1.0 - 0.6 * distances + rng.normal(0.0, 0.003, distances.size)
    lut_rows, lut_dists = fit_row_distance_table(list(zip(rows, distances)), points=8)
    assert len(lut_rows) >= 6
    assert all(np.diff(lut_rows) > 0) and all(np.diff(lut_dists) < 0)
    model = RowDistanceModel(lut_rows, lut_dists)
    assert model.distance(1.0 - 0.6 * 0.3) == pytest.approx(0.3, abs=0.01)
