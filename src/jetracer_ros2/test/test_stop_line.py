import cv2
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


@pytest.mark.parametrize('brightness', [150, 160, 170])
@pytest.mark.parametrize('angle', [-30., -20., -10., 0., 10., 20., 30.])
def test_dark_oblique_tape_is_not_rejected_as_a_tall_image_box(brightness, angle):
    image = np.full((480, 640, 3), 60, np.uint8)
    points = cv2.boxPoints(((320., 375.), (340., 14.), angle)).astype(np.int32)
    cv2.fillConvexPoly(image, points, (brightness,) * 3)
    detected = detect_stop_line(image, StopLineConfig(white_lightness_min=145))
    assert detected is not None
    # Preserve the opened contour's old lower edge when it already detects
    # the tape. Only the fallback uses the original visible strip extent.
    mask = (image[278:, :, 0] >= 145).astype(np.uint8) * 255
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 76), np.uint8))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    legacy = [cv2.boundingRect(c) for c in contours
              if cv2.boundingRect(c)[2] >= 640 * .42
              and cv2.contourArea(c) >= mask.size * .018]
    bottom = max(278 + y + h for x, y, w, h in legacy) if legacy else points[:, 1].max() + 1
    assert detected.row_ratio * 480 == pytest.approx(bottom, abs=2.)


def test_dark_wavy_tape_survives_horizontal_opening_loss():
    image = np.full((480, 640, 3), 60, np.uint8)
    for x in range(105, 535):
        y = 381 + round(7 * np.sin(x / 55))
        image[y:y + 10, x] = (167, 157, 150)
    assert detect_stop_line(image, StopLineConfig(white_lightness_min=145)) is not None
    # Colour threshold was not silently lowered.
    assert detect_stop_line(image, StopLineConfig()) is None


@pytest.mark.parametrize('filled', [False, True])
@pytest.mark.parametrize('center', [(320, 365), (480, 440)])
def test_diamond_and_clipped_diamond_are_not_stop_lines(filled, center):
    image = np.full((480, 640, 3), 60, np.uint8)
    cx, cy = center
    polygon = np.array([(cx, cy - 90), (cx + 180, cy), (cx, cy + 90), (cx - 180, cy)], np.int32)
    if filled:
        cv2.fillConvexPoly(image, polygon, (160,) * 3)
    else:
        cv2.polylines(image, [polygon], True, (160,) * 3, 35)
    assert detect_stop_line(image, StopLineConfig(white_lightness_min=145)) is None


def test_real_line_is_selected_even_with_a_nearer_false_patch():
    image = _image_with_band(310, 326, (155,) * 3)
    image[380:480, 368:640] = (160,) * 3
    detected = detect_stop_line(image, StopLineConfig(white_lightness_min=145))
    assert detected is not None and detected.row_ratio == pytest.approx(326 / 480)


@pytest.mark.parametrize('angle', [-50., 50., 90.])
def test_slanted_side_markings_are_not_promoted_by_fallback(angle):
    image = np.full((480, 640, 3), 60, np.uint8)
    box = cv2.boxPoints(((320., 375.), (500., 14.), angle)).astype(np.int32)
    cv2.fillConvexPoly(image, box, (160,) * 3)
    assert detect_stop_line(image, StopLineConfig(white_lightness_min=145)) is None


def test_bright_floor_patch_and_separated_short_strokes_are_rejected():
    image = _image_with_band(300, 480, (180,) * 3, left=0, right=640)
    assert detect_stop_line(image, StopLineConfig(white_lightness_min=145)) is None
    image[:] = 60
    image[380:396, 20:260] = (160,) * 3
    image[380:396, 380:620] = (160,) * 3
    assert detect_stop_line(image, StopLineConfig(white_lightness_min=145)) is None


def test_clipped_horizontal_tape_retains_existing_near_edge():
    detected = detect_stop_line(_image_with_band(465, 480), StopLineConfig())
    assert detected is not None and detected.row_ratio == 1.


def test_false_patch_cannot_seed_odom_tracking():
    config = StopLineConfig(white_lightness_min=145)
    tracker = _tracker()
    detection = detect_stop_line(_image_with_band(380, 480, (160,) * 3, left=368, right=640), config)
    assert detection is None
    assert tracker.update(None) is None
    tracker.update_odom(.1)
    assert tracker.update(None) is None and tracker.source == 'none'


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


# ---- 판정 기준 바퀴 (규정집 v3: 뒷바퀴) ------------------------------------- #

def test_rear_wheel_offset_added_only_when_calibrated():
    rear = RowDistanceModel([0.6, 0.8, 0.95], [0.50, 0.25, 0.12], reference_offset_m=0.20)
    assert rear.distance(0.8) == pytest.approx(0.45)
    assert rear.front_distance(0.8) == pytest.approx(0.25)
    legacy = RowDistanceModel([0.0], [0.0], 1.6, 0.58, reference_offset_m=0.20)
    assert legacy.distance(0.79) == pytest.approx(0.8)     # 보정 전은 예전 값 그대로


def test_rear_wheel_reaches_line_deep_in_blind_zone():
    """앞바퀴 기준 0.15 m 에서 선이 사라져도 뒷바퀴가 선에 닿을 때까지(0.35 m 더) 추적."""
    model = RowDistanceModel([0.6, 0.8, 0.95], [0.50, 0.25, 0.15], reference_offset_m=0.20)
    tracker = BlindZoneTracker(max_track_m=0.8)
    tracker.update_odom(0.0)
    assert tracker.update(model.distance(0.95)) == pytest.approx(0.35)
    tracker.update_odom(0.30)
    assert tracker.update(None) == pytest.approx(0.05)
    tracker.update_odom(0.35)
    assert tracker.update(None) == pytest.approx(0.0)
    assert tracker.source == 'odom'
