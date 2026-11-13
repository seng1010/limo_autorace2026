"""Read-only reproductions of independent near-field and curve-entry gates.

The synthetic ground points are the current *estimated* calibration, not a
substitute for camera/odometry measurements on the vehicle.
"""

import cv2
import numpy as np
import pytest
from dataclasses import replace

from jetracer_ros2.lane_pipeline import LanePipeline, LanePipelineConfig


def _metric_config():
    return LanePipelineConfig(
        detect_yellow=True,
        detect_white=True,
        metric_lookahead_enabled=True,
        camera_forward_offset_m=0.184,
        bev_ground_points_m=(-0.24, 1.326, 0.24, 1.326,
                             0.22, 0.421, -0.22, 0.421),
    )


def _paint_straight(offset_m=0.0, sides=(-1, 1), near_m=0.45):
    pipeline = LanePipeline(_metric_config())
    image = np.full((480, 640, 3), (40, 38, 36), np.uint8)
    for side in sides:
        previous = None
        for forward in np.linspace(near_m, 2.2, 200):
            lateral = side * 0.26 - offset_m
            bx, by = pipeline.ground.project_ground_point(lateral, forward)
            ix, iy = cv2.perspectiveTransform(
                np.float32([[[bx, by]]]), pipeline._inverse_matrix)[0, 0]
            point = (int(ix), int(iy))
            if previous is not None:
                cv2.line(image, previous, point, (0, 210, 230), 3)
            previous = point
    return pipeline, image


def test_centered_two_sided_straight_uses_nearest_observed_target():
    pipeline, image = _paint_straight()
    result = pipeline.process(image)
    # Both boundaries reach fitting; the target must be observed, not a
    # polynomial extrapolation into the camera's blind near-field.
    assert result.observation_state == 'accepted'
    assert result.left_type != 'unobserved'
    assert result.right_type != 'unobserved'
    assert result.valid
    assert result.target_forward_m == pytest.approx(.679, abs=.02)


@pytest.mark.parametrize('offset_m', (-0.10, -0.06, -0.02, 0.02, 0.06, 0.10))
def test_straight_offsets_keep_observed_boundaries(offset_m):
    pipeline, image = _paint_straight(offset_m)
    result = pipeline.process(image)
    # Each offset is a distinct camera/ROI visibility condition; a failure to
    # drive is recorded separately from whether paint was actually detected.
    assert result.observation_state == 'accepted'


def test_one_sided_short_observation_uses_metric_range_instead_of_fixed_row():
    pipeline, image = _paint_straight(sides=(-1,), near_m=0.68)
    result = pipeline.process(image)
    assert result.valid, result.reason
    assert result.recovered_side == 'left'
    assert result.target_forward_m == pytest.approx(.695, abs=.02)


def test_one_sided_line_still_stops_when_metric_target_is_too_far():
    pipeline, image = _paint_straight(sides=(-1,), near_m=0.78)
    result = pipeline.process(image)
    assert not result.valid
    assert result.reason == 'lookahead too close to or outside observed ground range'


def test_line_starting_above_bottom_windows_is_not_connected():
    pipeline = LanePipeline(LanePipelineConfig(
        src_points=(0, 0, 1, 0, 1, 1, 0, 1),
        dst_points=(0, 0, 1, 0, 1, 1, 0, 1),
        fit_min_points=80,
    ))
    binary = np.zeros((480, 640), np.uint8)
    cv2.line(binary, (140, 300), (140, 120), 255, 8)
    left_y, left_x, right_y, right_x, _ = pipeline._sliding_window_points(binary)
    assert left_x.size == right_x.size == 0
    assert left_y.size == right_y.size == 0


def test_right_curve_entry_uses_independently_observed_straight_left():
    config = LanePipelineConfig(
        src_points=(0, 0, 1, 0, 1, 1, 0, 1),
        dst_points=(0, 0, 1, 0, 1, 1, 0, 1),
        detect_yellow=True, metric_lookahead_enabled=True,
        bev_ground_points_m=(-.32, .9, .32, .9, .32, .421, -.32, .421),
        camera_forward_offset_m=.184, lane_width_px=270.0,
        curve_lane_width_m=.27, curve_right_min_clearance_m=.07,
        low_light_enabled=False, observed_outer_curve_entry_enabled=True,
    )
    image = np.zeros((480, 640, 3), np.uint8)
    ys = np.arange(190, 436)
    right_x = 560 - .003 * (ys - 190) ** 2
    cv2.polylines(image, [np.int32(np.column_stack((right_x, ys))).reshape(-1, 1, 2)],
                  False, (0, 255, 255), 12)
    cv2.line(image, (100, 90), (100, 435), (0, 255, 255), 12)
    legacy = LanePipeline(replace(config, observed_outer_curve_entry_enabled=False)).process(image)
    assert not legacy.valid and legacy.reason == 'outer left curve not observed'
    pipeline = LanePipeline(config)
    result = pipeline.process(image)
    # Pre-mask search preserves the left observation even though the right
    # boundary alone satisfies the curve classifier.
    left_y, left_x, _, _, _ = pipeline._sliding_window_points(
        pipeline._yellow_curve_binary, tracking_mode='left')
    assert left_x.size >= config.fit_min_points
    assert pipeline._coverage_ok(left_y)
    assert pipeline._yellow_curve_left_observed
    assert result.valid, result.reason
    assert result.recovered_side == 'left'
    left_only = np.zeros_like(image)
    cv2.line(left_only, (100, 90), (100, 435), (0, 255, 255), 12)
    continued = pipeline.process(left_only)
    assert continued.valid, continued.reason
    assert continued.recovered_side == 'left'


def test_pre_mask_hint_alone_cannot_authorize_inner_only_curve():
    config = LanePipelineConfig(
        src_points=(0, 0, 1, 0, 1, 1, 0, 1),
        dst_points=(0, 0, 1, 0, 1, 1, 0, 1),
        detect_yellow=True, yellow_curve_priority=False,
        metric_lookahead_enabled=True,
        bev_ground_points_m=(-.32, .9, .32, .9, .32, .421, -.32, .421),
        camera_forward_offset_m=.184, lane_width_px=270.0,
        curve_lane_width_m=.27, curve_right_min_clearance_m=.07,
        low_light_enabled=False, observed_outer_curve_entry_enabled=True,
    )
    image = np.zeros((480, 640, 3), np.uint8)
    ys = np.arange(190, 436)
    xs = 560 - .003 * (ys - 190) ** 2
    cv2.polylines(image, [np.int32(np.column_stack((xs, ys))).reshape(-1, 1, 2)],
                  False, (0, 255, 255), 12)
    pipeline = LanePipeline(config)
    pipeline._yellow_curve_left_observed = True  # stale pre-mask history
    result = pipeline.process(image)
    assert not result.valid
    assert result.reason == 'outer left curve not observed'
