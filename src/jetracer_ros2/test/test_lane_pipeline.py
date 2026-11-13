import cv2
import numpy as np

from jetracer_ros2.lane_pipeline import LanePipeline, LanePipelineConfig


def _synthetic_lane(offset=0, missing_left=False, missing_right=False):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    y = np.arange(480, dtype=np.float32)
    curve = 0.00035 * (y - 240.0) ** 2
    left = 140.0 + offset + curve
    right = left + 360.0
    left_points = np.int32(np.column_stack((left, y))).reshape(-1, 1, 2)
    if not missing_left:
        cv2.polylines(image, [left_points], False, (255, 255, 255), 10)
    if not missing_right:
        right_points = np.int32(np.column_stack((right, y))).reshape(-1, 1, 2)
        cv2.polylines(image, [right_points], False, (255, 255, 255), 10)
    return image


def _synthetic_zigzag():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    y = np.arange(480, dtype=np.float32)
    normalized_y = (y - 240.0) / 240.0
    center = 320.0 + 180.0 * (normalized_y ** 3 - 0.7 * normalized_y)
    left = center - 180.0
    right = center + 180.0
    left_points = np.int32(np.column_stack((left, y))).reshape(-1, 1, 2)
    right_points = np.int32(np.column_stack((right, y))).reshape(-1, 1, 2)
    cv2.polylines(image, [left_points], False, (255, 255, 255), 10)
    cv2.polylines(image, [right_points], False, (255, 255, 255), 10)
    return image


def _identity_config():
    return LanePipelineConfig(
        src_points=(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
        dst_points=(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
        white_lightness_min=150,
        gradient_min=20,
        lane_width_px=360.0,
        fit_min_points=80,
    )


def _partial_right_yellow_curve(max_forward_m=1.05):
    config = _identity_config()
    config.detect_yellow = True
    config.metric_lookahead_enabled = True
    config.bev_ground_points_m = (-.32, .9, .32, .9, .32, .421, -.32, .421)
    config.camera_forward_offset_m = .184
    config.lane_width_px = 270.0
    config.curve_lane_width_m = .27
    config.curve_right_min_clearance_m = .07
    config.curve_nearest_max_forward_m = max_forward_m
    config.low_light_enabled = False
    image = np.zeros((480, 640, 3), np.uint8)
    ys = np.arange(190, 436)
    xs = 560 - .003 * (ys - 190) ** 2
    cv2.polylines(image, [np.int32(np.column_stack((xs, ys))).reshape(-1, 1, 2)],
                  False, (0, 255, 255), 12)
    return config, image


def _observed_left_clockwise_curve(include_right=False):
    config = _identity_config()
    config.detect_yellow = True
    config.metric_lookahead_enabled = True
    config.bev_ground_points_m = (-.32, .9, .32, .9, .32, .421, -.32, .421)
    config.camera_forward_offset_m = .184
    config.lane_width_px = 270.0
    config.curve_lane_width_m = .27
    config.curve_right_min_clearance_m = .07
    config.low_light_enabled = False
    image = np.zeros((480, 640, 3), np.uint8)
    ys = np.arange(48, 436)
    left = 350 - .001 * ys ** 2
    for offset in ((0, 270) if include_right else (0,)):
        cv2.polylines(image, [np.int32(np.column_stack((left + offset, ys))).reshape(-1, 1, 2)],
                      False, (0, 255, 255), 12)
    return config, image


def test_clockwise_curve_prefers_observed_outer_left_line_and_early_preview():
    for include_right in (False, True):
        config, image = _observed_left_clockwise_curve(include_right)
        result = LanePipeline(config).process(image)
        assert result.valid and result.recovered_side == 'left'
        assert result.curve_turn == 'right' and not result.zigzag
        assert result.curve_left_target_active
        # This synthetic arc is already unwinding at the far end. The local
        # straight/curve split must not invent extra entry-steering demand.
        assert result.curve_left_preview_rad == 0.0
        assert .6 < result.target_forward_m <= config.curve_left_near_max_forward_m


def test_straight_entry_preserves_near_tangent_and_previews_far_right_bend():
    config, image = _observed_left_clockwise_curve()
    image[:] = 0
    ys = np.arange(48, 436)
    left = 115 + .002 * np.maximum(250 - ys, 0) ** 2
    cv2.polylines(image, [np.int32(np.column_stack((left, ys))).reshape(-1, 1, 2)],
                  False, (0, 255, 255), 12)
    result = LanePipeline(config).process(image)
    assert result.valid, result.reason
    assert result.curve_left_target_active
    assert result.curve_left_preview_rad > .03
    # Local near-row center must remain beside the observed straight, rather
    # than inherit the global cubic's bend before the corner actually starts.
    assert abs(result.target_lateral_m - (-.32 + 250 * .001)) < .07


def test_visible_right_paint_only_stops_when_measured_clearance_is_small():
    config, image = _observed_left_clockwise_curve()
    ys = np.arange(48, 436)
    left = 350 - .001 * ys ** 2
    for gap, expected_valid in ((270, True), (170, False)):
        painted = image.copy()
        cv2.polylines(painted, [np.int32(np.column_stack((left + gap, ys))).reshape(-1, 1, 2)],
                      False, (0, 255, 255), 12)
        result = LanePipeline(config).process(painted)
        assert result.valid == expected_valid, (gap, result.reason)
        if not expected_valid:
            assert result.reason == 'right curve boundary clearance too small'


def test_clockwise_curve_requires_near_and_far_observed_windows():
    config, image = _observed_left_clockwise_curve()
    # Clear only distant paint. Normal curve recovery may still use the
    # remaining line, but must not claim a measured far-window preview.
    image[:170] = 0
    result = LanePipeline(config).process(image)
    assert not result.curve_left_target_active
    assert result.curve_left_preview_rad is None


def test_zigzag_does_not_activate_curve_preview():
    config = _identity_config()
    config.detect_yellow = True
    config.metric_lookahead_enabled = True
    config.bev_ground_points_m = (-.32, .9, .32, .9, .32, .421, -.32, .421)
    config.camera_forward_offset_m = .184
    image = _synthetic_zigzag()
    image[np.all(image == 255, axis=2)] = (0, 220, 255)
    result = LanePipeline(config).process(image)
    assert result.zigzag and not result.curve_left_target_active


def test_inner_right_curve_is_not_used_when_outer_left_is_missing():
    for max_forward_m in (1.05, .70):
        config, image = _partial_right_yellow_curve(max_forward_m=max_forward_m)
        result = LanePipeline(config).process(image)
        assert not result.valid and result.reason == 'outer left curve not observed'
        # Keep diagnostic quality separate from authorization to drive.
        assert result.confidence >= .7 and result.curve_turn == 'right'
        assert result.left_type == 'unobserved'
        assert result.right_type == 'yellow/solid/curve'
        assert not result.curve_nearest_observation


def test_inner_only_curve_stops_even_without_yellow_priority_filter():
    config, image = _partial_right_yellow_curve()
    config.yellow_curve_priority = False
    result = LanePipeline(config).process(image)
    assert not result.valid and result.reason == 'outer left curve not observed'
    assert result.right_type == 'yellow/solid/curve'


def test_detects_two_curved_lanes():
    result = LanePipeline(_identity_config()).process(_synthetic_lane())
    assert result.valid
    assert result.confidence > 0.5
    assert abs(result.lateral_error) < 0.25


def test_recovers_one_missing_lane():
    pipeline = LanePipeline(_identity_config())
    assert pipeline.process(_synthetic_lane()).valid
    result = pipeline.process(_synthetic_lane(missing_right=True))
    assert result.valid
    assert result.reason == 'one-side recovery'
    # 노드 기본 임계값(0.48)을 넘겨 한쪽 선만으로도 계속 주행할 수 있어야 한다.
    assert result.confidence >= 0.48


def test_single_side_margin_never_crosses_measured_curve():
    """The inferred boundary and center keep the measured curve's shape."""
    pipeline = LanePipeline(_identity_config())
    observed = np.array([0.00008, -0.045, 5.0, 90.0])
    ys = np.linspace(0, 479, 100)
    for side in ('left', 'right'):
        left, right = pipeline._single_side(observed, side)
        gap = np.polyval(right, ys) - np.polyval(left, ys)
        assert np.all(gap > 0)
        assert np.ptp(gap) < 1e-6
        assert abs(float(gap[0]) - pipeline.config.lane_width_px) < 1e-6
        measured = left if side == 'left' else right
        assert np.allclose(np.polyval(measured, ys), np.polyval(observed, ys))


def test_strong_single_lane_can_recover_after_large_curve_shift():
    """곡선에서 중심이 크게 변해도 한쪽 선 자체가 강하면 정지하지 않아야 한다."""
    pipeline = LanePipeline(_identity_config())
    assert pipeline.process(_synthetic_lane()).valid

    result = pipeline.process(_synthetic_lane(offset=-110, missing_right=True))

    assert result.valid
    assert result.recovered_side == 'left'
    assert result.confidence >= 0.48


def test_rejects_empty_scene():
    result = LanePipeline(_identity_config()).process(
        np.zeros((480, 640, 3), dtype=np.uint8),
    )
    assert not result.valid
    assert result.confidence == 0.0


def test_detects_zigzag_for_slow_driving():
    result = LanePipeline(_identity_config()).process(_synthetic_zigzag())
    assert result.valid
    assert result.zigzag
    assert result.direction_changes >= 1
    assert result.zigzag_score > 0.5


def test_recovers_after_lane_shift_while_lost():
    """차선을 놓친 사이 위치가 변해도 강한 차선을 다시 찾아야 한다."""
    pipeline = LanePipeline(_identity_config())
    assert pipeline.process(_synthetic_lane()).valid
    for _ in range(5):
        pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    shifted = _synthetic_lane(offset=-110)
    assert pipeline.process(shifted).valid


def test_detects_yellow_lanes_when_enabled():
    """노란 차선은 흰색 검출만으로는 못 찾고, detect_yellow를 켜면 찾아야 한다."""
    image = _synthetic_lane()
    yellow = image.copy()
    yellow[np.all(image == 255, axis=2)] = (0, 220, 255)   # BGR 노란색

    white_only = _identity_config()
    white_only.gradient_min = 255     # 경계선 검출 영향 제거
    assert not LanePipeline(white_only).process(yellow).valid

    yellow_config = _identity_config()
    yellow_config.gradient_min = 255
    yellow_config.detect_white = False
    yellow_config.detect_yellow = True
    result = LanePipeline(yellow_config).process(yellow)
    assert result.valid
    assert abs(result.lateral_error) < 0.25


def test_gradient_does_not_create_lane_without_enabled_colour():
    """White road markings must not leak through gradient in yellow-only mode."""
    config = _identity_config()
    config.detect_white = False
    config.detect_yellow = True
    config.gradient_min = 20
    config.gradient_require_color_support = True

    result = LanePipeline(config).process(_synthetic_lane())

    assert not result.valid
    assert result.color_binary_image is not None
    assert cv2.countNonZero(result.color_binary_image) == 0
    assert cv2.countNonZero(result.gradient_binary_image) == 0


def test_left_mode_uses_left_anchor_only():
    pipeline = LanePipeline(_identity_config())
    image = _synthetic_lane(missing_right=True)
    cv2.line(image, (620, 0), (620, 479), (255, 255, 255), 12)

    result = pipeline.process(image, tracking_mode='left')

    assert result.valid
    assert result.recovered_side == 'left'
    assert result.tracking_mode == 'left'


def test_right_mode_uses_right_anchor_only():
    pipeline = LanePipeline(_identity_config())
    image = _synthetic_lane(missing_left=True)
    cv2.line(image, (20, 0), (20, 479), (255, 255, 255), 12)

    result = pipeline.process(image, tracking_mode='right')

    assert result.valid
    assert result.recovered_side == 'right'
    assert result.tracking_mode == 'right'


def test_upper_marking_without_bottom_connection_is_rejected():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(image, (150, 40), (150, 210), (255, 255, 255), 12)
    cv2.line(image, (510, 40), (510, 210), (255, 255, 255), 12)

    result = LanePipeline(_identity_config()).process(image)

    assert not result.valid


def test_mixed_yellow_solid_and_white_dashed_attributes():
    config = _identity_config()
    config.detect_yellow = True
    config.detect_white = True
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(image, (140, 0), (140, 479), (0, 220, 255), 8)
    for start in range(0, 480, 80):
        cv2.line(image, (500, start), (500, start + 39), (255, 255, 255), 8)
    result = LanePipeline(config).process(image)
    assert result.valid
    assert result.left_type == 'yellow/solid/straight'
    assert result.right_type == 'white/dashed/straight'


def test_missing_boundary_is_not_classified_as_paint():
    result = LanePipeline(_identity_config()).process(_synthetic_lane(missing_right=True))
    assert result.valid
    assert result.right_type == 'unobserved'


def test_reassigned_single_boundary_keeps_measured_pixels(monkeypatch):
    # During a turn, semantic left/right need not match the search half.
    # Reassigning a right-half observation to left used to delete all pixels.
    for target in ('left', 'right'):
        pipeline = LanePipeline(_identity_config())
        monkeypatch.setattr(pipeline, '_assign_side', lambda fit, fallback: target)
        image = _synthetic_lane(offset=-100 if target == 'left' else 100,
                                missing_left=target == 'left',
                                missing_right=target == 'right')
        result = pipeline.process(image)
        assert result.valid and result.confidence >= .48
        assert result.recovered_side == target
        observed_type = result.left_type if target == 'left' else result.right_type
        assert observed_type.startswith('white/')
        assert np.isfinite(result.preview_heading_error)
