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


def test_reset_recovers_after_lane_shift_while_lost():
    """차선을 놓친 사이 차가 밀리면 급변 검사에 계속 걸린다. reset()으로 복구돼야 한다."""
    pipeline = LanePipeline(_identity_config())
    assert pipeline.process(_synthetic_lane()).valid
    for _ in range(5):
        pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    shifted = _synthetic_lane(offset=-110)
    assert not pipeline.process(shifted).valid
    pipeline.reset()
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
