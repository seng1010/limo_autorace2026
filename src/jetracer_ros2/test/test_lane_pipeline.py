import cv2
import numpy as np

from jetracer_ros2.lane_pipeline import LanePipeline, LanePipelineConfig


def _synthetic_lane(offset=0, missing_right=False):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    y = np.arange(480, dtype=np.float32)
    curve = 0.00035 * (y - 240.0) ** 2
    left = 140.0 + offset + curve
    right = left + 360.0
    left_points = np.int32(np.column_stack((left, y))).reshape(-1, 1, 2)
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
