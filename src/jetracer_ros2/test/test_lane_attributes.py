import cv2
import numpy as np

from jetracer_ros2.lane_attributes import classify_lane, scale_steering, measured_heading_span
from jetracer_ros2.lane_pipeline import LanePipelineConfig


def scene(color, dashed=False, curved=False):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    fit = np.array([0.001 if curved else 0.0, -0.48 if curved else 0.0, 200.0])
    ys = np.arange(30, 450)
    if dashed:
        ys = ys[(ys // 40) % 2 == 0]
    for y in ys:
        x = int(np.polyval(fit, y))
        cv2.line(image, (x - 4, int(y)), (x + 4, int(y)), color, 1)
    return image, fit, ys


def test_yellow_solid_and_curve():
    for curved, expected in ((False, 'straight'), (True, 'curve')):
        image, fit, ys = scene((0, 220, 255), curved=curved)
        assert classify_lane(image, fit, ys, LanePipelineConfig()) == f'yellow/solid/{expected}'


def test_white_dashed():
    image, fit, ys = scene((255, 255, 255), dashed=True)
    assert classify_lane(image, fit, ys, LanePipelineConfig()) == 'white/dashed/straight'


def test_zigzag_and_reconstructed_boundary():
    image, fit, ys = scene((0, 220, 255), curved=True)
    assert classify_lane(image, fit, ys, LanePipelineConfig(), True) == 'yellow/solid/zigzag'
    assert classify_lane(image, fit, [], LanePipelineConfig()) == 'unobserved'


def test_missing_color_is_unknown():
    image, fit, ys = scene((0, 0, 0))
    assert classify_lane(image, fit, ys, LanePipelineConfig()) == 'unknown/unknown/straight'


def test_steering_gain_preserves_sign_and_limit():
    assert abs(scale_steering(0.5, 1.2) - 0.6) < 1e-9
    assert abs(scale_steering(-0.5, 1.2) + 0.6) < 1e-9
    assert scale_steering(0.9, 1.2) == 1.0
    assert scale_steering(-0.9, 1.2) == -1.0
    assert scale_steering(0.0, 1.2) == 0.0
    assert scale_steering(0.5, float('nan')) == 0.5


def test_shifted_or_tilted_straight_line_is_not_curve_evidence():
    ys = np.arange(30, 450)
    for intercept in (50, 300, 550):
        for slope in (-1.0, 0.0, 1.0):
            assert measured_heading_span(np.array([slope, intercept]), ys, 480) == 0.0
    assert measured_heading_span(np.array([.001, -.48, 200]), ys, 480) > .40
    assert measured_heading_span(np.array([.001, -.48, 200]), [], 480) == 0.0
