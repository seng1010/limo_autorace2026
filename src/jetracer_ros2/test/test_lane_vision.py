import cv2
import numpy as np
from types import SimpleNamespace

from jetracer_ros2.lane_pipeline import LanePipeline, LanePipelineConfig
from jetracer_ros2.lane_vision import GroundLookahead
from jetracer_ros2.lane_steering import tracking_error


def config():
    return LanePipelineConfig(
        src_points=(0, 0, 1, 0, 1, 1, 0, 1),
        dst_points=(0, 0, 1, 0, 1, 1, 0, 1),
        detect_white=True, detect_yellow=True)


def test_dim_yellow_and_white_lanes_retain_color():
    for color, label in (((0, 55, 64), 'yellow'), ((64, 64, 64), 'white')):
        image = np.zeros((480, 640, 3), np.uint8)
        for x in (140, 500):
            cv2.line(image, (x, 0), (x, 479), color, 8)
        result = LanePipeline(config()).process(image)
        assert result.valid and result.confidence >= 0.48
        assert result.left_type.startswith(label)
        assert 1 < result.illumination_gain <= 3.0
        disabled = config()
        disabled.low_light_enabled = False
        assert not LanePipeline(disabled).process(image).valid


def test_dark_uniform_floor_and_black_noise_do_not_become_lanes():
    for image in (np.full((480, 640, 3), 80, np.uint8),
                  np.random.default_rng(1).integers(0, 8, (480, 640, 3), dtype=np.uint8)):
        assert not LanePipeline(config()).process(image).valid


def test_roi_reaches_pixel_edges():
    pipeline = LanePipeline(LanePipelineConfig())
    points = pipeline._normalized_points(pipeline.config.src_points)
    assert points[2, 0] == 639
    assert points[3, 0] == 0
    projected = cv2.perspectiveTransform(points[None], pipeline._matrix)[0]
    assert np.allclose(projected, pipeline._normalized_points(pipeline.config.dst_points), atol=1e-3)


def ground(distance=0.4, offset=0.1):
    return GroundLookahead(
        [[0, 0], [639, 0], [639, 479], [0, 479]],
        [-0.32, 0.8, 0.32, 0.8, 0.32, 0.1, -0.32, 0.1], offset, distance, 0.30)


def test_metric_lookahead_30_to_50_cm_and_camera_offset():
    rows = []
    for distance in (0.3, 0.4, 0.5):
        lateral, heading, y = ground(distance).evaluate(np.array([0, 319.5]), 0, 479)
        assert abs(lateral) < 1e-5 and abs(heading) < 1e-5
        rows.append(y)
    assert rows[0] > rows[1] > rows[2]
    # Moving the camera forward means the same rear-axle target is nearer to it.
    assert ground(offset=0.2).evaluate(np.array([0, 319.5]), 0, 479)[2] > rows[1]
    x, heading, y = ground(distance=.4, offset=.184).evaluate_target(
        np.array([0, 319.5]), 0, 479)
    assert abs(x) < 1e-5 and abs(heading) < 1e-5 and 0 <= y <= 479


def test_metric_pipeline_exposes_rear_axle_target_only_when_calibrated():
    image = np.zeros((480, 640, 3), np.uint8)
    for x in (140, 500):
        cv2.line(image, (x, 0), (x, 479), (0, 255, 255), 8)
    raw = LanePipeline(config()).process(image)
    assert raw.valid and raw.target_lateral_m is None
    calibrated = config()
    calibrated.metric_lookahead_enabled = True
    calibrated.camera_forward_offset_m = .184
    calibrated.bev_ground_points_m = (-.32, .8, .32, .8, .32, .1, -.32, .1)
    result = LanePipeline(calibrated).process(image)
    assert result.valid and result.target_forward_m == .4
    assert abs(result.target_lateral_m) < .01


def test_metric_requires_calibration_and_visible_target():
    try:
        GroundLookahead([[0, 0], [639, 0], [639, 479], [0, 479]], [0.0]*8, 0, .4, .3)
    except ValueError:
        pass
    else:
        raise AssertionError('unmeasured calibration accepted')
    try:
        ground(distance=1.2).evaluate(np.array([0, 319.5]), 0, 479)
    except ValueError:
        pass
    else:
        raise AssertionError('unseen target accepted')


def test_preview_boost_only_for_same_direction_tightening():
    result = SimpleNamespace(lateral_error=.1, heading_error=.2,
                             preview_heading_error=.6, recovered_side='left')
    base = tracking_error(result, .95, .7, 1.35, 1., 0., 1.0)
    assert tracking_error(result, .95, .7, 1.35, 1., .25, 1.0) > base
    result.preview_heading_error = -.6
    assert tracking_error(result, .95, .7, 1.35, 1., .25, 1.0) == base


def test_video_estimate_rectifies_observed_parallel_boundaries():
    # Independent straight-line fits from raw screen video at 20s and 35s;
    # 55s checks another straight segment with small camera/track variation.
    boundaries = [
        ((-1.038518, 463.663611), (1.793026, 28.720665)),
        ((-1.560761, 537.902043), (1.219076, 105.213298)),
        ((-1.348182, 550.903898), (1.365397, 149.842283)),
    ]
    pipeline = LanePipeline(LanePipelineConfig())
    ys = np.float32([230, 265, 300])
    for left, right in boundaries:
        mapped = []
        for fit in (left, right):
            points = np.column_stack((np.polyval(fit, ys), ys)).astype(np.float32)
            mapped.append(cv2.perspectiveTransform(points[None], pipeline._matrix)[0])
        widths = mapped[1][:, 0] - mapped[0][:, 0]
        assert np.ptp(widths) / np.mean(widths) < 0.05
        assert abs(np.mean(widths) - pipeline.config.lane_width_px) < 30


def test_new_bev_preserves_camera_rows_of_control_targets():
    old = LanePipeline(LanePipelineConfig(
        src_points=(.18, .48, .82, .48, 1., .98, 0., .98),
        lookahead_y_ratio=.62, preview_y_ratio=.40))
    new = LanePipeline(LanePipelineConfig())
    for old_ratio, new_ratio in ((.62, new.config.lookahead_y_ratio),
                                  (.40, new.config.preview_y_ratio)):
        old_point = cv2.perspectiveTransform(np.float32([[[320, 480 * old_ratio]]]), old._inverse_matrix)
        new_point = cv2.perspectiveTransform(np.float32([[[320, 480 * new_ratio]]]), new._inverse_matrix)
        assert np.linalg.norm(old_point - new_point) < 2.0
