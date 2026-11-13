import cv2
import numpy as np
from jetracer_ros2.lane_pipeline import LanePipeline, LanePipelineConfig


def mask(lines):
    image = np.zeros((480, 640, 3), np.uint8)
    for x, color in lines:
        cv2.line(image, (x, 0), (x, 479), color, 8)
    return LanePipeline(LanePipelineConfig(detect_yellow=True))._binary_mask(image)


def test_outer_white_trim_is_rejected_but_inner_white_is_preserved():
    result = mask([(40, (255,)*3), (130, (0,255,255)),
                   (260, (255,)*3), (390, (255,)*3),
                   (510, (0,255,255)), (600, (255,)*3)])
    assert not result[:,40].any() and not result[:,600].any()
    assert all(result[:,x].any() for x in (130,260,390,510))


def test_parking_yellow_left_white_right_remains_usable():
    result = mask([(130, (0,255,255)), (510, (255,)*3)])
    assert result[:,130].any() and result[:,510].any()


def test_white_only_multilane_markings_are_not_globally_disabled():
    result = mask([(130, (255,)*3), (510, (255,)*3)])
    assert result[:,130].any() and result[:,510].any()


def test_connected_yellow_curve_overrides_outer_white_trim():
    cfg = LanePipelineConfig(
        src_points=(0,0,1,0,1,1,0,1),
        dst_points=(0,0,1,0,1,1,0,1),
        detect_yellow=True, lane_width_px=360.0, fit_min_points=80)
    pipeline = LanePipeline(cfg)
    image = np.zeros((480,640,3), np.uint8)
    # Only a short but connected yellow curve is visible near the vehicle.
    # Its left/right-half support is too short for the old white-trim gate.
    ys = np.arange(328, 480)
    xs = 290 + .005 * (479 - ys) ** 2
    cv2.polylines(image, [np.int32(np.column_stack((xs,ys))).reshape(-1,1,2)],
                  False, (0,255,255), 10)
    cv2.line(image, (90,0), (90,479), (255,255,255), 11)
    bird = image.copy()
    preferred = pipeline._yellow_curve_mask(bird, 'auto')
    assert preferred is not None
    assert preferred[:,90].sum() == 0
    result = pipeline.process(image)
    assert result.valid and result.yellow_curve_priority_active
    assert result.left_type == 'yellow/solid/curve'
    assert not result.binary_image[:,90].any()
    # If paint disappears briefly, do not jump straight onto the white trim.
    occluded = np.zeros_like(image)
    cv2.line(occluded, (90,0), (90,479), (255,255,255), 11)
    lost = pipeline.process(occluded)
    assert not lost.valid
    assert lost.yellow_curve_priority_active
    assert not lost.binary_image[:,90].any()


def test_yellow_priority_does_not_remove_white_straight_or_parking_lane():
    cfg = LanePipelineConfig(
        src_points=(0,0,1,0,1,1,0,1),
        dst_points=(0,0,1,0,1,1,0,1), detect_yellow=True)
    pipeline = LanePipeline(cfg)
    image = np.zeros((480,640,3), np.uint8)
    cv2.line(image, (140,0), (140,479), (0,255,255), 9)
    cv2.line(image, (500,0), (500,479), (255,255,255), 9)
    result = pipeline.process(image)
    assert result.valid and not result.yellow_curve_priority_active
    assert result.right_type.startswith('white')


def test_outer_curve_keeps_right_window_off_until_straight_is_confirmed():
    cfg = LanePipelineConfig(
        src_points=(0,0,1,0,1,1,0,1),
        dst_points=(0,0,1,0,1,1,0,1),
        detect_yellow=True, lane_width_px=360.0, fit_min_points=80)
    pipeline = LanePipeline(cfg)
    curve = np.zeros((480, 640, 3), np.uint8)
    ys = np.arange(328, 480)
    xs = 290 + .005 * (479 - ys) ** 2
    cv2.polylines(curve, [np.int32(np.column_stack((xs, ys))).reshape(-1, 1, 2)],
                  False, (0, 255, 255), 10)
    assert pipeline._yellow_curve_mask(curve, 'auto') is not None
    assert pipeline._yellow_curve_left_latched

    # Repeated missing frames may clear old fits, but must not re-enable the
    # opposing window or the white mat edge while the curve state is locked.
    missing = np.zeros_like(curve)
    cv2.line(missing, (90, 0), (90, 479), (255, 255, 255), 11)
    for _ in range(6):
        pipeline.reset(preserve_curve_lock=True)
        result = pipeline.process(missing)
        assert not result.valid and result.yellow_curve_priority_active
        assert pipeline._yellow_curve_side == 'left'
        assert not result.binary_image[:, 90].any()

    straight = np.zeros_like(curve)
    cv2.line(straight, (290, 0), (290, 479), (0, 255, 255), 10)
    assert pipeline._yellow_curve_mask(straight, 'auto') is not None
    assert pipeline._yellow_curve_left_latched
    assert pipeline._yellow_curve_mask(straight, 'auto') is None
    assert not pipeline._yellow_curve_left_latched


def test_outer_curve_releases_right_window_on_confirmed_yellow_zigzag():
    cfg = LanePipelineConfig(
        src_points=(0,0,1,0,1,1,0,1),
        dst_points=(0,0,1,0,1,1,0,1),
        detect_yellow=True, lane_width_px=360.0, fit_min_points=80)
    pipeline = LanePipeline(cfg)
    curve = np.zeros((480, 640, 3), np.uint8)
    ys = np.arange(328, 480)
    xs = 290 + .005 * (479 - ys) ** 2
    cv2.polylines(curve, [np.int32(np.column_stack((xs, ys))).reshape(-1, 1, 2)],
                  False, (0, 255, 255), 10)
    assert pipeline._yellow_curve_mask(curve, 'auto') is not None
    assert pipeline._yellow_curve_left_latched

    zigzag = np.zeros_like(curve)
    points = np.array([(290, 479), (370, 350), (260, 220), (340, 0)], np.int32)
    cv2.polylines(zigzag, [points], False, (0, 255, 255), 10)
    assert pipeline._yellow_curve_mask(zigzag, 'auto') is not None
    assert pipeline._yellow_curve_mask(zigzag, 'auto') is None
    assert not pipeline._yellow_curve_left_latched


def test_observed_mild_outer_bend_does_not_count_as_missing_during_latch():
    cfg = LanePipelineConfig(
        src_points=(0, 0, 1, 0, 1, 1, 0, 1),
        dst_points=(0, 0, 1, 0, 1, 1, 0, 1),
        detect_yellow=True, lane_width_px=360.0, fit_min_points=80,
        observed_outer_curve_entry_enabled=True)
    pipeline = LanePipeline(cfg)
    strong = np.zeros((480, 640, 3), np.uint8)
    ys = np.arange(328, 480)
    xs = 290 + .005 * (479 - ys) ** 2
    cv2.polylines(strong, [np.int32(np.column_stack((xs, ys))).reshape(-1, 1, 2)],
                  False, (0, 255, 255), 10)
    assert pipeline._yellow_curve_mask(strong, 'auto') is not None
    assert pipeline._yellow_curve_left_latched

    mild = np.zeros_like(strong)
    ys = np.arange(120, 480)
    xs = 290 + .0002 * (479 - ys) ** 2
    cv2.polylines(mild, [np.int32(np.column_stack((xs, ys))).reshape(-1, 1, 2)],
                  False, (0, 255, 255), 10)
    assert pipeline._yellow_curve_mask(mild, 'auto') is not None
    assert pipeline._yellow_curve_left_observed
    assert not pipeline._yellow_curve_outer_missing


def test_offscreen_reconstructed_center_is_rejected(monkeypatch):
    cfg = LanePipelineConfig(src_points=(0,0,1,0,1,1,0,1),
                             dst_points=(0,0,1,0,1,1,0,1))
    pipeline = LanePipeline(cfg)
    monkeypatch.setattr(pipeline, '_assign_side', lambda fit, fallback: 'left')
    image = np.zeros((480,640,3), np.uint8)
    cv2.line(image, (550,0), (550,479), (255,255,255), 10)
    result = pipeline.process(image)
    assert not result.valid
    assert result.reason == 'center target outside measured BEV road area'
    assert (result.observation_state, result.path_state, result.policy_state) == (
        'accepted', 'rejected', 'not_evaluated')
