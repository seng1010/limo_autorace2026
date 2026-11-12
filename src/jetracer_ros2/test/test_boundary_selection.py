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


def test_offscreen_reconstructed_center_is_rejected(monkeypatch):
    cfg = LanePipelineConfig(src_points=(0,0,1,0,1,1,0,1),
                             dst_points=(0,0,1,0,1,1,0,1))
    pipeline = LanePipeline(cfg)
    monkeypatch.setattr(pipeline, '_assign_side', lambda fit, fallback: 'left')
    image = np.zeros((480,640,3), np.uint8)
    cv2.line(image, (550,0), (550,479), (255,255,255), 10)
    result = pipeline.process(image)
    assert not result.valid
    assert result.reason == 'center target outside observable image'
