import cv2
import numpy as np

from jetracer_ros2.m8_observers import detect_m8_exit_stop_line, detect_m8_stop_line


def test_partial_oblique_exit_tape_does_not_change_strict_m8_entry_detection():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    box = cv2.boxPoints(((418., 330.), (185., 12.), 17.)).astype(np.int32)
    cv2.fillConvexPoly(image, box, (180, 180, 180))
    detected = detect_m8_exit_stop_line(image)
    assert detected is not None
    assert .60 < detected.row_ratio < .73
    assert detect_m8_stop_line(image, .58) is None


def test_exit_tape_rejects_yellow_and_steep_side_marking():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    steep = cv2.boxPoints(((275., 330.), (165., 13.), 42.)).astype(np.int32)
    cv2.fillConvexPoly(image, steep, (190, 190, 190))
    yellow = cv2.boxPoints(((480., 330.), (210., 13.), 8.)).astype(np.int32)
    cv2.fillConvexPoly(image, yellow, (0, 215, 240))
    assert detect_m8_exit_stop_line(image) is None


import pytest
from jetracer_ros2.stop_line_detector import detect_stop_line, StopLineConfig
WHITE = (255, 255, 255)

def _image_with_band(top, bottom, color=WHITE, left=60, right=580):
    image = np.full((480, 640, 3), 60, dtype=np.uint8)
    image[top:bottom, left:right] = color
    return image


# ---- 검출 ---------------------------------------------------------------- #

def test_m8_detects_shaded_wavy_tape_without_changing_generic_detector():
    image = np.full((480, 640, 3), 60, dtype=np.uint8)
    for x in range(105, 535):
        y = 381 + round(7*np.sin(x/55))
        image[y:y+10, x] = (167, 157, 150)
    assert detect_stop_line(image, StopLineConfig()) is None
    detection = detect_m8_stop_line(image, .58)
    assert detection is not None
    assert .80 <= detection.row_ratio <= .84


@pytest.mark.parametrize('box', [(360, 370, 15, 290), (370, 470, 60, 580), (465, 480, 60, 580)])
def test_m8_rejects_side_marking_large_patch_and_clipped_line(box):
    top, bottom, left, right = box
    assert detect_m8_stop_line(_image_with_band(top, bottom, left=left, right=right), .58) is None


@pytest.mark.parametrize('angle', [-17., 17.])
def test_oblique_m8_entry_uses_lower_edge_at_camera_center(angle):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    box = cv2.boxPoints(((370., 374.), (360., 13.), angle)).astype(np.int32)
    cv2.fillConvexPoly(image, box, (180, 180, 180))
    result = detect_m8_stop_line(image, .58)
    assert result is not None
    expected = 374.-50.*np.tan(np.radians(angle))+6.5/np.cos(np.radians(angle))
    assert result.row_ratio*480 == pytest.approx(expected, abs=3.)
    assert detect_m8_stop_line(image, .58, allow_oblique=False) is None


@pytest.mark.parametrize('center,shape,angle,color', [
    ((165., 375.), (260., 16.), -15., WHITE),
    ((445., 375.), (360., 16.), 15., WHITE),
    ((320., 375.), (300., 16.), -33., WHITE),
    ((320., 375.), (330., 49.), -10., WHITE),
    ((320., 375.), (330., 16.), -15., (0, 210, 230)),
])
def test_oblique_entry_rejects_side_steep_thick_and_colored_markings(center,shape,angle,color):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    box = cv2.boxPoints((center, shape, angle)).astype(np.int32)
    cv2.fillConvexPoly(image, box, color)
    assert detect_m8_stop_line(image, .58) is None
