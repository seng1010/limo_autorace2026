"""M1 신호등 색 판단 (ROS 없음, mission_traffic_light_node 가 사용).

입력: yolo_detection_node 의 perception/detections (모든 박스) + 같은 시점 카메라 이미지
출력: 'red' / 'yellow' / 'green' / 'unknown'

1. YOLO 라벨이 신호등 색인 박스만 고른다 (red/yellow/green, traffic_* 별칭 포함)
2. 너무 작은 박스(멀리 있는 것)·신뢰도 낮은 박스 제외, 여러 개면 가장 큰 것(가장 가까운 신호등)
3. (선택) 박스 안 HSV 로 켜진 불빛 색을 다시 확인 — 라벨과 다르면 그 프레임은 버림
4. 같은 색이 confirm_frames 번 연속이면 확정, lost_timeout_sec 동안 신호등이 없으면 unknown

광고 모니터: YOLO 학습 데이터에 모니터 화면을 배경(라벨 없음)으로 넣어서 1차로 거르고,
여기서는 HSV 확인과 연속 프레임 조건으로 한 번 더 거른다.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

COLORS = ('red', 'yellow', 'green')
LABEL_TO_COLOR = {
    'red': 'red', 'yellow': 'yellow', 'green': 'green',
    'traffic_red': 'red', 'traffic_yellow': 'yellow', 'traffic_green': 'green',
    'traffic light red': 'red', 'traffic light yellow': 'yellow',
    'traffic light green': 'green',
}


@dataclass
class TrafficLightConfig:
    min_confidence: float = 0.45
    min_area_ratio: float = 0.0005       # 박스 면적 / 화면 면적. 이보다 작으면 무시
    verify_hsv: bool = True
    min_lit_ratio: float = 0.04          # 박스 안에서 그 색 불빛 픽셀 비율 최소값
    lit_value_min: int = 150             # 켜진 불빛 밝기(V) 최소
    lit_saturation_min: int = 90         # 켜진 불빛 채도(S) 최소
    confirm_frames: int = 3
    lost_timeout_sec: float = 1.0


# OpenCV HSV hue 0~180
HUE_RANGES = {
    'red': ((0, 10), (165, 180)),
    'yellow': ((15, 35),),
    'green': ((40, 95),),
}


def lit_color_ratios(image_bgr: np.ndarray, bbox: Sequence[float],
                     config: TrafficLightConfig) -> Dict[str, float]:
    """박스 안에서 색별로 '켜진 불빛' 픽셀 비율."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {c: 0.0 for c in COLORS}
    hsv = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = (sat >= config.lit_saturation_min) & (val >= config.lit_value_min)
    total = float(hue.size)
    ratios = {}
    for color, ranges in HUE_RANGES.items():
        mask = np.zeros(hue.shape, dtype=bool)
        for lo, hi in ranges:
            mask |= (hue >= lo) & (hue <= hi)
        ratios[color] = float(np.count_nonzero(mask & lit)) / total
    return ratios


def frame_color(detections: List[dict], image_bgr: Optional[np.ndarray],
                image_shape, config: TrafficLightConfig) -> Optional[str]:
    """한 프레임의 신호등 색. 신호등이 없거나 확인에 실패하면 None."""
    h, w = image_shape[:2]
    best = None
    for det in detections:
        color = LABEL_TO_COLOR.get(str(det.get('label', '')).lower().strip())
        if color is None or float(det.get('confidence', 0.0)) < config.min_confidence:
            continue
        x1, y1, x2, y2 = det['bbox']
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(w * h)
        if area < config.min_area_ratio:
            continue
        if best is None or area > best[0]:
            best = (area, color, det['bbox'])
    if best is None:
        return None
    _, color, bbox = best
    if config.verify_hsv and image_bgr is not None:
        ratios = lit_color_ratios(image_bgr, bbox, config)
        if ratios[color] < config.min_lit_ratio or ratios[color] < max(ratios.values()):
            return None
    return color


class TrafficLightFilter:
    """프레임 색 -> 확정 상태 (연속 프레임 + 신호등 사라짐 타임아웃)."""

    def __init__(self, config: TrafficLightConfig):
        self.config = config
        self.state = 'unknown'
        self._candidate: Optional[str] = None
        self._count = 0
        self._last_seen: Optional[float] = None

    def update(self, now: float, color: Optional[str]) -> str:
        if color is None:
            if self._last_seen is None or now - self._last_seen > self.config.lost_timeout_sec:
                self.state = 'unknown'
                self._candidate, self._count = None, 0
            return self.state
        self._last_seen = now
        if color == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = color, 1
        if self._count >= self.config.confirm_frames:
            self.state = color
        return self.state

    def tick(self, now: float) -> str:
        """감지 메시지가 아예 안 올 때도 타임아웃을 적용하기 위한 주기 호출."""
        if self._last_seen is None or now - self._last_seen > self.config.lost_timeout_sec:
            self.state = 'unknown'
            self._candidate, self._count = None, 0
        return self.state
