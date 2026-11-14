"""M1 신호등 색 판단 (ROS 없음, mission_traffic_light_node 가 사용).

입력: yolo_detection_node 의 perception/detections (모든 박스) + 같은 시점 카메라 이미지
출력: 'red' / 'yellow' / 'green' / 'unknown'

**YOLO 라벨 방식 두 가지를 모두 받는다** (학습을 어느 쪽으로 하든 동작):
  (A) 색 클래스  : 'red' / 'yellow' / 'green' (불 켜진 색으로 라벨). traffic_* 별칭 포함
  (B) 몸체 클래스: 'traffic_light' 하나 (신호등 함체 전체에 박스). 색은 박스 안에서
      어느 칸에 불이 켜졌는지로 읽는다 (위=적, 가운데=황, 아래=녹) + HSV 색 확인.
      광고 모니터는 함체 모양이 아니라서 (B)가 더 강하다.

1. 신호등 박스만 고른다 (위 라벨 중 하나)
2. 너무 작은 박스(멀리 있는 것)·신뢰도 낮은 박스 제외, 여러 개면 가장 큰 것(가장 가까운 신호등)
3. (A) 박스 안 HSV 로 켜진 불빛 색을 다시 확인 — 라벨과 다르면 그 프레임은 버림
   (B) 켜진 칸의 위치 + 그 칸의 HSV 색이 둘 다 맞아야 인정
4. 같은 색이 confirm_frames 번 연속이면 확정, lost_timeout_sec 동안 신호등이 없으면 unknown

적+황이 동시에 켜지면 'red'(정지)로 본다. 어느 해석이든(전환 중 / "곧 출발") 멈추는 게
맞고, 녹색이 확정될 때까지 기다리면 된다. 세 칸이 모두 밝으면 광고 모니터로 보고 버린다.

광고 모니터: YOLO 학습 데이터에 모니터 화면을 배경(라벨 없음)으로 넣어서 1차로 거르고,
여기서는 HSV 확인과 연속 프레임 조건으로 한 번 더 거른다.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

COLORS = ('red', 'yellow', 'green')
HOUSING_LABELS = {'traffic_light', 'trafficlight', 'traffic light', 'light', 'signal'}
LABEL_TO_COLOR = {
    'red': 'red', 'yellow': 'yellow', 'green': 'green',
    'traffic_red': 'red', 'traffic_yellow': 'yellow', 'traffic_green': 'green',
    'traffic light red': 'red', 'traffic light yellow': 'yellow',
    'traffic light green': 'green',
}


@dataclass
class TrafficLightConfig:
    # (B) 몸체 클래스용: 함체를 세로 3칸으로 나눠 켜진 칸을 찾는다
    housing_lamp_margin: float = 0.08    # 함체 위·아래 테두리로 보고 버릴 비율
    housing_min_lit_ratio: float = 0.03  # 한 칸이 '켜졌다'고 볼 최소 밝은 픽셀 비율
    housing_lit_dominance: float = 1.5   # 1등 칸이 2등 칸보다 이 배 이상 밝아야 인정
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


def housing_color(image_bgr: np.ndarray, bbox: Sequence[float],
                  config: TrafficLightConfig) -> Optional[str]:
    """(B) 함체 박스 -> 켜진 칸의 색. 함체를 세로 3등분해 가장 밝은 칸을 찾는다.

    칸 위치(위=적, 가운데=황, 아래=녹)와 그 칸의 HSV 색이 **둘 다** 맞아야 인정한다.
    둘이 다르면(예: 위 칸인데 초록) 신호등이 아니거나 각도가 이상한 것으로 보고 버린다.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 6:
        return None
    margin = int(round((y2 - y1) * config.housing_lamp_margin))
    top, bottom = y1 + margin, y2 - margin
    if bottom - top < 3:
        top, bottom = y1, y2
    edges = np.linspace(top, bottom, 4).round().astype(int)
    scores = []
    for color, (lo, hi) in zip(COLORS, zip(edges[:-1], edges[1:])):
        if hi - lo < 1:
            return None
        ratios = lit_color_ratios(image_bgr, (x1, lo, x2, hi), config)
        scores.append((sum(ratios.values()), color, ratios))
    scores.sort(reverse=True, key=lambda item: item[0])
    best_total, best_color, best_ratios = scores[0]
    if best_total < config.housing_min_lit_ratio:
        return None                                     # 켜진 칸이 없음 (꺼져 있거나 오검출)
    lit = {color for total, color, _ in scores if total >= config.housing_min_lit_ratio}
    if len(lit) > 1 and best_total < scores[1][0] * config.housing_lit_dominance:
        # 1등 칸이 뚜렷하지 않음: 적+황 동시 점등이면 정지, 그 외(3칸 전부 등)는 버림
        return 'red' if lit == {'red', 'yellow'} else None
    if config.verify_hsv and max(best_ratios, key=best_ratios.get) != best_color:
        return None                                     # 칸 위치와 실제 색이 안 맞음
    return best_color


def frame_color(detections: List[dict], image_bgr: Optional[np.ndarray],
                image_shape, config: TrafficLightConfig) -> Optional[str]:
    """한 프레임의 신호등 색. 신호등이 없거나 확인에 실패하면 None."""
    h, w = image_shape[:2]
    best = None
    for det in detections:
        label = str(det.get('label', '')).lower().strip()
        color = LABEL_TO_COLOR.get(label)               # (A) 색 클래스, 몸체면 None
        if color is None and label not in HOUSING_LABELS:
            continue
        if float(det.get('confidence', 0.0)) < config.min_confidence:
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
    if color is None:                                   # (B) 몸체 -> 칸 위치로 색 읽기
        return housing_color(image_bgr, bbox, config) if image_bgr is not None else None
    if config.verify_hsv and image_bgr is not None:     # (A) 라벨 색을 HSV 로 재확인
        ratios = lit_color_ratios(image_bgr, bbox, config)
        if ratios[color] < config.min_lit_ratio:
            return None
        # 적+황 동시 점등이면 라벨이 뭐든, 둘 중 뭐가 더 밝든 정지
        both = min(ratios['red'], ratios['yellow']) >= config.min_lit_ratio
        if color in ('red', 'yellow') and both and ratios['green'] < config.min_lit_ratio:
            return 'red'
        if ratios[color] < max(ratios.values()):
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
