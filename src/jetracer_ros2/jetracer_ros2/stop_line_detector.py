"""정지선 검출 + 픽셀 행 -> 거리 보정 + 사각지대 odom 추적 (ROS 없이 동작).

stop_line_node 와 scripts/calibrate_stop_line.py 가 같이 쓴다. ROS 의존성이 없어서
합성 이미지로 단위 테스트할 수 있다.

거리 정의: **앞바퀴 접지점 -> 정지선의 가까운 쪽 가장자리** [m].
(규정: 모든 진입·정지 판정 기준점은 앞바퀴)

정지선이 차에 가까워지면 카메라 하단 사각지대로 사라진다. 예전에는 그 순간 거리가
-1 이 되어 "정지선에 서 있음" 상태가 풀렸다(M1 빨간불에 재출발하는 원인).
BlindZoneTracker 가 마지막으로 본 거리에서 odom 이동거리를 빼서 거리를 계속 낸다.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 검출
# --------------------------------------------------------------------------- #

@dataclass
class StopLineConfig:
    roi_top_ratio: float = 0.58
    detect_white: bool = True
    white_lightness_min: int = 180
    white_saturation_max: int = 90
    detect_yellow: bool = False
    yellow_hue_min: int = 15          # OpenCV HLS hue (0~180)
    yellow_hue_max: int = 35
    yellow_saturation_min: int = 80
    yellow_lightness_min: int = 60
    horizontal_kernel_ratio: float = 0.12
    minimum_width_ratio: float = 0.42
    minimum_area_ratio: float = 0.018


@dataclass
class StopLineDetection:
    row_ratio: float                     # 선의 가까운 쪽(아래) 가장자리, 전체 이미지 기준 0=위 1=아래
    bbox: Tuple[int, int, int, int]      # x, y, w, h (전체 이미지 좌표)


def color_mask(image_bgr: np.ndarray, config: StopLineConfig) -> np.ndarray:
    hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
    hue, lightness, saturation = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    mask = np.zeros(hue.shape, dtype=bool)
    if config.detect_white:
        mask |= ((lightness >= config.white_lightness_min)
                 & (saturation <= config.white_saturation_max))
    if config.detect_yellow:
        mask |= ((hue >= config.yellow_hue_min) & (hue <= config.yellow_hue_max)
                 & (saturation >= config.yellow_saturation_min)
                 & (lightness >= config.yellow_lightness_min))
    return mask.astype(np.uint8) * 255


def detect_stop_line(image_bgr: np.ndarray,
                     config: StopLineConfig) -> Optional[StopLineDetection]:
    """카메라 하단 ROI 에서 가장 가까운(가장 아래) 넓은 가로선을 찾는다."""
    height, width = image_bgr.shape[:2]
    roi_y = int(height * config.roi_top_ratio)
    roi = image_bgr[roi_y:, :]
    if roi.size == 0:
        return None
    mask = color_mask(roi, config)
    kernel_width = max(5, int(width * config.horizontal_kernel_ratio))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, kernel_width), np.uint8))
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    roi_area = float(max(1, roi.shape[0] * width))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w / width < config.minimum_width_ratio:
            continue
        if cv2.contourArea(contour) / roi_area < config.minimum_area_ratio:
            continue
        if best is None or y + h > best[1] + best[3]:
            best = (x, y, w, h)
    if best is None:
        return None
    x, y, w, h = best
    bottom = roi_y + y + h
    return StopLineDetection(row_ratio=bottom / float(height), bbox=(x, roi_y + y, w, h))


# --------------------------------------------------------------------------- #
# 픽셀 행 -> 거리
# --------------------------------------------------------------------------- #

class RowDistanceModel:
    """정지선 아래 가장자리의 행 비율 -> 앞바퀴까지 거리 [m].

    rows / distances 가 2개 이상이면 보정표(선형 보간, 양끝은 끝 구간 기울기로 외삽).
    2개 미만이면 보정 전 상태로 보고 예전 식을 그대로 쓴다:
        distance = (1 - ROI 안에서의 아래 가장자리 비율) * legacy_scale_m
    """

    def __init__(self, rows: Sequence[float], distances: Sequence[float],
                 legacy_scale_m: float = 1.6, roi_top_ratio: float = 0.58):
        rows = [float(r) for r in rows]
        distances = [float(d) for d in distances]
        self.legacy_scale_m = float(legacy_scale_m)
        self.roi_top_ratio = float(roi_top_ratio)
        self.calibrated = len(rows) >= 2 and len(distances) >= 2
        if not self.calibrated:
            self._rows = self._distances = None
            return
        if len(rows) != len(distances):
            raise ValueError('lut_rows 와 lut_distances_m 길이가 다릅니다')
        order = np.argsort(rows)
        self._rows = np.asarray(rows, dtype=float)[order]
        self._distances = np.asarray(distances, dtype=float)[order]
        if np.any(np.diff(self._rows) <= 0.0):
            raise ValueError('lut_rows 에 같은 값이 있습니다')
        if np.any(np.diff(self._distances) >= 0.0):
            raise ValueError('아래 행일수록 거리가 짧아야 합니다 (lut_distances_m 확인)')

    def distance(self, row_ratio: float) -> float:
        if not self.calibrated:
            span = max(1e-6, 1.0 - self.roi_top_ratio)
            bottom_in_roi = (row_ratio - self.roi_top_ratio) / span
            return max(0.0, (1.0 - bottom_in_roi) * self.legacy_scale_m)
        rows, dists = self._rows, self._distances
        if row_ratio <= rows[0]:
            i, j = 0, 1
        elif row_ratio >= rows[-1]:
            i, j = -2, -1
        else:
            return float(np.interp(row_ratio, rows, dists))
        slope = (dists[j] - dists[i]) / (rows[j] - rows[i])
        return max(0.0, float(dists[i] + slope * (row_ratio - rows[i])))


# --------------------------------------------------------------------------- #
# 사각지대 추적
# --------------------------------------------------------------------------- #

class BlindZoneTracker:
    """정지선이 화면에서 사라진 뒤에도 odom 으로 남은 거리를 계속 낸다.

    - 선이 보이면 그 거리와 그때의 odom 누적거리를 기준점으로 저장
    - 안 보이면: 남은 거리 = 기준 거리 - (지금 odom - 기준 odom)
      차가 멈춰 있으면 값도 그대로 -> 빨간불 대기 중 "정지선 위" 상태 유지
    - 기준 거리가 max_track_m 보다 멀면(멀리서 잠깐 놓친 것) 추적하지 않음
    - 남은 거리가 -pass_margin_m 보다 작아지면(선을 지나침) 추적 종료
    - 가까운 선을 따라가는 중(기준 거리 <= max_track_m)에 예상보다 jump_tolerance_m 이상
      먼 선이 보이면(가까운 선이 사각지대로 들어가고 횡단보도 반대편 선 등이 잡힘)
      무시하고 지금 선을 계속 따라감
    """

    def __init__(self, max_track_m: float = 0.6, pass_margin_m: float = 0.10,
                 jump_tolerance_m: float = 0.15):
        self.max_track_m = float(max_track_m)
        self.pass_margin_m = float(pass_margin_m)
        self.jump_tolerance_m = float(jump_tolerance_m)
        self.odom_s: Optional[float] = None
        self.source = 'none'                 # visual / odom / none
        self._anchor_distance: Optional[float] = None
        self._anchor_s: Optional[float] = None

    def update_odom(self, travelled_s: float) -> None:
        self.odom_s = float(travelled_s)

    def reset(self) -> None:
        self._anchor_distance = self._anchor_s = None
        self.source = 'none'

    def _estimate(self) -> Optional[float]:
        if self._anchor_distance is None or self.odom_s is None or self._anchor_s is None:
            return None
        return self._anchor_distance - (self.odom_s - self._anchor_s)

    def update(self, visual_distance: Optional[float]) -> Optional[float]:
        if visual_distance is not None:
            estimate = self._estimate()
            jumped_farther = (estimate is not None
                              and self._anchor_distance <= self.max_track_m
                              and visual_distance > estimate + self.jump_tolerance_m)
            if not jumped_farther:
                self._anchor_distance = float(visual_distance)
                self._anchor_s = self.odom_s
                self.source = 'visual'
                return float(visual_distance)

        estimate = self._estimate()
        if estimate is None or self._anchor_distance > self.max_track_m:
            self.reset()
            return None
        if estimate < -self.pass_margin_m:
            self.reset()
            return None
        self.source = 'odom'
        return max(0.0, estimate)


class OdomPathIntegrator:
    """nav_msgs/Odometry 위치를 누적 주행거리(전진 +, 후진 -)로 바꾼다."""

    def __init__(self, max_step_m: float = 0.5):
        self.max_step_m = float(max_step_m)
        self.s = 0.0
        self._last_xy: Optional[Tuple[float, float]] = None

    def update(self, x: float, y: float, linear_velocity: float) -> float:
        if self._last_xy is not None:
            step = float(np.hypot(x - self._last_xy[0], y - self._last_xy[1]))
            if step <= self.max_step_m:          # odom 리셋 같은 큰 점프는 무시
                self.s += -step if linear_velocity < 0.0 else step
        self._last_xy = (float(x), float(y))
        return self.s


# --------------------------------------------------------------------------- #
# 보정표 만들기 (scripts/calibrate_stop_line.py)
# --------------------------------------------------------------------------- #

def fit_row_distance_table(samples: Sequence[Tuple[float, float]],
                           points: int = 8) -> Tuple[List[float], List[float]]:
    """(row_ratio, distance_m) 샘플 -> 단조로운 보정표 (rows 오름차순, 거리 내림차순).

    거리 구간을 points 개로 나눠 구간별 중앙값을 쓰고, 단조성을 깨는 점은 버린다.
    """
    data = np.asarray([s for s in samples if s[1] >= 0.0], dtype=float)
    if len(data) < 2:
        raise ValueError('정지선이 검출된 프레임이 2개 미만입니다')
    near, far = float(data[:, 1].min()), float(data[:, 1].max())
    edges = np.linspace(near, far, max(2, points) + 1)
    rows, dists = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = data[(data[:, 1] >= low) & (data[:, 1] <= high)]
        if len(in_bin) == 0:
            continue
        rows.append(float(np.median(in_bin[:, 0])))
        dists.append(float(np.median(in_bin[:, 1])))
    # 가까운 거리부터: 거리가 늘수록 행은 줄어야 한다
    order = np.argsort(dists)
    kept_rows, kept_dists = [], []
    for k in order:
        if kept_rows and (rows[k] >= kept_rows[-1] or dists[k] <= kept_dists[-1]):
            continue
        kept_rows.append(rows[k])
        kept_dists.append(dists[k])
    if len(kept_rows) < 2:
        raise ValueError('단조로운 보정점이 2개 미만입니다 (검출이 흔들림)')
    return kept_rows[::-1], kept_dists[::-1]
