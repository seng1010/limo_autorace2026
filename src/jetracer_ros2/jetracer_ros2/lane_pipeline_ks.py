"""OpenCV lane-detection pipeline independent of ROS.

Image convention used by this module:
* x grows to the image right.
* y grows to the image bottom.
* lateral_error and heading_error are positive toward image right.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LanePipelineConfig:
    width: int = 640
    height: int = 480
    src_points: Tuple[float, ...] = (
        0.30, 0.52, 0.70, 0.52, 0.94, 0.92, 0.06, 0.92,
    )
    dst_points: Tuple[float, ...] = (
        0.22, 0.00, 0.78, 0.00, 0.78, 1.00, 0.22, 1.00,
    )
    detect_white: bool = True
    white_lightness_min: int = 170
    white_saturation_max: int = 100
    detect_yellow: bool = False
    yellow_hue_min: int = 15          # OpenCV HLS hue (0~180), 노란색 약 15~35
    yellow_hue_max: int = 35
    yellow_saturation_min: int = 80
    yellow_lightness_min: int = 60
    gradient_min: int = 35
    morphology_kernel: int = 3
    windows: int = 10
    window_margin: int = 55
    recenter_min_pixels: int = 35
    fit_min_points: int = 120
    polynomial_degree: int = 3
    ransac_iterations: int = 60
    ransac_residual_px: float = 7.0
    lane_width_px: float = 360.0
    lane_width_tolerance: float = 0.35
    max_lane_width_change_px: float = 70.0
    max_fit_jump_px: float = 90.0
    lookahead_y_ratio: float = 0.62
    smoothing_alpha: float = 0.35
    zigzag_min_step_px: float = 8.0
    zigzag_min_span_px: float = 55.0
    zigzag_min_direction_changes: int = 1
    # 같은 선을 왼쪽/오른쪽 탐색이 둘 다 잡았는지 판단하는 기준
    duplicate_overlap_ratio: float = 0.30   # 두 쪽 픽셀이 이 비율 이상 겹치면 같은 선
    duplicate_gap_ratio: float = 0.50       # 두 선 간격이 차선 폭의 이 비율보다 좁으면 같은 선
    random_seed: int = 7


@dataclass
class LaneResult:
    valid: bool
    lateral_error: float = 0.0
    heading_error: float = 0.0
    confidence: float = 0.0
    zigzag: bool = False
    zigzag_score: float = 0.0
    direction_changes: int = 0
    left_fit: Optional[np.ndarray] = None
    right_fit: Optional[np.ndarray] = None
    debug_image: Optional[np.ndarray] = None
    binary_image: Optional[np.ndarray] = None
    reason: str = ''
    recovered_side: str = ''   # 한쪽 복구 시 실제로 보고 간 선: 'left' / 'right'


class LanePipeline:
    """BEV, sliding-window and RANSAC polynomial lane detector."""

    def __init__(self, config: LanePipelineConfig):
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        self._previous_left: Optional[np.ndarray] = None
        self._previous_right: Optional[np.ndarray] = None
        self._previous_width: Optional[float] = None
        self._matrix, self._inverse_matrix = self._perspective_matrices()

    def reset(self) -> None:
        self._previous_left = None
        self._previous_right = None
        self._previous_width = None

    def _normalized_points(self, values: Tuple[float, ...]) -> np.ndarray:
        points = np.asarray(values, dtype=np.float32).reshape(4, 2).copy()
        points[:, 0] *= self.config.width
        points[:, 1] *= self.config.height
        return points

    def _perspective_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        src = self._normalized_points(self.config.src_points)
        dst = self._normalized_points(self.config.dst_points)
        return (
            cv2.getPerspectiveTransform(src, dst),
            cv2.getPerspectiveTransform(dst, src),
        )

    def _binary_mask(self, bird_image: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(bird_image, cv2.COLOR_BGR2HLS)
        hue = hls[:, :, 0]
        lightness = hls[:, :, 1]
        saturation = hls[:, :, 2]

        color = np.zeros(lightness.shape, dtype=bool)
        if self.config.detect_white:
            color |= (
                (lightness >= self.config.white_lightness_min)
                & (saturation <= self.config.white_saturation_max)
            )
        if self.config.detect_yellow:
            color |= (
                (hue >= self.config.yellow_hue_min)
                & (hue <= self.config.yellow_hue_max)
                & (saturation >= self.config.yellow_saturation_min)
                & (lightness >= self.config.yellow_lightness_min)
            )

        sobel = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
        absolute = np.abs(sobel)
        scaled = np.zeros_like(lightness)
        maximum = float(absolute.max())
        if maximum > 1e-6:
            scaled = np.uint8(np.clip(absolute * 255.0 / maximum, 0, 255))
        gradient = (scaled >= self.config.gradient_min) & (lightness >= 100)

        binary = np.uint8(color | gradient) * 255
        kernel_size = max(1, int(self.config.morphology_kernel))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _sliding_window_points(
        self, binary: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        histogram = np.sum(binary[binary.shape[0] // 2:, :] > 0, axis=0)
        midpoint = binary.shape[1] // 2
        left_current = int(np.argmax(histogram[:midpoint]))
        right_current = int(np.argmax(histogram[midpoint:]) + midpoint)

        nonzero_y, nonzero_x = binary.nonzero()
        left_indices = []
        right_indices = []
        window_height = max(1, binary.shape[0] // self.config.windows)
        margin = self.config.window_margin
        left_found = right_found = None      # 직전 창에서 실제로 찾은 x
        left_shift = right_shift = 0.0       # 창 하나 올라갈 때 선이 옆으로 움직인 양

        for window in range(self.config.windows):
            y_low = binary.shape[0] - (window + 1) * window_height
            y_high = binary.shape[0] - window * window_height
            left_low, left_high = left_current - margin, left_current + margin
            right_low, right_high = right_current - margin, right_current + margin

            left_good = np.flatnonzero(
                (nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= left_low) & (nonzero_x < left_high)
            )
            right_good = np.flatnonzero(
                (nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= right_low) & (nonzero_x < right_high)
            )
            left_indices.append(left_good)
            right_indices.append(right_good)

            # 다음 창 위치 = 이번에 찾은 위치 + 선이 휘어 온 방향(직전 이동량).
            # 곡선에서 선이 옆으로 빠르게 휘어도 창이 따라가게 한다.
            if left_good.size >= self.config.recenter_min_pixels:
                found = float(np.mean(nonzero_x[left_good]))
                if left_found is not None:
                    left_shift = float(np.clip(found - left_found, -margin, margin))
                left_found = found
                left_current = int(found + left_shift)
            else:
                left_shift *= 0.5
                left_current = int(left_current + left_shift)
            if right_good.size >= self.config.recenter_min_pixels:
                found = float(np.mean(nonzero_x[right_good]))
                if right_found is not None:
                    right_shift = float(np.clip(found - right_found, -margin, margin))
                right_found = found
                right_current = int(found + right_shift)
            else:
                right_shift *= 0.5
                right_current = int(right_current + right_shift)

            cv2.rectangle(debug, (left_low, y_low), (left_high, y_high), (255, 200, 0), 1)
            cv2.rectangle(debug, (right_low, y_low), (right_high, y_high), (255, 0, 255), 1)

        left_indices = np.concatenate(left_indices) if left_indices else np.array([], dtype=int)
        right_indices = np.concatenate(right_indices) if right_indices else np.array([], dtype=int)
        return (
            nonzero_y[left_indices], nonzero_x[left_indices],
            nonzero_y[right_indices], nonzero_x[right_indices], debug,
        )

    def _ransac_quadratic(
        self, y: np.ndarray, x: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        if x.size < self.config.fit_min_points:
            return None, 0.0

        best_inliers: Optional[np.ndarray] = None
        best_count = 0
        y_low, y_high = float(y.min()), float(y.max())
        sample_count = max(3, int(self.config.polynomial_degree) + 1)
        edges = np.linspace(y_low, y_high + 1e-6, sample_count + 1)
        strata = [
            np.flatnonzero((y >= edges[index]) & (y < edges[index + 1]))
            for index in range(sample_count)
        ]
        if any(indices.size == 0 for indices in strata):
            return None, 0.0
        for _ in range(self.config.ransac_iterations):
            # One point from each vertical third avoids ill-conditioned samples
            # made from three almost identical y coordinates.
            sample = np.asarray([
                self._rng.choice(indices) for indices in strata
            ])
            try:
                candidate = np.polyfit(
                    y[sample], x[sample], self.config.polynomial_degree,
                )
            except (np.linalg.LinAlgError, ValueError):
                continue
            residual = np.abs(x - np.polyval(candidate, y))
            inliers = residual <= self.config.ransac_residual_px
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers

        minimum_inliers = max(self.config.fit_min_points, int(x.size * 0.30))
        if best_inliers is None or best_count < minimum_inliers:
            return None, 0.0

        fit = np.polyfit(
            y[best_inliers], x[best_inliers], self.config.polynomial_degree,
        )
        return fit, float(best_count / x.size)

    def _offset_fit(self, fit: np.ndarray, width: float, direction: float) -> np.ndarray:
        """선과 평행하게, 선에 '수직 방향'으로 width 만큼 떨어진 선을 만든다.

        direction=+1 이면 오른쪽(왼쪽 선 -> 오른쪽 선), -1 이면 왼쪽.
        가로로만 width 를 밀면 기울어진(곡선) 구간에서 실제 차선보다 좁아지므로,
        기울기만큼 sqrt(1 + (dx/dy)^2) 를 곱해 가로 간격을 보정한다.
        """
        y_values = np.linspace(0.0, self.config.height - 1.0, 48)
        x_values = np.polyval(fit, y_values)
        slope = np.polyval(np.polyder(fit), y_values)
        norm = np.sqrt(1.0 + slope ** 2)
        # 선의 접선 (dx/dy, 1) 에 수직인 방향 (1, -dx/dy) 으로 width 만큼 이동한
        # 점들로 다시 곡선을 맞춘다 (= 정확한 평행 곡선).
        shifted_x = x_values + direction * width / norm
        shifted_y = y_values - direction * width * slope / norm
        return np.polyfit(shifted_y, shifted_x, len(fit) - 1)

    def _expected_width(self) -> float:
        return self._previous_width or self.config.lane_width_px

    def _assign_side(self, fit: np.ndarray, fallback: str) -> str:
        """선이 하나만 있을 때, 그게 왼쪽 선인지 오른쪽 선인지 정한다.

        1) 직전 프레임의 왼쪽/오른쪽 선 중 더 가까운 쪽
        2) 직전 기록이 없으면 화면 아래쪽에서 가운데보다 왼쪽이면 left
        """
        y_eval = self.config.height * self.config.lookahead_y_ratio
        x_eval = float(np.polyval(fit, y_eval))
        if self._previous_left is not None and self._previous_right is not None:
            to_left = abs(x_eval - float(np.polyval(self._previous_left, y_eval)))
            to_right = abs(x_eval - float(np.polyval(self._previous_right, y_eval)))
            return 'left' if to_left <= to_right else 'right'
        x_bottom = float(np.polyval(fit, self.config.height * 0.95))
        if abs(x_bottom - self.config.width * 0.5) < 1.0:
            return fallback
        return 'left' if x_bottom < self.config.width * 0.5 else 'right'

    def _single_side(
        self, fit: np.ndarray, side: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """선 하나(side 쪽)를 기준으로 반대쪽 선을 만들어 (left, right)를 돌려준다."""
        # 차선 폭 전체만큼 옮기면 급커브에서 곡선이 크게 일그러지므로,
        # 절반만 옮겨 '중심선'을 먼저 만들고 반대쪽 선 = 2*중심 - 보이는 선 으로 만든다.
        half = 0.5 * self._expected_width()
        if side == 'left':
            center = self._offset_fit(fit, half, +1.0)
            return fit, 2.0 * center - fit
        center = self._offset_fit(fit, half, -1.0)
        return 2.0 * center - fit, fit

    def _is_duplicate(
        self,
        left_y: np.ndarray, left_x: np.ndarray,
        right_y: np.ndarray, right_x: np.ndarray,
        left_fit: np.ndarray, right_fit: np.ndarray,
    ) -> bool:
        """왼쪽 탐색과 오른쪽 탐색이 사실은 '같은 선'을 잡았는지 확인한다.

        곡선에서 안쪽 선이 화면 밖으로 나가면, 남은 바깥쪽 선 하나를
        양쪽 탐색 창이 나눠 잡는 일이 생긴다(실차에서 확인된 증상).
        """
        if left_x.size == 0 or right_x.size == 0:
            return False
        width = self.config.width
        left_codes = left_y.astype(np.int64) * width + left_x.astype(np.int64)
        right_codes = right_y.astype(np.int64) * width + right_x.astype(np.int64)
        shared = np.intersect1d(left_codes, right_codes).size
        if shared / float(min(left_codes.size, right_codes.size)) >= self.config.duplicate_overlap_ratio:
            return True
        # 두 쪽 픽셀이 같이 있는 높이 구간에서 간격이 차선 폭보다 너무 좁으면 같은 선
        y_low = max(float(left_y.min()), float(right_y.min()))
        y_high = min(float(left_y.max()), float(right_y.max()))
        if y_high - y_low < self.config.height * 0.10:
            return False
        sample_y = np.linspace(y_low, y_high, 8)
        gaps = np.abs(np.polyval(right_fit, sample_y) - np.polyval(left_fit, sample_y))
        return float(np.median(gaps)) < self._expected_width() * self.config.duplicate_gap_ratio

    def _coverage_ok(self, ys: np.ndarray) -> bool:
        """선 픽셀이 화면 세로로 충분히 길게(30% 이상) 퍼져 있는지."""
        return ys.size > 0 and float(np.ptp(ys)) >= self.config.height * 0.30

    def _jump_ok(self, left: np.ndarray, right: np.ndarray) -> bool:
        """직전 프레임보다 차선 중심이 갑자기 튀지 않았는지."""
        if self._previous_left is None or self._previous_right is None:
            return True
        y_eval = self.config.height * self.config.lookahead_y_ratio
        previous_center = 0.5 * (np.polyval(self._previous_left, y_eval)
                                 + np.polyval(self._previous_right, y_eval))
        current_center = 0.5 * (np.polyval(left, y_eval) + np.polyval(right, y_eval))
        return abs(current_center - previous_center) <= self.config.max_fit_jump_px

    def _sanity_check(self, left: np.ndarray, right: np.ndarray) -> Tuple[bool, float]:
        sample_y = np.asarray([
            self.config.height * 0.35,
            self.config.height * self.config.lookahead_y_ratio,
            self.config.height * 0.90,
        ])
        # 차선 폭을 두 가지로 재서 하나라도 정상이면 통과시킨다.
        #  - 가로 간격: 기존 방식. 지그재그처럼 두 선이 평행하지 않은 구간에 맞음
        #  - 수직 간격: 곡선에서 선이 기울면 가로 간격이 넓어 보이는 것을 보정
        horizontal = np.polyval(right, sample_y) - np.polyval(left, sample_y)
        center_slope = np.polyval(np.polyder(0.5 * (left + right)), sample_y)
        perpendicular = horizontal / np.sqrt(1.0 + center_slope ** 2)
        expected = self._expected_width()
        lower = expected * (1.0 - self.config.lane_width_tolerance)
        upper = expected * (1.0 + self.config.lane_width_tolerance)

        def width_ok(values: np.ndarray) -> bool:
            return bool(
                np.all(values >= lower) and np.all(values <= upper)
                and float(np.ptp(values)) <= self.config.max_lane_width_change_px
            )

        if width_ok(perpendicular):
            widths = perpendicular
        elif width_ok(horizontal):
            widths = horizontal
        else:
            return False, float(np.mean(horizontal))

        y_eval = self.config.height * self.config.lookahead_y_ratio
        if self._previous_left is not None and self._previous_right is not None:
            previous_center = 0.5 * (
                np.polyval(self._previous_left, y_eval)
                + np.polyval(self._previous_right, y_eval)
            )
            current_center = 0.5 * (np.polyval(left, y_eval) + np.polyval(right, y_eval))
            if abs(current_center - previous_center) > self.config.max_fit_jump_px:
                return False, float(np.mean(widths))
        return True, float(np.mean(widths))

    def _smooth(self, current: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
        if previous is None:
            return current
        alpha = self.config.smoothing_alpha
        return alpha * current + (1.0 - alpha) * previous

    def _zigzag_metrics(
        self,
        left_y: np.ndarray,
        left_x: np.ndarray,
        right_y: np.ndarray,
        right_x: np.ndarray,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
    ) -> Tuple[bool, float, int, np.ndarray]:
        """Detect repeated left/right changes from vertical lane-center samples.

        Unlike the quadratic steering fit, this calculation uses the original
        sliding-window pixels so an S bend or repeated zigzag is not flattened
        into a single parabola.
        """
        sample_y = np.linspace(
            self.config.height * 0.25,
            self.config.height * 0.92,
            self.config.windows,
        )
        half_band = self.config.height / (2.0 * self.config.windows)
        centers = []
        for y_value in sample_y:
            left_band = left_x[np.abs(left_y - y_value) <= half_band]
            right_band = right_x[np.abs(right_y - y_value) <= half_band]
            left_value = (
                float(np.median(left_band))
                if left_band.size >= 3 else float(np.polyval(left_fit, y_value))
            )
            right_value = (
                float(np.median(right_band))
                if right_band.size >= 3 else float(np.polyval(right_fit, y_value))
            )
            centers.append((0.5 * (left_value + right_value), y_value))

        center_points = np.asarray(centers, dtype=np.float32)
        center_x = center_points[:, 0]
        if center_x.size >= 3:
            center_x = np.convolve(center_x, [0.25, 0.50, 0.25], mode='same')
            center_x[0] = center_points[0, 0]
            center_x[-1] = center_points[-1, 0]
            center_points[:, 0] = center_x

        delta = np.diff(center_x)
        significant = delta[np.abs(delta) >= self.config.zigzag_min_step_px]
        if significant.size:
            signs = np.sign(significant)
            compressed = [signs[0]]
            for sign in signs[1:]:
                if sign != compressed[-1]:
                    compressed.append(sign)
            direction_changes = max(0, len(compressed) - 1)
        else:
            direction_changes = 0

        span = float(np.ptp(center_x))
        change_score = min(
            1.0,
            direction_changes / max(1, self.config.zigzag_min_direction_changes + 1),
        )
        span_score = min(1.0, span / max(1.0, self.config.zigzag_min_span_px))
        score = 0.60 * change_score + 0.40 * span_score
        zigzag = (
            direction_changes >= self.config.zigzag_min_direction_changes
            and span >= self.config.zigzag_min_span_px
        )
        return zigzag, float(score), direction_changes, center_points

    def process(self, image: np.ndarray) -> LaneResult:
        if image is None or image.size == 0:
            return LaneResult(valid=False, reason='empty image')

        resized = cv2.resize(image, (self.config.width, self.config.height))
        bird = cv2.warpPerspective(
            resized, self._matrix, (self.config.width, self.config.height),
        )
        binary = self._binary_mask(bird)
        left_y, left_x, right_y, right_x, debug = self._sliding_window_points(binary)

        left_fit, left_ratio = self._ransac_quadratic(left_y, left_x)
        right_fit, right_ratio = self._ransac_quadratic(right_y, right_x)

        # [1] 왼쪽/오른쪽 탐색이 같은 선을 나눠 잡았으면 '선 하나'로 합친다.
        if (left_fit is not None and right_fit is not None
                and self._is_duplicate(left_y, left_x, right_y, right_x, left_fit, right_fit)):
            merged_y = np.concatenate((left_y, right_y))
            merged_x = np.concatenate((left_x, right_x))
            codes = merged_y.astype(np.int64) * self.config.width + merged_x.astype(np.int64)
            _, unique = np.unique(codes, return_index=True)
            merged_y, merged_x = merged_y[unique], merged_x[unique]
            merged_fit, merged_ratio = self._ransac_quadratic(merged_y, merged_x)
            if merged_fit is None:   # 합친 게 안 맞으면 픽셀 많은 쪽 하나만 쓴다
                use_left = left_x.size >= right_x.size
                merged_fit = left_fit if use_left else right_fit
                merged_ratio = left_ratio if use_left else right_ratio
            # [2] 남은 한 줄이 실제로 왼쪽 선인지 오른쪽 선인지 판단
            side = self._assign_side(merged_fit, fallback='left')
            empty = np.array([], dtype=merged_x.dtype)
            if side == 'left':
                left_fit, left_ratio, left_y, left_x = merged_fit, merged_ratio, merged_y, merged_x
                right_fit, right_ratio, right_y, right_x = None, 0.0, empty, empty
            else:
                right_fit, right_ratio, right_y, right_x = merged_fit, merged_ratio, merged_y, merged_x
                left_fit, left_ratio, left_y, left_x = None, 0.0, empty, empty

        if left_fit is None and right_fit is None:
            return LaneResult(
                valid=False, debug_image=debug, binary_image=binary,
                reason='not enough lane pixels',
            )

        recovered_side = ''
        mean_width = 0.0
        if left_fit is not None and right_fit is not None:
            sane, mean_width = self._sanity_check(left_fit, right_fit)
            if not sane:
                # [3] 두 선이 검사에 걸리면 포기하지 말고 한쪽씩만 믿어 본다.
                #     (곡선에서 한쪽 선이 화면 끝에 반쯤 걸쳐 엉터리로 맞춰지는 경우)
                y_eval = self.config.height * self.config.lookahead_y_ratio
                previous_center = None
                if self._previous_left is not None and self._previous_right is not None:
                    previous_center = 0.5 * (np.polyval(self._previous_left, y_eval)
                                             + np.polyval(self._previous_right, y_eval))
                candidates = []
                for side, fit, ys, points in (('left', left_fit, left_y, left_x.size),
                                              ('right', right_fit, right_y, right_x.size)):
                    # 화면 끝에 조금 걸친 짧은 조각은 믿지 않는다
                    if not self._coverage_ok(ys):
                        continue
                    cand_left, cand_right = self._single_side(fit, side)
                    # 직접 만든 반대쪽 선은 폭이 맞게 만들어졌으므로 '급변'만 검사한다
                    if not self._jump_ok(cand_left, cand_right):
                        continue
                    center = 0.5 * (np.polyval(cand_left, y_eval) + np.polyval(cand_right, y_eval))
                    jump = 0.0 if previous_center is None else abs(center - previous_center)
                    candidates.append((jump, -points, side, cand_left, cand_right, self._expected_width()))
                if not candidates:
                    return LaneResult(
                        valid=False, debug_image=debug, binary_image=binary,
                        reason='lane geometry sanity check failed',
                    )
                candidates.sort(key=lambda item: (item[0], item[1]))
                _, _, recovered_side, left_fit, right_fit, mean_width = candidates[0]
        else:
            # [2] 선이 하나만 보이면, 그게 왼쪽인지 오른쪽인지 판단해서 반대쪽을 만든다.
            only_fit = left_fit if left_fit is not None else right_fit
            fallback = 'left' if left_fit is not None else 'right'
            recovered_side = self._assign_side(only_fit, fallback)
            left_fit, right_fit = self._single_side(only_fit, recovered_side)
            mean_width = self._expected_width()
            if not self._jump_ok(left_fit, right_fit):
                return LaneResult(
                    valid=False, debug_image=debug, binary_image=binary,
                    reason='lane geometry sanity check failed',
                )

        recovered = bool(recovered_side)
        observed_sides = 1 if recovered else 2
        if recovered:
            # 한쪽 복구로 '만든' 선으로 폭을 다시 재면 매 프레임 조금씩 줄어든다.
            # 실제 두 선을 본 마지막 폭을 그대로 유지한다.
            mean_width = self._expected_width()
            # 실제로 보고 간 선의 픽셀/신뢰도만 남긴다
            if recovered_side == 'left' and right_x.size and left_x.size:
                right_ratio = 0.0
            elif recovered_side == 'right' and left_x.size and right_x.size:
                left_ratio = 0.0

        left_fit = self._smooth(left_fit, self._previous_left)
        right_fit = self._smooth(right_fit, self._previous_right)
        self._previous_left = left_fit
        self._previous_right = right_fit
        self._previous_width = mean_width

        zigzag, zigzag_score, direction_changes, sampled_centers = (
            self._zigzag_metrics(
                left_y, left_x, right_y, right_x, left_fit, right_fit,
            )
        )

        y_values = np.linspace(0, self.config.height - 1, self.config.height)
        left_values = np.polyval(left_fit, y_values)
        right_values = np.polyval(right_fit, y_values)
        center_values = 0.5 * (left_values + right_values)

        left_points = np.int32(np.column_stack((left_values, y_values))).reshape(-1, 1, 2)
        right_points = np.int32(np.column_stack((right_values, y_values))).reshape(-1, 1, 2)
        center_points = np.int32(np.column_stack((center_values, y_values))).reshape(-1, 1, 2)
        cv2.polylines(debug, [left_points], False, (255, 0, 0), 3)
        cv2.polylines(debug, [right_points], False, (0, 0, 255), 3)
        cv2.polylines(debug, [center_points], False, (0, 255, 255), 3)
        sampled_polyline = np.int32(sampled_centers).reshape(-1, 1, 2)
        cv2.polylines(debug, [sampled_polyline], False, (0, 165, 255), 2)

        y_eval = self.config.height * self.config.lookahead_y_ratio
        center_x = 0.5 * (np.polyval(left_fit, y_eval) + np.polyval(right_fit, y_eval))
        lateral_error = float(np.clip(
            (center_x - self.config.width * 0.5) / (self.config.width * 0.5), -1.5, 1.5,
        ))
        center_fit = 0.5 * (left_fit + right_fit)
        derivative = float(np.polyval(np.polyder(center_fit), y_eval))
        heading_error = float(np.arctan(-derivative))

        point_score = min(1.0, (left_x.size + right_x.size) / (2.0 * self.config.fit_min_points * 3.0))
        inlier_score = max(left_ratio, right_ratio) if recovered else 0.5 * (left_ratio + right_ratio)
        side_score = 1.0 if observed_sides == 2 else 0.65
        confidence = float(np.clip(0.35 * point_score + 0.40 * inlier_score + 0.25 * side_score, 0.0, 1.0))

        cv2.circle(debug, (int(center_x), int(y_eval)), 7, (0, 255, 0), -1)
        cv2.putText(
            debug,
            f'error={lateral_error:+.3f} heading={heading_error:+.3f} conf={confidence:.2f}',
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        if recovered:
            cv2.putText(
                debug, f'ONE-SIDE RECOVERY ({recovered_side.upper()})', (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2,
            )
        cv2.putText(
            debug,
            f'zigzag={int(zigzag)} score={zigzag_score:.2f} changes={direction_changes}',
            (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 80, 255) if zigzag else (180, 180, 180), 2,
        )

        return LaneResult(
            valid=True,
            lateral_error=lateral_error,
            heading_error=heading_error,
            confidence=confidence,
            zigzag=zigzag,
            zigzag_score=zigzag_score,
            direction_changes=direction_changes,
            left_fit=left_fit,
            right_fit=right_fit,
            debug_image=debug,
            binary_image=binary,
            reason='one-side recovery' if recovered else 'both lanes detected',
            recovered_side=recovered_side,
        )
