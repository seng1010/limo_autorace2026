"""Stable OpenCV lane-detection pipeline independent of ROS.

Image convention used by this module:
* x grows to the image right.
* y grows to the image bottom.
* lateral_error and heading_error are positive toward image right.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from jetracer_ros2.lane_attributes import classify_lane, measured_heading_span
from jetracer_ros2.lane_vision import compensate_light, GroundLookahead


@dataclass
class LanePipelineConfig:
    width: int = 640
    height: int = 480
    # Screen-video estimate; not a metric camera/ground calibration.
    src_points: Tuple[float, ...] = (
        0.3605, 0.46, 0.6395, 0.46, 1.00, 0.82, 0.00, 0.82,
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
    gradient_require_color_support: bool = True
    gradient_support_dilation: int = 9
    morphology_kernel: int = 3
    windows: int = 10
    window_margin: int = 55
    opposite_window_margin: int = 30
    search_opposite_in_single_mode: bool = False
    histogram_min_peak: int = 12
    bottom_tracking_windows: int = 3
    recenter_min_pixels: int = 35
    fit_min_points: int = 120
    polynomial_degree: int = 3
    ransac_iterations: int = 60
    ransac_residual_px: float = 7.0
    lane_width_px: float = 370.0
    lane_width_tolerance: float = 0.35
    max_lane_width_change_px: float = 70.0
    max_fit_jump_px: float = 90.0
    strong_single_inlier_ratio: float = 0.65
    strong_single_point_multiplier: float = 1.5
    lookahead_y_ratio: float = 0.92
    smoothing_alpha: float = 0.35
    zigzag_min_step_px: float = 8.0
    zigzag_min_span_px: float = 55.0
    zigzag_min_direction_changes: int = 1
    # 같은 선을 왼쪽/오른쪽 탐색이 둘 다 잡았는지 판단하는 기준
    duplicate_overlap_ratio: float = 0.30   # 두 쪽 픽셀이 이 비율 이상 겹치면 같은 선
    duplicate_gap_ratio: float = 0.50       # 두 선 간격이 차선 폭의 이 비율보다 좁으면 같은 선
    random_seed: int = 7
    low_light_enabled: bool = True
    low_light_max_gain: float = 3.0
    preview_y_ratio: float = 0.76
    metric_lookahead_enabled: bool = False
    lookahead_distance_m: float = 0.40
    camera_forward_offset_m: float = 0.0
    metric_lateral_scale_m: float = 0.30
    bev_ground_points_m: Tuple[float, ...] = (0.0,) * 8
    metric_max_near_clamp_m: float = 0.25
    metric_edge_margin_m: float = 0.02
    target_roi_margin_px: float = 20.0
    yellow_curve_priority: bool = True
    yellow_curve_min_span: float = 0.20
    curve_nearest_max_forward_m: float = 1.05
    curve_left_near_max_forward_m: float = 1.05
    curve_lane_width_m: float = 0.52  # Measured yellow-to-yellow track width.
    curve_right_min_clearance_m: float = 0.13  # Provisional center-to-right-paint guard.


@dataclass
class LaneResult:
    valid: bool
    # Diagnostic stages only. ``valid`` remains the existing drive gate.
    # accepted means an observed boundary survived pixel/geometry checks,
    # not that its colour or side identity is guaranteed correct.
    observation_state: str = 'not_evaluated'
    path_state: str = 'not_evaluated'
    policy_state: str = 'not_evaluated'
    lateral_error: float = 0.0
    heading_error: float = 0.0
    confidence: float = 0.0
    zigzag: bool = False
    zigzag_score: float = 0.0
    direction_changes: int = 0
    left_fit: Optional[np.ndarray] = None
    right_fit: Optional[np.ndarray] = None
    debug_image: Optional[np.ndarray] = None
    bev_image: Optional[np.ndarray] = None
    target_bev_xy: Optional[Tuple[float, float]] = None
    binary_image: Optional[np.ndarray] = None
    color_binary_image: Optional[np.ndarray] = None
    gradient_binary_image: Optional[np.ndarray] = None
    reason: str = ''
    recovered_side: str = ''   # 한쪽 복구 시 실제로 보고 간 선: 'left' / 'right'
    tracking_mode: str = 'auto'
    left_type: str = 'unknown'
    right_type: str = 'unknown'
    preview_heading_error: float = 0.0
    lookahead_description: str = 'uncalibrated'
    illumination_gain: float = 1.0
    curve_heading_span: float = 0.0  # Measured boundaries only; radians, not 1/metres.
    target_lateral_m: Optional[float] = None
    target_forward_m: Optional[float] = None
    yellow_curve_priority_active: bool = False
    curve_turn: str = 'unknown'
    curve_nearest_observation: bool = False
    curve_left_target_active: bool = False
    curve_left_preview_rad: Optional[float] = None
    ground_path: Tuple[Tuple[float, float], ...] = ()  # (lateral right, forward), near to far


class LanePipeline:
    """BEV, sliding-window and RANSAC polynomial lane detector."""

    def __init__(self, config: LanePipelineConfig):
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        self._previous_left: Optional[np.ndarray] = None
        self._previous_right: Optional[np.ndarray] = None
        self._previous_width: Optional[float] = None
        self._yellow_curve_hold = 0
        self._yellow_curve_side = ''
        self._yellow_curve_outer_missing = False
        self._yellow_curve_right_points = None
        self._yellow_curve_binary = None
        self._curve_right_clearance_failed = False
        self._matrix, self._inverse_matrix = self._perspective_matrices()
        self.ground = (GroundLookahead(
            self._normalized_points(config.dst_points), config.bev_ground_points_m,
            config.camera_forward_offset_m, config.lookahead_distance_m,
            config.metric_lateral_scale_m, config.metric_max_near_clamp_m,
            config.metric_edge_margin_m) if config.metric_lookahead_enabled else None)

    def reset(self) -> None:
        self._previous_left = None
        self._previous_right = None
        self._previous_width = None
        self._yellow_curve_hold = 0
        self._yellow_curve_side = ''
        self._yellow_curve_outer_missing = False
        self._yellow_curve_right_points = None
        self._yellow_curve_binary = None
        self._curve_right_clearance_failed = False

    def _yellow_curve_mask(self, bird: np.ndarray, tracking_mode: str) -> Optional[np.ndarray]:
        """Return yellow-only paint when a connected solid yellow curve exists.

        This check precedes the mixed-colour histogram so an outer white mat
        edge cannot become the curve anchor. Straight/mixed-colour lanes keep
        their original white candidates.
        """
        if not (self.config.detect_yellow and self.config.yellow_curve_priority):
            return None
        self._yellow_curve_outer_missing = False
        self._yellow_curve_right_points = None
        self._yellow_curve_binary = None
        hls = cv2.cvtColor(bird, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        yellow = np.uint8(
            (h >= self.config.yellow_hue_min) & (h <= self.config.yellow_hue_max)
            & (s >= self.config.yellow_saturation_min)
            & (l >= self.config.yellow_lightness_min)) * 255
        kernel = np.ones((max(1, self.config.morphology_kernel),) * 2, np.uint8)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
        self._yellow_curve_binary = yellow
        if cv2.countNonZero(yellow) < self.config.fit_min_points:
            # Brief occlusion must not immediately hand control to the bright
            # white mat edge. An empty yellow-only frame safely stops instead.
            if self._yellow_curve_hold:
                self._yellow_curve_hold -= 1
                return yellow
            return None
        confirmed_sides = []
        # Anchor each side independently: a tight curve can cross the image
        # midpoint, and AUTO would split one yellow line into two short tracks.
        modes = (tracking_mode,) if tracking_mode in ('left', 'right') else ('left', 'right')
        for mode in modes:
            ly, lx, ry, rx, _ = self._sliding_window_points(yellow, tracking_mode=mode)
            ys, xs = (ly, lx) if mode == 'left' else (ry, rx)
            if xs.size < self.config.fit_min_points or not self._coverage_ok(ys):
                continue
            fit = np.polyfit(ys, xs, self.config.polynomial_degree)
            if (measured_heading_span(fit, ys, self.config.height)
                    >= self.config.yellow_curve_min_span
                    and classify_lane(bird, fit, ys, self.config) == 'yellow/solid/curve'):
                sampled_x = np.polyval(fit, np.linspace(np.min(ys), np.max(ys), 12))
                steps = np.diff(sampled_x)
                reversal = bool(np.any(steps > self.config.zigzag_min_step_px)
                                and np.any(steps < -self.config.zigzag_min_step_px))
                confirmed_sides.append((
                    mode, fit, xs.size, float(np.polyval(fit, np.max(ys))),
                    float(np.polyval(fit, np.min(ys)) - np.polyval(fit, np.max(ys))),
                    reversal, ys, xs))
        if confirmed_sides:
            by_side = {candidate[0]: candidate for candidate in confirmed_sides}
            distinct = ('left' in by_side and 'right' in by_side
                        and by_side['right'][3] - by_side['left'][3]
                        >= 0.5 * self._expected_width())
            if distinct:
                self._yellow_curve_right_points = (by_side['right'][6], by_side['right'][7])
            clockwise = any(candidate[4] > 20 and not candidate[5]
                            for candidate in confirmed_sides)
            if distinct and clockwise:
                # Never substitute the inner/right curve when the left outer
                # boundary is missing or does not itself bend clockwise.
                left = by_side['left']
                self._yellow_curve_side = 'left' if left[4] > 20 and not left[5] else 'right'
            elif distinct:
                self._yellow_curve_side = 'right'
            else:
                # Both modes can follow the SAME paint. Position/history must
                # identify it as outer-left before it can command a right turn.
                best = max(confirmed_sides, key=lambda candidate: candidate[2])
                self._yellow_curve_side = self._assign_side(best[1], best[0])
            self._yellow_curve_outer_missing = clockwise and self._yellow_curve_side != 'left'
            self._yellow_curve_hold = 3
        elif self._yellow_curve_hold:
            self._yellow_curve_hold -= 1
            self._yellow_curve_outer_missing = True
        return yellow if self._yellow_curve_hold else None

    def _normalized_points(self, values: Tuple[float, ...]) -> np.ndarray:
        points = np.asarray(values, dtype=np.float32).reshape(4, 2).copy()
        points[:, 0] *= self.config.width - 1
        points[:, 1] *= self.config.height - 1
        return points

    def _perspective_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        src = self._normalized_points(self.config.src_points)
        dst = self._normalized_points(self.config.dst_points)
        return (
            cv2.getPerspectiveTransform(src, dst),
            cv2.getPerspectiveTransform(dst, src),
        )

    def _target_within_bev(self, x: float, y: float) -> bool:
        """Reject a driving target outside the four calibrated BEV corners."""
        dst = self._normalized_points(self.config.dst_points)
        near_y, far_y = float(dst[3, 1]), float(dst[0, 1])
        if not np.isfinite([x, y]).all() or abs(near_y - far_y) < 1e-6:
            return False
        fraction = float(np.clip((y - far_y) / (near_y - far_y), 0.0, 1.0))
        left_edge = float(dst[0, 0] + fraction * (dst[3, 0] - dst[0, 0]))
        right_edge = float(dst[1, 0] + fraction * (dst[2, 0] - dst[1, 0]))
        margin = max(0.0, float(self.config.target_roi_margin_px))
        return left_edge - margin <= x <= right_edge + margin

    def _binary_masks(
        self, bird_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hls = cv2.cvtColor(bird_image, cv2.COLOR_BGR2HLS)
        hue = hls[:, :, 0]
        lightness = hls[:, :, 1]
        saturation = hls[:, :, 2]
        local_contrast = lightness.astype(float) - cv2.GaussianBlur(lightness, (31, 31), 0)

        color = np.zeros(lightness.shape, dtype=bool)
        white = np.zeros_like(color)
        yellow = np.zeros_like(color)
        if self.config.detect_white:
            white = (
                (lightness >= self.config.white_lightness_min)
                & (saturation <= self.config.white_saturation_max)
                & ((local_contrast >= 8) | (lightness >= 230))
            )
        if self.config.detect_yellow:
            yellow = (
                (hue >= self.config.yellow_hue_min)
                & (hue <= self.config.yellow_hue_max)
                & (saturation >= self.config.yellow_saturation_min)
                & (lightness >= self.config.yellow_lightness_min)
            )

        # Yellow outside boundaries delimit the road. Preserve white paint
        # inside them (parking / multi-lane markings), reject outer mat trim.
        # Require longitudinal support so a small yellow sign cannot gate white.
        xs = np.arange(lightness.shape[1])
        mid = lightness.shape[1] / 2
        for left in (True, False):
            side = yellow & ((xs < mid) if left else (xs >= mid))[None, :]
            supported = side.sum(axis=1) >= 2
            if np.count_nonzero(supported) < lightness.shape[0] * .30:
                continue
            edge = (np.max(np.where(side, xs, -1), axis=1) if left
                    else np.min(np.where(side, xs, lightness.shape[1]), axis=1))
            outside = (xs[None, :] < edge[:, None] - 12 if left
                       else xs[None, :] > edge[:, None] + 12)
            white &= ~(outside & supported[:, None])
        color = white | yellow

        sobel = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
        absolute = np.abs(sobel)
        scaled = np.zeros_like(lightness)
        maximum = float(absolute.max())
        if maximum > 1e-6:
            scaled = np.uint8(np.clip(absolute * 255.0 / maximum, 0, 255))
        gradient = (scaled >= self.config.gradient_min) & (lightness >= 100)

        # Gradient alone also reacts strongly to arrows, text and stop lines.  Use
        # it only near a configured lane colour so it reinforces a boundary
        # instead of becoming an independent lane candidate.
        if self.config.gradient_require_color_support:
            dilation = max(1, int(self.config.gradient_support_dilation))
            if dilation % 2 == 0:
                dilation += 1
            support_kernel = np.ones((dilation, dilation), np.uint8)
            color_support = cv2.dilate(
                np.uint8(color) * 255, support_kernel, iterations=1,
            ) > 0
            gradient &= color_support

        binary = np.uint8(color | gradient) * 255
        kernel_size = max(1, int(self.config.morphology_kernel))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary, np.uint8(color) * 255, np.uint8(gradient) * 255

    def _binary_mask(self, bird_image: np.ndarray) -> np.ndarray:
        """Compatibility wrapper used by older callers and tests."""
        binary, _, _ = self._binary_masks(bird_image)
        return binary

    def _histogram_peak(
        self,
        histogram: np.ndarray,
        start: int,
        end: int,
        expected_x: Optional[float] = None,
        margin: Optional[int] = None,
    ) -> Optional[int]:
        """Return a real histogram peak, never an arbitrary argmax of zeros."""
        start = max(0, int(start))
        end = min(histogram.size, int(end))
        if expected_x is not None:
            radius = int(margin or self.config.window_margin)
            start = max(start, int(expected_x) - radius)
            end = min(end, int(expected_x) + radius + 1)
        if end <= start:
            return None
        local = histogram[start:end]
        index = int(np.argmax(local))
        if int(local[index]) < int(self.config.histogram_min_peak):
            return None
        return start + index

    def _sliding_window_points(
        self,
        binary: np.ndarray,
        tracking_mode: str = 'auto',
        color_binary: Optional[np.ndarray] = None,
        gradient_binary: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Track lane pixels from the image bottom with an explicit anchor policy.

        ``left`` and ``right`` modes trust only that boundary by default.  The
        opposite boundary is either searched near its expected lane-width
        position or reconstructed later; it is never seeded by an unrelated
        full-half-image peak.
        """
        if tracking_mode not in ('auto', 'both', 'left', 'right'):
            tracking_mode = 'auto'

        debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if color_binary is not None:
            debug[color_binary > 0] = (0, 150, 0)
        if gradient_binary is not None:
            gradient_only = (gradient_binary > 0) & (
                (color_binary == 0) if color_binary is not None else True
            )
            debug[gradient_only] = (255, 100, 0)
        # On a tight yellow bend, the lower-half histogram is dominated by
        # the distant portion of the arc and can seed the LEFT tracker on the
        # RIGHT paint. Seed from the bottom fifth, then follow all windows.
        seed_top = binary.shape[0] // 2
        if binary is self._yellow_curve_binary:
            bottom_histogram = np.sum(binary[int(binary.shape[0] * .8):, :] > 0, axis=0)
            split = binary.shape[1] // 2
            lower_left = self._histogram_peak(bottom_histogram, 0, split)
            lower_right = self._histogram_peak(bottom_histogram, split, binary.shape[1])
            if (lower_left is not None and lower_right is not None
                    and lower_right - lower_left >= .5 * self._expected_width()):
                seed_top = int(binary.shape[0] * .8)
        histogram = np.sum(binary[seed_top:, :] > 0, axis=0)
        midpoint = binary.shape[1] // 2
        bottom_y = self.config.height * 0.92

        previous_left_x = (
            float(np.polyval(self._previous_left, bottom_y))
            if self._previous_left is not None else None
        )
        previous_right_x = (
            float(np.polyval(self._previous_right, bottom_y))
            if self._previous_right is not None else None
        )

        left_current = self._histogram_peak(
            histogram,
            0,
            binary.shape[1] if previous_left_x is not None else midpoint,
            expected_x=previous_left_x,
        )
        if left_current is None and previous_left_x is not None:
            # 급곡선에서 선이 직전 탐색 반경 밖으로 이동했으면 화면 왼쪽 절반에서
            # 다시 잡는다. 이후 RANSAC/강한 한쪽 선 검사가 오검출을 걸러낸다.
            left_current = self._histogram_peak(histogram, 0, midpoint)
        right_current = self._histogram_peak(
            histogram,
            0 if previous_right_x is not None else midpoint,
            binary.shape[1],
            expected_x=previous_right_x,
        )
        if right_current is None and previous_right_x is not None:
            right_current = self._histogram_peak(
                histogram, midpoint, binary.shape[1],
            )

        search_opposite = bool(self.config.search_opposite_in_single_mode)
        expected_width = self._expected_width()
        opposite_margin = int(self.config.opposite_window_margin)
        if tracking_mode == 'left':
            if left_current is not None and search_opposite:
                right_current = self._histogram_peak(
                    histogram, 0, binary.shape[1],
                    expected_x=left_current + expected_width,
                    margin=opposite_margin,
                )
            else:
                right_current = None
        elif tracking_mode == 'right':
            if right_current is not None and search_opposite:
                left_current = self._histogram_peak(
                    histogram, 0, binary.shape[1],
                    expected_x=right_current - expected_width,
                    margin=opposite_margin,
                )
            else:
                left_current = None

        nonzero_y, nonzero_x = binary.nonzero()
        left_indices = []
        right_indices = []
        window_height = max(1, binary.shape[0] // self.config.windows)
        margin = int(self.config.window_margin)
        left_found = right_found = None
        left_shift = right_shift = 0.0
        left_connected = right_connected = False
        empty = np.array([], dtype=int)

        for window in range(self.config.windows):
            y_low = binary.shape[0] - (window + 1) * window_height
            y_high = binary.shape[0] - window * window_height

            def pixels_in_window(current, radius):
                if current is None:
                    return empty, None, None
                low = max(0, int(current) - radius)
                high = min(binary.shape[1], int(current) + radius)
                indices = np.flatnonzero(
                    (nonzero_y >= y_low) & (nonzero_y < y_high)
                    & (nonzero_x >= low) & (nonzero_x < high)
                )
                return indices, low, high

            left_good, left_low, left_high = pixels_in_window(left_current, margin)
            right_radius = (
                opposite_margin if tracking_mode == 'left' else margin
            )
            right_good, right_low, right_high = pixels_in_window(right_current, right_radius)
            if tracking_mode == 'right':
                left_good, left_low, left_high = pixels_in_window(left_current, opposite_margin)

            # A line appearing only in the upper image is usually an arrow,
            # text or crosswalk edge.  A valid track must connect to one of the
            # configured bottom windows first.
            if window >= int(self.config.bottom_tracking_windows):
                if not left_connected:
                    left_good = empty
                if not right_connected:
                    right_good = empty
            if window < int(self.config.bottom_tracking_windows):
                left_connected |= left_good.size >= self.config.recenter_min_pixels
                right_connected |= right_good.size >= self.config.recenter_min_pixels

            # Reject an impossible pair before its pixels enter RANSAC.  In an
            # anchored mode the opposite side is always the disposable one;
            # AUTO/BOTH keep the stronger candidate.
            if left_good.size and right_good.size:
                left_mean = float(np.mean(nonzero_x[left_good]))
                right_mean = float(np.mean(nonzero_x[right_good]))
                gap = right_mean - left_mean
                minimum = expected_width * (1.0 - self.config.lane_width_tolerance)
                # Apply only the minimum here.  A sharp curve can make the
                # horizontal image gap look wider even when perpendicular lane
                # width is correct; the full geometry check handles the upper
                # bound later.
                if gap < minimum:
                    if tracking_mode == 'left':
                        right_good = empty
                    elif tracking_mode == 'right':
                        left_good = empty
                    elif left_good.size >= right_good.size:
                        right_good = empty
                    else:
                        left_good = empty

            left_indices.append(left_good)
            right_indices.append(right_good)

            # Predict the next window from the most recent lateral shift.  The
            # shift is bounded by the search margin so a single marking cannot
            # pull the tracker across the image.
            if left_current is not None:
                if left_good.size >= self.config.recenter_min_pixels:
                    found = float(np.mean(nonzero_x[left_good]))
                    if left_found is not None:
                        left_shift = float(np.clip(found - left_found, -margin, margin))
                    left_found = found
                    left_current = int(found + left_shift)
                else:
                    left_shift *= 0.5
                    left_current = int(left_current + left_shift)
            if right_current is not None:
                if right_good.size >= self.config.recenter_min_pixels:
                    found = float(np.mean(nonzero_x[right_good]))
                    if right_found is not None:
                        right_shift = float(np.clip(found - right_found, -margin, margin))
                    right_found = found
                    right_current = int(found + right_shift)
                else:
                    right_shift *= 0.5
                    right_current = int(right_current + right_shift)

            if left_low is not None:
                cv2.rectangle(debug, (left_low, y_low), (left_high, y_high), (255, 200, 0), 1)
            if right_low is not None:
                cv2.rectangle(debug, (right_low, y_low), (right_high, y_high), (255, 0, 255), 1)

        left_indices = np.concatenate(left_indices) if left_indices else empty
        right_indices = np.concatenate(right_indices) if right_indices else empty
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
        """Infer a center from the measured line without refitting another curve.

        The former normal-offset/re-fit can fold at a tight bend and cross the
        measured line. A constant BEV lane-width margin preserves
        the measured polynomial's shape and the boundary ordering at every y.
        The returned opposite fit is only the mirror of that center, not an
        independent observation.
        """
        gap = self._expected_width()
        other = fit.copy()
        if side == 'left':
            other[-1] += gap
            return fit, other
        other[-1] -= gap
        return other, fit

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

    def _strong_single_line(
        self, ratio: float, ys: np.ndarray, point_count: int,
    ) -> bool:
        """이전 중심과 달라도 독립적으로 믿을 수 있는 한쪽 선인지 확인한다."""
        minimum_points = (
            self.config.fit_min_points * self.config.strong_single_point_multiplier
        )
        return bool(
            ratio >= self.config.strong_single_inlier_ratio
            and point_count >= minimum_points
            and self._coverage_ok(ys)
        )

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

    def _left_curve_window_target(self, left_y, left_x, right_fit, right_y):
        """Use measured near tangent and far bend, not the whole-lane cubic fit.

        The near pair defines the current straight/tangent and a normal-offset
        lane center. Only excess far displacement beyond that tangent creates
        right-turn preview. Zigzag detection is intentionally unchanged.
        """
        if self.ground is None or self.config.windows < 6:
            return None
        step = self.config.height / self.config.windows
        def sample(index):
            mask = (left_y >= index * step) & (left_y < (index + 1) * step)
            if np.count_nonzero(mask) < max(8, self.config.recenter_min_pixels):
                return None
            return (float(np.median(left_x[mask])), float(np.median(left_y[mask])))

        def first_pair(candidates):
            for indices in candidates:
                pair = tuple(sample(index) for index in indices)
                if all(point is not None for point in pair):
                    return pair
            return None

        far_pair = first_pair(((1, 2), (2, 3)))
        near_pair = first_pair(((self.config.windows - 3, self.config.windows - 2),
                                (self.config.windows - 4, self.config.windows - 3)))
        if far_pair is None or near_pair is None:
            return None
        try:
            near_world = [self.ground.project_bev_point(*point) for point in near_pair]
            far_world = [self.ground.project_bev_point(*point) for point in far_pair]
        except ValueError:
            return None
        near_lateral, near_forward = np.mean(near_world, axis=0)
        far_lateral, far_forward = np.mean(far_world, axis=0)
        if (far_forward <= near_forward + .10
                or near_forward > self.config.curve_left_near_max_forward_m):
            return None
        near_slope = ((near_world[0][0] - near_world[1][0])
                      / (near_world[0][1] - near_world[1][1]))
        width_m = float(self.config.curve_lane_width_m)
        if not np.isfinite(near_slope) or width_m <= 0:
            return None
        normal = 0.5 * width_m / np.hypot(1.0, near_slope)
        center_lateral = float(near_lateral + normal)
        center_forward = float(near_forward - near_slope * normal)
        # A visible right boundary is clearance evidence, not proof that the
        # car touches it. Do not follow its polynomial as a steering target.
        right_points = self._yellow_curve_right_points
        near_left_x, near_row = np.mean(near_pair, axis=0)
        if right_points is None and self._yellow_curve_binary is not None:
            row = int(np.clip(round(near_row), 0, self.config.height - 1))
            strip = self._yellow_curve_binary[max(0, row - 5):row + 6]
            candidates = np.flatnonzero(np.count_nonzero(strip, axis=0) >= 3)
            candidates = candidates[candidates > near_left_x + .4 * self._expected_width()]
            if candidates.size >= 5:
                # A same-row yellow cluster is enough for a clearance check;
                # the right sliding tracker need not follow the whole arc.
                right_points = (np.full(candidates.size, near_row), candidates)
        if right_points is None and right_y.size:
            right_points = (right_y, np.polyval(right_fit, right_y))
        if right_points is not None:
            ys, xs = right_points
            mask = np.abs(ys - near_row) < step
            if np.count_nonzero(mask) >= max(8, self.config.recenter_min_pixels):
                try:
                    right_lateral, _ = self.ground.project_bev_point(
                        float(np.median(xs[mask])), float(np.median(ys[mask])))
                except ValueError:
                    return None
                if right_lateral - center_lateral < self.config.curve_right_min_clearance_m:
                    self._curve_right_clearance_failed = True
                    return None
        try:
            center_bev = self.ground.project_ground_point(center_lateral, center_forward)
        except ValueError:
            return None
        if not self._target_within_bev(*center_bev):
            return None
        delta_forward = far_forward - near_forward
        far_excess = far_lateral - (near_lateral + near_slope * delta_forward)
        # A corner that starts after a straight has positive excess. On exit,
        # excess falls to zero, so the preview releases instead of bending the
        # entry straight with a whole-lane high-order polynomial.
        preview = float(np.arctan2(max(0.0, far_excess), delta_forward))
        return center_lateral, center_forward, center_bev, preview

    def process(self, image: np.ndarray, tracking_mode: str = 'auto') -> LaneResult:
        self._curve_right_clearance_failed = False
        tracking_mode = tracking_mode if tracking_mode in ('auto', 'both', 'left', 'right') else 'auto'
        if image is None or image.size == 0:
            return LaneResult(
                valid=False, observation_state='rejected',
                reason='empty image', tracking_mode=tracking_mode,
            )

        resized = cv2.resize(image, (self.config.width, self.config.height))
        bird = cv2.warpPerspective(
            resized, self._matrix, (self.config.width, self.config.height),
        )
        bird, illumination_gain = compensate_light(
            bird, self.config.low_light_enabled, self.config.low_light_max_gain)
        binary, color_binary, gradient_binary = self._binary_masks(bird)
        yellow_curve_binary = self._yellow_curve_mask(bird, tracking_mode)
        yellow_curve_priority_active = yellow_curve_binary is not None
        if yellow_curve_priority_active:
            binary = color_binary = yellow_curve_binary
            gradient_binary = np.zeros_like(binary)
        # Independent yellow anchors were checked above; AUTO must not split
        # the inner curve across the image midpoint or select the outer edge.
        search_mode = (self._yellow_curve_side if yellow_curve_priority_active
                       and tracking_mode in ('auto', 'both') and self._yellow_curve_side
                       else tracking_mode)
        left_y, left_x, right_y, right_x, debug = self._sliding_window_points(
            binary,
            tracking_mode=search_mode,
            color_binary=color_binary,
            gradient_binary=gradient_binary,
        )

        left_fit, left_ratio = self._ransac_quadratic(left_y, left_x)
        right_fit, right_ratio = self._ransac_quadratic(right_y, right_x)
        # Show the actual camera-space ROI even when geometry validation fails.
        roi_view = resized.copy()
        cv2.polylines(roi_view, [np.int32(self._normalized_points(self.config.src_points))],
                      True, (0, 255, 0), 2)
        thumb = cv2.resize(roi_view, (160, 120))
        if debug.shape[0] >= 120 and debug.shape[1] >= 160:
            debug[-120:, -160:] = thumb

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
            side = (
                search_mode if search_mode in ('left', 'right')
                else self._assign_side(merged_fit, fallback='left')
            )
            empty = np.array([], dtype=merged_x.dtype)
            if side == 'left':
                left_fit, left_ratio, left_y, left_x = merged_fit, merged_ratio, merged_y, merged_x
                right_fit, right_ratio, right_y, right_x = None, 0.0, empty, empty
            else:
                right_fit, right_ratio, right_y, right_x = merged_fit, merged_ratio, merged_y, merged_x
                left_fit, left_ratio, left_y, left_x = None, 0.0, empty, empty

        if left_fit is None and right_fit is None:
            return LaneResult(
                valid=False, observation_state='rejected',
                debug_image=debug, bev_image=bird, binary_image=binary,
                color_binary_image=color_binary,
                gradient_binary_image=gradient_binary,
                reason='not enough lane pixels', tracking_mode=tracking_mode,
                yellow_curve_priority_active=yellow_curve_priority_active,
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
                for side, fit, ratio, ys, points in (
                        ('left', left_fit, left_ratio, left_y, left_x.size),
                        ('right', right_fit, right_ratio, right_y, right_x.size)):
                    if search_mode in ('left', 'right') and side != search_mode:
                        continue
                    # 화면 끝에 조금 걸친 짧은 조각은 믿지 않는다
                    if not self._coverage_ok(ys):
                        continue
                    cand_left, cand_right = self._single_side(fit, side)
                    # 직접 만든 반대쪽 선은 폭이 맞게 만들어졌으므로 '급변'만 검사한다
                    if (not self._jump_ok(cand_left, cand_right)
                            and not self._strong_single_line(ratio, ys, points)):
                        continue
                    center = 0.5 * (np.polyval(cand_left, y_eval) + np.polyval(cand_right, y_eval))
                    jump = 0.0 if previous_center is None else abs(center - previous_center)
                    candidates.append((jump, -points, side, cand_left, cand_right, self._expected_width()))
                if not candidates:
                    return LaneResult(
                        valid=False, observation_state='rejected',
                        debug_image=debug, bev_image=bird, binary_image=binary,
                        color_binary_image=color_binary,
                        gradient_binary_image=gradient_binary,
                        reason='lane geometry sanity check failed',
                        tracking_mode=tracking_mode,
                        yellow_curve_priority_active=yellow_curve_priority_active,
                    )
                candidates.sort(key=lambda item: (item[0], item[1]))
                _, _, recovered_side, left_fit, right_fit, mean_width = candidates[0]
        else:
            # [2] 선이 하나만 보이면, 그게 왼쪽인지 오른쪽인지 판단해서 반대쪽을 만든다.
            only_fit = left_fit if left_fit is not None else right_fit
            only_ratio = left_ratio if left_fit is not None else right_ratio
            only_y = left_y if left_fit is not None else right_y
            only_x = left_x if left_fit is not None else right_x
            only_points = only_x.size
            fallback = 'left' if left_fit is not None else 'right'
            recovered_side = (
                search_mode if search_mode in ('left', 'right')
                else self._assign_side(only_fit, fallback)
            )
            # A line can be reassigned across the image midpoint. Move its
            # observations with it before discarding the inferred side below.
            if recovered_side == 'left':
                left_y, left_x, left_ratio = only_y, only_x, only_ratio
            else:
                right_y, right_x, right_ratio = only_y, only_x, only_ratio
            left_fit, right_fit = self._single_side(only_fit, recovered_side)
            mean_width = self._expected_width()
            if (not self._jump_ok(left_fit, right_fit)
                    and not self._strong_single_line(only_ratio, only_y, only_points)):
                return LaneResult(
                    valid=False, observation_state='rejected',
                    debug_image=debug, bev_image=bird, binary_image=binary,
                    color_binary_image=color_binary,
                    gradient_binary_image=gradient_binary,
                    reason='lane geometry sanity check failed',
                    tracking_mode=tracking_mode,
                    yellow_curve_priority_active=yellow_curve_priority_active,
                )

        recovered = bool(recovered_side)
        if recovered:
            # 한쪽 복구로 '만든' 선으로 폭을 다시 재면 매 프레임 조금씩 줄어든다.
            # 실제 두 선을 본 마지막 폭을 그대로 유지한다.
            mean_width = self._expected_width()
            # 실제로 보고 간 선의 픽셀/신뢰도만 남긴다
            if recovered_side == 'left':
                right_ratio = 0.0
                right_y = np.array([], dtype=left_y.dtype)
                right_x = np.array([], dtype=left_x.dtype)
            elif recovered_side == 'right':
                left_ratio = 0.0
                left_y = np.array([], dtype=right_y.dtype)
                left_x = np.array([], dtype=right_x.dtype)

        measured_left_fit, measured_right_fit = left_fit, right_fit
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
        if not np.isfinite(center_x):
            self.reset()
            return LaneResult(valid=False, observation_state='accepted',
                              path_state='rejected', debug_image=debug, bev_image=bird,
                              binary_image=binary, color_binary_image=color_binary,
                              gradient_binary_image=gradient_binary,
                              reason='center target is not finite', tracking_mode=tracking_mode,
                              yellow_curve_priority_active=yellow_curve_priority_active)
        lateral_error = float(np.clip(
            (center_x - self.config.width * 0.5) / (self.config.width * 0.5), -1.5, 1.5,
        ))
        center_fit = 0.5 * (left_fit + right_fit)
        derivative = float(np.polyval(np.polyder(center_fit), y_eval))
        heading_error = float(np.arctan(-derivative))
        observed_ys = np.concatenate((left_y, right_y))
        left_type = classify_lane(bird, measured_left_fit, left_y, self.config, zigzag)
        right_type = classify_lane(bird, measured_right_fit, right_y, self.config, zigzag)
        curve_span = max(
            measured_heading_span(measured_left_fit, left_y, self.config.height),
            measured_heading_span(measured_right_fit, right_y, self.config.height))
        curve_turn = 'unknown'
        for fit, ys, lane_type in ((measured_right_fit, right_y, right_type),
                                   (measured_left_fit, left_y, left_type)):
            if fit is None or ys.size < 2 or lane_type != 'yellow/solid/curve':
                continue
            displacement = float(np.polyval(fit, np.min(ys)) - np.polyval(fit, np.max(ys)))
            if abs(displacement) >= 20.0 and curve_span >= self.config.yellow_curve_min_span:
                curve_turn = 'right' if displacement > 0 else 'left'
                break
        if recovered:
            observed_x = left_x if recovered_side == 'left' else right_x
            observed_y = left_y if recovered_side == 'left' else right_y
            point_score = min(1.0, observed_x.size / (self.config.fit_min_points * 3.0))
            inlier_score = max(left_ratio, right_ratio)
            coverage_score = (min(1.0, float(np.ptp(observed_y)) / (self.config.height * 0.65))
                              if observed_y.size else 0.0)
            confidence = float(np.clip(
                0.30 * point_score + 0.50 * inlier_score + 0.20 * coverage_score, 0.0, 1.0))
        else:
            point_score = min(1.0, (left_x.size + right_x.size)
                              / (2.0 * self.config.fit_min_points * 3.0))
            inlier_score = 0.5 * (left_ratio + right_ratio)
            confidence = float(np.clip(
                0.35 * point_score + 0.40 * inlier_score + 0.25, 0.0, 1.0))

        left_curve_target = (
            self._left_curve_window_target(left_y, left_x, right_fit, right_y)
            if (self.ground is not None and yellow_curve_priority_active and not zigzag
                and curve_turn == 'right' and left_type == 'yellow/solid/curve'
                and confidence >= 0.70) else None)
        if self._curve_right_clearance_failed:
            return LaneResult(valid=False, observation_state='accepted',
                              policy_state='blocked', debug_image=debug, bev_image=bird,
                              binary_image=binary, color_binary_image=color_binary,
                              gradient_binary_image=gradient_binary,
                              reason='right curve boundary clearance too small',
                              tracking_mode=tracking_mode, confidence=confidence,
                              curve_heading_span=curve_span, curve_turn=curve_turn,
                              left_type=left_type, right_type=right_type,
                              yellow_curve_priority_active=yellow_curve_priority_active)
        observed_nearest = False
        fixed_row_missing = recovered and not (
            float(observed_ys.min()) <= y_eval <= float(observed_ys.max()))
        curve_recovery_allowed = bool(
            self.ground is not None and yellow_curve_priority_active
            and curve_turn == 'right' and left_type == 'yellow/solid/curve'
            and not zigzag and confidence >= 0.70)
        inner_only_clockwise = (
            tracking_mode in ('auto', 'both') and curve_turn == 'right'
            and right_type == 'yellow/solid/curve'
            and left_type != 'yellow/solid/curve')
        if (yellow_curve_priority_active and self._yellow_curve_outer_missing
                or inner_only_clockwise):
            return LaneResult(valid=False, observation_state='accepted',
                              policy_state='blocked', debug_image=debug, bev_image=bird,
                              binary_image=binary, color_binary_image=color_binary,
                              gradient_binary_image=gradient_binary,
                              reason='outer left curve not observed',
                              tracking_mode=tracking_mode, confidence=confidence,
                              curve_heading_span=curve_span, curve_turn=curve_turn,
                              left_type=left_type, right_type=right_type,
                              yellow_curve_priority_active=True)
        if fixed_row_missing and left_curve_target is None and not curve_recovery_allowed:
            self.reset()
            return LaneResult(valid=False, observation_state='accepted',
                              path_state='rejected', debug_image=debug, bev_image=bird,
                              binary_image=binary, color_binary_image=color_binary,
                              gradient_binary_image=gradient_binary,
                              reason='one-side target outside observed lane rows',
                              tracking_mode=tracking_mode, confidence=confidence,
                              curve_heading_span=curve_span, curve_turn=curve_turn,
                              left_type=left_type, right_type=right_type,
                              yellow_curve_priority_active=yellow_curve_priority_active)
        preview_y = float(np.clip(self.config.height * self.config.preview_y_ratio,
                                 observed_ys.min(), observed_ys.max()))
        preview_heading = float(np.arctan(-np.polyval(np.polyder(center_fit), preview_y)))
        lookahead_description = f'uncalibrated y={y_eval:.0f}px'
        target_lateral_m = None
        target_forward_m = None
        if self.ground is not None:
            try:
                if left_curve_target is not None:
                    target_lateral_m, target_forward_m, (center_x, y_eval), _ = left_curve_target
                    heading_error = float(np.arctan(-np.polyval(np.polyder(center_fit), y_eval)))
                elif fixed_row_missing:
                    raise ValueError('fixed target outside observed curve rows')
                else:
                    target_lateral_m, heading_error, y_eval, target_forward_m = (
                        self.ground.evaluate_target(center_fit, observed_ys.min(), observed_ys.max()))
            except ValueError as error:
                if curve_recovery_allowed:
                    try:
                        target_lateral_m, heading_error, y_eval, target_forward_m = (
                            self.ground.evaluate_nearest_observed(
                                center_fit, observed_ys.min(), observed_ys.max(),
                                self.config.curve_nearest_max_forward_m))
                        observed_nearest = True
                    except ValueError as recovery_error:
                        error = recovery_error
                if target_lateral_m is None:
                    return LaneResult(valid=False, observation_state='accepted',
                                      path_state='rejected', debug_image=debug, bev_image=bird,
                                      binary_image=binary, color_binary_image=color_binary,
                                      gradient_binary_image=gradient_binary, reason=str(error),
                                      confidence=confidence, curve_heading_span=curve_span,
                                      curve_turn=curve_turn, left_type=left_type,
                                      right_type=right_type,
                                      yellow_curve_priority_active=yellow_curve_priority_active)
            lateral_error = float(np.clip(
                target_lateral_m / self.config.metric_lateral_scale_m, -1.5, 1.5))
            if left_curve_target is None:
                center_x = float(np.polyval(center_fit, y_eval))
            preview_heading = heading_error  # Do not mix image and metric angles.
            lookahead_source = ("left curve near-window center" if left_curve_target is not None
                                else "nearest observed curve" if observed_nearest
                                else f'{self.config.lookahead_distance_m:.2f}m requested')
            lookahead_description = (
                f'{lookahead_source}, {target_forward_m:.2f}m used from rear axle y={y_eval:.0f}px')

        # A polynomial can have many inliers yet extrapolate the driving
        # target outside the measured BEV road area (seen at the fork in the
        # manual recording). Confidence alone must not make it driveable.
        if not self._target_within_bev(center_x, y_eval):
            self.reset()
            return LaneResult(valid=False, observation_state='accepted',
                              path_state='rejected', debug_image=debug, bev_image=bird,
                              binary_image=binary, color_binary_image=color_binary,
                              gradient_binary_image=gradient_binary,
                              reason='center target outside measured BEV road area',
                              tracking_mode=tracking_mode, confidence=confidence,
                              curve_heading_span=curve_span, curve_turn=curve_turn,
                              left_type=left_type, right_type=right_type,
                              yellow_curve_priority_active=yellow_curve_priority_active)

        # Keep an observed-range metric path for optional bounded recovery.
        # This is diagnostic data only until the odometry feature is enabled.
        ground_path = []
        if self.ground is not None:
            for row in np.linspace(float(observed_ys.max()), float(observed_ys.min()), 24):
                x = float(np.polyval(center_fit, row))
                if not self._target_within_bev(x, row):
                    continue
                try:
                    lateral, forward = self.ground.project_bev_point(x, row)
                except ValueError:
                    continue
                if (forward <= self.config.curve_nearest_max_forward_m
                        and (not ground_path or forward > ground_path[-1][1] + 1e-4)):
                    ground_path.append((lateral, forward))

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
        cv2.putText(
            debug,
            f'tracking={tracking_mode}',
            (10, 109), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 255, 160), 2,
        )

        cv2.putText(debug, f'L: {left_type}', (10, 136),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(debug, f'R: {right_type}', (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(debug, f'lookahead: {lookahead_description}', (10, 184),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.line(debug, (0, int(y_eval)), (self.config.width - 1, int(y_eval)), (0, 255, 0), 1)
        target_camera = cv2.perspectiveTransform(
            np.float32([[[center_x, y_eval]]]), self._inverse_matrix)[0, 0]
        if np.all(np.isfinite(target_camera)):
            cv2.circle(roi_view, tuple(np.int32(target_camera)), 6, (0, 0, 255), -1)
            if debug.shape[0] >= 120 and debug.shape[1] >= 160:
                debug[-120:, -160:] = cv2.resize(roi_view, (160, 120))
        return LaneResult(
            valid=True,
            observation_state='accepted',
            path_state='accepted',
            policy_state='allowed',
            curve_heading_span=curve_span,
            curve_turn=curve_turn,
            curve_nearest_observation=observed_nearest,
            curve_left_target_active=left_curve_target is not None,
            curve_left_preview_rad=(left_curve_target[3] if left_curve_target is not None else None),
            ground_path=tuple(ground_path),
            preview_heading_error=preview_heading,
            lookahead_description=lookahead_description,
            target_lateral_m=target_lateral_m,
            target_forward_m=target_forward_m,
            yellow_curve_priority_active=yellow_curve_priority_active,
            illumination_gain=illumination_gain,
            left_type=left_type,
            right_type=right_type,
            lateral_error=lateral_error,
            heading_error=heading_error,
            confidence=confidence,
            zigzag=zigzag,
            zigzag_score=zigzag_score,
            direction_changes=direction_changes,
            left_fit=left_fit,
            right_fit=right_fit,
            debug_image=debug,
            bev_image=bird,
            target_bev_xy=(float(center_x), float(y_eval)),
            binary_image=binary,
            color_binary_image=color_binary,
            gradient_binary_image=gradient_binary,
            reason='one-side recovery' if recovered else 'both lanes detected',
            recovered_side=recovered_side,
            tracking_mode=tracking_mode,
        )
