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
    white_lightness_min: int = 170
    white_saturation_max: int = 100
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
        lightness = hls[:, :, 1]
        saturation = hls[:, :, 2]

        white = (
            (lightness >= self.config.white_lightness_min)
            & (saturation <= self.config.white_saturation_max)
        )

        sobel = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
        absolute = np.abs(sobel)
        scaled = np.zeros_like(lightness)
        maximum = float(absolute.max())
        if maximum > 1e-6:
            scaled = np.uint8(np.clip(absolute * 255.0 / maximum, 0, 255))
        gradient = (scaled >= self.config.gradient_min) & (lightness >= 100)

        binary = np.uint8(white | gradient) * 255
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

            if left_good.size >= self.config.recenter_min_pixels:
                left_current = int(np.mean(nonzero_x[left_good]))
            if right_good.size >= self.config.recenter_min_pixels:
                right_current = int(np.mean(nonzero_x[right_good]))

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

    def _recover_missing_side(
        self,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        width = self._previous_width or self.config.lane_width_px
        recovered = False
        if left is not None and right is None:
            right = left.copy()
            right[-1] += width
            recovered = True
        elif right is not None and left is None:
            left = right.copy()
            left[-1] -= width
            recovered = True
        return left, right, recovered

    def _sanity_check(self, left: np.ndarray, right: np.ndarray) -> Tuple[bool, float]:
        sample_y = np.asarray([
            self.config.height * 0.35,
            self.config.height * self.config.lookahead_y_ratio,
            self.config.height * 0.90,
        ])
        widths = np.polyval(right, sample_y) - np.polyval(left, sample_y)
        expected = self._previous_width or self.config.lane_width_px
        lower = expected * (1.0 - self.config.lane_width_tolerance)
        upper = expected * (1.0 + self.config.lane_width_tolerance)
        if np.any(widths < lower) or np.any(widths > upper):
            return False, float(np.mean(widths))
        if float(np.ptp(widths)) > self.config.max_lane_width_change_px:
            return False, float(np.mean(widths))

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
        observed_sides = int(left_fit is not None) + int(right_fit is not None)
        left_fit, right_fit, recovered = self._recover_missing_side(left_fit, right_fit)

        if left_fit is None or right_fit is None:
            return LaneResult(
                valid=False, debug_image=debug, binary_image=binary,
                reason='not enough lane pixels',
            )

        sane, mean_width = self._sanity_check(left_fit, right_fit)
        if not sane:
            return LaneResult(
                valid=False, debug_image=debug, binary_image=binary,
                reason='lane geometry sanity check failed',
            )

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
                debug, 'ONE-SIDE RECOVERY', (10, 55),
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
        )
