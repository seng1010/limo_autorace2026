"""Bounded illumination compensation and measured ground-plane lookahead."""
import cv2
import numpy as np


def compensate_light(image, enabled=True, max_gain=3.0):
    if not enabled:
        return image, 1.0
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    light = hls[:, :, 1]
    values = light[light > 8]  # Ignore black warp padding and near-black noise.
    if values.size < 32:
        return image, 1.0
    reference = float(np.percentile(values, 95))
    gain = float(np.clip(180.0 / max(reference, 1.0), 1.0, max(1.0, max_gain)))
    hls[:, :, 1] = np.uint8(np.clip(light.astype(float) * gain, 0, 255))
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR), gain


class GroundLookahead:
    """Ground coordinates: x right, y forward from camera ground projection.

    Four measured ground points correspond to the four bev_src corners. Offset
    is the camera's forward displacement from the rear axle (not viewing range).
    """
    def __init__(self, bird_corners, ground_points, offset, distance, lateral_scale):
        ground = np.asarray(ground_points, np.float32).reshape(4, 2)
        if (not np.all(np.isfinite(ground)) or not cv2.isContourConvex(ground)
                or abs(cv2.contourArea(ground)) < 1e-4):
            raise ValueError('metric lookahead requires four measured, convex ground points')
        if (not np.all(np.isfinite([distance, lateral_scale, offset]))
                or distance <= 0 or lateral_scale <= 0):
            raise ValueError('invalid metric lookahead distance/scale/offset')
        self.matrix = cv2.getPerspectiveTransform(np.float32(bird_corners), ground)
        self.offset, self.distance, self.scale = offset, distance, lateral_scale

    def evaluate(self, fit, lo, hi):
        x, heading, y = self.evaluate_target(fit, lo, hi)
        return float(np.clip(x / self.scale, -1.5, 1.5)), heading, y

    def evaluate_target(self, fit, lo, hi):
        """Return metric lateral target from the rear axle, heading and BEV row."""
        ys = np.linspace(lo, hi, 200)
        points = np.column_stack((np.polyval(fit, ys), ys)).astype(np.float32)
        world = cv2.perspectiveTransform(points[None], self.matrix)[0]
        world[:, 1] += self.offset
        if not np.all(np.isfinite(world)) or not np.all(np.diff(world[:, 1]) < 0):
            raise ValueError('ground calibration must map upper image to farther ground')
        forward, lateral = world[::-1, 1], world[::-1, 0]
        if not forward[0] <= self.distance <= forward[-1]:
            raise ValueError('lookahead distance outside observed ground range')
        x = float(np.interp(self.distance, forward, lateral))
        slope = float(np.interp(self.distance, forward, np.gradient(lateral, forward)))
        y = float(np.interp(self.distance, forward, ys[::-1]))
        return x, float(np.arctan(slope)), y
