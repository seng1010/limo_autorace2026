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
    def __init__(self, bird_corners, ground_points, offset, distance, lateral_scale,
                 max_near_clamp_m=0.0, edge_margin_m=0.0):
        ground = np.asarray(ground_points, np.float32).reshape(4, 2)
        if (not np.all(np.isfinite(ground)) or not cv2.isContourConvex(ground)
                or abs(cv2.contourArea(ground)) < 1e-4):
            raise ValueError('metric lookahead requires four measured, convex ground points')
        if (not np.all(np.isfinite([distance, lateral_scale, offset]))
                or distance <= 0 or lateral_scale <= 0):
            raise ValueError('invalid metric lookahead distance/scale/offset')
        if (not np.all(np.isfinite([max_near_clamp_m, edge_margin_m]))
                or max_near_clamp_m < 0 or edge_margin_m < 0):
            raise ValueError('invalid metric edge allowance')
        self.matrix = cv2.getPerspectiveTransform(np.float32(bird_corners), ground)
        self.inverse_matrix = cv2.getPerspectiveTransform(ground, np.float32(bird_corners))
        self.offset, self.distance, self.scale = offset, distance, lateral_scale
        self.max_near_clamp_m = max_near_clamp_m
        self.edge_margin_m = edge_margin_m

    def project_bev_point(self, x, y):
        """Map an observed BEV point to lateral/forward metres from rear axle."""
        if not np.all(np.isfinite([x, y])):
            raise ValueError('non-finite BEV point')
        lateral, forward = cv2.perspectiveTransform(
            np.float32([[[x, y]]]), self.matrix)[0, 0]
        forward = float(forward) + self.offset
        if not np.all(np.isfinite([lateral, forward])) or forward <= 0:
            raise ValueError('BEV point outside forward ground range')
        return float(lateral), forward

    def project_ground_point(self, lateral, forward_from_rear):
        """Project a rear-axle ground target into BEV for the existing ROI guard."""
        if not np.all(np.isfinite([lateral, forward_from_rear])):
            raise ValueError('non-finite ground point')
        point = np.float32([[[lateral, forward_from_rear - self.offset]]])
        x, y = cv2.perspectiveTransform(point, self.inverse_matrix)[0, 0]
        if not np.all(np.isfinite([x, y])):
            raise ValueError('ground target cannot be projected into BEV')
        return float(x), float(y)

    def evaluate(self, fit, lo, hi):
        x, heading, y, _ = self.evaluate_target(fit, lo, hi)
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
        observed = (f'requested={self.distance:.2f}m, '
                    f'observed={forward[0]:.2f}..{forward[-1]:.2f}m, '
                    f'near_allowance={self.max_near_clamp_m:.2f}m')
        if self.distance > forward[-1]:
            raise ValueError(f'lookahead distance outside observed ground range ({observed})')
        effective = max(self.distance, float(forward[0]) + self.edge_margin_m)
        if (effective > forward[-1]
                or effective - self.distance > self.max_near_clamp_m + 1e-8):
            raise ValueError(f'lookahead too close to or outside observed ground range ({observed})')
        x = float(np.interp(effective, forward, lateral))
        slope = float(np.interp(effective, forward, np.gradient(lateral, forward)))
        y = float(np.interp(effective, forward, ys[::-1]))
        return x, float(np.arctan(slope)), y, effective

    def evaluate_nearest_observed(self, fit, lo, hi, max_forward_m):
        """Use the closest *measured* curve row, never a polynomial extension.

        This is a separate, bounded low-speed target for a confirmed curve;
        it does not relax the normal lookahead's near-edge allowance.
        """
        if not np.isfinite(max_forward_m) or max_forward_m <= 0:
            raise ValueError('invalid curve observation range')
        ys = np.linspace(lo, hi, 200)
        points = np.column_stack((np.polyval(fit, ys), ys)).astype(np.float32)
        world = cv2.perspectiveTransform(points[None], self.matrix)[0]
        world[:, 1] += self.offset
        if not np.all(np.isfinite(world)) or not np.all(np.diff(world[:, 1]) < 0):
            raise ValueError('ground calibration must map upper image to farther ground')
        forward, lateral = world[::-1, 1], world[::-1, 0]
        effective = float(forward[0]) + self.edge_margin_m
        if effective > min(float(forward[-1]), max_forward_m):
            raise ValueError('observed curve is farther than safe recovery range')
        x = float(np.interp(effective, forward, lateral))
        slope = float(np.interp(effective, forward, np.gradient(lateral, forward)))
        y = float(np.interp(effective, forward, ys[::-1]))
        return x, float(np.arctan(slope)), y, effective
