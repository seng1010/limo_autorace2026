"""Rear-axle pure-pursuit target to an inner-wheel Ackermann steering angle.

The image x-axis points right; ROS positive yaw/steering points left.
"""

import math


def metric_target_ready(lateral_m, forward_m):
    return (lateral_m is not None and forward_m is not None
            and math.isfinite(lateral_m) and math.isfinite(forward_m)
            and forward_m > 0)


def target_to_inner_angle(lateral_right_m, forward_from_rear_m,
                          wheelbase_m=0.20, track_m=0.172,
                          maximum_inner_angle_rad=0.42):
    values = (lateral_right_m, forward_from_rear_m, wheelbase_m,
              track_m, maximum_inner_angle_rad)
    if (not all(math.isfinite(value) for value in values)
            or forward_from_rear_m <= 0.0 or wheelbase_m <= 0.0
            or track_m < 0.0 or maximum_inner_angle_rad <= 0.0):
        raise ValueError('invalid Ackermann target or vehicle geometry')
    lateral = float(lateral_right_m)
    if abs(lateral) < 1e-9:
        return 0.0
    # Pure pursuit curvature is at the rear axle. Convert its center turning
    # radius to the inside front-wheel angle expected by limo_base.
    curvature = 2.0 * abs(lateral) / (
        forward_from_rear_m ** 2 + lateral ** 2)
    center_radius = 1.0 / curvature
    inside_radius = max(1e-6, center_radius - track_m / 2.0)
    angle = math.atan(wheelbase_m / inside_radius)
    return math.copysign(min(angle, maximum_inner_angle_rad), -lateral)
