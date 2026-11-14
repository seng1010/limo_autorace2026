"""Steering helpers independent of ROS; angles remain limited by the node."""

import math


def reference_side(result):
    return result.recovered_side if result.recovered_side in ('left', 'right') else 'both'


def limited_steering(previous, target, dt, attack, release, reversal):
    reversing = previous * target < 0.0
    increasing = not reversing and abs(target) > abs(previous)
    rate = reversal if reversing else attack if increasing else release
    step = max(0.0, rate) * min(0.20, max(0.0, dt))
    return min(1.0, max(-1.0, previous + min(step, max(-step, target - previous))))


def right_curve_metric_target(target, span, max_angle, minimum_angle, span_gain,
                              confirmed, enabled=True):
    """Keep a measured clockwise curve from commanding a counter-turn.

    Negative is right steering. The floor is applied only after consecutive
    confirmation; it never overrides the hardware's maximum steering angle.
    """
    if not math.isfinite(target) or not math.isfinite(span) or max_angle <= 0:
        return 0.0
    if not enabled:
        return target
    if not confirmed:
        return min(0.0, target)
    floor = min(max_angle, max(0.0, minimum_angle) + max(0.0, span_gain) * max(0.0, span))
    return max(-1.0, min(target, -floor / max_angle))


def left_curve_preview_target(target, preview_rad, max_angle, gain, max_preview_rad,
                              enabled=True):
    """Anticipate a measured right bend of the outer left line, with a hard cap."""
    if not enabled:
        return target
    if (not all(math.isfinite(v) for v in
                (target, preview_rad, max_angle, gain, max_preview_rad))
            or max_angle <= 0 or gain < 0 or max_preview_rad < 0
            or preview_rad <= 0):
        return target
    anticipation = min(max_preview_rad, gain * preview_rad)
    return max(-1.0, min(1.0, target - anticipation / max_angle))
