"""Steering helpers independent of ROS; angles remain limited by the node."""

import math


def reference_side(result):
    return result.recovered_side if result.recovered_side in ('left', 'right') else 'both'


def tracking_error(result, lateral_gain, heading_gain, left_gain, right_gain):
    # The pipeline already offsets the observed boundary along its normal to
    # reconstruct a center path. Do not replace it with a horizontal half-width.
    error = lateral_gain * result.lateral_error + heading_gain * result.heading_error
    side = reference_side(result)
    gain = left_gain if side == 'left' else right_gain if side == 'right' else 1.0
    gain = gain if math.isfinite(gain) and gain > 0.0 else 1.0
    return error * gain


def curve_release_target(magnitude, enter, exit_error, peak):
    """Blend from peak turn to ordinary demand as error falls through hysteresis."""
    span = max(1e-6, enter - exit_error)
    t = min(1.0, max(0.0, (magnitude - exit_error) / span))
    smooth = t * t * (3.0 - 2.0 * t)
    # Never reduce a larger raw demand simply because peak was configured lower.
    return min(1.0, magnitude + (max(magnitude, peak) - magnitude) * smooth)


def limited_steering(previous, target, dt, attack, release, reversal):
    reversing = previous * target < 0.0
    increasing = not reversing and abs(target) > abs(previous)
    rate = reversal if reversing else attack if increasing else release
    step = max(0.0, rate) * min(0.20, max(0.0, dt))
    return min(1.0, max(-1.0, previous + min(step, max(-step, target - previous))))
