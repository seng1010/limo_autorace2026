"""Ground-plane path measurements shared by perception and control experiments."""

import math

import numpy as np


def signed_curvature(points, at_fraction=0.3, window_fraction=0.5):
    """Return local curvature in 1/m; positive bends toward the vehicle's right.

    Points are (lateral-right, forward) in metres, ordered near to far. This
    measures a *validated observed path*, never an image-wide displacement or
    a polynomial extrapolation beyond the camera's visible rows.
    """
    path = np.asarray(points, dtype=float)
    if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 5
            or not np.all(np.isfinite(path)) or not 0 <= at_fraction <= 1
            or not 0 < window_fraction <= 1):
        return None
    lateral, forward = path[:, 0], path[:, 1]
    if np.any(np.diff(forward) <= 0) or forward[-1] - forward[0] < 0.15:
        return None
    span = forward[-1] - forward[0]
    center = forward[0] + at_fraction * span
    radius = max(0.075, 0.5 * window_fraction * span)
    selected = np.abs(forward - center) <= radius
    if np.count_nonzero(selected) < 5 or np.ptp(forward[selected]) < 0.12:
        return None
    local_forward = forward[selected] - center
    coefficients = np.polyfit(local_forward, lateral[selected], 2)
    derivative = coefficients[1]
    second_derivative = 2.0 * coefficients[0]
    curvature = second_derivative / (1.0 + derivative * derivative) ** 1.5
    return float(curvature) if math.isfinite(curvature) else None
