"""Signed ground curvature is local, not the line's total image displacement."""

import numpy as np

from jetracer_ros2.lane_geometry import signed_curvature


def _path(lateral):
    forward = np.linspace(0.60, 1.45, 30)
    return tuple(zip(lateral(forward), forward))


def test_tilted_straight_has_zero_curvature():
    path = _path(lambda s: 0.2 * s - 0.1)
    assert abs(signed_curvature(path)) < 1e-10


def test_right_and_left_bends_have_opposite_sign():
    right = _path(lambda s: 0.30 * (s - .6) ** 2)
    left = _path(lambda s: -0.30 * (s - .6) ** 2)
    assert signed_curvature(right) > .4
    assert signed_curvature(left) < -.4


def test_s_bend_can_reverse_local_sign_without_changing_path():
    path = _path(lambda s: (s - 1.0) ** 3)
    assert signed_curvature(path, at_fraction=.20) < 0
    assert signed_curvature(path, at_fraction=.80) > 0


def test_unordered_or_short_path_is_not_classified():
    assert signed_curvature(((0, .6), (0, .7))) is None
    path = list(_path(lambda s: 0 * s))
    path[5] = (0, path[4][1])
    assert signed_curvature(path) is None
