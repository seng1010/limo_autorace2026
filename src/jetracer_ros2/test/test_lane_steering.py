"""Test the steering helpers used by the live metric Pure Pursuit path."""

import ast
from pathlib import Path
from types import SimpleNamespace

from jetracer_ros2.lane_steering import (
    limited_steering, right_curve_metric_target, left_curve_preview_target,
)


def test_outer_left_curve_preview_is_bounded_and_never_applies_opposite_bend():
    assert left_curve_preview_target(0.0, .2, .42, .5, .16) < 0.0
    assert left_curve_preview_target(0.0, 10.0, .42, .5, .16) == -.16 / .42
    assert left_curve_preview_target(.1, -.2, .42, .5, .16) == .1
    assert left_curve_preview_target(-.9, 1., .42, .5, .16) == -1.0


def test_metric_right_curve_feedforward_never_countersteers_or_exceeds_limit():
    assert right_curve_metric_target(.4, .3, .42, .10, .20, False) == 0.0
    target = right_curve_metric_target(.4, .3, .42, .10, .20, True)
    assert -.42 < target < -.3
    assert right_curve_metric_target(-.9, .3, .42, .10, .20, True) == -.9
    assert right_curve_metric_target(.2, 10., .42, .10, .20, True) == -1.0


def controller():
    path = Path(__file__).parents[1] / 'jetracer_ros2' / 'lane_following_node.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    names = {'_rate_limit_steering', '_reset_steering'}
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(limited_steering=limited_steering)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), scope)
    params = dict(steering_attack_rate=4.0, steering_release_rate=1.2,
                  steering_reversal_rate=4.0)
    cls = type('Controller', (), {name: scope[name] for name in names})
    instance = cls()
    instance.get_parameter = lambda name: SimpleNamespace(value=params[name])
    instance._reset_steering()
    return instance


def test_release_rate_is_time_based():
    values = []
    for hz in (10, 20, 40):
        node = controller()
        node._applied_steering = 1.0
        node._last_steering_ns = 1_000_000_000
        for i in range(1, hz // 2 + 1):
            value = node._rate_limit_steering(0.0, int(1e9 + i * 1e9 / hz))
        values.append(value)
    assert all(abs(v - 0.4) < 1e-8 for v in values)


def test_reversal_and_reset_do_not_retain_old_turn():
    node = controller()
    node._applied_steering = 1.0
    node._last_steering_ns = 1_000_000_000
    for i in range(1, 7):
        value = node._rate_limit_steering(-1.0, int(1e9 + i * 0.05e9))
    assert value < 0.0
    node._reset_steering()
    assert node._applied_steering == 0.0
    assert node._last_valid_cmd is None
