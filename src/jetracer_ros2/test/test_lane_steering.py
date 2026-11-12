"""Exercise real node steering methods without requiring a ROS installation."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jetracer_ros2.lane_steering import (
    tracking_error, curve_release_target, limited_steering,
    right_curve_metric_target,
)


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
    names = {'_curve_steering_target', '_rate_limit_steering', '_mode_error', '_reset_steering',
             '_update_curve_geometry'}
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(np=np, tracking_error=tracking_error,
                 curve_release_target=curve_release_target, limited_steering=limited_steering)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), scope)
    params = dict(curve_enter_error=0.55, curve_exit_error=0.25,
                  curve_enter_frames=2, curve_max_steering_ratio=1.0,
                  curve_shape_min_span=0.20, curve_shape_full_span=0.40,
                  curve_shape_confirm_frames=2,
                  steering_attack_rate=4.0, steering_release_rate=1.2,
                  steering_reversal_rate=4.0, lateral_gain=0.95, heading_gain=0.70,
                  left_reference_gain=1.15, right_reference_gain=1.0, preview_heading_gain=0.25)
    cls = type('Controller', (), {name: scope[name] for name in names})
    instance = cls()
    instance.get_parameter = lambda name: SimpleNamespace(value=params[name])
    instance._reset_steering()
    return instance


def test_left_reference_boost_and_center_path_used():
    node = controller()
    node._curve_weight = 1.0
    # Deliberately no boundary polynomials: the reconstructed center errors are
    # the only inputs, in explicit single-side mode as well as automatic mode.
    result = SimpleNamespace(lateral_error=0.2, heading_error=0.3, recovered_side='left')
    node._mode = 'left'
    left = node._mode_error(result)
    result.recovered_side = 'right'
    node._mode = 'right'
    right = node._mode_error(result)
    assert abs(left - right * 1.15) < 1e-9
    result.recovered_side = ''
    assert abs(node._mode_error(result) - right) < 1e-9


def test_curve_release_starts_before_exit_threshold_on_both_turns():
    for sign in (-1, 1):
        node = controller()
        node._curve_weight = 1.0
        assert node._curve_steering_target(sign * 0.6) == sign * 0.6
        assert node._curve_steering_target(sign * 0.6) == sign
        values = [abs(node._curve_steering_target(sign * e)) for e in (0.50, 0.40, 0.30)]
        assert 1.0 > values[0] > values[1] > values[2] > 0.25
        assert node._curve_active
        assert node._curve_steering_target(sign * 0.20) == sign * 0.20
        assert not node._curve_active


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
    assert not node._curve_active


def test_large_straight_error_cannot_activate_curve_or_side_boost():
    for side in ('left', 'right', ''):
        for sign in (-1, 1):
            node = controller()
            result = SimpleNamespace(lateral_error=sign * .8, heading_error=sign * .2,
                                     recovered_side=side, curve_heading_span=0.0,
                                     preview_heading_error=sign * .6)
            expected = .95 * result.lateral_error + .70 * result.heading_error
            for _ in range(10):
                node._update_curve_geometry(result)
                assert abs(node._mode_error(result) - expected) < 1e-9
                assert node._curve_steering_target(-expected) == -expected
                assert not node._curve_active
                assert node._curve_weight == 0


def test_shape_confirmation_and_gradual_boost():
    node = controller()
    result = SimpleNamespace(curve_heading_span=.30)
    node._update_curve_geometry(result)
    assert node._curve_weight == 0.0
    node._update_curve_geometry(result)
    assert abs(node._curve_weight - .5) < 1e-9
    result.curve_heading_span = .40
    node._update_curve_geometry(result)
    assert node._curve_weight == 1.0
    result.curve_heading_span = .0
    node._update_curve_geometry(result)
    assert node._curve_weight == 0.0
    result.curve_heading_span = .40
    node._update_curve_geometry(result)
    assert node._curve_weight == 0.0  # A new run must be confirmed again.


def test_curve_to_straight_drops_target_but_releases_output_gradually():
    node = controller()
    result = SimpleNamespace(curve_heading_span=.5)
    node._update_curve_geometry(result)
    node._update_curve_geometry(result)
    node._curve_steering_target(-.7)
    assert node._curve_steering_target(-.7) == -1.0
    node._applied_steering = -1.0
    node._last_steering_ns = 1_000_000_000
    result.curve_heading_span = 0.0
    node._update_curve_geometry(result)
    target = node._curve_steering_target(0.0)
    assert target == 0.0 and not node._curve_active
    assert abs(node._rate_limit_steering(target, 1_050_000_000) + .94) < 1e-9
