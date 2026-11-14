"""Explicit manual-inspection tuning contract; not vehicle-control authority."""
import math

# group, type, minimum, maximum. Units remain ROS parameter units (m, s, rad).
SPECS = {}
def _group(group, kind, entries):
    for name, bounds in entries.items():
        SPECS[name] = (group, kind, *bounds)

_group('Color / light', bool, {n: (0, 1) for n in (
    'detect_white', 'detect_yellow', 'low_light_enabled', 'gradient_require_color_support')})
_group('Color / light', int, {n: (0, 255) for n in (
    'white_lightness_min', 'white_saturation_max', 'yellow_saturation_min',
    'yellow_lightness_min', 'gradient_min')})
_group('Color / light', int, {'yellow_hue_min': (0, 180), 'yellow_hue_max': (0, 180),
                           'morphology_kernel': (1, 9), 'gradient_support_dilation': (1, 25)})
_group('Color / light', float, {'low_light_max_gain': (1, 3)})
_group('Windows / fit', int, {'windows': (5, 30), 'window_margin': (10, 120),
    'opposite_window_margin': (5, 80), 'histogram_min_peak': (1, 100),
    'bottom_tracking_windows': (1, 10), 'recenter_min_pixels': (5, 200),
    'fit_min_points': (40, 1000), 'ransac_iterations': (10, 100), 'polynomial_degree': (2, 3)})
_group('Windows / fit', float, {'ransac_residual_px': (1, 20),
    'smoothing_alpha': (.05, 1), 'lane_width_px': (100, 600)})
_group('Target / curve', float, {'lookahead_distance_m': (.3, 1.2),
    'metric_max_near_clamp_m': (0, .30), 'metric_edge_margin_m': (0, .05),
    'yellow_curve_min_span': (.1, .6), 'minimum_confidence': (.48, .95)})
_group('Target / curve', bool, {'observed_outer_curve_entry_enabled': (0, 1)})
_group('Speed / steering (calculated only)', float, {'normal_speed': (0, .7),
    'curve_speed': (0, .3), 'zigzag_speed': (0, .18),
    'max_steer_angle': (.05, math.radians(27)), 'steering_attack_rate': (.1, 4),
    'steering_release_rate': (.1, 4), 'steering_reversal_rate': (.1, 4),
    'right_curve_min_steer_rad': (0, .2), 'right_curve_span_gain': (0, .5),
    'curve_left_preview_gain': (0, 1), 'curve_left_preview_max_rad': (0, .16)})


def validate_changes(changes, current, commands_enabled):
    if commands_enabled:
        raise ValueError('Tuning is locked while command output is enabled. Use lane_debug.launch.py.')
    if not isinstance(changes, dict) or not changes:
        raise ValueError('Expected a non-empty JSON object')
    result = {}
    for name, value in changes.items():
        if name not in SPECS:
            raise ValueError(f'{name}: not an editable parameter')
        _, kind, low, high = SPECS[name]
        if kind is bool:
            if type(value) is not bool:
                raise ValueError(f'{name}: expected true/false')
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name}: expected finite number')
            if kind is int and value != int(value):
                raise ValueError(f'{name}: expected integer')
            if not low <= value <= high:
                raise ValueError(f'{name}: allowed [{low}, {high}]')
            value = kind(value)
        result[name] = value
    combined = dict(current, **result)
    if combined['yellow_hue_min'] > combined['yellow_hue_max']:
        raise ValueError('yellow_hue_min must not exceed yellow_hue_max')
    if combined['bottom_tracking_windows'] > combined['windows']:
        raise ValueError('bottom_tracking_windows must not exceed windows')
    return result
