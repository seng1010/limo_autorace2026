"""Viewer projection, overload handling, and manual-mode command isolation."""
import ast
from pathlib import Path
import queue
from types import SimpleNamespace

import cv2
import numpy as np

from jetracer_ros2.live_debug_view import LiveViewer, render_view, render_grid, path_layers, status_text, VIEW_NAMES
from jetracer_ros2.lane_pipeline import LanePipeline, LanePipelineConfig
from jetracer_ros2.driving_profile import ZigzagTracker
from jetracer_ros2.lane_tuning import SPECS, validate_changes


def sample_packet(valid=True):
    image = np.zeros((480, 640, 3), np.uint8)
    return dict(raw=image, camera=image, bev=image, windows=image,
                binary=np.zeros((480, 640), np.uint8), color=None, gradient=None,
                state=dict(valid=valid, reference='both'),
                left_fit=np.array([0., 140.]), right_fit=np.array([0., 500.]),
                target_bev_xy=(320., 300.), inverse_matrix=np.eye(3),
                roi=np.float32([[100, 240], [540, 240], [639, 479], [0, 479]]))


def test_camera_panel_labels_distinguish_raw_and_undistorted_roi():
    assert 'before undistort' in VIEW_NAMES['raw']
    assert 'Undistorted' in VIEW_NAMES['overlay']


def test_projection_uses_actual_pipeline_transform():
    packet = sample_packet()
    packet['inverse_matrix'] = np.array([[.5, 0, 20], [0, .5, 30], [0, 0, 1.]])
    layers = path_layers(packet, camera=True)
    assert np.allclose(layers['center'][:, 0], 180.)
    assert np.allclose(layers['center'][0], [180., 30.])
    result = render_view(packet, 'overlay', dict(roi=False))
    assert np.any(result != 0)
    assert not packet['camera'].any()  # Rendering must not mutate inference buffers.


def test_status_separates_lane_validity_from_control_decision():
    packet = sample_packet(valid=False)
    packet['state'].update(
        commands_enabled=True, mode='center', usable=False, confidence=0.89,
        left_type='unobserved', right_type='yellow/solid/curve', lateral=0.0,
        heading_rad=0.0, shape_span=0.516,
        curve_control=False, angle_cmd_deg=0.0, lane_v=0.0, lane_omega=0.0,
        lookahead='unavailable', reason='outer left curve not observed',
        observation_state='accepted', path_state='not_evaluated', policy_state='blocked',
        control_state='SAFETY_STOP', control_reason='outer left curve not observed')
    status = status_text(packet)
    assert 'confidence=0.89' in status
    assert 'control state: SAFETY_STOP' in status
    assert 'control reason: outer left curve not observed' in status
    assert 'observation/path/policy: accepted/not_evaluated/blocked' in status


def test_invalid_result_does_not_show_old_path():
    packet = sample_packet(valid=False)
    assert path_layers(packet, camera=True) == {}
    assert not render_view(packet, 'overlay', dict(roi=False)).any()


def test_layer_toggles_and_inferred_boundary():
    packet = sample_packet()
    off = dict(roi=False, left=False, right=False, center=False, target=False)
    assert not render_view(packet, 'overlay', off).any()
    on = {**off, 'left': True}
    measured = render_view(packet, 'bev', on)
    packet['state']['reference'] = 'right'
    inferred = render_view(packet, 'bev', on)
    assert 0 < np.count_nonzero(inferred) < np.count_nonzero(measured)


def test_all_layouts_keep_dimensions_and_handle_missing_images():
    packet = sample_packet()
    for columns in (0, 1, 2, 3):
        result = render_grid(packet, list(VIEW_NAMES), {}, 1000, 700, columns)
        assert result.ndim == 3 and result.shape[2] == 3
        assert result.shape[0] <= 700 and result.shape[1] <= 1000
    assert render_grid(packet, [], {}, 1000, 700).shape == (700, 1000, 3)


def test_projection_handles_nonfinite_and_offscreen_points():
    packet = sample_packet()
    packet['left_fit'] = np.array([np.nan, 1e30])
    result = render_view(packet, 'overlay', {})
    assert result.shape == (480, 640, 3)


def test_viewer_queue_never_waits_when_full():
    viewer = LiveViewer.__new__(LiveViewer)
    viewer.frames = queue.Queue(maxsize=2)
    for _ in range(1000):
        viewer.submit(sample_packet(False))
    assert viewer.frames.qsize() == 2


def test_invalid_frame_keeps_bev_and_masks():
    result = LanePipeline(LanePipelineConfig()).process(np.zeros((480, 640, 3), np.uint8))
    assert not result.valid and result.target_bev_xy is None
    assert result.bev_image.shape == (480, 640, 3)
    assert result.binary_image.shape == (480, 640)


def test_camera_watchdog_timestamp_follows_command_publication():
    path = Path(__file__).parents[1] / 'jetracer_ros2' / 'lane_following_node.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'LaneFollowingNode')
    callback = next(n for n in node.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_image_callback')
    published = [n.lineno for n in ast.walk(callback) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == '_publish_command']
    completed = [n.lineno for n in ast.walk(callback) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == '_last_image_ns'
                         for t in n.targets)]
    assert len(published) == len(completed) == 1
    assert published[0] < completed[0]


def test_manual_mode_creates_no_command_publisher_and_watchdog_sends_nothing():
    path = Path(__file__).parents[1] / 'jetracer_ros2' / 'lane_following_node.py'
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    names = {'__init__', '_declare_parameters', '_pipeline_config', '_publish_command',
             '_watchdog_callback', '_safe_stop', '_reset_steering', '_tuning_values',
             '_guard_tuning_parameters', '_poll_tuning'}
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]

    class FakeNode:
        def __init__(self, name):
            self.params = {'publish_commands': False}
            self.topics, self.sent = [], []

        def declare_parameter(self, name, value):
            self.params.setdefault(name, value)

        def get_parameter(self, name):
            return SimpleNamespace(value=self.params[name])

        def add_on_set_parameters_callback(self, callback):
            self.parameter_callback = callback

        def set_parameters_atomically(self, parameters):
            result = self.parameter_callback(parameters)
            if result.successful:
                self.params.update({p.name: p.value for p in parameters})
            return result

        def create_publisher(self, msg, name, qos):
            self.topics.append(name)
            return SimpleNamespace(publish=lambda msg: self.sent.append((name, msg)))

        def create_subscription(self, *args):
            pass

        def create_timer(self, *args):
            pass

        def get_logger(self):
            return SimpleNamespace(info=lambda text: None)

        def get_clock(self):
            return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1000000000))

        def _warn(self, *args):
            pass

        def _mode_callback(self, *args):
            pass

        _image_callback = _camera_info_callback = _odom_callback = _mode_callback

    definition = ast.ClassDef(name='TestNode', bases=[ast.Name(id='FakeNode', ctx=ast.Load())],
                              keywords=[], body=methods, decorator_list=[])
    scope = dict(FakeNode=FakeNode, np=np, LanePipeline=LanePipeline, LanePipelineConfig=LanePipelineConfig,
                 SPECS=SPECS, validate_changes=validate_changes, queue=queue, cv2=cv2,
                 runtime_identity=lambda path: ('loaded.py', 'unknown'),
                 SetParametersResult=SimpleNamespace,
                 Parameter=lambda name, value: SimpleNamespace(name=name, value=value),
                 ZigzagTracker=ZigzagTracker,
                 Twist=object, Vector3Stamped=object, CompressedImage=object, CameraInfo=object,
                 Odometry=object,
                 Bool=object, String=object, qos_profile_sensor_data=object())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), str(path), 'exec'), scope)
    node = scope['TestNode']()
    assert 'inference/cmd_vel' not in node.topics and '/cmd_vel' not in node.topics
    node._publish_command(object())
    node._watchdog_callback()
    node._safe_stop('test')
    assert node.sent == []
    node.viewer = SimpleNamespace(tuning_requests=queue.Queue(), tuning_responses=queue.Queue())
    node.viewer.tuning_requests.put({'windows': 20})
    node._poll_tuning()
    assert node.pipeline.config.windows == 20 and node.params['windows'] == 20
    assert 'Applied' in node.viewer.tuning_responses.get()['message']
    previous = node.pipeline
    node.viewer.tuning_requests.put({'windows': 0, 'normal_speed': .1})
    node._poll_tuning()
    assert node.pipeline is previous and node.params['normal_speed'] == .4
    assert 'Rejected' in node.viewer.tuning_responses.get()['message']
    node.commands_enabled = True
    node.viewer.tuning_requests.put({'windows': 10})
    node._poll_tuning()
    assert node.pipeline is previous
    assert 'Rejected' in node.viewer.tuning_responses.get()['message']
    assert node.sent == []


def test_launch_arguments_override_defaults_and_viewer_closes_on_shutdown():
    import os
    path = Path(__file__).parents[1] / 'jetracer_ros2' / 'lane_following_node.py'
    main = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = []

    class Shutdown(Exception):
        pass

    def spin(node):
        raise Shutdown()

    node = SimpleNamespace(close_viewer=lambda: calls.append('close'), destroy_node=lambda: calls.append('destroy'))
    scope = dict(os=os, sys=SimpleNamespace(argv=['lane_following_node']),
                 get_package_share_directory=lambda _: '/share', LaneFollowingNode=lambda: node,
                 ExternalShutdownException=Shutdown,
                 rclpy=SimpleNamespace(init=lambda **kw: calls.append(kw['args']), spin=spin, ok=lambda: False))
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), 'exec'), scope)
    args = ['--ros-args', '-p', 'publish_commands:=false', '-p', 'live_debug_enabled:=true']
    scope['main'](args)
    assert calls[0][-len(args):] == args
    assert calls[0].index('--params-file') < calls[0].index('-p')
    assert calls[1:] == ['close', 'destroy']


def test_bev_roi_is_visible_even_when_lane_is_invalid_and_can_be_hidden():
    packet = sample_packet(valid=False)
    packet['bev_roi'] = np.float32([[140, 0], [500, 0], [500, 479], [140, 479]])
    visible = render_view(packet, 'bev', dict(roi=True))
    assert np.any(visible[:, 140] != 0)
    assert np.any(visible[:, 500] != 0)
    assert not render_view(packet, 'bev', dict(roi=False)).any()
    assert not packet['bev'].any()
