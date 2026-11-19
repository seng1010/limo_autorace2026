"""stop_line_node: M1 에서만 흰선 밝기 기준을 낮추는지 (실제 노드 콜백, 가짜 ROS)."""

import importlib
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from jetracer_ros2.stop_line_detector import StopLineConfig, detect_stop_line


class Node:
    overrides = {}

    def __init__(self, name):
        self.params = dict(self.overrides)
        self.subs = {}

    def declare_parameter(self, key, value):
        self.params.setdefault(key, value)

    def get_parameter(self, key):
        return NS(value=self.params[key])

    def create_publisher(self, *args, **kwargs):
        return NS(publish=lambda msg: None)

    def create_subscription(self, kind, topic, callback, *args, **kwargs):
        self.subs[topic] = callback

    def get_logger(self):
        return NS(info=lambda *a: None, warn=lambda *a: None)


@pytest.fixture
def node_cls(monkeypatch):
    stubs = {
        'rclpy': dict(),
        'rclpy.node': dict(Node=Node),
        'rclpy.qos': dict(qos_profile_sensor_data=10),
        'ament_index_python.packages': dict(get_package_share_directory=lambda _: ''),
        'nav_msgs.msg': dict(Odometry=NS),
        'sensor_msgs.msg': dict(CompressedImage=NS),
        'std_msgs.msg': dict(Bool=NS, Float32=NS, String=NS),
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    observer = types.ModuleType('jetracer_ros2.m8_observers')
    observer.M8StopLineObserver = lambda *args: NS(update=lambda *a: None)
    monkeypatch.setitem(sys.modules, 'jetracer_ros2.m8_observers', observer)
    sys.modules.pop('jetracer_ros2.stop_line_node', None)
    module = importlib.import_module('jetracer_ros2.stop_line_node')
    yield module.StopLineNode
    sys.modules.pop('jetracer_ros2.stop_line_node', None)
    Node.overrides = {}


def _state(node, text):
    node.subs['/mission_manager/debug/state'](NS(data=text))


def test_m1_uses_lower_threshold_only_while_m1(node_cls):
    Node.overrides = {'white_lightness_min': 75, 'm1_white_lightness_min': 65}
    node = node_cls()
    assert node._line_config().white_lightness_min == 75          # 미션 모름 = 기본
    for word, expected in (('M8', 75), ('M1', 65), ('M4', 75)):
        _state(node, f'{word} state=approach line=none')
        assert node._line_config().white_lightness_min == expected
    assert node.config.white_lightness_min == 75                  # M8 관측기가 쓰는 설정은 그대로


def test_m1_threshold_off_by_default(node_cls):
    Node.overrides = {'white_lightness_min': 75}
    node = node_cls()
    assert '/mission_manager/debug/state' not in node.subs
    assert node._line_config() is node.config


def test_dim_line_detected_only_with_m1_threshold():
    """11/19 경기장 M1 선 밝기(L 약 70)를 흉내 낸 합성 영상."""
    image = np.full((480, 640, 3), 16, np.uint8)                  # 어두운 바닥
    image[380:392, 60:600] = 70                                   # 흐린 흰 선 (회색 = 채도 0)
    assert detect_stop_line(image, StopLineConfig(white_lightness_min=75)) is None
    assert detect_stop_line(image, StopLineConfig(white_lightness_min=65)) is not None
