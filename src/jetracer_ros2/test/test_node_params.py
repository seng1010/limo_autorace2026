"""노드가 설정 dataclass 의 항목을 전부 ROS 파라미터로 선언하는지 확인.

2026-11-15 실차에서 mission_traffic_light_node 가 뜨자마자 죽었다:
TrafficLightConfig 에 housing_* 항목을 추가하면서 노드의 선언 목록에 빠뜨려
`ParameterNotDeclaredException: housing_lamp_margin` 이 났다.
노드 파일은 rclpy 가 필요해 import 할 수 없으므로 소스를 읽어서 확인한다.
"""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from jetracer_ros2.stop_line_detector import StopLineConfig
from jetracer_ros2.traffic_light_logic import TrafficLightConfig

NODES = Path(__file__).resolve().parent.parent / 'jetracer_ros2'
CASES = [('stop_line_node.py', StopLineConfig),
         ('mission_traffic_light_node.py', TrafficLightConfig)]


@pytest.mark.parametrize('filename,config', CASES)
def test_node_declares_every_config_field(filename, config):
    source = (NODES / filename).read_text()
    if 'fields(' in source:
        return                                  # dataclass 에서 자동 생성 — 빠질 수 없음
    block = source.split('defaults = {', 1)[1].split('}', 1)[0]
    declared = set(re.findall(r"'([a-z_0-9]+)'\s*:", block))
    missing = [f.name for f in fields(config) if f.name not in declared]
    assert not missing, f'{filename} 에서 선언 안 된 파라미터: {missing}'


@pytest.mark.parametrize('filename,config', CASES)
def test_every_config_field_has_default(filename, config):
    """declare_parameter 는 None 을 기본값으로 받지 못한다."""
    assert [f.name for f in fields(config) if f.default is None] == []
