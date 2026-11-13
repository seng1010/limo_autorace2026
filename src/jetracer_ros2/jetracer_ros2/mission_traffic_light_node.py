"""M1 신호등 색 인식 전용 노드. 정지·출발 판단은 mission_manager_node(m1_traffic_light)가 한다.

구독
  perception/detections   String(JSON)  yolo_detection_node 의 모든 박스
  /camera/color/image_raw/compressed     HSV 확인용 (최근 프레임)
발행
  mission/traffic_light_state  String  red / yellow / green / unknown  (publish_rate_hz)

판단 로직은 traffic_light_logic.py (ROS 없음, 단위 테스트 있음).
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from jetracer_ros2.traffic_light_logic import (
    TrafficLightConfig, TrafficLightFilter, frame_color)


class MissionTrafficLightNode(Node):
    def __init__(self):
        super().__init__('mission_traffic_light_node')
        defaults = {
            'detections_topic': 'perception/detections',
            'image_topic': '/camera/color/image_raw/compressed',
            'output_topic': 'mission/traffic_light_state',
            'publish_rate_hz': 10.0,
            'image_max_age_sec': 0.3,        # HSV 확인에 쓸 이미지가 이보다 오래되면 확인 생략
            'min_confidence': 0.45,
            'min_area_ratio': 0.0005,
            'verify_hsv': True,
            'min_lit_ratio': 0.04,
            'lit_value_min': 150,
            'lit_saturation_min': 90,
            'confirm_frames': 3,
            'lost_timeout_sec': 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.config = TrafficLightConfig(**{
            field: p(field) for field in TrafficLightConfig.__dataclass_fields__})
        self.filter = TrafficLightFilter(self.config)
        self._image = None
        self._image_time = 0.0
        self._last_state = None

        self.pub = self.create_publisher(String, p('output_topic'), 10)
        self.create_subscription(String, p('detections_topic'), self._detections_callback, 10)
        self.create_subscription(
            CompressedImage, p('image_topic'), self._image_callback, qos_profile_sensor_data)
        self.create_timer(1.0 / max(1.0, p('publish_rate_hz')), self._publish)

    def _image_callback(self, msg):
        self._image_msg = msg
        self._image_time = time.monotonic()

    def _latest_image(self):
        if not hasattr(self, '_image_msg'):
            return None
        if time.monotonic() - self._image_time > self.get_parameter('image_max_age_sec').value:
            return None
        return cv2.imdecode(np.frombuffer(self._image_msg.data, np.uint8), cv2.IMREAD_COLOR)

    def _detections_callback(self, msg):
        try:
            detections = json.loads(msg.data).get('detections', [])
        except (ValueError, AttributeError):
            return
        image = self._latest_image() if self.config.verify_hsv else None
        shape = image.shape if image is not None else (480, 640)
        color = frame_color(detections, image, shape, self.config)
        self.filter.update(time.monotonic(), color)

    def _publish(self):
        state = self.filter.tick(time.monotonic())
        if state != self._last_state:
            self.get_logger().info(f'신호등: {self._last_state} -> {state}')
            self._last_state = state
        self.pub.publish(String(data=state))


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params] + list(argv[1:]))
    node = MissionTrafficLightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
