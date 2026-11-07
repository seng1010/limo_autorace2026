"""Fuse YOLO traffic-light classes with the stop-line distance for M1."""

import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float32, String


class MissionTrafficLightNode(Node):
    def __init__(self):
        super().__init__('mission_traffic_light_node')
        self.declare_parameter('yolo_event_topic', 'mission/yolo_event')
        self.declare_parameter('stop_line_topic', 'mission/stop_line_distance')
        self.declare_parameter('output_topic', 'mission/traffic_light_state')
        self.declare_parameter('stop_distance_m', 0.35)
        self.declare_parameter('require_stop_line', True)
        self.declare_parameter('confirm_events', 2)
        self._distance = -1.0
        self._candidate = 'unknown'
        self._count = 0
        self.pub = self.create_publisher(String, self.get_parameter('output_topic').value, 10)
        self.event_sub = self.create_subscription(
            String, self.get_parameter('yolo_event_topic').value, self._event_callback, 10)
        self.line_sub = self.create_subscription(
            Float32, self.get_parameter('stop_line_topic').value, self._line_callback, 10)

    def _line_callback(self, msg):
        self._distance = msg.data

    def _event_callback(self, msg):
        colors = {'traffic_red': 'red', 'traffic_yellow': 'yellow', 'traffic_green': 'green'}
        color = colors.get(msg.data)
        if color is None:
            return
        if color == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = color, 1
        if self._count < self.get_parameter('confirm_events').value:
            return
        line_required = self.get_parameter('require_stop_line').value
        at_line = 0.0 <= self._distance <= self.get_parameter('stop_distance_m').value
        if color in ('red', 'yellow') or not line_required or at_line:
            self.pub.publish(String(data=color))


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params])
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
