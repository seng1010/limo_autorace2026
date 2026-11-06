"""RViz text summary for the 2026 perception and mission stack."""

import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray


class DebugAutoraceNode(Node):
    def __init__(self):
        super().__init__('debug_autorace_node')
        self.declare_parameter('frame_id', 'base_link')
        self.values = {
            'mission': '-', 'lane_mode': '-', 'lane_conf': '-', 'zigzag': 'false', 'stop_line': '-',
            'yolo': '-', 'obstacle': '-', 'cone': '-', 'parking': '-',
        }
        self.pub = self.create_publisher(MarkerArray, '/autorace/debug_markers', 10)
        self.create_subscription(String, 'mission/active', lambda m: self._set('mission', m.data), 10)
        self.create_subscription(String, '/lane/mode_state', lambda m: self._set('lane_mode', m.data), 10)
        self.create_subscription(Bool, '/lane/zigzag', lambda m: self._set('zigzag', str(m.data)), 10)
        self.create_subscription(Float32, 'mission/stop_line_distance',
                                 lambda m: self._set('stop_line', f'{m.data:.2f} m'), 10)
        self.create_subscription(String, 'mission/yolo_event', lambda m: self._set('yolo', m.data), 10)
        self.create_subscription(String, 'mission/obstacle_avoid/state',
                                 lambda m: self._set('obstacle', m.data), 10)
        self.create_subscription(String, 'mission/cone/state', lambda m: self._set('cone', m.data), 10)
        self.create_subscription(String, 'mission/parking/state', lambda m: self._set('parking', m.data), 10)
        self.create_subscription(Vector3Stamped, '/lane/model', self._lane_model, 10)
        self.timer = self.create_timer(0.2, self._publish)

    def _set(self, key, value):
        self.values[key] = value

    def _lane_model(self, msg):
        self.values['lane_conf'] = f'{msg.vector.z:.2f}'

    def _publish(self):
        markers = MarkerArray()
        for index, (key, value) in enumerate(self.values.items()):
            marker = Marker()
            marker.header.frame_id = self.get_parameter('frame_id').value
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'autorace_status'
            marker.id = index
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = 0.2
            marker.pose.position.z = 1.0 + 0.12 * index
            marker.scale.z = 0.10
            marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
            marker.text = f'{key}: {value}'
            markers.markers.append(marker)
        self.pub.publish(markers)


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params])
    node = DebugAutoraceNode()
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
