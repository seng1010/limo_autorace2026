"""LiDAR-only low-speed path command through a cone corridor."""

import math
import os
import sys

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega


class MissionConeAvoidNode(Node):
    def __init__(self):
        super().__init__('mission_cone_avoid_node')
        defaults = {
            'scan_topic': '/scan', 'output_topic': 'mission/cone/cmd_vel',
            'speed': 0.10, 'max_steer_angle': 0.42, 'lookahead_m': 1.0,
            'field_of_view_deg': 80.0, 'maximum_cone_range_m': 2.5,
            'minimum_points_each_side': 2, 'steering_gain': 1.3,
            'emergency_distance_m': 0.25,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 10)
        self.state_pub = self.create_publisher(String, 'mission/cone/state', 10)
        self.sub = self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self._callback, qos_profile_sensor_data)

    def _callback(self, scan):
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
        fov = math.radians(self.get_parameter('field_of_view_deg').value)
        valid = (np.isfinite(ranges) & (ranges >= scan.range_min)
                 & (ranges <= self.get_parameter('maximum_cone_range_m').value)
                 & (np.abs(angles) <= fov))
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        cmd = Twist()
        if x.size and np.any((x > 0.0) & (x < self.get_parameter('emergency_distance_m').value)
                             & (np.abs(y) < 0.25)):
            self.state_pub.publish(String(data='emergency_stop'))
            self.cmd_pub.publish(cmd)
            return
        min_points = self.get_parameter('minimum_points_each_side').value
        left = y[y > 0.0]
        right = y[y < 0.0]
        if left.size < min_points or right.size < min_points:
            self.state_pub.publish(String(data='lost'))
            self.cmd_pub.publish(cmd)
            return
        # The nearer quartile is robust to wall/background points behind cones.
        left_edge = float(np.percentile(left, 25))
        right_edge = float(np.percentile(right, 75))
        target_y = 0.5 * (left_edge + right_edge)
        heading = math.atan2(target_y, self.get_parameter('lookahead_m').value)
        steering = np.clip(
            heading * self.get_parameter('steering_gain').value,
            -self.get_parameter('max_steer_angle').value,
            self.get_parameter('max_steer_angle').value)
        cmd.linear.x = self.get_parameter('speed').value
        cmd.angular.z = inner_angle_to_omega(float(steering), cmd.linear.x)
        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data='tracking'))


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params])
    node = MissionConeAvoidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
