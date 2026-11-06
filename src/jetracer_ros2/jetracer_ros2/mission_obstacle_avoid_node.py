"""Detect a stopped LIMO and recommend a safe lane-change direction."""

import math
import os
import sys

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class MissionObstacleAvoidNode(Node):
    def __init__(self):
        super().__init__('mission_obstacle_avoid_node')
        defaults = {
            'scan_topic': '/scan', 'front_angle_deg': 24.0,
            'side_angle_min_deg': 20.0, 'side_angle_max_deg': 75.0,
            'obstacle_distance_m': 1.15, 'clearance_distance_m': 1.4,
            'minimum_points': 4, 'confirm_scans': 3,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.pub = self.create_publisher(String, 'mission/obstacle_avoid/state', 10)
        self._candidate = 'clear'
        self._count = 0
        self.sub = self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self._callback, qos_profile_sensor_data)

    def _sector(self, scan, low_deg, high_deg):
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
        valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        low, high = math.radians(low_deg), math.radians(high_deg)
        return ranges[valid & (angles >= low) & (angles <= high)]

    def _callback(self, scan):
        front_angle = self.get_parameter('front_angle_deg').value
        side_min = self.get_parameter('side_angle_min_deg').value
        side_max = self.get_parameter('side_angle_max_deg').value
        obstacle_distance = self.get_parameter('obstacle_distance_m').value
        min_points = self.get_parameter('minimum_points').value
        front = self._sector(scan, -front_angle, front_angle)
        if int(np.count_nonzero(front < obstacle_distance)) < min_points:
            candidate = 'clear'
        else:
            left = self._sector(scan, side_min, side_max)
            right = self._sector(scan, -side_max, -side_min)
            clearance = self.get_parameter('clearance_distance_m').value
            left_clear = left.size > 0 and float(np.percentile(left, 25)) >= clearance
            right_clear = right.size > 0 and float(np.percentile(right, 25)) >= clearance
            if left_clear and (not right_clear or np.median(left) >= np.median(right)):
                candidate = 'change_left'
            elif right_clear:
                candidate = 'change_right'
            else:
                candidate = 'stop'
        if candidate == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = candidate, 1
        if self._count >= self.get_parameter('confirm_scans').value:
            self.pub.publish(String(data=candidate))


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params])
    node = MissionObstacleAvoidNode()
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
