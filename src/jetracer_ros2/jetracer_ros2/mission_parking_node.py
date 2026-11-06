"""Conservative odometry-based T-parking controller for M4.

The manoeuvre is disabled until ``mission/parking/start`` is true.  It emits a
proposal on ``mission/parking/cmd_vel``; only the mission manager may forward it
to the vehicle.  Parameters must be calibrated at low speed on the real course.
"""

import math
import os
import sys

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega


class MissionParkingNode(Node):
    def __init__(self):
        super().__init__('mission_parking_node')
        defaults = {
            'output_topic': 'mission/parking/cmd_vel',
            'search_speed': 0.08, 'reverse_speed': -0.07,
            'max_steer_angle': 0.40, 'opening_distance_m': 1.0,
            'opening_confirm_scans': 4, 'advance_distance_m': 0.42,
            'reverse_turn_yaw_deg': 72.0, 'reverse_straight_distance_m': 0.48,
            'rear_stop_distance_m': 0.22, 'right_sector_deg': [-100.0, -55.0],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 10)
        self.state_pub = self.create_publisher(String, 'mission/parking/state', 10)
        self.create_subscription(Bool, 'mission/parking/start', self._start_callback, 10)
        self.create_subscription(Odometry, '/odom', self._odom_callback, 20)
        self.create_subscription(LaserScan, '/scan', self._scan_callback, qos_profile_sensor_data)
        self._state = 'waiting'
        self._pose = None
        self._state_origin = None
        self._opening_hits = 0
        self._rear_distance = float('inf')
        self.timer = self.create_timer(0.05, self._tick)

    def _start_callback(self, msg):
        if msg.data and self._state == 'waiting':
            self._opening_hits = 0
            self._transition('searching')

    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (p.x, p.y, yaw)
        if self._state_origin is None:
            self._state_origin = self._pose

    def _scan_callback(self, scan):
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = np.degrees(scan.angle_min + np.arange(ranges.size) * scan.angle_increment)
        valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        sector = self.get_parameter('right_sector_deg').value
        right = ranges[valid & (angles >= sector[0]) & (angles <= sector[1])]
        opening = right.size > 0 and float(np.percentile(right, 30)) >= self.get_parameter('opening_distance_m').value
        self._opening_hits = self._opening_hits + 1 if opening else 0
        rear = ranges[valid & (np.abs(np.abs(angles) - 180.0) <= 20.0)]
        self._rear_distance = float(np.min(rear)) if rear.size else float('inf')

    def _tick(self):
        cmd = Twist()
        if self._state == 'searching':
            cmd.linear.x = self.get_parameter('search_speed').value
            if self._opening_hits >= self.get_parameter('opening_confirm_scans').value:
                self._transition('advance')
        elif self._state == 'advance':
            cmd.linear.x = self.get_parameter('search_speed').value
            if self._distance_from_origin() >= self.get_parameter('advance_distance_m').value:
                self._transition('reverse_turn')
        elif self._state == 'reverse_turn':
            cmd.linear.x = self.get_parameter('reverse_speed').value
            steering = -self.get_parameter('max_steer_angle').value
            cmd.angular.z = inner_angle_to_omega(steering, cmd.linear.x)
            if self._yaw_from_origin() >= math.radians(self.get_parameter('reverse_turn_yaw_deg').value):
                self._transition('reverse_straight')
        elif self._state == 'reverse_straight':
            cmd.linear.x = self.get_parameter('reverse_speed').value
            if (self._distance_from_origin() >= self.get_parameter('reverse_straight_distance_m').value
                    or self._rear_distance <= self.get_parameter('rear_stop_distance_m').value):
                self._transition('complete')
                cmd = Twist()
        self.cmd_pub.publish(cmd)

    def _transition(self, state):
        self._state = state
        self._state_origin = self._pose
        self.state_pub.publish(String(data=state))
        self.get_logger().info(f'parking state -> {state}')

    def _distance_from_origin(self):
        if self._pose is None or self._state_origin is None:
            return 0.0
        return math.hypot(self._pose[0] - self._state_origin[0],
                          self._pose[1] - self._state_origin[1])

    def _yaw_from_origin(self):
        if self._pose is None or self._state_origin is None:
            return 0.0
        delta = self._pose[2] - self._state_origin[2]
        return abs(math.atan2(math.sin(delta), math.cos(delta)))


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params])
    node = MissionParkingNode()
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
