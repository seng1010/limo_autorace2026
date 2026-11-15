"""M8-only LiDAR cone corridor; proposes commands to the mission manager."""

import json
import os
import sys
import time
from collections import deque

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PolygonStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from jetracer_ros2.cone_corridor import (
    Params, Planner, ResumeGate, checked_exit_lane, safe_lane_arc)
from jetracer_ros2.m8_entry import (
    M8_COMMAND_TIMEOUT, M8_EXIT_CHECK_TOPIC, M8_EXIT_COMMAND_TOPIC,
    M8_LANE_HINT_TOPIC, M8_LANE_PATH_FRAME, M8_LANE_PATH_TOPIC, M8_PHASE_TOPIC,
    M8_PLANNER_TIMING_TOPIC, MAX_M8_SPEED, MIN_M8_SPEED, OBSERVED_LANE_TOPIC)
from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega


def _stamp_ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


class MissionConeAvoidNode(Node):
    def __init__(self):
        super().__init__('mission_cone_avoid_node')
        defaults = {
            'scan_topic': '/scan', 'output_topic': 'mission/cone/cmd_vel',
            'validated_output_topic': 'mission/cone/safe_cmd_vel',
            'speed': 0.10, 'max_steer_angle': 0.42,
            'lidar_x_m': 0.203, 'lidar_y_m': 0.0, 'lidar_yaw_rad': 0.0,
            'scan_frame': 'laser_link', 'scan_timeout_sec': 0.25,
            'vehicle_half_width_m': 0.12, 'vehicle_front_m': 0.29, 'vehicle_rear_m': 0.09,
            'cone_padding_m': 0.045, 'minimum_clearance_m': 0.025,
            'resume_scans': 2,
            'minimum_turn_radius_m': 0.50,
            'motion_enabled': False,
            'odom_topic': '/odom', 'target_hold_sec': .50,
            'target_hold_distance_m': .10, 'rear_from_odom_x_m': -.10,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        self.config = Params(
            lidar_x=p('lidar_x_m'), lidar_y=p('lidar_y_m'), lidar_yaw=p('lidar_yaw_rad'),
            half_width=p('vehicle_half_width_m'), front=p('vehicle_front_m'),
            rear=p('vehicle_rear_m'), cone_padding=p('cone_padding_m'),
            clearance=p('minimum_clearance_m'), minimum_radius=p('minimum_turn_radius_m'),
            max_steer_rad=p('max_steer_angle'), speed=p('speed'),
            target_hold_sec=p('target_hold_sec'), target_hold_distance=p('target_hold_distance_m'),
            rear_from_odom_x=p('rear_from_odom_x_m'))
        self.planner = Planner(self.config)
        self.odom = deque(maxlen=100)
        self.create_subscription(Odometry, p('odom_topic'), self._observe_odom, 5)
        self.scan_timeout, self.scan_frame = p('scan_timeout_sec'), p('scan_frame')
        self.resume = ResumeGate(p('resume_scans'))
        self.exit_resume = ResumeGate(p('resume_scans'))
        self.motion_enabled = p('motion_enabled')
        v = self.config
        if not (np.isfinite([*v.__dict__.values(), self.scan_timeout]).all()
                and MIN_M8_SPEED <= v.speed <= MAX_M8_SPEED
                and v.clearance > 0.0
                and 0 <= v.cone_padding and v.half_width > 0 and v.front > 0 and v.rear >= 0
                and v.minimum_radius >= 0.50 and 0 < v.max_steer_rad <= 0.42
                and 0 <= v.target_hold_sec <= .5 and 0 <= v.target_hold_distance <= .10
                and -.20 <= v.rear_from_odom_x <= 0.
                and 1 <= self.resume.required <= 10
                and 0 < self.scan_timeout <= 0.30 and self.scan_frame):
            raise ValueError('Invalid M8 geometry, clearance, speed or timing parameters')
        self.last_scan, self.last_stamp, self.last_reason = None, None, None
        self.last_log = 0.
        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 10)
        self.safe_cmd_pub = self.create_publisher(TwistStamped, p('validated_output_topic'), 1)
        self.exit_cmd_pub = self.create_publisher(TwistStamped, M8_EXIT_COMMAND_TOPIC, 1)
        self.timing_pub = self.create_publisher(String, M8_PLANNER_TIMING_TOPIC, 10)
        self.exit_check = False
        self.exit_lane = None
        self.m8_phase = 'off'
        self.lane_center = None
        self.lane_hint = None
        self.create_subscription(TwistStamped, OBSERVED_LANE_TOPIC, self._observe_lane, 1)
        self.create_subscription(PolygonStamped, M8_LANE_PATH_TOPIC, self._observe_center, 1)
        self.create_subscription(TwistStamped, M8_LANE_HINT_TOPIC, self._observe_hint, 1)
        self.create_subscription(
            String, M8_PHASE_TOPIC, self._change_phase,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(
            Bool, M8_EXIT_CHECK_TOPIC, self._enable_exit_check,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.state_pub = self.create_publisher(String, 'mission/cone/state', 10)
        self.sub = self.create_subscription(
            LaserScan, p('scan_topic'), self._callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.timer = self.create_timer(0.05, self._watchdog)
        if not self.motion_enabled:
            self.get_logger().warning('M8 motion disabled: confirm geometry before enabling')

    def _publish(self, state, reason, speed=0.0, steer=0.0, scan_stamp=None):
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = inner_angle_to_omega(steer, speed)
        # A single message carries the planner's decision and sensor freshness.
        # Stops use the current time even if no valid scan was received.
        safe_cmd = TwistStamped(twist=cmd)
        safe_cmd.header.stamp = scan_stamp if scan_stamp is not None else self.get_clock().now().to_msg()
        published_ns = self.get_clock().now().nanoseconds
        self.safe_cmd_pub.publish(safe_cmd)
        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=state))
        if reason != self.last_reason or time.monotonic()-self.last_log >= 1.:
            self.get_logger().info(f'M8 {reason}: v={speed:.3f} m/s steer={np.degrees(steer):+.1f} deg')
            self.last_reason = reason
            self.last_log = time.monotonic()
        header = safe_cmd.header.stamp
        return published_ns, header.sec*10**9 + header.nanosec

    def _stop(self, reason, state='lost'):
        self.resume.update(False)
        return self._publish(state, reason)

    def _observe_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        stamp = _stamp_ns(msg.header.stamp)
        values = (p.x,p.y,q.x,q.y,q.z,q.w)
        if stamp <= 0 or not np.isfinite(values).all():
            self.odom.clear()
            return
        yaw = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        if self.odom and stamp <= self.odom[-1][0]:
            self.odom.clear()
        self.odom.append((stamp,(p.x,p.y,float(yaw))))

    def _pose_at_scan(self, stamp):
        # Never use future odometry or infer motion from a commanded speed.
        for t,pose in reversed(self.odom):
            if t <= stamp:
                return pose if stamp-t <= 200_000_000 else None
        return None

    def _observe_lane(self, msg):
        self.exit_lane = msg

    def _observe_center(self, msg):
        points = msg.polygon.points
        if msg.header.frame_id == M8_LANE_PATH_FRAME and 8 <= len(points) <= 256:
            self.lane_center = (_stamp_ns(msg.header.stamp), [(p.x, p.y) for p in points])
        else:
            self.lane_center = None

    def _observe_hint(self, msg):
        if msg.header.frame_id == M8_LANE_PATH_FRAME:
            self.lane_hint = msg
        else:
            self.lane_hint = None

    def _change_phase(self, msg):
        if msg.data != self.m8_phase:
            self.planner.reset()
            self.m8_phase = msg.data
            self.exit_resume.update(False)
            self.lane_center = None
            self.lane_hint = None

    def _enable_exit_check(self, msg):
        self.exit_check = msg.data
        self.exit_resume.update(False)
        if not self.exit_check:
            self._publish_exit(force=True)

    def _publish_exit(self, result=None, scan_stamp=None, *, force=False):
        if not self.exit_check and not force:
            return
        cmd = TwistStamped()
        now = self.get_clock().now()
        cmd.header.stamp = now.to_msg()
        if self.motion_enabled and result is not None and result.get('scan') is not None:
            lidar = _stamp_ns(scan_stamp)
            stamp = scan_stamp
            command = (0., 0.)

            def fresh(t):
                return t > 0 and 0 <= (now.nanoseconds-t)/1e9 <= M8_COMMAND_TIMEOUT

            lane = self.exit_lane
            if lane is not None and fresh(lidar) and fresh(_stamp_ns(lane.header.stamp)):
                command = checked_exit_lane(
                    (lane.twist.linear.x, lane.twist.angular.z), result['points'], self.config,
                    scan=result['scan'], approach=self.m8_phase in ('off','watch_entry','approach'))
                stamp = min((scan_stamp, lane.header.stamp), key=_stamp_ns)

            if (not command[0] and self.m8_phase == 'exit_lane' and fresh(lidar)
                    and self.lane_center is not None):
                geom_stamp, points = self.lane_center
                hint = next((m for m in (lane, self.lane_hint) if m is not None
                             and _stamp_ns(m.header.stamp) == geom_stamp
                             and m.twist.linear.x > 0), None)
                if hint is not None and fresh(geom_stamp):
                    command = safe_lane_arc(result['scan'], result['points'], self.config, points,
                                            (hint.twist.linear.x, hint.twist.angular.z))
                    stamp = min((scan_stamp, hint.header.stamp), key=_stamp_ns)
            if (command[0] and 0 <= (self.get_clock().now().nanoseconds-_stamp_ns(stamp))/1e9
                    <= M8_COMMAND_TIMEOUT):
                cmd.twist.linear.x, cmd.twist.angular.z = command
                cmd.header.stamp = stamp
        if not self.exit_resume.update(cmd.twist.linear.x > 0):
            cmd.twist = Twist()
        self.exit_cmd_pub.publish(cmd)

    def _watchdog(self):
        if self.last_scan is None or time.monotonic() - self.last_scan > self.scan_timeout:
            self.planner.reset()
            self._publish_exit()
            self._stop('scan_timeout')

    def _callback(self, scan):
        rx_mono = time.monotonic_ns()
        rx_ros = self.get_clock().now().nanoseconds
        stamp = scan.header.stamp.sec * 10**9 + scan.header.stamp.nanosec
        age = (rx_ros - stamp) / 1e9
        a = (scan.angle_min, scan.angle_max, scan.angle_increment,
             scan.time_increment, scan.scan_time, scan.range_min, scan.range_max)
        if (scan.header.frame_id != self.scan_frame or not np.isfinite(a).all()
                or a[2] <= 0.0 or a[5] < 0.0 or a[6] <= a[5]
                or not 4 <= len(scan.ranges) <= 4096):
            self._finish_scan(stamp, rx_ros, rx_mono, 0., None, scan.header.stamp,
                              'invalid_scan_geometry')
            return
        if (stamp <= 0 or not -.02 <= age <= self.scan_timeout
                or (self.last_stamp is not None and stamp <= self.last_stamp)):
            self._finish_scan(stamp, rx_ros, rx_mono, 0., None, scan.header.stamp,
                              'stale_or_unordered_scan')
            return
        self.last_scan, self.last_stamp = rx_mono/1e9, stamp
        started = time.monotonic_ns()
        try:
            ranges = np.asarray(scan.ranges, dtype=float)
            result = self.planner.update(a, ranges, stamp/1e9, self._pose_at_scan(stamp))
            if self.exit_check:
                result['scan'] = a, ranges
            speed = result['speed']
        except (ValueError, ArithmeticError) as exc:
            self.get_logger().error(f'M8 planner error: {exc}')
            self._finish_scan(stamp, rx_ros, rx_mono, (time.monotonic_ns()-started)/1e6,
                              None, scan.header.stamp, 'planning_error')
            return
        plan_ms = (time.monotonic_ns()-started)/1e6
        if (time.monotonic_ns() - rx_mono > self.scan_timeout*1e9 or
                (self.get_clock().now().nanoseconds - stamp)/1e9 >
                min(self.scan_timeout, M8_COMMAND_TIMEOUT)):
            reason, state, result = 'planning_result_stale', 'lost', None
        elif result['state'] != 'tracking':
            state = ('emergency_stop' if result['state'] in ('footprint_blocked', 'path_blocked')
                     else 'lost')
            reason = result['state']
        elif not self.motion_enabled:
            reason, state = 'motion_disabled', 'lost'
        elif speed <= 0.0:
            reason, state = 'command_prefix_blocked', 'emergency_stop'
        else:
            if not self.resume.update(True, result['road']['gate']):
                # Keep the valid streak; waiting for the next scan is not a failure.
                self._finish_scan(stamp, rx_ros, rx_mono, plan_ms, result, scan.header.stamp,
                                  'waiting_for_consecutive_safe_scans', keep_streak=True)
                return
            self._finish_scan(stamp, rx_ros, rx_mono, plan_ms, result, scan.header.stamp,
                              result['mode'], state='tracking', speed=speed)
            return
        self._finish_scan(stamp, rx_ros, rx_mono, plan_ms, result, scan.header.stamp, reason, state)

    def _finish_scan(self, stamp, rx_ros, rx_mono, plan_ms, result, scan_stamp,
                     reason, state='lost', speed=0., keep_streak=False):
        # Publish cone control before optional approach/exit lane checks or diagnostics.
        if speed > 0:
            published_ns, command_stamp_ns = self._publish(
                state, reason, speed, result['steer'], scan_stamp)
        elif keep_streak:
            published_ns, command_stamp_ns = self._publish(state, reason)
        else:
            published_ns, command_stamp_ns = self._stop(reason, state)
        exit_ms = None
        if self.exit_check:
            t = time.monotonic_ns()
            self._publish_exit(result, scan_stamp)
            exit_ms = (time.monotonic_ns()-t)/1e6
        if self.timing_pub.get_subscription_count():
            self.timing_pub.publish(String(data=json.dumps(dict(
                scan_ns=stamp, command_stamp_ns=command_stamp_ns,
                received_ns=rx_ros, published_ns=published_ns,
                arrival_age_ms=(rx_ros-stamp)/1e6, plan_ms=plan_ms, exit_ms=exit_ms,
                callback_ms=(time.monotonic_ns()-rx_mono)/1e6, reason=reason,
                speed=speed, groups=len(result['groups']) if result else 0,
                goal=result['road']['gate'].tolist() if result and result['road'] else None,
                clearance=result.get('min_clearance') if result else None), separators=(',', ':'))))


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'm8.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params] + list(argv[1:]))
    node = MissionConeAvoidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._publish_exit()
            node._publish('lost', 'shutdown')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
