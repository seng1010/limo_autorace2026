"""정지선 검출 연동 T자 주차 노드.

동작 순서:
  1) APPROACH  : 정지선이 목표 거리(stop_distance_m) 이내로 올 때까지 직진 접근
                 (정지선이 아직 안 보이면 approach_speed 로 탐색 주행)
  2) STOP_WAIT : 정지선 도달 후 settle_sec 동안 완전 정지 (차체 흔들림 안정화)
  3) FORWARD   : forward_sec 동안 직진 (a)
  4) TURN      : turn_sec 동안 우회전 (b)
  5) ENTER     : enter_sec 동안 주차 칸 진입 (c)
  6) DONE      : 최종 정지
  
"""

import os
import re
import sys
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from jetracer_ros2.utils.ackermann_utils import HARDWARE_MAX_INNER_ANGLE, inner_angle_to_omega

TICK_SEC = 0.05  # 20 Hz

DEFAULTS = {
    'output_topic': '/cmd_vel',
    'approach_speed': 0.25,     # 정지선 접근 속도 [m/s] (너무 빠르면 정지선 지나침)
    'stop_distance_m': 0.05,    # 정지선 멈춤 기준 거리 [m] (0.05m = 5cm 앞)
    'settle_sec': 1.0,          # 정지선 도착 후 주차 시작 전 대기 시간 [s]
    'forward_sec': 1.7,         # a: 정지선에서 직진 시간 [s]
    'turn_sec': 1.8,            # b: 우회전 시간 [s]
    'enter_sec': 0.1,           # c: 칸 안으로 직진 시간 [s]
    'speed': 0.50,              # 주차 직진 속도 [m/s]
    'turn_speed': 0.50,         # 주차 우회전 속도 [m/s]
    'turn_steer_rad': 0.45,     # 우회전 조향각 크기 [rad]
    'start_delay_sec': 0.0,     # 실행 후 출발까지 대기 [s]
    'wait_enter': False,        # True면 Enter를 눌러야 출발
}


class ParkingMissionNode(Node):
    def __init__(self):
        super().__init__('parking_mission_node')
        for name, value in DEFAULTS.items():
            self.declare_parameter(name, value)
        p = {name: self.get_parameter(name).value for name in DEFAULTS}

        steer = abs(float(p['turn_steer_rad']))
        if steer > HARDWARE_MAX_INNER_ANGLE:
            self.get_logger().warn(
                f'turn_steer_rad {steer:.3f} > 최대 {HARDWARE_MAX_INNER_ANGLE:.3f}, 제한합니다.')
            steer = HARDWARE_MAX_INNER_ANGLE

        self.approach_speed = float(p['approach_speed'])
        self.stop_distance = float(p['stop_distance_m'])
        self.settle_sec = float(p['settle_sec'])
        self.start_delay = float(p['start_delay_sec'])

        # 정지선 정지 이후 실행할 주차 단계들
        self.parking_phases = [
            ('forward', float(p['forward_sec']), float(p['speed']), 0.0),
            ('turn', float(p['turn_sec']), float(p['turn_speed']), -steer),
            ('enter', float(p['enter_sec']), float(p['speed']), 0.0),
        ]

        # 상태 변수
        self.stop_line_detected = False
        self.stop_line_distance = -1.0

        # 상태 머신: WAIT_START -> APPROACH -> STOP_WAIT -> PARKING -> DONE
        self._state = 'WAIT_START'
        self._state_start = None
        self._phase_index = 0
        self._go_at = None

        # Pub / Sub
        self.cmd_pub = self.create_publisher(Twist, p['output_topic'], 10)
        self.state_pub = self.create_publisher(String, 'parking_mission/state', 10)
        self.create_subscription(
            Bool, 'mission/stop_line_detected', self._detected_cb, 10)
        self.create_subscription(
            Float32, 'mission/stop_line_distance', self._distance_cb, 10)

        self.timer = self.create_timer(TICK_SEC, self._tick)
        self.get_logger().info(
            f'정지선 연동 주차 노드 준비 완료 (정지 기준: {self.stop_distance:.2f}m, '
            f'접근 속도: {self.approach_speed:.2f}m/s)')

    def _detected_cb(self, msg: Bool):
        self.stop_line_detected = msg.data

    def _distance_cb(self, msg: Float32):
        self.stop_line_distance = msg.data

    def go(self):
        if self._go_at is None:
            self._go_at = time.monotonic() + self.start_delay
            self.get_logger().info(f'{self.start_delay:.1f}초 뒤 출발합니다.')

    def _set_state(self, new_state: str, now: float):
        self._state = new_state
        self._state_start = now
        self.state_pub.publish(String(data=new_state))
        self.get_logger().info(f'[상태 전환] -> {new_state}')

    def _publish_cmd(self, speed: float, steer: float):
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = inner_angle_to_omega(steer, speed)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def _tick(self):
        now = time.monotonic()

        # 0. 출발 전 대기
        if self._state == 'WAIT_START':
            self.stop()
            if self._go_at is not None and now >= self._go_at:
                self._set_state('APPROACH', now)
            return

        # 1. 정지선까지 접근
        if self._state == 'APPROACH':
            if self.stop_line_detected and 0.0 <= self.stop_line_distance <= self.stop_distance:
                self.stop()
                self.get_logger().info(
                    f'정지선 도달! (남은 거리: {self.stop_line_distance:.3f}m)')
                self._set_state('STOP_WAIT', now)
            else:
                # 정지선이 안 보이거나 아직 멀면 approach_speed로 직진
                self._publish_cmd(self.approach_speed, 0.0)
            return

        # 2. 정지선 위에서 잠시 대기 (관성/흔들림 안정화)
        if self._state == 'STOP_WAIT':
            self.stop()
            if now - self._state_start >= self.settle_sec:
                self._phase_index = 0
                self._set_state('PARKING', now)
                self.get_logger().info(f'주차 시작 -> {self.parking_phases[0][0]}')
            return

        # 3. 기존 T자 주차 시퀀스 (forward -> turn -> enter)
        if self._state == 'PARKING':
            while (self._phase_index < len(self.parking_phases)
                   and now - self._state_start >= self.parking_phases[self._phase_index][1]):
                elapsed_phase = self.parking_phases[self._phase_index][1]
                self._phase_index += 1
                self._state_start += elapsed_phase
                if self._phase_index < len(self.parking_phases):
                    self.get_logger().info(
                        f'주차 단계 -> {self.parking_phases[self._phase_index][0]}')

            if self._phase_index >= len(self.parking_phases):
                self.stop()
                self._set_state('DONE', now)
                self.get_logger().info('주차 미션 완료! (정지). Ctrl+C로 종료하세요.')
                return

            _, _, speed, steer = self.parking_phases[self._phase_index]
            self._publish_cmd(speed, steer)
            return

        # 4. 완료 후 정지 유지
        if self._state == 'DONE':
            self.stop()


def _plain_overrides(argv):
    ros_args = []
    for arg in argv[1:]:
        m = re.match(r'^([A-Za-z_]+):=(.*)$', arg)
        if m and m.group(1) in DEFAULTS:
            ros_args += ['-p', f'{m.group(1)}:={m.group(2)}']
    return ros_args


def main(args=None):
    argv = sys.argv if args is None else args
    ros_args = ['--ros-args']
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    if os.path.exists(params):
        ros_args += ['--params-file', params]
    ros_args += _plain_overrides(argv)
    if '--ros-args' in argv:
        ros_args += argv[argv.index('--ros-args') + 1:]

    rclpy.init(args=[argv[0]] + ros_args)
    node = None
    try:
        node = ParkingMissionNode()
        if node.get_parameter('wait_enter').value and sys.stdin.isatty():
            input('[parking_mission_node] Enter를 누르면 출발합니다...')
        node.go()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            for _ in range(5):
                node.stop()
                time.sleep(0.02)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
