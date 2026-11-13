"""T자 주차 기본 주행 테스트 노드 (시간 기반, 라이다/정지선 없음).

정지선 위에 리모를 세워 두고 실행하면 아래 순서로 움직인 뒤 멈춘다.
  1) forward : forward_sec(a) 초 동안 직진
  2) turn    : turn_sec(b)    초 동안 우회전
  3) enter   : enter_sec(c)   초 동안 직진해서 주차 칸 안으로 진입
  4) done    : 정지

라이다 회피 없이 순수하게 a/b/c 시간과 속도/조향각만 맞춰 보는 용도다.
mission manager 없이 /cmd_vel 로 바로 내보낸다 (다른 주행 노드는 꺼 둘 것).

사용법:
  ros2 run jetracer_ros2 parking_test_node
  ros2 run jetracer_ros2 parking_test_node forward_sec:=1.2 turn_sec:=2.8 enter_sec:=1.0
  ros2 run jetracer_ros2 parking_test_node turn_steer_rad:=0.40 speed:=0.08

  - 실행 후 start_delay_sec 초 기다렸다가 출발한다 (손 뗄 시간)
  - wait_enter:=true 로 실행하면 터미널에서 Enter를 눌러야 출발한다
  - Ctrl+C 를 누르면 즉시 정지
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
from std_msgs.msg import String

from jetracer_ros2.utils.ackermann_utils import HARDWARE_MAX_INNER_ANGLE, inner_angle_to_omega

TICK_SEC = 0.05  # 20 Hz

DEFAULTS = {
    'output_topic': '/cmd_vel',
    'forward_sec': 1.5,       # a: 정지선에서 직진 시간 [s]
    'turn_sec': 2.5,          # b: 우회전 시간 [s]
    'enter_sec': 1.5,         # c: 칸 안으로 직진 시간 [s]
    'speed': 0.10,            # 직진 속도 [m/s]
    'turn_speed': 0.08,       # 우회전 중 속도 [m/s]
    'turn_steer_rad': 0.45,   # 우회전 조향각 크기 [rad] (최대 약 0.488 = 28도)
    'start_delay_sec': 2.0,   # 실행 후 출발까지 대기 [s]
    'wait_enter': False,      # True면 Enter를 눌러야 출발
}


class ParkingTestNode(Node):
    def __init__(self):
        super().__init__('parking_test_node')
        for name, value in DEFAULTS.items():
            self.declare_parameter(name, value)
        p = {name: self.get_parameter(name).value for name in DEFAULTS}

        steer = abs(float(p['turn_steer_rad']))
        if steer > HARDWARE_MAX_INNER_ANGLE:
            self.get_logger().warn(
                f'turn_steer_rad {steer:.3f} > 하드웨어 최대 {HARDWARE_MAX_INNER_ANGLE:.3f}, '
                '최대값으로 제한합니다.')
            steer = HARDWARE_MAX_INNER_ANGLE

        # (단계 이름, 시간, 속도, 조향각) — 조향각 음수 = 오른쪽
        self.phases = [
            ('forward', float(p['forward_sec']), float(p['speed']), 0.0),
            ('turn', float(p['turn_sec']), float(p['turn_speed']), -steer),
            ('enter', float(p['enter_sec']), float(p['speed']), 0.0),
        ]
        self.start_delay = float(p['start_delay_sec'])

        self.cmd_pub = self.create_publisher(Twist, p['output_topic'], 10)
        self.state_pub = self.create_publisher(String, 'parking_test/state', 10)

        self._index = -1              # -1: 출발 대기
        self._phase_start = None
        self._go_at = None            # 출발 시각 (monotonic). None이면 아직 출발 신호 없음
        self.done = False
        self.timer = self.create_timer(TICK_SEC, self._tick)

        plan = ' -> '.join(
            f'{name} {sec:.2f}s @ {spd:.2f}m/s' + (f' steer {st:+.2f}rad' if st else '')
            for name, sec, spd, st in self.phases)
        self.get_logger().info(f'주차 테스트: {plan}  (output={p["output_topic"]})')

    def go(self):
        """출발 신호. start_delay_sec 뒤에 첫 단계 시작."""
        if self._go_at is None:
            self._go_at = time.monotonic() + self.start_delay
            self.get_logger().info(f'{self.start_delay:.1f}초 뒤 출발')

    def _publish(self, speed, steer):
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = inner_angle_to_omega(steer, speed)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def _enter_phase(self, index, now):
        self._index = index
        self._phase_start = now
        name = self.phases[index][0] if index < len(self.phases) else 'done'
        self.state_pub.publish(String(data=name))
        self.get_logger().info(f'-> {name}')

    def _tick(self):
        now = time.monotonic()
        if self.done:
            self.stop()
            return
        if self._index < 0:
            self.stop()
            if self._go_at is not None and now >= self._go_at:
                self._enter_phase(0, now)
            else:
                return

        # 시간이 다 된 단계는 건너뛴다 (0초짜리 단계 포함)
        while (self._index < len(self.phases)
               and now - self._phase_start >= self.phases[self._index][1]):
            self._enter_phase(self._index + 1, self._phase_start + self.phases[self._index][1])

        if self._index >= len(self.phases):
            self.done = True
            self.stop()
            self.get_logger().info('주차 테스트 완료 (정지). Ctrl+C로 종료')
            return

        _, _, speed, steer = self.phases[self._index]
        self._publish(speed, steer)


def _plain_overrides(argv):
    """`forward_sec:=1.2`처럼 --ros-args 없이 넘긴 인자를 -p 인자로 바꾼다."""
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
        node = ParkingTestNode()
        if node.get_parameter('wait_enter').value and sys.stdin.isatty():
            input('[parking_test_node] Enter를 누르면 출발합니다...')
        node.go()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            # 종료 시 확실히 멈추도록 정지 명령을 몇 번 보낸다
            for _ in range(5):
                node.stop()
                time.sleep(0.02)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
