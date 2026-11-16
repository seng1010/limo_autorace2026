"""M5 서행 테스트 노드: 정지선에 닿으면 서행.

조향은 lane_following_node(inference/cmd_vel)값을 그대로 쓰고, 속도만 M5 로직으로 정함.
차선 명령이 들어올 때마다 /cmd_vel 발행 (차선 명령이 없으면 발행 안 함).
"""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32

from jetracer_ros2.slow_bump_mission import M5Config, SlowBumpMission


class M5RunnerNode(Node):
    def __init__(self):
        super().__init__('m5_runner_node')
        self.cfg = M5Config(
            base_speed=0.35,         # 기본 주행 속도 [m/s]
            slow_duration_sec=7.0,   # 서행 유지 시간 [초]
            trigger_distance_m=0.03, # 정지선 접촉 판정 거리 [m]
        )
        self.mission = SlowBumpMission(self.cfg)
        self.stop_line_m = None
        self.prev_state = None

        self.create_subscription(Float32, 'mission/stop_line_distance', self._line_cb, 10)
        self.create_subscription(Twist, 'inference/cmd_vel', self._lane_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f'M5 시작: 기본속도={self.cfg.base_speed}, 서행속도={self.cfg.computed_slow_speed}, '
            f'서행시간={self.cfg.slow_duration_sec}초')

    def _line_cb(self, msg: Float32):
        # stop_line_node 는 선이 없으면 -1.0 발행
        self.stop_line_m = None if msg.data < 0.0 else float(msg.data)

    def _lane_cb(self, msg: Twist):
        speed = self.mission.update(time.monotonic(), self.stop_line_m)
        if self.mission.state != self.prev_state:
            self.get_logger().info(f'상태: {self.mission.state} (속도 {speed} m/s)')
            self.prev_state = self.mission.state

        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = msg.angular.z
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = M5RunnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())  # 종료 시 정지
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
