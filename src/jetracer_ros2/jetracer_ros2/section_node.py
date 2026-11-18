"""구간 전환 노드: 정지선 + odom 거리로 미션 구간을 넘기고, 서행 구간에서 속도 제한을 낸다.

매니저는 M5·M6·M2·M8 을 스스로 넘기지 않는다 (`mission/event` 로 `{미션}_DONE` 을 받아야 넘어감).
이 노드가 그 신호를 보낸다. 판단은 section_logic.py (ROS 없음, 단위 테스트 있음).

구독
  /odom                           누적 주행거리
  mission/stop_line_distance      앞바퀴 -> 정지선 [m], 없으면 -1 (stop_line_node)
  /mission_manager/debug/state    맨 앞 단어 = 지금 미션 (매니저가 20 Hz 로 발행)
발행
  mission/event                   String  'M5_DONE' 등
  mission/speed_limit             Float32 서행 구간이면 slow_speed, 아니면 -1 (매니저가 차선 속도에 적용)
  /section/debug                  String  미션, 간 거리, 서행 여부, 마지막 정지선

설정 (params.yaml section_node): 미션마다 line_after / timeout / slow_from / slow_to [m], 음수 = 끄기
"""

import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String

from jetracer_ros2.section_logic import LinePassDetector, SectionRule, SectionTracker
from jetracer_ros2.stop_line_detector import OdomPathIntegrator

MISSIONS = ('M5', 'M6', 'M3', 'M7', 'M2', 'M8', 'M1', 'M4')
RULE_MISSIONS = ('M5', 'M6', 'M2', 'M8')   # 매니저가 스스로 넘기지 않는 미션


class SectionNode(Node):
    def __init__(self):
        super().__init__('section_node')
        defaults = {
            'odom_topic': '/odom',
            'slow_speed': 0.13,
            'line_min_frames': 2,
            'line_refractory_m': 0.30,
            'stop_line_timeout_sec': 0.5,
            'event_period_sec': 0.5,
            'm5_slow_mode': 'line',   # line: 정지선에 닿으면 서행 (못 보면 거리로 대신) / distance: 거리로만
        }
        for mission in RULE_MISSIONS:
            for field in SectionRule.__dataclass_fields__:
                defaults[f'{mission.lower()}_{field}'] = -1.0
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value  # noqa: E731

        rules = {mission: SectionRule(**{field: float(p(f'{mission.lower()}_{field}'))
                                         for field in SectionRule.__dataclass_fields__})
                 for mission in RULE_MISSIONS}
        mode = str(p('m5_slow_mode')).strip().lower()
        if mode not in ('line', 'distance'):
            raise ValueError("m5_slow_mode 는 'line' 또는 'distance'")
        if mode == 'distance':
            rules['M5'].slow_line_after_m = -1.0      # 선은 안 보고 slow_from~slow_to 거리로만 서행
        self.get_logger().info(f'M5 서행 방식: {mode}')
        self.tracker = SectionTracker(rules, LinePassDetector(
            min_frames=p('line_min_frames'), refractory_m=p('line_refractory_m')))
        for mission, rule in rules.items():
            self.get_logger().info(f'{mission}: {rule}')

        self.odom = OdomPathIntegrator()
        self._have_odom = False
        self._active = None
        self._stop_line_m = None
        self._stop_line_ns = 0
        self._last_event_ns = 0
        self._last_event = ''
        self._last_line_s = None
        self._slow = False

        self.event_pub = self.create_publisher(String, 'mission/event', 10)
        self.speed_pub = self.create_publisher(Float32, 'mission/speed_limit', 10)
        self.debug_pub = self.create_publisher(String, '/section/debug', 10)
        self.create_subscription(Odometry, p('odom_topic'), self._odom_callback, 20)
        self.create_subscription(Float32, 'mission/stop_line_distance', self._stop_line_callback, 10)
        self.create_subscription(String, '/mission_manager/debug/state', self._manager_callback, 10)
        self.timer = self.create_timer(0.05, self._tick)

    def _odom_callback(self, msg):
        pose = msg.pose.pose.position
        self.odom.update(pose.x, pose.y, msg.twist.twist.linear.x)
        self._have_odom = True

    def _stop_line_callback(self, msg):
        self._stop_line_m = float(msg.data) if msg.data >= 0.0 else None
        self._stop_line_ns = self.get_clock().now().nanoseconds

    def _manager_callback(self, msg):
        word = msg.data.split(' ', 1)[0]
        if word in MISSIONS:
            self._active = word

    def _tick(self):
        now = self.get_clock().now().nanoseconds
        if not self._have_odom:
            self.speed_pub.publish(Float32(data=-1.0))
            self.debug_pub.publish(String(data='odom 없음'))
            return
        fresh = (now - self._stop_line_ns) / 1e9 <= self.get_parameter('stop_line_timeout_sec').value
        out = self.tracker.update(self.odom.s, self._active,
                                  self._stop_line_m if fresh else None)
        if out.line_s is not None:
            self._last_line_s = out.line_s
            self.get_logger().info(
                f'정지선: 누적 {out.line_s:.2f} m ({self._active} 시작 후 '
                f'{out.line_s - self.tracker.start_s:.2f} m)')
        if out.slow != self._slow:
            self._slow = out.slow
            self.get_logger().info(f'서행 {"시작" if out.slow else "끝"} '
                                   f'({self._active} 시작 후 {out.travelled_m:.2f} m)')
        self.speed_pub.publish(Float32(
            data=float(self.get_parameter('slow_speed').value) if out.slow else -1.0))
        if out.event:
            period = self.get_parameter('event_period_sec').value
            if out.event != self._last_event or (now - self._last_event_ns) / 1e9 >= period:
                if out.event != self._last_event:
                    self.get_logger().info(
                        f'{out.event} ({out.reason}, {out.travelled_m:.2f} m)')
                self.event_pub.publish(String(data=out.event))
                self._last_event, self._last_event_ns = out.event, now
        line_text = f'{self._last_line_s:.2f}' if self._last_line_s is not None else 'none'
        self.debug_pub.publish(String(
            data=f'{self._active or "?"} s={self.odom.s:.2f} in={out.travelled_m:.2f} '
                 f'slow={int(out.slow)} last_line={line_text} event={out.event or "-"}'))


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params] + list(argv[1:]))
    node = SectionNode()
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
