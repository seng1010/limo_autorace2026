import math
import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# 라이다로 전방 장애물(사람 등)을 감지해서 mission/people_estop_state 토픽(String)에
# 'stop'/'clear'를 발행하는 노드. 실제 cmd_vel 게이팅은 이 노드가 아니라
# mission_manager_pestop_test_node가 이 토픽을 구독해서 처리한다
# (mission_traffic_light_node/mission_tunnel_node와 같은 "센서 노드" 역할 분리 —
# /cmd_vel엔 mission_manager 계열 노드 하나만 발행해야 경쟁 문제가 안 생기기 때문).
#
# 흐름:
#   1. 노드가 뜨면 0.5초 간격으로 'clear'를 총 4번 발행 — manager가 이걸 "센서 켜짐"
#      신호로 보고 그때부터 주행을 시작한다. 한 번만 보내면 DDS discovery가 아직 안
#      끝난 순간과 겹쳤을 때 그 유일한 메시지가 유실돼서 manager가 영원히 대기
#      상태에 갇히는 문제가 있었다(실측으로 확인) — 여러 번 보내면 그중 하나라도
#      discovery 완료 이후에 도착할 확률이 훨씬 높아져서 이 레이스를 사실상 없앤다.
#   2. /cmd_vel의 linear.x가 처음으로 0이 아니게 되는 순간(=실제로 출발하는 순간)을
#      0초로 잡는다 — 이 노드 자신의 프로세스 기동 시각을 기준으로 삼으면 subprocess
#      기동/DDS discovery 지연 때문에 "출발 후 몇 초"가 실행할 때마다 들쭉날쭉해져서,
#      대신 실제 주행 명령(mission_manager_node가 내보내는 /cmd_vel)이 나가는 순간에
#      직접 동기화한다. 그 뒤 obstacle_min_time초 동안은 라이다를 무시하고(막 출발했을
#      때 근처 물체로 바로 오탐하지 않도록), 정확히 obstacle_min_time초가 되는 순간
#      'active'를 한 번 발행한다 — manager는 이 신호를 받고서야(자기 타이머를 따로
#      재지 않고) 선제 감속을 시작하므로, "라이다 활성화"와 "감속 시작"이 오차 없이
#      정확히 같은 순간에 일어난다.
#   3. 그 이후 check_rate_hz(기본 1Hz)마다 전방을 0도로 뒀을 때
#      obstacle_angle_deg_range=[min, max] 범위 안에서 obstacle_distance_m보다
#      가까운 포인트가 obstacle_min_points개 이상인지 판단한다.
#      - 장애물이 있으면: 'stop'을 (판단할 때마다 계속) 발행한다.
#      - 장애물이 있었다가 사라지면: 'clear'를 발행하고 이 노드 자신이 종료된다
#        (manager는 이걸 받아서 원래 속도로 주행을 재개한다).
#
# 사용법:
#   ros2 run jetracer_ros2 mission_people_estop_node

_ANNOUNCE_INTERVAL_SEC = 0.5
_ANNOUNCE_COUNT = 4


class MissionPeopleEstopNode(Node):
    def __init__(self):
        super().__init__('mission_people_estop_node')

        self.declare_parameter('obstacle_angle_deg_range', [-10.0, 10.0])
        self.declare_parameter('obstacle_distance_m', 1.0)
        self.declare_parameter('obstacle_min_points', 1)
        self.declare_parameter('obstacle_min_time', 7.0)  # 출발(cmd_vel nonzero) 후 이 시간(초) 전에는 라이다 무시
        self.declare_parameter('check_rate_hz', 1.0)       # 판단 주기

        angle_deg_range = self.get_parameter('obstacle_angle_deg_range').value
        self.obstacle_angle_min = math.radians(angle_deg_range[0])
        self.obstacle_angle_max = math.radians(angle_deg_range[1])
        self.obstacle_distance_m = self.get_parameter('obstacle_distance_m').value
        self.obstacle_min_points = self.get_parameter('obstacle_min_points').value
        self.obstacle_min_time = self.get_parameter('obstacle_min_time').value
        check_rate_hz = self.get_parameter('check_rate_hz').value

        self._state = 'clear'  # 'stop' | 'clear' — debug 서브클래스의 마커 색상용
        self._departed = False  # /cmd_vel이 처음 0이 아니게 된 시점(출발) 감지 여부
        self._start_time = None  # 출발 감지 시각 — _departed=True가 되는 순간 채워짐
        self._latest_scan = None
        self._was_obstacle = False  # 장애물이 한 번이라도 감지된 적 있는지 (종료 조건용)
        self.done = False  # True가 되면 main()의 spin 루프가 알아서 멈춘다
        self._announce_count = 0

        self.state_pub = self.create_publisher(String, 'mission/people_estop_state', 10)
        # 라이다 드라이버는 보통 BEST_EFFORT로 발행하므로 반드시 sensor_data QoS로 구독해야 한다.
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._on_scan_msg, qos_profile_sensor_data)
        # mission_manager_node가 실제로 주행 명령을 내보내는 순간(=출발)을 감지하기
        # 위한 구독 (위 흐름 2번 참고, _on_cmd_vel_msg에서 처리).
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._on_cmd_vel_msg, 10)

        # 생성자에서 바로 발행하면 manager와의 DDS discovery가 아직 안 끝나 유실될 수
        # 있어서, 짧게 지연 후 여러 번 발행한다 (위 흐름 1번 참고).
        self._ready_timer = self.create_timer(_ANNOUNCE_INTERVAL_SEC, self._announce_ready)
        # 출발 감지 시점에 _on_cmd_vel_msg가 생성한다 (위 흐름 2번 참고) — 그 전에는
        # obstacle_min_time을 셀 기준 시각 자체가 없으므로 타이머도 만들 수 없다.
        self._activate_timer = None
        self._check_timer = self.create_timer(1.0 / check_rate_hz, self._check_tick)

        self.get_logger().info(
            f'PeopleEstop ready — cmd_vel 출발 감지 후 {self.obstacle_min_time}초 뒤부터 '
            f'{check_rate_hz}Hz로 전방 {angle_deg_range}도, {self.obstacle_distance_m}m 이내 '
            f'포인트 {self.obstacle_min_points}개 이상 판단, 장애물 해소되면 종료')

    def _on_scan_msg(self, msg: LaserScan):
        self._latest_scan = msg

    def _on_cmd_vel_msg(self, msg: Twist):
        if self._departed or abs(msg.linear.x) < 1e-6:
            return
        self._departed = True
        self._start_time = self.get_clock().now()
        # 이 시점부터 obstacle_min_time초 뒤 정확히 한 번 'active' 발행 (위 흐름 2번
        # 참고). 이미 출발한 뒤라 announce도 여러 번 끝났을 시간이라, discovery race
        # 걱정 없이 단발성으로 보내도 안전하다.
        self._activate_timer = self.create_timer(self.obstacle_min_time, self._on_activate)
        self.get_logger().info(
            f'cmd_vel 출발 감지 — {self.obstacle_min_time}초 뒤 라이다 활성화(active) 발행')

    def _announce_ready(self):
        self.state_pub.publish(String(data='clear'))
        self._announce_count += 1
        self.get_logger().info(f'센서 켜짐 신호(clear) 발행 ({self._announce_count}/{_ANNOUNCE_COUNT})')
        if self._announce_count >= _ANNOUNCE_COUNT:
            self._ready_timer.cancel()

    def _on_activate(self):
        self._activate_timer.cancel()
        self.state_pub.publish(String(data='active'))
        self.get_logger().info(f'{self.obstacle_min_time}초 경과 — 라이다 활성화(active) 발행')

    def _count_obstacle_points(self, msg: LaserScan) -> int:
        count = 0
        angle = msg.angle_min
        for r in msg.ranges:
            if (self.obstacle_angle_min <= angle <= self.obstacle_angle_max
                    and msg.range_min <= r <= self.obstacle_distance_m):
                count += 1
                if count >= self.obstacle_min_points:
                    break
            angle += msg.angle_increment
        return count

    def _check_tick(self):
        if not self._departed:
            self._on_scan(self._latest_scan, False, 0)
            return
        elapsed_sec = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        if elapsed_sec < self.obstacle_min_time:
            self._on_scan(self._latest_scan, False, 0)
            return
        if self._latest_scan is None:
            return

        count = self._count_obstacle_points(self._latest_scan)
        obstacle = count >= self.obstacle_min_points

        if obstacle:
            self._was_obstacle = True
            self._state = 'stop'
            self.get_logger().warn(f'장애물 감지 ({count}개) — stop 발행')
            self.state_pub.publish(String(data='stop'))
        elif self._was_obstacle:
            # 장애물이 있었다가 사라짐 -> 최종 clear 발행하고 종료
            self._state = 'clear'
            self.get_logger().warn('장애물 해소 — clear 발행 후 노드 종료')
            self.state_pub.publish(String(data='clear'))
            self._check_timer.cancel()
            self._on_scan(self._latest_scan, obstacle, count)
            self.done = True
            return
        # else: 아직 한 번도 장애물이 없었음 -> 조용히 계속 감시만 한다.

        self._on_scan(self._latest_scan, obstacle, count)

    def _on_scan(self, msg, obstacle, count):
        """디버그 시각화 훅 (기본 노드는 아무것도 안 함, debug 서브클래스가 오버라이드)."""
        pass


def main(args=None):
    argv = sys.argv if args is None else args

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
    node = MissionPeopleEstopNode()
    try:
        # rclpy.spin(node) 대신 명시적으로 spin_once 루프를 도는 이유: 콜백
        # 안에서 rclpy.shutdown()을 불러도 spin()이 항상 바로 반환된다는 보장이
        # 없어서(실측 결과 안 죽고 계속 떠 있는 경우가 있었음), node.done 플래그를
        # 직접 체크하는 쪽이 훨씬 확실하다.
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
