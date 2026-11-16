"""Autorace 2026 미션 4: T자 주차 노드 (6칸 중 빈 칸 탐색 및 주차 차량 회피).

트랙 구조:
  - 주행 차로(폭 550mm) 우측에 3열 x 2행(총 6칸) 주차 구역 배치.
  - 정지선에서 출발해 직진하면서 우측 라이다로 1열 -> 2열 -> 3열을 스캔한다.
  - 주차된 LIMO 차량이 있는 칸은 지나치고, 비어 있는 칸의 중심을 찾아 우회전 진입한다.
  - 진입 시 양옆에 주차된 LIMO와 충돌하지 않도록 중앙 정렬하며 주차 구역 내부에 정지한다.

사용법:
  ros2 run jetracer_ros2 parking_test_node
  ros2 run jetracer_ros2 parking_test_node empty_dist:=0.75 turn_sec:=2.5 enter_sec:=1.6
  ros2 run jetracer_ros2 parking_test_node target_slot:=2  # (테스트용) 특정 번째 빈 칸 선택
"""

import math
import os
import re
import sys
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from jetracer_ros2.utils.ackermann_utils import HARDWARE_MAX_INNER_ANGLE, inner_angle_to_omega

TICK_SEC = 0.05  # 20 Hz

DEFAULTS = {
    'output_topic': '/cmd_vel',
    'scan_topic': '/scan',
    'speed': 0.12,               # 직진 탐색 및 진입 속도 [m/s] (규정 서행 0.2m/s 이하 준수)
    'turn_speed': 0.08,          # 우회전 속도 [m/s]
    'turn_steer_rad': 0.45,      # 우회전 조향각 크기 [rad]
    'turn_sec': 2.5,             # 90도 우회전 소요 시간 [s]
    'enter_sec': 1.6,            # 칸 진입 직진 최대 시간 [s]
    'start_delay_sec': 1.5,      # 실행 후 출발 대기 시간 [s] (규정 15초 타임아웃 이내)
    'wait_enter': False,

    # --- 6칸 빈 자리 탐색 및 회피 파라미터 ---
    'lidar_front_deg': 0.0,      # 라이다 정면 각도 보정 [deg]
    'min_search_sec': 0.4,       # 정지선 통과 직후 1열 진입 전 최소 직진 시간 [s]
    'max_search_sec': 6.0,       # 3열 끝까지 탐색하는 최대 직진 시간 [s] (막다른 벽 보호)
    'empty_dist': 0.72,          # 우측(-90도) 거리가 이 값 이상이면 '빈 칸'으로 판정 [m]
    'slot_width_m': 0.45,        # 주차 1칸의 가로 폭 [m] (빈 칸 중앙 정렬 및 회전 시작점 계산용)
    'turn_offset_m': -0.10,      # 빈 칸 중심 대비 회전 시작 위치 미세 보정 [m] (+면 더 가서 회전)
    'target_empty_idx': 1,       # 발견한 빈 칸 중 몇 번째 빈 칸에 들어갈지 (1 = 첫 번째 빈 칸)

    # --- 충돌 방지 및 차간 정렬 파라미터 ---
    'front_stop_dist': 0.25,     # 탐색 중 전방 막다른 벽/차량 앞 정지 거리 [m]
    'enter_stop_dist': 0.22,     # 주차 칸 안쪽 벽 또는 2행 주차 차량 앞 정지 거리 [m]
    'side_safe_dist': 0.22,      # 양옆 주차 LIMO와 유지할 최소 측면 안전 거리 [m]
    'avoid_gain': 1.3,           # 양옆 주차 LIMO 회피 조향 게인
}


class TSlotParkingNode(Node):
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

        self.speed = float(p['speed'])
        self.turn_speed = float(p['turn_speed'])
        self.turn_steer = -steer  # 음수 = 우회전
        self.turn_sec = float(p['turn_sec'])
        self.enter_sec = float(p['enter_sec'])
        self.start_delay = float(p['start_delay_sec'])

        self.lidar_front_rad = math.radians(float(p['lidar_front_deg']))
        self.min_search_sec = float(p['min_search_sec'])
        self.max_search_sec = float(p['max_search_sec'])
        self.empty_dist = float(p['empty_dist'])
        self.slot_width_m = float(p['slot_width_m'])
        self.turn_offset_m = float(p['turn_offset_m'])
        self.target_empty_idx = int(p['target_empty_idx'])

        self.front_stop_dist = float(p['front_stop_dist'])
        self.enter_stop_dist = float(p['enter_stop_dist'])
        self.side_safe_dist = float(p['side_safe_dist'])
        self.avoid_gain = float(p['avoid_gain'])

        # 라이다 구역별 거리 (inf = 장애물 없음)
        self.dist_front = float('inf')        # 정면 (-15° ~ +15°)
        self.dist_front_left = float('inf')   # 좌전방 (+15° ~ +65°)
        self.dist_front_right = float('inf')  # 우전방 (-65° ~ -15°)
        self.dist_right_side = float('inf')   # 우측 직각 (-102° ~ -78°): 주차 칸 깊이 측정용
        self.dist_left_side = float('inf')    # 좌측 직각 (+78° ~ +102°): 주행 차로 벽 거리 유지용
        self.last_scan_time = 0.0

        # 상태 머신 변수: WAIT -> SEARCH -> ALIGN_TURN -> TURN -> ENTER -> DONE
        self.state = 'WAIT'
        self.state_elapsed = 0.0
        self._last_tick = None
        self._go_at = None

        # 빈 칸 탐색 추적 변수
        self._in_empty_zone = False
        self._empty_start_dist = 0.0
        self._traveled_dist = 0.0
        self._found_empty_count = 0
        self._target_turn_dist = None

        self.cmd_pub = self.create_publisher(Twist, p['output_topic'], 10)
        self.state_pub = self.create_publisher(String, 'parking_test/state', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, p['scan_topic'], self._scan_cb, qos_profile_sensor_data)

        self.timer = self.create_timer(TICK_SEC, self._tick)
        self.get_logger().info(
            f'T주차 탐색 노드 준비 완료 (empty_dist={self.empty_dist}m, '
            f'target_empty_idx={self.target_empty_idx})')

    @staticmethod
    def _normalize_angle(rad):
        return (rad + math.pi) % (2.0 * math.pi) - math.pi

    def _scan_cb(self, msg: LaserScan):
        """라이다 스캔에서 정면, 좌/우전방, 좌/우측면(주차칸 깊이) 최소 거리를 추출한다."""
        f_min = float('inf')
        fl_min = float('inf')
        fr_min = float('inf')
        r_side_vals = []
        l_side_min = float('inf')

        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and msg.range_min < r < msg.range_max:
                rel_deg = math.degrees(self._normalize_angle(angle - self.lidar_front_rad))
                if -15.0 <= rel_deg <= 15.0:
                    if r < f_min:
                        f_min = r
                elif 15.0 < rel_deg <= 65.0:
                    if r < fl_min:
                        fl_min = r
                elif -65.0 <= rel_deg < -15.0:
                    if r < fr_min:
                        fr_min = r
                elif -102.0 <= rel_deg <= -78.0:
                    r_side_vals.append(r)
                elif 78.0 <= rel_deg <= 102.0:
                    if r < l_side_min:
                        l_side_min = r
            angle += msg.angle_increment

        self.dist_front = f_min
        self.dist_front_left = fl_min
        self.dist_front_right = fr_min
        self.dist_left_side = l_side_min
        # 측면 노이즈 방지를 위해 중앙값(median) 사용
        if r_side_vals:
            r_side_vals.sort()
            self.dist_right_side = r_side_vals[len(r_side_vals) // 2]
        else:
            self.dist_right_side = float('inf')

        self.last_scan_time = time.monotonic()

    def _set_state(self, new_state):
        self.state = new_state
        self.state_elapsed = 0.0
        self.state_pub.publish(String(data=new_state))
        self.get_logger().info(f'=== 상태 전환: {new_state} ===')

    def go(self):
        if self._go_at is None:
            self._go_at = time.monotonic() + self.start_delay
            self.get_logger().info(f'{self.start_delay:.1f}초 뒤 출발')

    def _publish(self, speed, steer):
        steer = max(-HARDWARE_MAX_INNER_ANGLE, min(HARDWARE_MAX_INNER_ANGLE, steer))
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = inner_angle_to_omega(steer, speed)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def _tick(self):
        now = time.monotonic()
        dt = TICK_SEC if self._last_tick is None else max(0.0, now - self._last_tick)
        self._last_tick = now

        if self.state == 'DONE':
            self.stop()
            return

        if self.state == 'WAIT':
            self.stop()
            if self._go_at is not None and now >= self._go_at:
                self._set_state('SEARCH')
            return

        # 라이다 데이터 수신 확인
        if self.last_scan_time == 0.0 or (now - self.last_scan_time > 0.5):
            self.get_logger().warn('라이다(/scan) 수신 대기 중...', throttle_duration_sec=1.0)
            self.stop()
            return

        self.state_elapsed += dt

        # ------------------------------------------------------------------
        # 1. SEARCH: 직진하면서 우측 3개 열 중 빈 칸(주차된 차가 없는 칸)을 찾는다
        # ------------------------------------------------------------------
        if self.state == 'SEARCH':
            # 막다른 길(3열 끝 벽)에 도달하면 충돌 전 정지
            if self.dist_front < self.front_stop_dist or self.state_elapsed > self.max_search_sec:
                self.get_logger().error('탐색 구간 끝 도달(빈 칸 미발견) -> 안전 정지')
                self._set_state('DONE')
                self.stop()
                return

            self._traveled_dist += self.speed * dt

            # 최소 진입 시간 이후부터 우측 측면 거리로 빈 칸 판별
            if self.state_elapsed >= self.min_search_sec:
                is_empty = self.dist_right_side >= self.empty_dist

                if is_empty and not self._in_empty_zone:
                    # 주차된 차를 지나쳐 빈 칸 시작 지점 진입
                    self._in_empty_zone = True
                    self._empty_start_dist = self._traveled_dist
                    self.get_logger().info(
                        f'빈 칸 시작 감지 (우측 거리={self.dist_right_side:.2f}m, '
                        f'누적 이동={self._traveled_dist:.2f}m)')

                elif self._in_empty_zone:
                    # 빈 칸 폭의 절반 이상 지났거나, 다음 칸의 주차된 차가 다시 감지된 경우
                    empty_span = self._traveled_dist - self._empty_start_dist
                    if (not is_empty and empty_span > self.slot_width_m * 0.45) or (
                            empty_span >= self.slot_width_m * 0.85):
                        self._found_empty_count += 1
                        self._in_empty_zone = False
                        slot_center_dist = self._empty_start_dist + (self.slot_width_m * 0.5)
                        self.get_logger().info(
                            f'[{self._found_empty_count}번째 빈 칸 확정] '
                            f'중심 위치={slot_center_dist:.2f}m')

                        if self._found_empty_count >= self.target_empty_idx:
                            # 목표 회전 시작 지점 설정 후 ALIGN_TURN 상태로 전환
                            self._target_turn_dist = slot_center_dist + self.turn_offset_m
                            self._set_state('ALIGN_TURN')
                            return

            # 직진 중 좌측 차선 벽과 너무 가까워지지 않도록 미세 보정
            steer = 0.0
            if self.dist_left_side < 0.22:
                steer = -0.15 * (0.22 - self.dist_left_side)
            self._publish(self.speed, steer)

        # ------------------------------------------------------------------
        # 2. ALIGN_TURN: 빈 칸 중앙 회전 시작 지점까지 미세 이동
        # ------------------------------------------------------------------
        elif self.state == 'ALIGN_TURN':
            if self.dist_front < self.front_stop_dist:
                self._set_state('TURN')
                return

            self._traveled_dist += self.speed * dt
            if self._traveled_dist >= self._target_turn_dist:
                self._set_state('TURN')
                return

            self._publish(self.speed, 0.0)

        # ------------------------------------------------------------------
        # 3. TURN: 빈 칸 방향으로 우회전 (양옆 주차 LIMO 회피 보정 포함)
        # ------------------------------------------------------------------
        elif self.state == 'TURN':
            if self.state_elapsed >= self.turn_sec:
                self._set_state('ENTER')
                return

            steer = self.turn_steer
            # 우회전 안쪽(우전방)에 주차된 LIMO가 너무 가까우면 회전 반경을 살짝 넓힘
            if self.dist_front_right < self.side_safe_dist:
                steer += (self.side_safe_dist - self.dist_front_right) * self.avoid_gain
            # 좌전방(맞은편 벽 또는 옆 차)이 너무 가까우면 안쪽으로 더 꺾음
            if self.dist_front_left < self.side_safe_dist:
                steer -= (self.side_safe_dist - self.dist_front_left) * self.avoid_gain

            self._publish(self.turn_speed, steer)

        # ------------------------------------------------------------------
        # 4. ENTER: 양옆 LIMO 중앙을 맞추며 칸 안쪽으로 직진 진입 후 정지
        # ------------------------------------------------------------------
        elif self.state == 'ENTER':
            # 1) 주차 칸 끝 벽이나 뒷줄(2행)에 세워진 차량 감지 시 즉시 정지
            if self.dist_front < self.enter_stop_dist:
                self.get_logger().info(
                    f'전방 경계/차량 감지({self.dist_front:.2f}m) -> 주차 완료 정지')
                self._set_state('DONE')
                self.stop()
                return

            # 2) 지정된 진입 시간(enter_sec) 완료 시 정지
            if self.state_elapsed >= self.enter_sec:
                self.get_logger().info('진입 시간 완료 -> 주차 완료 정지')
                self._set_state('DONE')
                self.stop()
                return

            # 3) 좌·우측에 주차된 LIMO 차량과의 거리를 비교해 중앙 정렬(Centering)
            steer = 0.0
            left_err = max(0.0, self.side_safe_dist - self.dist_front_left)
            right_err = max(0.0, self.side_safe_dist - self.dist_front_right)
            if left_err > 0.0 or right_err > 0.0:
                # 우측 차가 가까우면 좌측(+steer)으로, 좌측 차가 가까우면 우측(-steer)으로 보정
                steer = (right_err - left_err) * self.avoid_gain

            self._publish(self.speed * 0.85, steer)


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
        node = TSlotParkingNode()
        if node.get_parameter('wait_enter').value and sys.stdin.isatty():
            input('[parking_test_node] Enter를 누르면 출발합니다...')
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
