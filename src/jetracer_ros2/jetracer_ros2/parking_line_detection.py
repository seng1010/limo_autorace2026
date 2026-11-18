"""정지선 검출 연동 T자 주차 노드.

동작 순서:
  1) APPROACH  : 정지선이 목표 거리(stop_distance_m) 이내로 올 때까지 직진 접근
                 (정지선이 아직 안 보이면 approach_speed 로 탐색 주행)
  2) STOP_WAIT : 정지선 도달 후 settle_sec 동안 완전 정지 (차체 흔들림 안정화)
  2-1) SCAN    : scan_speed 로 직진하며 우측 LiDAR(P30)+odom 으로 빈 칸(1/2/3) 판정
  2-2) RETURN  : 정지선(odom 0 m)까지 직진 후진 -> 판정된 칸에 맞춰 forward 시간 선택
                 (slot_scan_enabled=false 면 2-1/2-2 생략, 기존처럼 2번 칸 기준)
  3) FORWARD   : forward_sec 동안 직진 (a)  [1번/3번 칸은 slot1_*/slot3_* 값 사용]
  4) TURN      : turn_sec 동안 우회전 (b)
  5) ENTER     : enter_sec 동안 주차 칸 진입 (c)
  6) DONE      : 최종 정지
  
"""

import os
import re
import sys
import time

import math

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String

from jetracer_ros2.parking_logic import BaySlotClassifier, scan_sector_percentile
from jetracer_ros2.utils.ackermann_utils import HARDWARE_MAX_INNER_ANGLE, inner_angle_to_omega

TICK_SEC = 0.05  # 20 Hz

DEFAULTS = {
    # ===== 빈 칸 판별 (정지선 0 m 기준 odom 누적 거리) =====
    'slot_ignore_until_m': 0.45,    # 정지선 직후 여기까지는 항상 열려 보여서 무시
    'slot_min_open_m': 0.30,        # 이만큼 연속으로 열려야 빈칸 확정 (차 사이 틈 0.01~0.06 m 무시)
    # 확정 거리 실측: 1번 0.75 m / 2번 0.995~1.04 m / 3번 1.46 m -> 경계는 그 중간값
    'slot1_max_confirm_m': 0.87,    # 확정 거리 < 0.87 m -> 1번 칸
    'slot2_max_confirm_m': 1.25,    # 확정 거리 < 1.25 m -> 2번 칸, 그 이상 -> 3번 칸
    'scan_give_up_m': 2.00,         # 여기까지 확정 못 하면 fallback_slot
    'open_min_range_m': 1.0,        # 우측 P30 >= 이 값이면 열림 (막힘 0.46~0.90, 열림 1.9~5.2)
    # ===== 칸별 주차 시간 (실차에서 안 맞으면 여기만 조정) =====
    # 2번 칸 = 기존에 맞춰 둔 값. 정지선에서 speed(0.5 m/s)로 forward_sec 만큼 직진.
    # 칸 간격 실측 ~0.47 m (차1 중심 0.58 m, 차2 1.04 m, 차3 1.52 m)
    #   -> 0.47 m / 0.5 m/s = 0.94 s 를 2번 칸 forward 에서 빼고(1번) 더함(3번).
    # 1번 칸: 정지선에서 ~0.47 m 덜 감
    'slot1_forward_sec': 0.76,  # a: 1.7 - 0.94
    'slot1_turn_sec': 1.8,      # b
    'slot1_enter_sec': 0.1,     # c
    # 2번 칸 (기존 값 그대로)
    'forward_sec': 1.7,         # a: 정지선에서 직진 시간 [s]
    'turn_sec': 1.8,            # b: 우회전 시간 [s]
    'enter_sec': 0.1,           # c: 칸 안으로 직진 시간 [s]
    # 3번 칸: 정지선에서 ~0.47 m 더 감
    'slot3_forward_sec': 2.64,  # a: 1.7 + 0.94
    'slot3_turn_sec': 1.8,      # b
    'slot3_enter_sec': 0.1,     # c
    # ==========================================================
    'output_topic': '/cmd_vel',
    'approach_speed': 0.25,     # 정지선 접근 속도 [m/s] (너무 빠르면 정지선 지나침)
    'stop_distance_m': 0.05,    # 정지선 멈춤 기준 거리 [m] (0.05m = 5cm 앞)
    'settle_sec': 1.0,          # 정지선 도착 후 주차 시작 전 대기 시간 [s]
    'speed': 0.50,              # 주차 직진 속도 [m/s]
    'turn_speed': 0.50,         # 주차 우회전 속도 [m/s]
    'turn_steer_rad': 0.45,     # 우회전 조향각 크기 [rad]
    'start_delay_sec': 0.0,     # 실행 후 출발까지 대기 [s]
    'wait_enter': False,        # True면 Enter를 눌러야 출발
    'managed_mode': False,     # True: 접근은 매니저 소유, 시작 신호 뒤 STOP_WAIT부터 실행
    # --- 빈 칸 탐색 (정지선 0 m 기준 실측: docs 참고) ---
    'slot_scan_enabled': True,  # False: 탐색 없이 기존 2번 칸 시퀀스
    'scan_only': False,         # True: 칸 판정까지만 하고 그 자리에 정지 (주차 안 함, 판정 시험용)
    'scan_topic': '/scan',
    'odom_topic': '/odom',
    'lidar_front_deg': 0.0,
    'right_angle_min_deg': -105.0,
    'right_angle_max_deg': -75.0,
    'scan_speed': 0.10,          # 탐색 직진 속도 [m/s] (실측과 같은 0.10 권장)
    'return_speed': 0.15,        # 정지선 복귀 후진 속도 크기 [m/s]
    'fallback_slot': 2,
    'sensor_timeout_sec': 0.5,   # odom/scan 이 이보다 오래되면 정지 후 대기
    'sensor_fallback_sec': 3.0,  # 센서 없이 이만큼 지나면 fallback_slot 으로 진행
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
        self.managed = bool(p['managed_mode'])
        self.p = p
        self.slot_scan_enabled = bool(p['slot_scan_enabled'])
        self.classifier = BaySlotClassifier(
            open_min_range_m=float(p['open_min_range_m']),
            ignore_until_m=float(p['slot_ignore_until_m']),
            min_open_m=float(p['slot_min_open_m']),
            slot1_max_confirm_m=float(p['slot1_max_confirm_m']),
            slot2_max_confirm_m=float(p['slot2_max_confirm_m']),
            give_up_m=float(p['scan_give_up_m']))
        self.slot = None
        self._odom_pose = None      # (x, y, yaw)
        self._odom_at = None
        self._scan_p30 = None
        self._scan_at = None
        self._scan_seq = 0
        self._scan_seen_seq = 0
        self._origin = None         # 정지선 위치 (x, y, yaw)
        self._stale_since = None
        self._enabled = False
        self._enable_at = None
        self._active_mission = ''
        self._active_at = None
        self._pause_since = None

        # 칸별 (forward, turn, enter) 시간 [s]. 2번 칸 = 기존 forward/turn/enter_sec
        self.phase_sec_by_slot = {
            1: (p['slot1_forward_sec'], p['slot1_turn_sec'], p['slot1_enter_sec']),
            2: (p['forward_sec'], p['turn_sec'], p['enter_sec']),
            3: (p['slot3_forward_sec'], p['slot3_turn_sec'], p['slot3_enter_sec']),
        }
        self._steer = steer

        # 정지선 정지 이후 실행할 주차 단계들 (칸 판정 후 _set_slot_phases 로 교체)
        self._set_slot_phases(2)

        # 상태 변수
        self.stop_line_detected = False
        self.stop_line_distance = -1.0

        # 상태 머신: WAIT_START -> APPROACH -> STOP_WAIT -> PARKING -> DONE
        self._state = 'WAIT_START'
        self._state_start = None
        self._phase_index = 0
        self._go_at = None

        # Pub / Sub
        output_topic = 'mission/parking/cmd_vel' if self.managed else p['output_topic']
        state_topic = 'mission/parking/state' if self.managed else 'parking_mission/state'
        self.cmd_pub = self.create_publisher(Twist, output_topic, 10)
        self.state_pub = self.create_publisher(String, state_topic, 10)
        if self.managed:
            self.create_subscription(Bool, 'mission/parking/start', self._start_cb, 10)
            self.create_subscription(String, 'mission/active', self._active_cb, 10)
        self.create_subscription(
            Bool, 'mission/stop_line_detected', self._detected_cb, 10)
        self.create_subscription(
            Float32, 'mission/stop_line_distance', self._distance_cb, 10)
        if self.slot_scan_enabled:
            self.create_subscription(Odometry, p['odom_topic'], self._odom_cb, 20)
            self.create_subscription(
                LaserScan, p['scan_topic'], self._scan_cb, qos_profile_sensor_data)

        self.timer = self.create_timer(TICK_SEC, self._tick)
        self.get_logger().info(
            f'정지선 연동 주차 노드 준비 완료 (정지 기준: {self.stop_distance:.2f}m, '
            f'접근 속도: {self.approach_speed:.2f}m/s)')

    def _detected_cb(self, msg: Bool):
        self.stop_line_detected = msg.data

    def _distance_cb(self, msg: Float32):
        self.stop_line_distance = msg.data

    def _odom_cb(self, msg: Odometry):
        pose = msg.pose.pose
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._odom_pose = (pose.position.x, pose.position.y, yaw)
        self._odom_at = time.monotonic()

    def _scan_cb(self, scan: LaserScan):
        self._scan_p30 = scan_sector_percentile(
            scan, float(self.p['right_angle_min_deg']), float(self.p['right_angle_max_deg']),
            percentile=30.0, front_angle_deg=float(self.p['lidar_front_deg']))
        self._scan_at = time.monotonic()
        self._scan_seq += 1

    def _travel(self):
        """정지선(_origin)부터 처음 방향 기준 전진 거리 [m] (후진이면 감소)."""
        x0, y0, yaw0 = self._origin
        x, y, _ = self._odom_pose
        return (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)

    def _sensors_fresh(self, now, need_scan):
        timeout = float(self.p['sensor_timeout_sec'])
        ok = self._odom_at is not None and now - self._odom_at <= timeout
        if need_scan:
            ok = ok and self._scan_at is not None and now - self._scan_at <= timeout
        if ok:
            self._stale_since = None
        elif self._stale_since is None:
            self._stale_since = now
            self.get_logger().warn('odom/scan 없음 - 정지 후 대기')
        return ok

    def _sensor_gave_up(self, now):
        return (self._stale_since is not None
                and now - self._stale_since >= float(self.p['sensor_fallback_sec']))

    def _set_slot_phases(self, slot):
        forward, turn, enter = (float(v) for v in self.phase_sec_by_slot[slot])
        speed, turn_speed = float(self.p['speed']), float(self.p['turn_speed'])
        self.parking_phases = [
            ('forward', forward, speed, 0.0),
            ('turn', turn, turn_speed, -self._steer),
            ('enter', enter, speed, 0.0),
        ]

    def _choose_slot(self, slot, now, reason):
        if slot not in (1, 2, 3):
            slot = int(self.p['fallback_slot'])
        self.slot = slot
        self._set_slot_phases(slot)
        f, t, e = (ph[1] for ph in self.parking_phases)
        self.get_logger().info(
            f'>>> {slot}번 칸 주차 ({reason}): forward {f:.2f}s / turn {t:.2f}s / enter {e:.2f}s')

    def _start_parking(self, now):
        self._phase_index = 0
        self._set_state('PARKING', now)
        self.get_logger().info(f'주차 시작 -> {self.parking_phases[0][0]}')

    def _start_cb(self, msg):
        self._enabled = bool(msg.data)
        self._enable_at = time.monotonic()
        if not self._enabled:
            self._pause()

    def _active_cb(self, msg):
        self._active_mission = msg.data
        self._active_at = time.monotonic()
        if self._active_mission != 'M4':
            self._pause()

    def _pause(self):
        if self._pause_since is None:
            self._pause_since = time.monotonic()
        self.stop()

    def _publish_state(self):
        state = 'complete' if self.managed and self._state == 'DONE' else self._state
        self.state_pub.publish(String(data=state))

    def go(self):
        if self.managed:
            return  # 단독 실행용 go()/Enter로 매니저의 정지 확인을 우회하지 않는다.
        if self._go_at is None:
            self._go_at = time.monotonic() + self.start_delay
            self.get_logger().info(f'{self.start_delay:.1f}초 뒤 출발합니다.')

    def _set_state(self, new_state: str, now: float):
        self._state = new_state
        self._state_start = now
        self._publish_state()
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
        if self.managed:
            allowed = (self._enabled and self._active_mission == 'M4'
                       and self._enable_at is not None and now - self._enable_at <= 0.5
                       and self._active_at is not None and now - self._active_at <= 1.5)
            if not allowed:
                self._pause()
                if self._state == 'DONE':
                    self._publish_state()
                return
            if self._pause_since is not None:
                if self._state_start is not None:
                    self._state_start += now - self._pause_since
                self._pause_since = None
            if self._state == 'WAIT_START':
                self.stop()
                self._set_state('STOP_WAIT', now)
                return

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
                if not self.slot_scan_enabled:
                    self._start_parking(now)
                elif self._sensors_fresh(now, need_scan=True):
                    self._origin = self._odom_pose
                    self.classifier.reset()
                    self._scan_seen_seq = self._scan_seq
                    self._set_state('SCAN', now)
                elif self._sensor_gave_up(now):
                    self._choose_slot(None, now, '센서 없음 - 기본 칸')
                    self._start_parking(now)
            return

        # 2-1. 빈 칸 탐색: 천천히 직진하며 판정
        if self._state == 'SCAN':
            if not self._sensors_fresh(now, need_scan=True):
                self.stop()
                if self._sensor_gave_up(now):
                    self._choose_slot(None, now, '센서 끊김 - 기본 칸')
                    self._set_state('RETURN', now)
                return
            d = self._travel()
            if self._scan_seq != self._scan_seen_seq:
                self._scan_seen_seq = self._scan_seq
                result = self.classifier.update(d, self._scan_p30)
                if result is not None:
                    self.stop()
                    self._choose_slot(result, now, f'판정 @ {d:+.3f}m' if result else
                                      f'{d:.2f}m 까지 빈 칸 없음 - 기본 칸')
                    if bool(self.p['scan_only']):
                        self.get_logger().info(
                            f'scan_only: 판정 결과 {result if result else "없음"}번 칸, 정지')
                        self._set_state('DONE', now)
                        return
                    self._set_state('RETURN', now)
                    return
            self._publish_cmd(float(self.p['scan_speed']), 0.0)
            return

        # 2-2. 정지선(0 m)까지 직진 후진 -> 기존 시퀀스와 같은 출발점
        if self._state == 'RETURN':
            if not self._sensors_fresh(now, need_scan=False):
                self.stop()
                if self._sensor_gave_up(now):
                    self.get_logger().warn('odom 끊김 - 현재 위치에서 주차 시작')
                    self._start_parking(now)
                return
            if self._travel() <= 0.0:
                self.stop()
                self._set_state('RETURN_WAIT', now)
                return
            self._publish_cmd(-abs(float(self.p['return_speed'])), 0.0)
            return

        if self._state == 'RETURN_WAIT':
            self.stop()
            if now - self._state_start >= self.settle_sec:
                self._start_parking(now)
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
            if self.managed:
                self._publish_state()


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
        if not node.managed and node.get_parameter('wait_enter').value and sys.stdin.isatty():
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


def managed_main(args=None):
    """전체 주행용 진입점. 팀원이 조정하는 주차 단계·파라미터는 공유한다."""
    argv = list(sys.argv if args is None else args)
    if '--ros-args' not in argv:
        argv += ['--ros-args']
    main(args=argv + ['-p', 'managed_mode:=true'])


if __name__ == '__main__':
    main()
