"""라이다(/scan) 구역별 거리 및 주차 빈 칸 감지 실시간 진단 노드.

모터 명령(/cmd_vel)을 보내지 않으므로 차량을 손으로 밀거나 수동 주행하면서
정면/좌/우 장애물 거리와 주차 빈 칸 판정 여부를 터미널에서 실시간 확인할 수 있다.

사용법:
  ros2 run jetracer_ros2 lidar_check_node
  ros2 run jetracer_ros2 lidar_check_node lidar_front_deg:=180.0 empty_dist:=0.72
"""

import math
import os
import re
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

DEFAULTS = {
    'scan_topic': '/scan',
    'lidar_front_deg': 0.0,   # 라이다 정면 각도 보정 [deg] (뒤집혀 있으면 180.0)
    'empty_dist': 0.72,       # 우측(-90도) 거리가 이 값 이상이면 '빈 칸'으로 표시 [m]
    'warn_dist': 0.25,        # 충돌 경고 표시 기준 거리 [m]
    'print_hz': 5.0,          # 터미널 출력 주기 [Hz] (초당 5회 출력)
}


class LidarCheckNode(Node):
    def __init__(self):
        super().__init__('lidar_check_node')
        for name, value in DEFAULTS.items():
            self.declare_parameter(name, value)
        p = {name: self.get_parameter(name).value for name in DEFAULTS}

        self.lidar_front_rad = math.radians(float(p['lidar_front_deg']))
        self.empty_dist = float(p['empty_dist'])
        self.warn_dist = float(p['warn_dist'])

        self.dists = {
            'left_side': float('inf'),    # 좌측 직각 (+78° ~ +102°)
            'front_left': float('inf'),   # 좌전방 (+15° ~ +65°)
            'front': float('inf'),        # 정면 (-15° ~ +15°)
            'front_right': float('inf'),  # 우전방 (-65° ~ -15°)
            'right_side': float('inf'),   # 우측 직각 (-102° ~ -78°) : 주차칸 깊이 측정
        }
        self.valid_points = 0
        self.total_points = 0
        self.scan_received = False

        self.scan_sub = self.create_subscription(
            LaserScan, p['scan_topic'], self._scan_cb, qos_profile_sensor_data)

        period = 1.0 / max(1.0, float(p['print_hz']))
        self.timer = self.create_timer(period, self._print_status)

        self.get_logger().info(
            f'라이다 진단 시작 (topic={p["scan_topic"]}, front_deg={p["lidar_front_deg"]}, '
            f'empty_dist={self.empty_dist}m)')
        self.get_logger().info('차량 앞/좌/우에 물체나 LIMO를 대보면서 거리를 확인하세요. (Ctrl+C 종료)')

    @staticmethod
    def _normalize_angle(rad):
        return (rad + math.pi) % (2.0 * math.pi) - math.pi

    def _scan_cb(self, msg: LaserScan):
        l_side = float('inf')
        fl = float('inf')
        f = float('inf')
        fr = float('inf')
        r_side_vals = []

        valid = 0
        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and msg.range_min < r < msg.range_max:
                valid += 1
                rel_deg = math.degrees(self._normalize_angle(angle - self.lidar_front_rad))
                if -15.0 <= rel_deg <= 15.0:
                    if r < f:
                        f = r
                elif 15.0 < rel_deg <= 65.0:
                    if r < fl:
                        fl = r
                elif -65.0 <= rel_deg < -15.0:
                    if r < fr:
                        fr = r
                elif 78.0 <= rel_deg <= 102.0:
                    if r < l_side:
                        l_side = r
                elif -102.0 <= rel_deg <= -78.0:
                    r_side_vals.append(r)
            angle += msg.angle_increment

        self.dists['left_side'] = l_side
        self.dists['front_left'] = fl
        self.dists['front'] = f
        self.dists['front_right'] = fr
        if r_side_vals:
            r_side_vals.sort()
            self.dists['right_side'] = r_side_vals[len(r_side_vals) // 2]
        else:
            self.dists['right_side'] = float('inf')

        self.valid_points = valid
        self.total_points = len(msg.ranges)
        self.scan_received = True

    def _fmt(self, val):
        if not math.isfinite(val):
            return '  INF  '
        mark = '!' if val < self.warn_dist else ' '
        return f'{val:5.2f}m{mark}'

    def _print_status(self):
        if not self.scan_received:
            print('[대기중] /scan 토픽 데이터가 아직 들어오지 않았습니다...', flush=True)
            return

        r_val = self.dists['right_side']
        if not math.isfinite(r_val) or r_val >= self.empty_dist:
            slot_status = '빈 칸 (진입 가능 O)'
        else:
            slot_status = '차량 있음 (회피 X)'

        line = (
            f'[유효점:{self.valid_points:3d}/{self.total_points:3d}] '
            f'좌측(90°):{self._fmt(self.dists["left_side"])} | '
            f'좌전:{self._fmt(self.dists["front_left"])} | '
            f'정면(0°):{self._fmt(self.dists["front"])} | '
            f'우전:{self._fmt(self.dists["front_right"])} | '
            f'우측(-90°):{self._fmt(r_val)} -> [{slot_status}]'
        )
        print(line, flush=True)


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
        node = LidarCheckNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
