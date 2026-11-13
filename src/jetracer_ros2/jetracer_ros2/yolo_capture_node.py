"""YOLO 학습용 이미지 자동 촬영 노드.

실행하면 바로, 로봇이 움직일 때마다 카메라 이미지를 자동으로
<save_dir>/<label>/ 폴더에 저장한다. (기본: ~/limo_autorace2026/yolo_data/slow_sign/)
파일명은 기존 yolo_data 형식과 같다: <label>_<HHMMSS>_<번호4자리>.jpg
라벨링(박스 그리기)은 나중에 따로 하고, 여기서는 물체별로 폴더만 나눠서 모은다.

촬영 조건 (trigger 파라미터):
  - 'cmd_vel' (기본): 마지막으로 받은 /cmd_vel이 0이 아니면(=리모가 움직이는 중)
                  min_interval_sec마다 1장 저장. 정지 명령(0)이 오면 촬영도 멈춤
                  (limo_base는 마지막 명령을 계속 유지하고, teleop_twist_keyboard는
                   키를 누를 때만 발행하므로 '마지막 명령' 기준으로 판단한다)
                  ※ wego_ws의 limo_base는 아커만 모드에서 /odom을 발행하지 않으므로
                    (publishOdometry에 ACKERMANN 케이스 없음) 기본값을 이걸로 둔다.
  - 'move'      : /odom 기준으로 min_distance_m 이상 이동했거나
                  min_rotation_rad 이상 회전했을 때 1장 저장
                  → 정지해 있으면 같은 사진이 쌓이지 않음
  - 'time'      : 움직임과 상관없이 min_interval_sec마다 1장 저장
                  (odom이 없거나, 손으로 물체를 옮기며 찍을 때)
어느 모드든 min_interval_sec보다 빠르게 저장하지는 않는다.

일시정지 (선택):
  - 실행하면 바로 녹화 중이다 (start_paused:=true 로 일시정지 상태 시작 가능)
  - 실행한 터미널에서 [스페이스바] 또는 [Enter] → 녹화 일시정지/재개
  - 터미널 입력을 못 쓸 때(launch 등)는 토픽으로도 가능:
      ros2 topic pub --once /yolo_capture/enable std_msgs/Bool "data: true"

주행은 이 노드가 하지 않는다. 조이스틱 텔레옵 등 다른 노드로 움직이면 된다.

사용법:
  ros2 run jetracer_ros2 yolo_capture_node                    # → yolo_data/slow_sign/
  ros2 run jetracer_ros2 yolo_capture_node label:=parking_sign # → yolo_data/parking_sign/
  ros2 run jetracer_ros2 yolo_capture_node label:=pedestrian trigger:=time
  ros2 run jetracer_ros2 yolo_capture_node label:=background   # 물체 없는 배경 사진
"""

import math
import os
import re
import select
import sys
import termios
import threading
import tty
from datetime import datetime

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

# README의 mission/yolo_event 값과 동일하게 맞춘 권장 클래스 이름
KNOWN_LABELS = {
    'traffic_red', 'traffic_yellow', 'traffic_green', 'sign_a', 'sign_b',
    'pedestrian', 'slow_sign', 'parking_sign', 'background',
}


class YoloCaptureNode(Node):
    def __init__(self):
        super().__init__('yolo_capture_node')
        defaults = {
            'label': 'slow_sign',
            'save_dir': os.path.expanduser('~/limo_autorace2026/yolo_data'),
            'image_topic': '/camera/color/image_raw/compressed',
            'odom_topic': '/odom',
            'cmd_vel_topic': '/cmd_vel',
            'trigger': 'cmd_vel',       # 'cmd_vel' | 'move' | 'time'
            'cmd_vel_timeout_sec': 0.0, # >0이면 마지막 명령 후 이 시간이 지나면 정지로 판단
            'min_distance_m': 0.05,     # 이만큼 이동할 때마다 1장
            'min_rotation_rad': 0.15,   # 또는 이만큼 회전할 때마다 1장 (약 8.6도)
            'min_interval_sec': 0.2,    # 최대 저장 속도 제한 (5장/초)
            'jpeg_quality': 95,
            'start_paused': False,      # True면 스페이스바를 눌러야 녹화 시작
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        label = str(self.get_parameter('label').value).strip()
        if not label:
            raise ValueError('label이 비어 있습니다. 예: label:=slow_sign')
        if not re.fullmatch(r'[A-Za-z0-9_\-]+', label):
            raise ValueError(f'label에는 영문/숫자/_/- 만 쓸 수 있습니다: {label!r}')
        if label not in KNOWN_LABELS:
            self.get_logger().warn(
                f"'{label}'은 README 권장 클래스 이름이 아닙니다 "
                f'(권장: {", ".join(sorted(KNOWN_LABELS))}). 그대로 진행합니다.')
        self.label = label

        self.trigger = self.get_parameter('trigger').value
        if self.trigger not in ('cmd_vel', 'move', 'time'):
            raise ValueError(
                f"trigger는 'cmd_vel', 'move', 'time' 중 하나여야 합니다: {self.trigger!r}")

        self.save_dir = os.path.join(
            os.path.expanduser(self.get_parameter('save_dir').value), label)
        os.makedirs(self.save_dir, exist_ok=True)
        existing = [f for f in os.listdir(self.save_dir) if f.endswith('.jpg')]
        # 기존 파일 뒤에 이어서 번호를 매긴다 (예: slow_sign_170716_0012.jpg → 다음은 0013)
        numbers = [int(m.group(1)) for f in existing
                   if (m := re.search(r'_(\d+)\.jpg$', f))]
        self._next_index = max(numbers) + 1 if numbers else 0

        self.saved = 0
        self._last_save = 0.0
        self._pose = None          # 현재 (x, y, yaw)
        self._last_pose = None     # 마지막으로 저장했을 때의 (x, y, yaw)
        self._odom_warned = False
        self._cmd_moving = False      # 마지막으로 받은 /cmd_vel이 이동 명령인지
        self._last_cmd_time = 0.0     # 마지막 /cmd_vel 수신 시각 [s]
        self._start = self.get_clock().now()
        self.recording = not self.get_parameter('start_paused').value
        self._session_saved = 0     # 이번 ON 구간에서 저장한 장수

        self.create_subscription(
            CompressedImage, self.get_parameter('image_topic').value,
            self._image_callback, qos_profile_sensor_data)
        if self.trigger == 'move':
            self.create_subscription(
                Odometry, self.get_parameter('odom_topic').value, self._odom_callback, 20)
        if self.trigger == 'cmd_vel':
            self.create_subscription(
                Twist, self.get_parameter('cmd_vel_topic').value, self._cmd_vel_callback, 10)
        self.create_subscription(Bool, 'yolo_capture/enable', self._enable_callback, 10)

        self.get_logger().info(
            f'촬영 준비 완료  label={label}  trigger={self.trigger}  '
            f'save_dir={self.save_dir}  (기존 {len(existing)}장)')
        self.get_logger().info(
            f"현재 {'녹화 중' if self.recording else '일시정지'}  "
            '— [스페이스바/Enter]로 녹화 ON/OFF, Ctrl+C로 종료')

    def set_recording(self, on):
        if on == self.recording:
            return
        self.recording = on
        if on:
            self._session_saved = 0
            self._last_pose = None  # 켜자마자 첫 장 저장
            self.get_logger().info(f'● 녹화 ON  [{self.label}]')
        else:
            self.get_logger().info(
                f'■ 녹화 OFF  (이번 구간 {self._session_saved}장, 누적 {self.saved}장)')

    def _enable_callback(self, msg):
        self.set_recording(bool(msg.data))

    def _cmd_vel_callback(self, msg):
        self._cmd_moving = abs(msg.linear.x) > 1e-3 or abs(msg.angular.z) > 1e-3
        self._last_cmd_time = self.get_clock().now().nanoseconds / 1e9

    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (p.x, p.y, yaw)

    def _should_capture(self, now):
        if not self.recording:
            return False
        if now - self._last_save < self.get_parameter('min_interval_sec').value:
            return False
        if self.trigger == 'time':
            return True
        if self.trigger == 'cmd_vel':
            timeout = self.get_parameter('cmd_vel_timeout_sec').value
            if timeout > 0 and now - self._last_cmd_time > timeout:
                return False
            return self._cmd_moving
        if self._pose is None:
            waited = (self.get_clock().now() - self._start).nanoseconds / 1e9
            if waited > 3.0 and not self._odom_warned:
                self._odom_warned = True
                self.get_logger().warn(
                    f"{self.get_parameter('odom_topic').value} 수신이 없습니다. "
                    "base 드라이버가 켜져 있는지 확인하거나 trigger:=time 으로 실행하세요.")
            return False
        if self._last_pose is None:
            return True  # 첫 장은 바로 저장
        dx = self._pose[0] - self._last_pose[0]
        dy = self._pose[1] - self._last_pose[1]
        dyaw = abs(math.atan2(math.sin(self._pose[2] - self._last_pose[2]),
                              math.cos(self._pose[2] - self._last_pose[2])))
        return (math.hypot(dx, dy) >= self.get_parameter('min_distance_m').value
                or dyaw >= self.get_parameter('min_rotation_rad').value)

    def _image_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if not self._should_capture(now):
            return
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        timestamp = datetime.now().strftime('%H%M%S')
        path = os.path.join(self.save_dir, f'{self.label}_{timestamp}_{self._next_index:04d}.jpg')
        quality = int(self.get_parameter('jpeg_quality').value)
        if not cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            self.get_logger().error(f'저장 실패: {path}')
            return
        self._last_save = now
        self._next_index += 1
        self._last_pose = self._pose
        self.saved += 1
        self._session_saved += 1
        if self.saved % 20 == 0:
            self.get_logger().info(f'[{self.label}] {self.saved}장 저장')


def _keyboard_loop(node, stop_event):
    """터미널에서 스페이스바/Enter를 누를 때마다 녹화를 토글한다."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # 엔터 없이 한 글자씩 읽기 (Ctrl+C는 그대로 동작)
        while not stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready and sys.stdin.read(1) in (' ', '\n', '\r'):
                node.set_recording(not node.recording)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _plain_overrides(argv, names):
    """`label:=slow_sign`처럼 --ros-args 없이 넘긴 인자를 -p 인자로 바꾼다."""
    ros_args = []
    for arg in argv[1:]:
        m = re.match(r'^([A-Za-z_]+):=(.*)$', arg)
        if m and m.group(1) in names:
            ros_args += ['-p', f'{m.group(1)}:={m.group(2)}']
    return ros_args


def main(args=None):
    argv = sys.argv if args is None else args
    names = {'label', 'save_dir', 'image_topic', 'odom_topic', 'trigger',
             'cmd_vel_topic', 'cmd_vel_timeout_sec',
             'min_distance_m', 'min_rotation_rad', 'min_interval_sec', 'jpeg_quality',
             'start_paused'}
    ros_args = ['--ros-args']
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    if os.path.exists(params):
        ros_args += ['--params-file', params]
    ros_args += _plain_overrides(argv, names)
    # 사용자가 직접 --ros-args 를 붙여 실행한 경우 그것도 그대로 전달
    if '--ros-args' in argv:
        ros_args += argv[argv.index('--ros-args') + 1:]

    rclpy.init(args=[argv[0]] + ros_args)
    node = None
    stop_event = threading.Event()
    key_thread = None
    try:
        node = YoloCaptureNode()
        if sys.stdin.isatty():
            key_thread = threading.Thread(
                target=_keyboard_loop, args=(node, stop_event), daemon=True)
            key_thread.start()
        else:
            node.get_logger().warn(
                '터미널 입력이 없어 키보드 토글을 쓸 수 없습니다. '
                '/yolo_capture/enable 토픽으로 켜고 끄세요.')
        rclpy.spin(node)
    except ValueError as error:
        print(f'[yolo_capture_node] {error}', file=sys.stderr)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        stop_event.set()
        if key_thread is not None:
            key_thread.join(timeout=1.0)  # 터미널 설정 원복
        if node is not None:
            print(f'[yolo_capture_node] 종료: [{node.label}] 이번 실행에서 {node.saved}장 저장 '
                  f'-> {node.save_dir}')
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
