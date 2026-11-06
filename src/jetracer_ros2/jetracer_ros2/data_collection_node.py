import math
import os
import re
import subprocess
import sys
from datetime import datetime

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Joy

from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega, HARDWARE_MAX_INNER_ANGLE

# 조이스틱으로 직접 주행하면서 (이미지, 조향값) 데이터를 모으는 노드.
# X버튼(deadman)을 누르고 있는 동안만 실제로 주행하고 저장도 한다.
#
# 라벨 저장 방식: 파일명에 cx(0~300 정수)를 인코딩한다.
#   cx = 150 - (inner_angle / max_steer_angle) * 150
#   → 150=직진, 0=완전좌회전, 300=완전우회전
# 별도 CSV 없이 파일명만으로 라벨이 따라다니는 NVIDIA JetRacer 표준 관례를
# 그대로 따른다. train.py가 이 형식을 그대로 읽어서 -1~1 float으로 변환해 학습한다.
#
# 전처리: 640×480 원본 → 하단 몇 %만 크롭해서 그대로 저장 (리사이즈 없음, 비율은 params.yaml의 crop_ratio)
# 학습/추론 시 224×224로 단 한 번만 리사이즈 → 리사이즈를 여러 번 거치며
# 생기는 품질 손실을 최소화하기 위함.
#
# joy_node는 이 스크립트가 서브프로세스로 함께 띄운다 (예전 data_collection.launch.py가
# joy_node + data_collector_node 두 개를 띄우던 것과 동일한 구성).
#
# 사용법:
#   ros2 run jetracer_ros2 data_collection_node
#   ros2 run jetracer_ros2 data_collection_node section_id:=2


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')

        # section_id: 코스 구간 번호. 구간별로 별도 폴더(section_N)에 저장하고,
        # 나중에 구간마다 별도 모델을 학습시켜 inference_node가 구간별로 바꿔가며 씀.
        self.declare_parameter('section_id', 1)
        self.declare_parameter('save_dir', '/home/wego/third_impact/src/jetracer_ros2/jetracer_dataset')
        self.declare_parameter('joy_steering_axis', 0)
        self.declare_parameter('deadman_button', 3)   # 이 패드에서 X버튼 = buttons[3]
        self.declare_parameter('max_steer_angle', 0.42)  # LIMO 실제 최대 조향(안쪽 바퀴) 각도, rad
        # 이 패드는 axes[0] > 0 이 왼쪽이다 (일반적인 조이스틱 관례와 반대, 실측 확인됨).
        # 실제 좌회전이 우회전보다 약하게 꺾이는 게 확인되면 1.0보다 크게 올려서 보정한다.
        self.declare_parameter('left_steer_boost', 1.0)
        self.declare_parameter('linear_x', 1.0)   # X버튼 눌렀을 때 고정 주행 속도 [m/s]
        self.declare_parameter('save_rate_hz', 10.0)  # 저장 최대 주기 (X를 누르고 있을 때만 실제 저장됨)
        # 640×480 원본에서 위쪽 몇 %를 버릴지 (0.4 = 위 40% 제거, 0.0 = 크롭 없음).
        # inference_node의 crop_ratio와 반드시 같은 값을 써야 한다.
        self.declare_parameter('crop_ratio', 0.0)

        section_id = self.get_parameter('section_id').value
        base_dir = self.get_parameter('save_dir').value
        self.joy_axis = self.get_parameter('joy_steering_axis').value
        self.deadman_btn = self.get_parameter('deadman_button').value
        self.max_steer_angle = self.get_parameter('max_steer_angle').value
        self.left_steer_boost = self.get_parameter('left_steer_boost').value
        self.linear_x = self.get_parameter('linear_x').value
        self.crop_ratio = self.get_parameter('crop_ratio').value

        self.save_dir = os.path.join(base_dir, f'section_{section_id}')
        os.makedirs(self.save_dir, exist_ok=True)

        self.current_image = None
        self.inner_angle = 0.0  # 의도한 실제 앞바퀴 조향각 (rad). cx 라벨 계산에도 이 값을 그대로 씀.
        self.deadman_active = False  # X버튼을 누르고 있는지 여부
        self.frame_index = 0

        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed',
            self.image_callback, 10)
        self.joy_sub = self.create_subscription(
            Joy, '/joy',
            self.joy_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 이미지 저장 전용 타이머. 이 타이머 자체는 X버튼과 무관하게 항상 돌아가지만,
        # 콜백 안에서 deadman_active를 확인하기 때문에 실제 저장은 X를 누르고 있을 때만 일어난다.
        period = 1.0 / self.get_parameter('save_rate_hz').value
        self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'DataCollector ready  section={section_id}  save_dir={self.save_dir}')
        self.get_logger().info(
            f'Hold button[{self.deadman_btn}] to start driving and saving.')

    # ------------------------------------------------------------------ #

    def image_callback(self, msg: CompressedImage): 
        """카메라 프레임을 받아서 하단 40%만 잘라 최신 이미지로 보관한다."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            crop_y = int(h * self.crop_ratio)
            # crop_ratio만큼 위쪽을 잘라내고 나머지는 원본 해상도 그대로 유지 (리사이즈 없음)
            self.current_image = img[crop_y:h, :]

    def joy_callback(self, msg: Joy):
        """조이스틱 입력을 받아서 조향의도(inner_angle)와 deadman 상태를 갱신하고,
        joy 주기(30~50Hz)에 맞춰 곧바로 /cmd_vel을 publish한다 (조향 지연 최소화)."""
        if len(msg.axes) > self.joy_axis:
            # 스틱 위치(-1~1) × 최대 조향각 = 이번 프레임에 원하는 실제 조향각
            inner_angle = float(msg.axes[self.joy_axis]) * self.max_steer_angle
            if inner_angle > 0:  # 이 패드에서 양수 축값 = 왼쪽
                inner_angle *= self.left_steer_boost
                inner_angle = min(inner_angle, HARDWARE_MAX_INNER_ANGLE)
            self.inner_angle = inner_angle

        if len(msg.buttons) > self.deadman_btn:
            self.deadman_active = bool(msg.buttons[self.deadman_btn])
        else:
            self.deadman_active = False

        cmd = Twist()
        if self.deadman_active:
            cmd.linear.x = self.linear_x
            # limo_base는 Ackermann 모드에서 r = linear.x / angular.z 로
            # 실제 조향각을 역산하기 때문에, 우리가 원하는 "조향각"을 그대로
            # angular.z에 넣으면 안 되고, 그 공식을 거꾸로 계산해서 원하는
            # 조향각이 나오게 하는 angular.z를 만들어 보내야 한다.
            cmd.angular.z = inner_angle_to_omega(self.inner_angle, self.linear_x)
        else:
            # X를 안 누르고 있을 때: linear.x=0으로 두되, angular.z는 부호가 있는
            # 값을 그대로 보낸다. linear.x==0이면 limo_base의 r=v/angular.z 공식이
            # 성립하지 않아서 무조건 최대 조향각으로 튀는데(비례 조향 불가), 그 특성을
            # 이용해서 "정지 상태에서 스틱 방향으로 최대 조향각 미리보기"를 구현한다.
            if abs(self.inner_angle) > 1e-3:
                cmd.angular.z = math.copysign(1.0, self.inner_angle)
        self.cmd_pub.publish(cmd)

    def timer_callback(self):
        """이미지 저장 전용 타이머 콜백 (save_rate_hz 주기, 기본 10Hz)."""
        if self.deadman_active and self.current_image is not None:
            self._save_frame()

    def _save_frame(self):
        # cx ∈ [0, 300]: 150=직진, 0=완전좌회전, 300=완전우회전
        cx = int(150.0 - self.inner_angle / self.max_steer_angle * 150.0)
        cx = max(0, min(300, cx))

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        # 기존 데이터셋과 동일한 파일명 형식: {cx1}_{cx2}_{인덱스}_{타임스탬프}.jpg
        # (JetRacer 원본은 (x,y) 두 좌표를 쓰지만 여기서는 조향값 하나만 쓰므로 cx1==cx2)
        filename = f'{cx:03d}_{cx:03d}_{self.frame_index:06d}_{timestamp}.jpg'
        cv2.imwrite(os.path.join(self.save_dir, filename), self.current_image)

        self.frame_index += 1
        if self.frame_index % 100 == 0:
            self.get_logger().info(
                f'Saved {self.frame_index} frames  '
                f'inner_angle={self.inner_angle:.3f}  cx={cx}')


# ------------------------------------------------------------------ #

def _extract_plain_arg(argv, name, default):
    """`name:=value` 형태로 넘어온 인자를 찾는다.

    ros2 run으로 `--ros-args` 없이 바로 넘긴 `section_id:=2` 같은 인자는
    rclpy가 파라미터로 자동 인식하지 않으므로 직접 파싱한다.
    """
    pattern = re.compile(rf'^{re.escape(name)}:=(.*)$')
    for arg in argv:
        m = pattern.match(arg)
        if m:
            return m.group(1)
    return default


def main(args=None):
    argv = sys.argv if args is None else args
    section_id = int(_extract_plain_arg(argv, 'section_id', 1))

    # params.yaml에서 data_collector_node 설정을 그대로 읽어와서,
    # section_id만 이번 실행에 맞게 덮어쓴다.
    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    with open(params_path) as f:
        all_params = yaml.safe_load(f)
    node_params = dict(all_params['data_collector_node']['ros__parameters'])
    node_params['section_id'] = section_id

    ros_args = ['--ros-args']
    for key, value in node_params.items():
        ros_args += ['-p', f'{key}:={value}']

    # 조이스틱 드라이버 — /joy 토픽 발행
    joy_proc = subprocess.Popen(
        ['ros2', 'run', 'joy', 'joy_node',
         '--ros-args', '-p', 'autorepeat_rate:=20.0'])

    rclpy.init(args=[argv[0]] + ros_args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        joy_proc.terminate()
        joy_proc.wait()


if __name__ == '__main__':
    main()
