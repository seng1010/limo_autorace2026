import os
import sys

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

# 카메라 이미지의 평균 밝기(조도 근사치)를 mission/tunnel_brightness 토픽으로
# 실시간 발행하고, 터미널에도 주기적으로 출력하는 노드. mission_manager_tunnel_test_node가
# 이 토픽을 구독해서 터널 진입(조도 급락) 여부를 판단한다.
#
# 밝기 = 그레이스케일 변환 후 픽셀 평균값 (0~255, 어두울수록 0에 가까움).
#
# 사용법:
#   ros2 run jetracer_ros2 mission_tunnel_node


class TunnelNode(Node):
    def __init__(self):
        super().__init__('mission_tunnel_node')

        self.declare_parameter('print_rate_hz', 2.0)  # 터미널 출력 주기 (카메라 프레임 속도와 무관)

        print_rate_hz = self.get_parameter('print_rate_hz').value

        self._latest_brightness = None

        self.brightness_pub = self.create_publisher(Float32, 'mission/tunnel_brightness', 10)
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed',
            self.image_callback, 10)
        self.create_timer(1.0 / print_rate_hz, self._print_tick)

        self.get_logger().info(f'Tunnel(조도 관측) ready  print_rate_hz={print_rate_hz}')

    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._latest_brightness = float(np.mean(gray))
        # 카메라 프레임 속도 그대로 실시간 발행 (터미널 출력은 별도 타이머로 throttle됨)
        self.brightness_pub.publish(Float32(data=self._latest_brightness))

    def _print_tick(self):
        if self._latest_brightness is not None:
            print(f'brightness: {self._latest_brightness:.1f}')


def main(args=None):
    argv = sys.argv if args is None else args

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
    node = TunnelNode()
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
