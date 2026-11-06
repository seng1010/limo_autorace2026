import os
import sys

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# 좌측 하단에 놓인 핸드폰 신호등(빨강/초록 화면)을 카메라 이미지의 ROI 영역에서
# HSV 색상 임계값으로 감지해서, 확정되면 터미널에 'red'/'green'을 출력하고
# mission/traffic_light_state 토픽(String)으로도 발행하는 노드. mission_manager_node가
# 이 토픽을 구독해서 주행 여부를 결정한다. ML 모델 없이 순수 색상 기반이라
# 가볍고 빠르다 (핸드폰 화면은 자체발광이라 조명 반사에 덜 민감하고 채도가
# 높아서 임계값으로 충분히 구분됨).
#
# 판정 방식: ROI 안에서 빨강/초록 각각의 픽셀 비율을 구해서 min_pixel_ratio를 넘는
# 쪽이 우세하면 후보로 삼고, hold_sec 동안 같은 후보가 계속 유지돼야 확정한다
# (깜빡임/노이즈로 인한 오탐 방지).
#
# 사용법:
#   ros2 run jetracer_ros2 mission_traffic_light_node


class TrafficLightNode(Node):
    def __init__(self):
        super().__init__('mission_traffic_light_node')

        # ROI: 이미지 좌측 하단 (기본값). 0.0~1.0 비율로 지정.
        self.declare_parameter('roi_x_start_ratio', 0.0)
        self.declare_parameter('roi_x_end_ratio', 0.35)
        self.declare_parameter('roi_y_start_ratio', 0.65)
        self.declare_parameter('roi_y_end_ratio', 1.0)

        # HSV 범위 (OpenCV 기준: H 0~179, S/V 0~255).
        # 빨강은 색상환 양 끝(0 근처 / 180 근처)에 걸쳐 있어서 두 구간이 필요하다.
        self.declare_parameter('red_low1', [0, 100, 100])
        self.declare_parameter('red_high1', [10, 255, 255])
        self.declare_parameter('red_low2', [170, 100, 100])
        self.declare_parameter('red_high2', [180, 255, 255])
        self.declare_parameter('green_low', [40, 100, 100])
        self.declare_parameter('green_high', [90, 255, 255])

        self.declare_parameter('min_pixel_ratio', 0.15)  # ROI 내 이 비율 이상이어야 후보
        self.declare_parameter('hold_sec', 0.3)           # 이 시간 연속 같은 색이어야 확정

        self.roi_x_start = self.get_parameter('roi_x_start_ratio').value
        self.roi_x_end = self.get_parameter('roi_x_end_ratio').value
        self.roi_y_start = self.get_parameter('roi_y_start_ratio').value
        self.roi_y_end = self.get_parameter('roi_y_end_ratio').value

        self.red_low1 = self.get_parameter('red_low1').value
        self.red_high1 = self.get_parameter('red_high1').value
        self.red_low2 = self.get_parameter('red_low2').value
        self.red_high2 = self.get_parameter('red_high2').value
        self.green_low = self.get_parameter('green_low').value
        self.green_high = self.get_parameter('green_high').value

        self.min_pixel_ratio = self.get_parameter('min_pixel_ratio').value
        self.hold_sec = self.get_parameter('hold_sec').value

        self._candidate = None
        self._candidate_since = self.get_clock().now()
        self._confirmed = None
        self._print_output = True  # debug 서브클래스가 False로 끔 (터미널 출력 중복 방지)

        self.state_pub = self.create_publisher(String, 'mission/traffic_light_state', 10)
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed',
            self.image_callback, 10)

        self.get_logger().info(
            f'TrafficLight ready  roi=x[{self.roi_x_start},{self.roi_x_end}] '
            f'y[{self.roi_y_start},{self.roi_y_end}]  '
            f'min_pixel_ratio={self.min_pixel_ratio}  hold_sec={self.hold_sec}')

    # ------------------------------------------------------------------ #

    def _compute_ratios(self, frame):
        """ROI를 잘라 HSV로 변환하고 빨강/초록 픽셀 비율을 계산한다."""
        h, w = frame.shape[:2]
        x0 = int(w * self.roi_x_start)
        x1 = int(w * self.roi_x_end)
        y0 = int(h * self.roi_y_start)
        y1 = int(h * self.roi_y_end)
        roi = frame[y0:y1, x0:x1]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, np.array(self.red_low1), np.array(self.red_high1))
        red_mask2 = cv2.inRange(hsv, np.array(self.red_low2), np.array(self.red_high2))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, np.array(self.green_low), np.array(self.green_high))

        total = roi.shape[0] * roi.shape[1]
        if total == 0:
            return 0.0, 0.0, (x0, y0, x1, y1)

        red_ratio = float(np.count_nonzero(red_mask)) / total
        green_ratio = float(np.count_nonzero(green_mask)) / total
        return red_ratio, green_ratio, (x0, y0, x1, y1)

    def _update_state(self, red_ratio: float, green_ratio: float):
        """min_pixel_ratio/hold_sec 조건을 거쳐 확정되면 토픽 발행(+터미널 출력)한다."""
        candidate = None
        if red_ratio >= self.min_pixel_ratio and red_ratio > green_ratio:
            candidate = 'red'
        elif green_ratio >= self.min_pixel_ratio and green_ratio > red_ratio:
            candidate = 'green'

        now = self.get_clock().now()
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = now

        if candidate is not None:
            elapsed = (now - self._candidate_since).nanoseconds / 1e9
            if elapsed >= self.hold_sec and candidate != self._confirmed:
                self._confirmed = candidate
                self.state_pub.publish(String(data=candidate))
                if self._print_output:
                    print(candidate)

    def _on_frame(self, frame, red_ratio, green_ratio, roi_box):
        """디버그 시각화 훅 (기본 노드는 아무것도 안 함, debug 서브클래스가 오버라이드)."""
        pass

    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        red_ratio, green_ratio, roi_box = self._compute_ratios(frame)
        self._update_state(red_ratio, green_ratio)
        self._on_frame(frame, red_ratio, green_ratio, roi_box)


def main(args=None):
    argv = sys.argv if args is None else args

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
    node = TrafficLightNode()
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
