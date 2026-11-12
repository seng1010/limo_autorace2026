"""정지선 검출 -> 판정 기준 바퀴(기본 뒷바퀴)까지 거리 발행 (M1, M3 공용).

발행
  mission/stop_line_distance  Float32  기준 바퀴 -> 정지선 거리 [m], 없으면 -1.0
                                       (규정집 v3: 뒷바퀴. reference_wheel 로 front/rear 선택)
  mission/stop_line_detected  Bool     거리 값이 유효한지 (화면 검출 또는 odom 추적)
  /stop_line/debug/compressed          디버그 이미지 (거리, 출처, 행 비율 표시)
  /stop_line/debug/row_ratio  Float32  정지선 아래 가장자리 행 비율 (보정용, 없으면 -1.0)

거리 계산
  - lut_rows / lut_distances_m 이 2개 이상이면 보정표 사용 (scripts/calibrate_stop_line.py)
  - 2개 미만이면 보정 전: 예전 식 (1 - 하단 비율) * distance_scale_m
  - 선이 사각지대로 사라지면 /odom 이동거리로 계속 추정 (use_odom_tracking)
"""

import os
import sys

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32

from jetracer_ros2.stop_line_detector import (
    BlindZoneTracker, OdomPathIntegrator, RowDistanceModel, StopLineConfig, detect_stop_line)


class StopLineNode(Node):
    def __init__(self):
        super().__init__('stop_line_node')
        defaults = {
            'image_topic': '/camera/color/image_raw/compressed',
            'odom_topic': '/odom',
            'roi_top_ratio': 0.58,
            'detect_white': True,
            'white_lightness_min': 180,
            'white_saturation_max': 90,
            'detect_yellow': False,
            'yellow_hue_min': 15,
            'yellow_hue_max': 35,
            'yellow_saturation_min': 80,
            'yellow_lightness_min': 60,
            'horizontal_kernel_ratio': 0.12,
            'minimum_width_ratio': 0.42,
            'minimum_area_ratio': 0.018,
            'distance_scale_m': 1.6,
            'lut_rows': [0.0],               # 2개 미만 = 보정 전
            'lut_distances_m': [0.0],
            'reference_wheel': 'rear',       # 규정집 v3 판정 기준점 = 뒷바퀴
            'wheelbase_m': 0.20,             # 앞바퀴 축 -> 뒷바퀴 축 거리
            'use_odom_tracking': True,
            'blind_track_max_m': 0.8,
            'blind_pass_margin_m': 0.10,
            'blind_jump_tolerance_m': 0.15,
            'publish_debug': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value  # noqa: E731

        self.config = StopLineConfig(**{
            field: p(field) for field in StopLineConfig.__dataclass_fields__})
        reference = str(p('reference_wheel')).strip().lower()
        if reference not in ('front', 'rear'):
            raise ValueError("reference_wheel 은 'front' 또는 'rear'")
        offset = float(p('wheelbase_m')) if reference == 'rear' else 0.0
        self.model = RowDistanceModel(
            p('lut_rows'), p('lut_distances_m'), p('distance_scale_m'), p('roi_top_ratio'),
            reference_offset_m=offset)
        self.tracker = BlindZoneTracker(
            p('blind_track_max_m'), p('blind_pass_margin_m'), p('blind_jump_tolerance_m'))
        self.odom = OdomPathIntegrator()
        self.use_odom = bool(p('use_odom_tracking'))
        self.publish_debug = bool(p('publish_debug'))
        self.get_logger().info(
            f'정지선 거리 기준: {reference} 바퀴, '
            + ('보정표 사용' if self.model.calibrated
                          else '보정 전 (distance_scale_m 선형식)')
            + (', odom 사각지대 추적 ON' if self.use_odom else ''))

        self.distance_pub = self.create_publisher(Float32, 'mission/stop_line_distance', 10)
        self.detected_pub = self.create_publisher(Bool, 'mission/stop_line_detected', 10)
        self.debug_pub = self.create_publisher(CompressedImage, '/stop_line/debug/compressed', 2)
        self.row_pub = self.create_publisher(Float32, '/stop_line/debug/row_ratio', 10)
        self.create_subscription(
            CompressedImage, p('image_topic'), self._image_callback, qos_profile_sensor_data)
        if self.use_odom:
            self.create_subscription(Odometry, p('odom_topic'), self._odom_callback, 20)

    def _odom_callback(self, msg):
        pose = msg.pose.pose.position
        self.tracker.update_odom(self.odom.update(pose.x, pose.y, msg.twist.twist.linear.x))

    def _publish(self, distance):
        valid = distance is not None
        self.detected_pub.publish(Bool(data=valid))
        self.distance_pub.publish(Float32(data=float(distance) if valid else -1.0))

    def _image_callback(self, msg):
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        detection = detect_stop_line(image, self.config)
        visual = self.model.distance(detection.row_ratio) if detection else None
        distance = self.tracker.update(visual) if self.use_odom else visual
        self._publish(distance)
        self.row_pub.publish(Float32(data=detection.row_ratio if detection else -1.0))
        if self.publish_debug:
            self._publish_debug(msg, image, detection, distance)

    def _publish_debug(self, msg, image, detection, distance):
        debug = image
        height = image.shape[0]
        cv2.line(debug, (0, int(height * self.config.roi_top_ratio)),
                 (image.shape[1], int(height * self.config.roi_top_ratio)), (255, 120, 0), 1)
        if detection:
            x, y, w, h = detection.bbox
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 3)
        source = self.tracker.source if self.use_odom else ('visual' if detection else 'none')
        text = (f'{distance:.3f} m ({source})' if distance is not None else 'no line')
        if detection:
            text += f'  row={detection.row_ratio:.3f}'
        cv2.putText(debug, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, encoded = cv2.imencode('.jpg', debug)
        if ok:
            self.debug_pub.publish(
                CompressedImage(header=msg.header, format='jpeg', data=encoded.tobytes()))


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params])
    node = StopLineNode()
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
