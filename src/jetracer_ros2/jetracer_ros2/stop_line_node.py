"""Detect a horizontal stop line in the lower camera image."""

import os
import sys

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32


class StopLineNode(Node):
    def __init__(self):
        super().__init__('stop_line_node')
        defaults = {
            'image_topic': '/camera/color/image_raw/compressed',
            'roi_top_ratio': 0.58,
            'white_lightness_min': 180,
            'white_saturation_max': 90,
            'horizontal_kernel_ratio': 0.12,
            'minimum_width_ratio': 0.42,
            'minimum_area_ratio': 0.018,
            'distance_scale_m': 1.6,
            'publish_debug': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.distance_pub = self.create_publisher(Float32, 'mission/stop_line_distance', 10)
        self.detected_pub = self.create_publisher(Bool, 'mission/stop_line_detected', 10)
        self.debug_pub = self.create_publisher(CompressedImage, '/stop_line/debug/compressed', 2)
        self.sub = self.create_subscription(
            CompressedImage, self.get_parameter('image_topic').value,
            self._callback, qos_profile_sensor_data)

    def _callback(self, msg):
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.detected_pub.publish(Bool(data=False))
            self.distance_pub.publish(Float32(data=-1.0))
            return
        height, width = image.shape[:2]
        roi_y = int(height * self.get_parameter('roi_top_ratio').value)
        roi = image[roi_y:, :]
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        mask = np.uint8(
            (hls[:, :, 1] >= self.get_parameter('white_lightness_min').value)
            & (hls[:, :, 2] <= self.get_parameter('white_saturation_max').value)) * 255
        kernel_width = max(5, int(width * self.get_parameter('horizontal_kernel_ratio').value))
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, kernel_width), np.uint8))
        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = cv2.contourArea(contour) / float(max(1, roi.shape[0] * width))
            if (w / width >= self.get_parameter('minimum_width_ratio').value
                    and area_ratio >= self.get_parameter('minimum_area_ratio').value):
                if best is None or y + h > best[1] + best[3]:
                    best = (x, y, w, h)
        detected = best is not None
        distance = -1.0
        debug = image.copy()
        if detected:
            x, y, w, h = best
            bottom_ratio = (y + h) / float(max(1, roi.shape[0]))
            distance = max(0.0, (1.0 - bottom_ratio) * self.get_parameter('distance_scale_m').value)
            cv2.rectangle(debug, (x, roi_y + y), (x + w, roi_y + y + h), (0, 255, 0), 3)
            cv2.putText(debug, f'{distance:.2f} m', (x, max(20, roi_y + y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.detected_pub.publish(Bool(data=detected))
        self.distance_pub.publish(Float32(data=float(distance)))
        if self.get_parameter('publish_debug').value:
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
