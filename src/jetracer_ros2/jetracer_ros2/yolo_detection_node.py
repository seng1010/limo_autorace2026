"""Single YOLO front end for all camera-based mission classes.

Output ``perception/detections`` is JSON so the detector can evolve without a
custom message build. ``mission/yolo_event`` carries the best stable semantic
event for the mission manager.
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from jetracer_ros2.yolo_event_tracker import StableEventTracker


class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        defaults = {
            'image_topic': '/camera/color/image_raw/compressed',
            'model_path': '',
            'confidence': 0.45,
            'iou': 0.5,
            'image_size': 640,
            'max_rate_hz': 10.0,
            'confirm_frames': 3,
            'publish_debug': True,
            'traffic_roi': [0.0, 0.0, 1.0, 0.65],
            'advertisement_exclusion_roi': [0.0, 0.65, 1.0, 1.0],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._model = None
        self._last_inference = 0.0
        self._tracker = StableEventTracker(self.get_parameter('confirm_frames').value)
        self._load_model()
        self.detections_pub = self.create_publisher(String, 'perception/detections', 10)
        self.event_pub = self.create_publisher(String, 'mission/yolo_event', 10)
        self.debug_pub = self.create_publisher(CompressedImage, '/yolo/debug/compressed', 2)
        self.sub = self.create_subscription(
            CompressedImage, self.get_parameter('image_topic').value,
            self._callback, qos_profile_sensor_data)

    def _load_model(self):
        model_path = self.get_parameter('model_path').value
        if not model_path:
            self.get_logger().warn('YOLO model_path is empty; detector will publish no detections')
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self.get_logger().info(f'loaded YOLO model: {model_path}')
        except (ImportError, OSError, RuntimeError) as error:
            self.get_logger().error(f'failed to load YOLO model: {error}')

    def _callback(self, msg):
        now = time.monotonic()
        if now - self._last_inference < 1.0 / max(0.1, self.get_parameter('max_rate_hz').value):
            return
        self._last_inference = now
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        detections = []
        annotated = image.copy()
        if self._model is not None:
            try:
                result = self._model.predict(
                    source=image, verbose=False,
                    conf=self.get_parameter('confidence').value,
                    iou=self.get_parameter('iou').value,
                    imgsz=self.get_parameter('image_size').value)[0]
                names = result.names
                for box in result.boxes:
                    xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    confidence = float(box.conf[0].cpu())
                    class_id = int(box.cls[0].cpu())
                    label = str(names[class_id]).lower().strip()
                    if self._excluded_advertisement(label, xyxy, image.shape):
                        continue
                    detections.append({'label': label, 'confidence': confidence, 'bbox': xyxy})
                if self.get_parameter('publish_debug').value:
                    annotated = result.plot()
            except Exception as error:  # backend errors differ between ONNX/TRT/PyTorch
                self.get_logger().error(f'YOLO inference failed: {error}')
        payload = {'stamp_ns': self.get_clock().now().nanoseconds, 'detections': detections}
        self.detections_pub.publish(String(data=json.dumps(payload, separators=(',', ':'))))
        self._publish_stable_event(detections)
        if self.get_parameter('publish_debug').value:
            ok, encoded = cv2.imencode('.jpg', annotated)
            if ok:
                self.debug_pub.publish(
                    CompressedImage(header=msg.header, format='jpeg', data=encoded.tobytes()))

    def _excluded_advertisement(self, label, box, shape):
        if label not in {'red', 'yellow', 'green', 'traffic_red', 'traffic_yellow', 'traffic_green'}:
            return False
        height, width = shape[:2]
        cx = (box[0] + box[2]) * 0.5 / width
        cy = (box[1] + box[3]) * 0.5 / height
        traffic = self.get_parameter('traffic_roi').value
        exclusion = self.get_parameter('advertisement_exclusion_roi').value
        in_traffic = traffic[0] <= cx <= traffic[2] and traffic[1] <= cy <= traffic[3]
        in_exclusion = exclusion[0] <= cx <= exclusion[2] and exclusion[1] <= cy <= exclusion[3]
        return (not in_traffic) or in_exclusion

    @staticmethod
    def _semantic_event(label):
        aliases = {
            'red': 'traffic_red', 'yellow': 'traffic_yellow', 'green': 'traffic_green',
            'traffic light red': 'traffic_red', 'traffic light yellow': 'traffic_yellow',
            'traffic light green': 'traffic_green', 'a': 'sign_a', 'b': 'sign_b',
            'pedestrian': 'pedestrian', 'person': 'pedestrian',
            'slow': 'slow_sign', 'slow_sign': 'slow_sign',
            'parking': 'parking_sign', 'parking_sign': 'parking_sign',
        }
        return aliases.get(label, label if label.startswith(('traffic_', 'sign_')) else '')

    def _publish_stable_event(self, detections):
        """클래스마다 따로 세서, confirm_frames 연속으로 보인 것을 각각 발행한다.

        한 프레임에 여러 종류가 보여도 서로 묻히지 않는다 (예: 보행자 표지판 + 신호등).
        """
        events = (self._semantic_event(det['label']) for det in detections)
        for event in self._tracker.update(events):
            self.event_pub.publish(String(data=event))


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params])
    node = YoloDetectionNode()
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
