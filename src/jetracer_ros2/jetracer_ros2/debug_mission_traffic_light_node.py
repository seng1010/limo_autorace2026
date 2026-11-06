import os
import sys

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import CompressedImage

from jetracer_ros2.mission_traffic_light_node import TrafficLightNode

# mission_traffic_light_node를 그대로 재사용하되(감지/판정 로직은 100% 동일),
# ROI 박스와 red/green 픽셀 비율, 확정 상태를 이미지에 그려서
# /mission_traffic_light/debug/compressed로 추가 발행하는 디버그용 노드.
# RViz(Image 디스플레이) 또는 rqt_image_view로 구독해서 ROI가 실제로 신호등
# 위치에 잘 맞는지, 지금 red/green 비율이 얼마인지 눈으로 확인할 수 있다.
#
# 사용법:
#   ros2 run jetracer_ros2 debug_mission_traffic_light_node
#   rqt_image_view  (또는 RViz에서 /mission_traffic_light/debug/compressed 구독)


class DebugTrafficLightNode(TrafficLightNode):
    def __init__(self):
        super().__init__()
        self._print_output = False  # 터미널 출력은 mission_traffic_light_node 몫으로 남겨둠
        self.debug_pub = self.create_publisher(
            CompressedImage, '/mission_traffic_light/debug/compressed', 10)

    def _on_frame(self, frame, red_ratio, green_ratio, roi_box):
        x0, y0, x1, y1 = roi_box
        vis = frame.copy()

        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)
        cv2.putText(vis, f'red:   {red_ratio * 100:.1f}%', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f'green: {green_ratio * 100:.1f}%', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f'state: {self._confirmed or "-"}', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ok, enc = cv2.imencode('.jpg', vis)
        if not ok:
            return

        out_msg = CompressedImage()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'camera_color_optical_frame'
        out_msg.format = 'jpeg'
        out_msg.data = enc.tobytes()
        self.debug_pub.publish(out_msg)


def main(args=None):
    argv = sys.argv if args is None else args

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
    node = DebugTrafficLightNode()
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
