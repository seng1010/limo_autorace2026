import math
import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from jetracer_ros2.mission_track_following_node import MissionTrackFollowingNode

# mission_track_following_node를 그대로 재사용하되(감지 로직은 100% 동일), 실제로
# 감시하고 있는 좌측 범위(각도×거리)를 부채꼴 모양의 Marker로 그려서
# /mission_track_following/debug/fov_marker로 발행하는 디버그용 노드. RViz에서 이
# 토픽(Marker)과 원래 있는 /scan(LaserScan 디스플레이)을 같이 구독하면, 라이다 점들과
# 실제 감지 범위가 겹쳐 보여서 파라미터가 실제 위치에 잘 맞는지 눈으로 확인할 수 있다.
#
# debug_mission_people_estop_node의 FOV 마커는 각도 범위가 좁아서(±10도 등) 삼각형
# 하나로 근사해도 충분했지만, 여기는 좌측 15~70도처럼 각도 폭이 훨씬 넓어서 삼각형
# 하나로 이으면 실제 부채꼴보다 훨씬 좁아 보인다. 그래서 각도를 잘게 나눠 여러 개의
# 삼각형을 이어붙인 부채꼴(fan)로 그린다.
#
# 부채꼴 색: 현재 상태가 'stop'이면 빨강, 'clear'면 초록, 아직 모르면 회색.
#
# 사용법:
#   ros2 run jetracer_ros2 debug_mission_track_following_node
#   RViz에서 LaserScan(/scan) + Marker(/mission_track_following/debug/fov_marker) 둘 다 구독

_FAN_SEGMENTS = 12  # 부채꼴을 몇 개의 삼각형으로 나눠 근사할지


class DebugMissionTrackFollowingNode(MissionTrackFollowingNode):
    def __init__(self):
        super().__init__()
        self.declare_parameter('laser_frame', 'laser_link')
        self.laser_frame = self.get_parameter('laser_frame').value

        self.marker_pub = self.create_publisher(
            Marker, '/mission_track_following/debug/fov_marker', 10)

    def _on_scan(self, msg, obstacle, count):
        marker = Marker()
        marker.header.frame_id = self.laser_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'track_following_fov'
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        if self._state == 'stop':
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        elif self._state == 'clear':
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 0.5, 0.5, 0.5
        marker.color.a = 0.35  # 반투명 — 뒤의 라이다 점이 같이 보이게

        origin = Point(x=0.0, y=0.0, z=0.0)
        span = self.track_angle_max - self.track_angle_min
        points = []
        for i in range(_FAN_SEGMENTS):
            a0 = self.track_angle_min + span * i / _FAN_SEGMENTS
            a1 = self.track_angle_min + span * (i + 1) / _FAN_SEGMENTS
            p0 = Point(
                x=self.track_distance_m * math.cos(a0),
                y=self.track_distance_m * math.sin(a0),
                z=0.0)
            p1 = Point(
                x=self.track_distance_m * math.cos(a1),
                y=self.track_distance_m * math.sin(a1),
                z=0.0)
            points += [origin, p0, p1]
        marker.points = points

        self.marker_pub.publish(marker)


def main(args=None):
    argv = sys.argv if args is None else args

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
    node = DebugMissionTrackFollowingNode()
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
