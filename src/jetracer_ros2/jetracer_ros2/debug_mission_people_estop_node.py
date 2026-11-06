import math
import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from jetracer_ros2.mission_people_estop_node import MissionPeopleEstopNode

# mission_people_estop_node를 그대로 재사용하되(감지 로직은 100% 동일), 로봇이
# 실제로 감시하고 있는 전방 범위(각도×거리)를 세모(부채꼴을 각도가 작아서 직선으로
# 근사한 삼각형) 모양의 Marker로 그려서 /mission_people_estop/debug/fov_marker로
# 발행하는 디버그용 노드. RViz에서 이 토픽(Marker)과 원래 있는 /scan(LaserScan
# 디스플레이)을 같이 구독하면, 라이다 점들과 실제 감지 범위가 겹쳐 보여서 파라미터가
# 실제 위치에 잘 맞는지 눈으로 확인할 수 있다. 라이다 원본 점들은 이미 RViz의
# LaserScan 디스플레이로 /scan을 그대로 볼 수 있어서 여기서 다시 발행하지 않는다.
#
# 삼각형 색: 현재 상태가 'stop'이면 빨강, 'clear'면 초록, 아직 모르면 회색.
#
# 사용법:
#   ros2 run jetracer_ros2 debug_mission_people_estop_node
#   RViz에서 LaserScan(/scan) + Marker(/mission_people_estop/debug/fov_marker) 둘 다 구독


class DebugMissionPeopleEstopNode(MissionPeopleEstopNode):
    def __init__(self):
        super().__init__()
        self.declare_parameter('laser_frame', 'laser_link')
        self.laser_frame = self.get_parameter('laser_frame').value

        self.marker_pub = self.create_publisher(
            Marker, '/mission_people_estop/debug/fov_marker', 10)

    def _on_scan(self, msg, obstacle, count):
        marker = Marker()
        marker.header.frame_id = self.laser_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'people_estop_fov'
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
        p_min = Point(
            x=self.obstacle_distance_m * math.cos(self.obstacle_angle_min),
            y=self.obstacle_distance_m * math.sin(self.obstacle_angle_min),
            z=0.0)
        p_max = Point(
            x=self.obstacle_distance_m * math.cos(self.obstacle_angle_max),
            y=self.obstacle_distance_m * math.sin(self.obstacle_angle_max),
            z=0.0)
        # 각도 범위가 좁으면(±10도 등) 부채꼴 호를 직선으로 근사해도 시각적으로
        # 충분히 정확하다 — 그래서 원점+양끝점 3개짜리 삼각형 하나로 표현한다.
        marker.points = [origin, p_min, p_max]

        self.marker_pub.publish(marker)


def main(args=None):
    argv = sys.argv if args is None else args

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
    node = DebugMissionPeopleEstopNode()
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
