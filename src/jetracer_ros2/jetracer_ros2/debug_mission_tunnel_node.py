import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory

from jetracer_ros2.mission_tunnel_node import TunnelNode

# mission_tunnel_node를 그대로 재사용하는 디버그용 노드. mission_tunnel_node는 원래도
# print_rate_hz 주기로 터미널에 조도(brightness)를 출력하므로 로직을 따로 바꿀 게
# 없다 — 이 노드는 debug_all_in_node.launch.py에서 mission_manager_node가 TUNNEL
# 단계에 도달하기 전부터도 조도를 계속 확인할 수 있도록 별도 프로세스로 미리
# 띄워두는 용도일 뿐이다.
#
# 사용법:
#   ros2 run jetracer_ros2 debug_mission_tunnel_node


class DebugTunnelNode(TunnelNode):
    pass


def main(args=None):
    argv = sys.argv if args is None else args

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
    node = DebugTunnelNode()
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
