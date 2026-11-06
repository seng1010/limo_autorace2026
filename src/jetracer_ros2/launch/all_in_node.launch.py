from launch import LaunchDescription
from launch_ros.actions import Node

# 전체 미션 파이프라인 시작점. 처음부터 항상 떠 있어야 하는 세 노드만 띄운다:
#   - mission_manager_node       : 전체 상태머신, /cmd_vel에 실제로 발행하는 유일한 노드
#   - mission_inference_node     : go_stop.engine으로 조향+정지판단, inference/cmd_vel로만 발행
#   - mission_traffic_light_node : 신호등 인식, green 확정 시 mission_manager_node가 직접 종료시킴
#
# 이후 단계(사람 e-stop, 터널, 트럭 추종, 주차)에 필요한 노드는 mission_manager_node가
# 상태 전환 시점마다 subprocess로 직접 켜고 끈다 — 여기서 미리 띄우지 않는다.
#
# 사용법:
#   ros2 launch jetracer_ros2 all_in_node.launch.py


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='jetracer_ros2',
            executable='mission_manager_node',
            output='screen',
        ),
        Node(
            package='jetracer_ros2',
            executable='mission_inference_node',
            output='screen',
        ),
        Node(
            package='jetracer_ros2',
            executable='mission_traffic_light_node',
            output='screen',
        ),
    ])
