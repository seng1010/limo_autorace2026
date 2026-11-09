from launch import LaunchDescription
from launch_ros.actions import Node

# Always-on 2026 stack. Mission-prefixed nodes are started and stopped by the
# manager only while their mission is active.
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
            executable='lane_following_node_ks',
            output='screen',
        ),
        Node(
            package='jetracer_ros2',
            executable='stop_line_node',
            output='screen',
        ),
        Node(
            package='jetracer_ros2',
            executable='yolo_detection_node',
            output='screen',
        ),
    ])
