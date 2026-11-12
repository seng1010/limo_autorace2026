from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Always-on 2026 stack. Mission-prefixed nodes are started and stopped by the
# manager only while their mission is active.
#
# 사용법:
#   ros2 launch jetracer_ros2 all_in_node.launch.py


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('debug_view', default_value='false'),
        Node(
            package='jetracer_ros2',
            executable='mission_manager_node',
            output='screen',
        ),
        Node(
            package='jetracer_ros2',
            executable='lane_following_node',
            parameters=[{'live_debug_enabled': ParameterValue(LaunchConfiguration('debug_view'), value_type=bool)}],
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
