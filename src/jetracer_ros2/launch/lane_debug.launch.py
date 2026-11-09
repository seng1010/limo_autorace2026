from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Run perception only; no node in this launch publishes /cmd_vel."""
    return LaunchDescription([
        Node(
            package='jetracer_ros2',
            executable='lane_following_node_ks',
            output='screen',
        ),
    ])
