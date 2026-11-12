from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Manual-drive inspection: GUI + real lane pipeline, NO driving publisher."""
    return LaunchDescription([
        Node(
            package='jetracer_ros2',
            executable='lane_following_node',
            parameters=[{'live_debug_enabled': True, 'publish_commands': False}],
            output='screen',
        ),
    ])
