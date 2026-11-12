from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Manual-drive inspection: GUI + real lane pipeline, NO driving publisher."""
    return LaunchDescription([
        # Inspection now defaults to the measured rear-axle target. This launch
        # never creates a driving publisher, so it is safe to compare on screen.
        DeclareLaunchArgument('metric_ackermann', default_value='true'),
        Node(
            package='jetracer_ros2',
            executable='lane_following_node',
            parameters=[{'live_debug_enabled': True, 'publish_commands': False,
                         'metric_lookahead_enabled': ParameterValue(
                             LaunchConfiguration('metric_ackermann'), value_type=bool)}],
            output='screen',
        ),
    ])
