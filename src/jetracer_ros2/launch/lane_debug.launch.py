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
        DeclareLaunchArgument('near_clamp_m', default_value='0.30'),
        DeclareLaunchArgument('outer_curve_entry', default_value='false'),
        Node(
            package='jetracer_ros2',
            executable='lane_following_node',
            parameters=[{'live_debug_enabled': True, 'publish_commands': False,
                         'metric_lookahead_enabled': ParameterValue(
                             LaunchConfiguration('metric_ackermann'), value_type=bool),
                         'metric_max_near_clamp_m': ParameterValue(
                             LaunchConfiguration('near_clamp_m'), value_type=float),
                         'observed_outer_curve_entry_enabled': ParameterValue(
                             LaunchConfiguration('outer_curve_entry'), value_type=bool)}],
            output='screen',
        ),
    ])
