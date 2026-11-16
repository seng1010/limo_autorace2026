"""M8-only approach/drive/record harness. Hardware runs separately, exactly once."""
import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bag = os.path.expanduser('~/bag_m8_local_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    topics = [
        '/scan', '/odom', '/limo_status', '/camera/color/image_raw/compressed',
        '/camera/color/camera_info', '/inference/cmd_vel', '/cmd_vel', '/limo/ack_cmd',
        '/mission/active', '/mission/event', '/mission/stop_line_visual_distance',
        '/mission/lane/observed_cmd_vel', '/mission/lane/observed_center',
        '/mission/lane/geometry_hint', '/mission/cone/phase', '/mission/cone/exit_line',
        '/mission/cone/approach_risk', '/mission/cone/partial_entry_line', '/mission/cone/exit_progress',
        '/mission/cone/safe_cmd_vel', '/mission/cone/safe_exit_cmd_vel', '/mission/cone/recovery_cmd_vel',
        '/mission/cone/state', '/mission/cone/timing', '/mission/cone/control_timing',
        '/mission_manager/debug/state']
    return LaunchDescription([
        DeclareLaunchArgument('motion', default_value='false', choices=['true', 'false'],
                              description='true permits immediate movement; false is display/record only'),
        DeclareLaunchArgument('start', default_value='M2', choices=['M2', 'M8'],
                              description='M2: lane approach from just before entrance; M8: already at cones'),
        DeclareLaunchArgument('record', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('entry_near_m', default_value='0.45',
                              description='Transfer to guarded approach this far from the entry line (front axle, m)'),
        DeclareLaunchArgument('entry_precheck_m', default_value='0.55',
                              description='Start LiDAR preflight while keeping race lane control (front axle, m)'),
        DeclareLaunchArgument('bag', default_value=bag),
        ExecuteProcess(cmd=['ros2', 'bag', 'record', '-o', LaunchConfiguration('bag'), *topics],
                       output='screen', condition=IfCondition(LaunchConfiguration('record'))),
        Node(package='jetracer_ros2', executable='mission_cone_avoid_node', output='screen',
             parameters=[{'motion_enabled': ParameterValue(LaunchConfiguration('motion'), value_type=bool)}]),
        Node(package='jetracer_ros2', executable='simple_lane', output='screen', emulate_tty=True,
             arguments=['drive', '--topic', 'inference/cmd_vel']),
        Node(package='jetracer_ros2', executable='stop_line_node', output='screen'),
        Node(package='jetracer_ros2', executable='mission_manager_node', output='screen',
             parameters=[{'start_mission': LaunchConfiguration('start'),
                          'auto_start_mission_nodes': False,
                          'm8_entry_arm_near_line_m': ParameterValue(
                              LaunchConfiguration('entry_near_m'), value_type=float),
                          'm8_entry_precheck_m': ParameterValue(
                              LaunchConfiguration('entry_precheck_m'), value_type=float),
                          # In no-motion mode keep the existing immediate guard.
                          # Otherwise the unarmed race lane path bypasses that guard.
                          'm8_entry_arm_after_m': ParameterValue(PythonExpression(
                              ["-1.0 if '", LaunchConfiguration('motion'),
                               "' == 'true' else 0.0"]), value_type=float)}]),
    ])
