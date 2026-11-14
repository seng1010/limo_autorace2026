from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Always-on 2026 stack. Mission-prefixed nodes are started and stopped by the
# manager only while their mission is active.
#
# 사용법:
#   ros2 launch jetracer_ros2 all_in_node.launch.py
#
# 차선 주행 (race 브랜치):
#   기본은 팀 lane_following_node 를 켜지 않는다. 대신 simple_lane.py 를 따로 켠다:
#     python3 simple_lane.py drive --topic inference/cmd_vel
#   (매니저가 inference/cmd_vel 을 받아 /cmd_vel 로 내보내므로 둘을 동시에 켜면 명령이 섞인다)
#   팀 차선 노드로 돌리려면:  ros2 launch jetracer_ros2 all_in_node.launch.py team_lane:=true


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'team_lane', default_value='false',
            description='true: 팀 lane_following_node 사용 / false: simple_lane.py 를 따로 켬'),
        Node(
            package='jetracer_ros2',
            executable='mission_manager_node',
            output='screen',
        ),
        Node(
            package='jetracer_ros2',
            executable='lane_following_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('team_lane')),
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
