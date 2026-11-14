from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# Always-on 2026 stack. Mission-prefixed nodes are started and stopped by the
# manager only while their mission is active.
#
# 사용법:
#   ros2 launch jetracer_ros2 all_in_node.launch.py
#
# 차선 주행 (race 브랜치):
#   기본은 simple_lane 이 차선 주행을 맡는다 (inference/cmd_vel 로 매니저에게 제안, 매니저가 /cmd_vel 발행).
#   팀 lane_following_node 로 돌리려면:  ros2 launch jetracer_ros2 all_in_node.launch.py team_lane:=true
#   (둘을 동시에 켜면 inference/cmd_vel 에 명령이 섞이므로 하나만 켠다)
#   슬라이더 창을 보며 튜닝하려면 lane:=false 로 켜고 NoMachine 터미널에서 따로 실행:
#     ros2 run jetracer_ros2 simple_lane drive --topic inference/cmd_vel


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'team_lane', default_value='false',
            description='true: 팀 lane_following_node 사용 / false: simple_lane 사용'),
        DeclareLaunchArgument(
            'lane', default_value='true',
            description='false: 차선 노드를 이 launch 에서 켜지 않음 (따로 켤 때)'),
        Node(
            package='jetracer_ros2',
            executable='mission_manager_node',
            output='screen',
        ),
        Node(
            package='jetracer_ros2',
            executable='lane_following_node',
            output='screen',
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('lane'), "' == 'true' and '",
                LaunchConfiguration('team_lane'), "' == 'true'"])),
        ),
        Node(
            package='jetracer_ros2',
            executable='simple_lane',
            arguments=['drive', '--topic', 'inference/cmd_vel'],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('lane'), "' == 'true' and '",
                LaunchConfiguration('team_lane'), "' != 'true'"])),
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
