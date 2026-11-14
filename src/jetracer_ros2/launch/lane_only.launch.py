from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# 차선 주행만 시험 (매니저 없음). simple_lane 이 /cmd_vel 로 바로 보낸다.
# 하드웨어(jetracer_teleop.launch.py: 드라이버 + 카메라 + 라이다 + 변환 노드)를 먼저 켜 둘 것.
#
# 사용법:
#   ros2 launch jetracer_ros2 lane_only.launch.py            주행
#   ros2 launch jetracer_ros2 lane_only.launch.py rec:=true  주행 + 기록 (~/simple_lane_log)
#
# 슬라이더 창은 화면(DISPLAY)이 있는 터미널(NoMachine)에서 켰을 때만 뜬다.
# all_in_node.launch.py 와 동시에 켜지 말 것 (둘 다 차를 움직이려 한다).


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('rec', default_value='false', description='true: 프레임과 상태를 기록'),
        Node(
            package='jetracer_ros2', executable='simple_lane', arguments=['drive'],
            output='screen', emulate_tty=True,
            condition=UnlessCondition(LaunchConfiguration('rec')),
        ),
        Node(
            package='jetracer_ros2', executable='simple_lane', arguments=['drive', 'rec'],
            output='screen', emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('rec')),
        ),
    ])
