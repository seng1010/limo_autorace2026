from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# 3차선(M7) 단독 시험: 단순 버전 3차선 주행 + YOLO(갈림길 표지판). 하드웨어는 따로 켠다.
#   ros2 launch jetracer_ros2 jetracer_teleop.launch.py      (다른 터미널, 한 번만)
#   ros2 launch jetracer_ros2 lane3_test.launch.py           3차선 입구(시작선 0.3~1 m 앞)에 놓고
#   ros2 launch jetracer_ros2 lane3_test.launch.py goal:=b   표지판 대신 갈림길 방향을 정해 둠 (a / b)
#   ros2 launch jetracer_ros2 lane3_test.launch.py yolo:=false rec:=false
# 기록은 ~/simple_lane_log/ (rec:=true 기본)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('goal', default_value='none', description='a / b / none(표지판으로 판단)'),
        DeclareLaunchArgument('rec', default_value='true', description='true: 프레임과 상태 기록'),
        DeclareLaunchArgument('yolo', default_value='true', description='false: YOLO 를 켜지 않음'),
        Node(
            package='jetracer_ros2', executable='simple_lane', output='screen', emulate_tty=True,
            arguments=['drive', 'lane3', 'l3simple', ['goal_', LaunchConfiguration('goal')],
                       ['rec_', LaunchConfiguration('rec')]],
        ),
        Node(
            package='jetracer_ros2', executable='yolo_detection_node', output='screen',
            condition=IfCondition(LaunchConfiguration('yolo')),
        ),
    ])
