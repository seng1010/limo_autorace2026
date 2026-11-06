import os
import subprocess

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

# all_in_node.launch.py(미션 파이프라인)가 실제로 구독/제어하는 하드웨어만 띄운다:
#   - limo_base     : /cmd_vel 구독, 실제 모터 구동
#   - orbbec_camera : /camera/color/image_raw/compressed 발행
#   - ydlidar       : /scan 발행
#
# wego_ws의 teleop_launch.py에 있던 URDF/tf, camera_tilt(서보), robot_localization(EKF),
# rviz2는 jetracer_ros2의 mission_*.py 코드가 전혀 사용하지 않아 여기서는 제외했다.
#
# limo_base/orbbec_camera/ydlidar_ros2_driver는 third_impact가 아니라 wego_ws에
# 설치되어 있어서, 터미널에서 wego_ws를 따로 source하지 않아도 되도록 이 launch
# 파일이 직접 source해서 필요한 환경변수(AMENT_PREFIX_PATH 등)를 주입한다.
#
# 사용법 (터미널 2개, wego_ws source 불필요):
#   터미널1) ros2 launch jetracer_ros2 all_in_node.launch.py
#   터미널2) ros2 launch jetracer_ros2 jetracer_teleop.launch.py

WEGO_WS_SETUP = os.path.expanduser('~/wego_ws/install/setup.bash')


def _source_wego_ws():
    if not os.path.isfile(WEGO_WS_SETUP):
        raise RuntimeError(f'wego_ws setup 스크립트를 찾을 수 없음: {WEGO_WS_SETUP}')

    output = subprocess.check_output(
        ['bash', '-c', f'source {WEGO_WS_SETUP} && env -0'])
    for entry in output.decode().split('\0'):
        if not entry:
            continue
        key, _, value = entry.partition('=')
        os.environ[key] = value


def generate_launch_description():
    _source_wego_ws()

    return LaunchDescription([
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('limo_base'), 'launch', 'limo_base.launch.py'])
        ),
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('orbbec_camera'), 'launch', 'astra_stereo_u3.launch.py'])
        ),
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('ydlidar_ros2_driver'), 'launch', 'ydlidar.launch.py'])
        ),
    ])
