from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'jetracer_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'params'),
            glob(os.path.join('params', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wego',
    maintainer_email='wego@todo.todo',
    description='Event-driven perception and mission control for LIMO Autorace 2026',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_collection_node = jetracer_ros2.data_collection_node:main',
            'lane_following_node = jetracer_ros2.lane_following_node:main',
            'stop_line_node = jetracer_ros2.stop_line_node:main',
            'yolo_detection_node = jetracer_ros2.yolo_detection_node:main',
            'mission_manager_node = jetracer_ros2.mission_manager_node:main',
            'mission_traffic_light_node = jetracer_ros2.mission_traffic_light_node:main',
            'mission_obstacle_avoid_node = jetracer_ros2.mission_obstacle_avoid_node:main',
            'mission_cone_avoid_node = jetracer_ros2.mission_cone_avoid_node:main',
            'mission_parking_node = jetracer_ros2.mission_parking_node:main',
            'debug_autorace_node = jetracer_ros2.debug_autorace_node:main',
        ],
    },
)
