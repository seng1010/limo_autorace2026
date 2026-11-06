"""Event-driven 2026 mission state machine and sole ``/cmd_vel`` publisher.

Mission order: M5 -> M6 -> M3 -> M7 -> M2 -> M8 -> M1 -> M4.
Always-on perception nodes publish proposals or events.  This node selects one
command source, applies mission speed/stop constraints, and owns ``/cmd_vel``.
"""

import os
import signal
import subprocess
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


MISSION_ORDER = ('M5', 'M6', 'M3', 'M7', 'M2', 'M8', 'M1', 'M4')
DYNAMIC_NODE = {
    'M7': 'mission_obstacle_avoid_node',
    'M8': 'mission_cone_avoid_node',
    'M1': 'mission_traffic_light_node',
    'M4': 'mission_parking_node',
}


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')
        defaults = {
            'lane_command_topic': 'inference/cmd_vel',
            'cone_command_topic': 'mission/cone/cmd_vel',
            'parking_command_topic': 'mission/parking/cmd_vel',
            'output_topic': '/cmd_vel',
            'mission_event_topic': 'mission/event',
            'active_mission_topic': 'mission/active',
            'lane_mode_topic': 'mission/lane_mode',
            'normal_speed': 0.25,
            'slow_speed': 0.12,
            'command_timeout_sec': 0.40,
            'stop_line_distance_m': 0.35,
            'pedestrian_hold_sec': 1.0,
            'auto_start_mission_nodes': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._index = 0
        self._active = MISSION_ORDER[0]
        self._lane_cmd = Twist()
        self._cone_cmd = Twist()
        self._parking_cmd = Twist()
        self._last_cmd_ns = {'lane': 0, 'cone': 0, 'parking': 0}
        self._slow = False
        self._blocked = False
        self._at_stop_line = False
        self._traffic = 'unknown'
        self._pedestrian_until_ns = 0
        self._obstacle_seen = False
        self._processes = {}

        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 10)
        self.active_pub = self.create_publisher(
            String, self.get_parameter('active_mission_topic').value, 10)
        self.lane_mode_pub = self.create_publisher(
            String, self.get_parameter('lane_mode_topic').value, 10)
        self.parking_start_pub = self.create_publisher(Bool, 'mission/parking/start', 10)
        self.create_subscription(Twist, self.get_parameter('lane_command_topic').value,
                                 lambda msg: self._store_cmd('lane', msg), 10)
        self.create_subscription(Twist, self.get_parameter('cone_command_topic').value,
                                 lambda msg: self._store_cmd('cone', msg), 10)
        self.create_subscription(Twist, self.get_parameter('parking_command_topic').value,
                                 lambda msg: self._store_cmd('parking', msg), 10)
        self.create_subscription(String, self.get_parameter('mission_event_topic').value,
                                 self._mission_event_callback, 10)
        self.create_subscription(String, 'mission/yolo_event', self._yolo_callback, 10)
        self.create_subscription(Float32, 'mission/stop_line_distance', self._stop_line_callback, 10)
        self.create_subscription(String, 'mission/traffic_light_state', self._traffic_callback, 10)
        self.create_subscription(String, 'mission/obstacle_avoid/state', self._obstacle_callback, 10)
        self.create_subscription(String, 'mission/cone/state', self._cone_state_callback, 10)
        self.create_subscription(String, 'mission/parking/state', self._parking_state_callback, 10)
        self.timer = self.create_timer(0.05, self._control_tick)
        self._enter_mission(self._active, initial=True)

    def _store_cmd(self, source, msg):
        setattr(self, f'_{source}_cmd', msg)
        self._last_cmd_ns[source] = self.get_clock().now().nanoseconds

    def _mission_event_callback(self, msg):
        event = msg.data.strip().upper()
        if self._active == 'M4' and event in ('NEXT', 'M4_DONE'):
            self._blocked = True
            self.cmd_pub.publish(Twist())
            self.get_logger().info('M4 complete event received; vehicle held stopped')
            return
        if event in ('NEXT', f'{self._active}_DONE'):
            self._advance(f'operator/event {event}')
        elif event in ('LANE_CENTER', 'LANE_LEFT', 'LANE_RIGHT', 'HEADING_HOLD'):
            mode = {'LANE_CENTER': 'center', 'LANE_LEFT': 'left',
                    'LANE_RIGHT': 'right', 'HEADING_HOLD': 'heading_hold'}[event]
            self.lane_mode_pub.publish(String(data=mode))
        elif event == 'ESTOP':
            self._blocked = True
        elif event == 'ESTOP_CLEAR':
            self._blocked = False

    def _yolo_callback(self, msg):
        event = msg.data
        if self._active == 'M5' and event == 'slow_sign':
            self._slow = True
        elif self._active == 'M3' and event == 'pedestrian':
            hold = self.get_parameter('pedestrian_hold_sec').value
            self._pedestrian_until_ns = self.get_clock().now().nanoseconds + int(hold * 1e9)
        elif self._active == 'M2' and event in ('sign_a', 'sign_b'):
            self.lane_mode_pub.publish(String(data='left' if event == 'sign_a' else 'right'))
        elif self._active == 'M4' and event == 'parking_sign':
            self.parking_start_pub.publish(Bool(data=True))

    def _stop_line_callback(self, msg):
        distance = msg.data
        self._at_stop_line = 0.0 <= distance <= self.get_parameter('stop_line_distance_m').value

    def _traffic_callback(self, msg):
        if self._active != 'M1':
            return
        self._traffic = msg.data
        if msg.data == 'green' and self._at_stop_line:
            self._advance('green traffic light at stop line')

    def _obstacle_callback(self, msg):
        if self._active != 'M7':
            return
        if msg.data == 'change_left':
            self._obstacle_seen = True
            self._blocked = False
            self.lane_mode_pub.publish(String(data='left'))
        elif msg.data == 'change_right':
            self._obstacle_seen = True
            self._blocked = False
            self.lane_mode_pub.publish(String(data='right'))
        elif msg.data == 'stop':
            self._obstacle_seen = True
            self._blocked = True
        elif msg.data == 'clear':
            self._blocked = False
            if self._obstacle_seen:
                self.lane_mode_pub.publish(String(data='center'))
                self._advance('M7 obstacle cleared')

    def _cone_state_callback(self, msg):
        if self._active == 'M8':
            self._blocked = msg.data in ('lost', 'emergency_stop')

    def _parking_state_callback(self, msg):
        if self._active == 'M4' and msg.data == 'complete':
            self._blocked = True
            self.get_logger().info('M4 parking complete; mission sequence complete')

    def _advance(self, reason):
        if self._index >= len(MISSION_ORDER) - 1:
            self.get_logger().info(f'final mission retained ({reason})')
            return
        previous = self._active
        self._leave_mission(previous)
        self._index += 1
        self._active = MISSION_ORDER[self._index]
        self.get_logger().info(f'{previous} -> {self._active}: {reason}')
        self._enter_mission(self._active)

    def _enter_mission(self, mission, initial=False):
        self._slow = False
        self._blocked = False
        self._at_stop_line = False
        self._traffic = 'unknown'
        self._obstacle_seen = False
        self.lane_mode_pub.publish(String(data='center'))
        self.active_pub.publish(String(data=mission))
        process = DYNAMIC_NODE.get(mission)
        if process and self.get_parameter('auto_start_mission_nodes').value:
            self._start_process(process)
        if mission == 'M4':
            # Parking node remains stationary until its sign or an explicit start event.
            self.parking_start_pub.publish(Bool(data=False))
        if initial:
            self.get_logger().info(f'2026 mission manager ready; starting {mission}')

    def _leave_mission(self, mission):
        process = DYNAMIC_NODE.get(mission)
        if process:
            self._stop_process(process)
        self.cmd_pub.publish(Twist())

    def _control_tick(self):
        if self._blocked:
            self.cmd_pub.publish(Twist())
            return
        now = self.get_clock().now().nanoseconds
        if self._active == 'M3' and now < self._pedestrian_until_ns:
            self.cmd_pub.publish(Twist())
            return
        if self._active == 'M1' and self._at_stop_line and self._traffic != 'green':
            self.cmd_pub.publish(Twist())
            return
        source = 'cone' if self._active == 'M8' else ('parking' if self._active == 'M4' else 'lane')
        age = (now - self._last_cmd_ns[source]) / 1e9 if self._last_cmd_ns[source] else float('inf')
        if age > self.get_parameter('command_timeout_sec').value:
            self.cmd_pub.publish(Twist())
            return
        command = getattr(self, f'_{source}_cmd')
        if source == 'lane':
            target = (self.get_parameter('slow_speed').value
                      if self._slow else self.get_parameter('normal_speed').value)
            command = self._limit_speed(command, target)
        self.cmd_pub.publish(command)

    @staticmethod
    def _limit_speed(command, limit):
        if command.linear.x <= 0.0 or command.linear.x <= limit:
            return command
        scale = limit / command.linear.x
        output = Twist()
        output.linear.x = limit
        output.angular.z = command.angular.z * scale
        return output

    def _start_process(self, name):
        if name in self._processes and self._processes[name].poll() is None:
            return
        self._processes[name] = subprocess.Popen(['ros2', 'run', 'jetracer_ros2', name])
        self.get_logger().info(f'started {name}')

    def _stop_process(self, name):
        process = self._processes.pop(name, None)
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()

    def destroy_node(self):
        self.cmd_pub.publish(Twist())
        for name in list(self._processes):
            self._stop_process(name)
        return super().destroy_node()


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params])
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
