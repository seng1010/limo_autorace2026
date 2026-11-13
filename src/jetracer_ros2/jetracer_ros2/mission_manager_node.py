"""Event-driven 2026 mission state machine and sole ``/cmd_vel`` publisher.

Mission order: M5 -> M6 -> M3 -> M7 -> M2 -> M8 -> M1 -> M4.
Always-on perception nodes publish proposals or events.  This node selects one
command source, applies mission speed/stop constraints, and owns ``/cmd_vel``.

M3 (보행자 표지판 정지) 판단은 ``m3_pedestrian.PedestrianStopMission`` 이 하고,
이 노드는 입력(정지선 거리, odom 속도·이동거리, YOLO 이벤트)을 모아 매 주기 넘긴다.

테스트용: ``ros2 run jetracer_ros2 mission_manager_node --ros-args -p start_mission:=M3``
디버그:   ``/mission_manager/debug/state`` (String) — 현재 미션·상태·정지선 거리·속도 (모든 미션, 20 Hz)
"""

import os
import signal
import subprocess
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from jetracer_ros2.m3_pedestrian import M3Config, PedestrianStopMission
from jetracer_ros2.stop_approach import (
    FullStopDetector, StopApproachConfig, StopApproachController)
from jetracer_ros2.stop_line_detector import OdomPathIntegrator


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
            'auto_start_mission_nodes': True,
            'start_mission': 'M5',                  # 테스트할 때 M3 등으로 바로 시작
            'odom_topic': '/odom',
            'odom_timeout_sec': 0.5,
            'stop_line_timeout_sec': 0.5,           # 정지선 거리 메시지가 이보다 오래되면 없음 처리
            # 정지선 감속·정밀 정지 (M1·M3 공용)
            'approach_start_m': 0.50,
            'approach_gain': 0.8,
            'approach_creep_speed': 0.06,
            'approach_stop_offset_m': 0.01,
            'full_stop_speed': 0.01,
            'full_stop_sec': 0.3,
            # M3
            'm3_trigger': 'sign',
            'm3_line_arm_distance_m': 0.80,
            'm3_sign_valid_distance_m': 1.5,
            'm3_max_approach_distance_m': 1.5,
            'm3_hold_sec': 3.5,
            'm3_done_distance_m': 0.50,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        start = str(p('start_mission')).strip().upper()
        if start not in MISSION_ORDER:
            raise ValueError(f'start_mission 은 {MISSION_ORDER} 중 하나')
        self._index = MISSION_ORDER.index(start)
        self._active = start
        self._lane_cmd = Twist()
        self._cone_cmd = Twist()
        self._parking_cmd = Twist()
        self._last_cmd_ns = {'lane': 0, 'cone': 0, 'parking': 0}
        self._slow = False
        self._blocked = False
        self._at_stop_line = False
        self._traffic = 'unknown'
        self._obstacle_seen = False
        self._processes = {}
        self._stop_line_m = None
        self._stop_line_ns = 0
        self._odom = OdomPathIntegrator()
        self._odom_speed = 0.0
        self._odom_ns = 0
        self._odom_warned = False
        self._last_out_speed = 0.0
        self._last_pedestrian_s = None    # 보행자 표지판을 마지막으로 본 odom 위치 (미션 무관)
        approach = StopApproachController(StopApproachConfig(
            approach_start_m=p('approach_start_m'), gain=p('approach_gain'),
            creep_speed=p('approach_creep_speed'), max_speed=p('normal_speed'),
            stop_offset_m=p('approach_stop_offset_m')))
        self._m3 = PedestrianStopMission(
            M3Config(trigger=p('m3_trigger'),
                     line_arm_distance_m=p('m3_line_arm_distance_m'),
                     sign_valid_distance_m=p('m3_sign_valid_distance_m'),
                     max_approach_distance_m=p('m3_max_approach_distance_m'),
                     hold_sec=p('m3_hold_sec'), done_distance_m=p('m3_done_distance_m')),
            approach, FullStopDetector(p('full_stop_speed'), p('full_stop_sec')))
        self._m3_last_state = ''

        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 10)
        self.active_pub = self.create_publisher(
            String, self.get_parameter('active_mission_topic').value, 10)
        self.lane_mode_pub = self.create_publisher(
            String, self.get_parameter('lane_mode_topic').value, 10)
        self.parking_start_pub = self.create_publisher(Bool, 'mission/parking/start', 10)
        self.debug_pub = self.create_publisher(String, '/mission_manager/debug/state', 10)
        self.create_subscription(Odometry, p('odom_topic'), self._odom_callback, 20)
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
        if event == 'pedestrian':
            # M3 직전(M6 구간)에서 먼저 보일 수 있어서 미션과 상관없이 위치를 기억
            self._last_pedestrian_s = self._odom.s
        elif self._active == 'M2' and event in ('sign_a', 'sign_b'):
            self.lane_mode_pub.publish(String(data='left' if event == 'sign_a' else 'right'))
        elif self._active == 'M4' and event == 'parking_sign':
            self.parking_start_pub.publish(Bool(data=True))

    def _odom_callback(self, msg):
        pose = msg.pose.pose.position
        self._odom_speed = float(msg.twist.twist.linear.x)
        self._odom.update(pose.x, pose.y, self._odom_speed)
        self._odom_ns = self.get_clock().now().nanoseconds

    def _stop_line_callback(self, msg):
        distance = msg.data
        self._stop_line_m = float(distance) if distance >= 0.0 else None
        self._stop_line_ns = self.get_clock().now().nanoseconds
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
        if mission == 'M3':
            self._m3.reset()
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

    def _fresh_stop_line(self, now_ns):
        age = (now_ns - self._stop_line_ns) / 1e9
        if self._stop_line_ns == 0 or age > self.get_parameter('stop_line_timeout_sec').value:
            return None
        return self._stop_line_m

    def _measured_speed(self, now_ns):
        """odom 속도. odom 이 끊기면 마지막으로 내보낸 명령 속도로 대신한다 (경고 1회)."""
        age = (now_ns - self._odom_ns) / 1e9
        if self._odom_ns and age <= self.get_parameter('odom_timeout_sec').value:
            return self._odom_speed
        if not self._odom_warned:
            self.get_logger().warn('odom 없음 — 완전정지 판정을 명령 속도로 대신함')
            self._odom_warned = True
        return self._last_out_speed

    def _publish(self, command):
        self._last_out_speed = float(command.linear.x)
        self.cmd_pub.publish(command)

    def _debug(self, text):
        self.debug_pub.publish(String(data=f'{self._active} {text}'))

    def _m3_tick(self, now_ns):
        """M3 판단. 속도 제한(None=평소, 0=정지)을 돌려주고, 끝나면 다음 미션으로."""
        stop_line = self._fresh_stop_line(now_ns)
        speed = self._measured_speed(now_ns)
        decision = self._m3.update(now_ns / 1e9, stop_line, speed, self._odom.s,
                                   last_sign_s=self._last_pedestrian_s)
        if decision.state != self._m3_last_state:
            self.get_logger().info(f'M3 상태: {self._m3_last_state or "-"} -> {decision.state}'
                                   + (f' (정지선 {stop_line:.3f} m)' if stop_line is not None else ''))
            self._m3_last_state = decision.state
        line_text = f'{stop_line:.3f}' if stop_line is not None else 'none'
        self._debug(f'{decision.state} line={line_text} v={speed:.3f}')
        if decision.done:
            self._m3_last_state = ''
            if decision.skipped:
                self.get_logger().warn('M3 보행자 표지판을 못 보고 구간을 지나침 — M3 건너뜀')
                self._advance('M3 skipped (표지판 미인식)')
            else:
                self._advance('M3 정지·3초 대기·출발 완료')
            return None
        return decision.speed_limit

    def _control_tick(self):
        now = self.get_clock().now().nanoseconds
        if self._active != 'M3':          # M3는 _m3_tick 안에서 상태까지 같이 발행
            line = self._fresh_stop_line(now)
            line_text = f'{line:.3f}' if line is not None else 'none'
            self._debug(f'{"blocked" if self._blocked else "run"} line={line_text} '
                        f'v={self._measured_speed(now):.3f}')
        if self._blocked:
            self._publish(Twist())
            return
        speed_limit = None
        if self._active == 'M3':
            speed_limit = self._m3_tick(now)
            if speed_limit == 0.0:
                self._publish(Twist())
                return
        if self._active == 'M1' and self._at_stop_line and self._traffic != 'green':
            self._publish(Twist())
            return
        source = 'cone' if self._active == 'M8' else ('parking' if self._active == 'M4' else 'lane')
        age = (now - self._last_cmd_ns[source]) / 1e9 if self._last_cmd_ns[source] else float('inf')
        if age > self.get_parameter('command_timeout_sec').value:
            self._publish(Twist())
            return
        command = getattr(self, f'_{source}_cmd')
        if source == 'lane':
            target = (self.get_parameter('slow_speed').value
                      if self._slow else self.get_parameter('normal_speed').value)
            if speed_limit is not None:
                target = min(target, speed_limit)
            command = self._limit_speed(command, target)
        self._publish(command)

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
        # Ctrl+C 로 이미 통신이 내려간 뒤면 정지 명령 발행이 실패하므로 무시한다
        try:
            if rclpy.ok():
                self.cmd_pub.publish(Twist())
        except Exception:  # noqa: BLE001
            pass
        for name in list(self._processes):
            self._stop_process(name)
        return super().destroy_node()


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    # params.yaml 뒤에 명령줄 인자를 붙여서 `--ros-args -p start_mission:=M3` 같은 값이 이기게 한다
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params] + list(argv[1:]))
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
