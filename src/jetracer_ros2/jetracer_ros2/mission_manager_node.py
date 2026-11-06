import os
import signal
import subprocess
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

# 전체 미션을 순서대로 진행시키는 단일 상태머신 노드. /cmd_vel에 실제로 발행하는
# 건 이 노드 하나뿐이다(파킹 단계 제외 — 아래 참고). 각 단계에 필요한 "센서/판단"
# 노드(traffic_light, people_estop, tunnel, track_following)는 이 노드가 필요할
# 때 subprocess로 직접 켜고 끈다. mission_inference_node는 all_in_node.launch.py가
# 처음부터 켜둔 채로 계속 떠 있고, inference/cmd_vel로만 발행하니 이 노드가 단계별로
# 속도를 재조정해서 /cmd_vel로 중계한다.
#
# 단계 흐름:
#   1) WAIT_LIGHT        — mission_traffic_light_node의 green 확정까지 정지 대기.
#                           green 확정 시 traffic_light_node 종료, people_estop_node 시작.
#   2) WAIT_ESTOP_READY   — people_estop_node가 뜨고 첫 신호(clear=센서 켜짐)를 보낼
#                           때까지 정지 대기.
#   3) PEOPLE_ESTOP       — cruise_speed로 주행. people_estop_node가 obstacle_min_time초
#                           경과 시점에 'active'를 발행하면 그 즉시 people_estop_slow_speed로
#                           선제 감속(라이다 활성화와 감속이 같은 사건으로 완전히 동기화됨 —
#                           manager가 따로 자기 타이머를 재는 게 아니라 센서 쪽 타이머 하나만
#                           신뢰의 원천으로 씀). 'stop' 오면 즉시 정지, 그 뒤 최종
#                           'clear'(=people_estop_node 자체 종료 신호) 오면 people_estop_node
#                           종료 + tunnel_node 시작, cruise_speed로 복귀.
#   4) TUNNEL             — cruise_speed로 주행하며 조도 관찰. threshold 이하로 떨어지면
#                           (1회성) tunnel_node 즉시 종료 후 cruise_speed 유지한 채
#                           tunnel_slow_delay_sec만큼 대기. 그 뒤 track_following_node
#                           시작 + tunnel_slow_speed로 감속, 그 시점부터
#                           tunnel_slow_duration_sec 카운트 시작
#                           + track_following_state로 게이팅(stop=정지/clear=주행).
#                           clear로 바뀌어도 즉시 재출발하지 않고 track_resume_delay_sec만큼
#                           대기 후 재출발(트럭이 stop 임계값 근처에서 흔들릴 때 덜컹거림 방지).
#   5) TRACK_WINDOW       — 위 감속+게이팅 창. tunnel_slow_duration_sec 지나면
#                           track_following_node 종료, cruise_speed로 복귀.
#   6) PARKING_WAIT       — 이 단계 진입과 동시에 mission_parking_node를 미리
#                           띄운다(warm standby). 이 노드는 /scan 구독과 프로세스
#                           기동을 미리 끝내두기만 하고 실제 주행은 시작하지 않은
#                           채 자기 자신도 /parking_start를 구독해서 대기한다.
#                           cruise_speed로 주행하며 /parking_start(Bool) 대기.
#                           True 수신 시(1회성) mission_inference_node만 종료
#                           (mission_parking_node는 이미 떠 있으므로 새로 안 띄움 —
#                           프로세스 기동/DDS discovery 지연으로 인한 /cmd_vel
#                           공백·순간정지를 없애기 위함).
#   7) PARKING_ACTIVE     — mission_parking_node가 /cmd_vel을 직접 발행하므로 이
#                           노드는 더 이상 아무것도 publish하지 않는다(경쟁 방지).
#
# 재조정 원리: linear.x와 angular.z를 같은 비율로 스케일하면 r = v/ω(회전반경)이
# 그대로 보존된다 (자세한 배경은 ackermann_utils.inner_angle_to_omega 참고).
#
# 사용법:
#   ros2 launch jetracer_ros2 all_in_node.launch.py


PHASE_WAIT_LIGHT = 'wait_light'
PHASE_WAIT_ESTOP_READY = 'wait_estop_ready'
PHASE_PEOPLE_ESTOP = 'people_estop'
PHASE_TUNNEL = 'tunnel'
PHASE_TRACK_WINDOW = 'track_window'
PHASE_PARKING_WAIT = 'parking_wait'
PHASE_PARKING_ACTIVE = 'parking_active'


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')

        self.declare_parameter('input_topic', 'inference/cmd_vel')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('traffic_light_topic', 'mission/traffic_light_state')
        self.declare_parameter('people_estop_topic', 'mission/people_estop_state')
        self.declare_parameter('tunnel_topic', 'mission/tunnel_brightness')
        self.declare_parameter('track_following_topic', 'mission/track_following_state')
        self.declare_parameter('parking_start_topic', '/parking_start')

        self.declare_parameter('cruise_speed', 1.0)               # 기본 순항 속도 [m/s]
        # 선제 감속 시점(몇 초 뒤)은 여기서 따로 관리하지 않는다 — people_estop_node의
        # obstacle_min_time과 어긋나면 안 되므로, 그 노드가 obstacle_min_time초 뒤
        # 'active'를 발행하는 순간을 그대로 신뢰해서 감속한다(아래 people_estop_callback 참고).
        self.declare_parameter('people_estop_slow_speed', 0.5)      # 선제 감속 목표 속도 [m/s]
        self.declare_parameter('tunnel_brightness_threshold', 75.0)  # 이 이하로 떨어지면 터널 진입
        self.declare_parameter('tunnel_slow_delay_sec', 2.0)         # 조도 감지 후 감속 시작까지 대기 시간 [sec]
        self.declare_parameter('tunnel_slow_speed', 0.5)             # 터널/트럭 추종 구간 속도 [m/s]
        self.declare_parameter('tunnel_slow_duration_sec', 30.0)     # 감속 시작 후 track_following 유지 시간 [sec]
        self.declare_parameter('track_resume_delay_sec', 1.0)        # 트럭 회피 해제(clear) 후 재출발까지 대기 시간 [sec]

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        traffic_light_topic = self.get_parameter('traffic_light_topic').value
        people_estop_topic = self.get_parameter('people_estop_topic').value
        tunnel_topic = self.get_parameter('tunnel_topic').value
        track_following_topic = self.get_parameter('track_following_topic').value
        parking_start_topic = self.get_parameter('parking_start_topic').value

        self.cruise_speed = self.get_parameter('cruise_speed').value
        self.people_estop_slow_speed = self.get_parameter('people_estop_slow_speed').value
        self.tunnel_brightness_threshold = self.get_parameter('tunnel_brightness_threshold').value
        self.tunnel_slow_delay_sec = self.get_parameter('tunnel_slow_delay_sec').value
        self.tunnel_slow_speed = self.get_parameter('tunnel_slow_speed').value
        self.tunnel_slow_duration_sec = self.get_parameter('tunnel_slow_duration_sec').value
        self.track_resume_delay_sec = self.get_parameter('track_resume_delay_sec').value

        self._phase = PHASE_WAIT_LIGHT
        self._people_estop_blocked = False
        self._people_estop_slow = False
        self._people_estop_saw_stop = False
        self._tunnel_triggered = False
        self._tunnel_slow_delay_timer = None
        self._track_blocked = False
        self._track_window_timer = None
        self._track_resume_timer = None
        self._parking_triggered = False

        self.cmd_pub = self.create_publisher(Twist, output_topic, 10)
        self.cmd_sub = self.create_subscription(
            Twist, input_topic, self.cmd_callback, 10)
        self.traffic_light_sub = self.create_subscription(
            String, traffic_light_topic, self.traffic_light_callback, 10)
        self.people_estop_sub = self.create_subscription(
            String, people_estop_topic, self.people_estop_callback, 10)
        self.tunnel_sub = self.create_subscription(
            Float32, tunnel_topic, self.tunnel_callback, 10)
        self.track_following_sub = self.create_subscription(
            String, track_following_topic, self.track_following_callback, 10)
        self.parking_start_sub = self.create_subscription(
            Bool, parking_start_topic, self.parking_start_callback, 10)

        self.get_logger().info(
            f'MissionManager ready  {input_topic} -> {output_topic}  '
            f'cruise_speed={self.cruise_speed}  '
            '(초록불 확정 전까지 정지 상태로 대기)')

    # ------------------------------------------------------------------ #
    # 1) 신호등
    # ------------------------------------------------------------------ #

    def traffic_light_callback(self, msg: String):
        if self._phase != PHASE_WAIT_LIGHT or msg.data != 'green':
            return
        self.get_logger().info(
            'GREEN 확정 — mission_traffic_light_node 종료, mission_people_estop_node 시작')
        self._kill_process('mission_traffic_light_node')
        self._start_process('mission_people_estop_node')
        self._phase = PHASE_WAIT_ESTOP_READY

    # ------------------------------------------------------------------ #
    # 2)/3) 사람 e-stop
    # ------------------------------------------------------------------ #

    def people_estop_callback(self, msg: String):
        if self._phase == PHASE_WAIT_ESTOP_READY:
            if msg.data != 'clear':
                return
            # 첫 'clear' = mission_people_estop_node가 막 켜졌다는 신호(센서 켜짐).
            self._phase = PHASE_PEOPLE_ESTOP
            self.get_logger().info('mission_people_estop_node 켜짐 감지 — 주행 시작')
            return

        if self._phase != PHASE_PEOPLE_ESTOP:
            return

        if msg.data == 'active':
            # people_estop_node가 obstacle_min_time초 경과 시점에 발행하는 신호.
            # manager가 따로 타이머를 재지 않고 이 신호 하나만 신뢰해서 감속하므로,
            # "라이다 활성화"와 "감속 시작"이 오차 없이 같은 순간에 일어난다.
            if not self._people_estop_slow:
                self._people_estop_slow = True
                self.get_logger().info(
                    f'라이다 활성화(active) 수신 — {self.people_estop_slow_speed}m/s로 감속')
            return

        if msg.data == 'stop':
            self._people_estop_saw_stop = True
            if not self._people_estop_blocked:
                self._people_estop_blocked = True
                self.get_logger().warn('사람 감지 — 정지')
            return

        # mission_people_estop_node는 시작 시 'clear'를 여러 번 반복 발행한다(DDS
        # discovery race 방지용 — 켜짐 알림 참고). 'stop'을 한 번도 본 적 없는데
        # 온 'clear'는 진짜 장애물 해소가 아니라 그 반복 알림 중 하나이므로 무시한다.
        if not self._people_estop_saw_stop:
            return

        # 'clear': 장애물이 있었다가 해소됨 -> people_estop_node 자체가 종료된다는 신호.
        self.get_logger().info(
            '장애물 해소 — mission_people_estop_node 종료, mission_tunnel_node 시작, '
            f'{self.cruise_speed}m/s로 복귀')
        self._people_estop_blocked = False
        self._people_estop_slow = False
        self._people_estop_saw_stop = False
        self._kill_process('mission_people_estop_node')
        self._start_process('mission_tunnel_node')
        self._phase = PHASE_TUNNEL

    # ------------------------------------------------------------------ #
    # 4)/5) 터널(조도) + 트럭 추종
    # ------------------------------------------------------------------ #

    def tunnel_callback(self, msg: Float32):
        if self._phase != PHASE_TUNNEL or self._tunnel_triggered:
            return
        if msg.data > self.tunnel_brightness_threshold:
            return

        self._tunnel_triggered = True
        self.get_logger().info(
            f'조도 {msg.data:.1f} <= {self.tunnel_brightness_threshold} 감지 — '
            f'mission_tunnel_node 종료, {self.tunnel_slow_delay_sec}초 뒤 '
            f'mission_track_following_node 시작 + {self.tunnel_slow_speed}m/s 감속')
        self._kill_process('mission_tunnel_node')
        self._tunnel_slow_delay_timer = self.create_timer(
            self.tunnel_slow_delay_sec, self._start_track_window)

    def _start_track_window(self):
        self._tunnel_slow_delay_timer.cancel()
        self._tunnel_slow_delay_timer = None
        self.get_logger().info(
            f'mission_track_following_node 시작, '
            f'{self.tunnel_slow_duration_sec}초 동안 {self.tunnel_slow_speed}m/s')
        self._start_process('mission_track_following_node')
        self._track_blocked = False
        self._phase = PHASE_TRACK_WINDOW
        self._track_window_timer = self.create_timer(
            self.tunnel_slow_duration_sec, self._end_track_window)

    def track_following_callback(self, msg: String):
        if self._phase != PHASE_TRACK_WINDOW:
            return

        if msg.data == 'stop':
            # 트럭이 다시 가까워지면 대기 중이던 재출발 타이머부터 취소하고 즉시 정지.
            if self._track_resume_timer is not None:
                self._track_resume_timer.cancel()
                self._track_resume_timer = None
            self._track_blocked = True
            return

        # msg.data == 'clear': 이미 안 막혀있거나 재출발 타이머가 이미 돌고 있으면 무시
        # (트럭 노드는 매 스캔마다 발행하므로 clear가 연속으로 여러 번 들어온다).
        if not self._track_blocked or self._track_resume_timer is not None:
            return
        self._track_resume_timer = self.create_timer(
            self.track_resume_delay_sec, self._resume_after_track_clear)

    def _resume_after_track_clear(self):
        self._track_resume_timer.cancel()
        self._track_resume_timer = None
        self._track_blocked = False
        self.get_logger().info(
            f'트럭 회피 해제 — {self.track_resume_delay_sec}초 대기 후 재출발')

    def _end_track_window(self):
        self._track_window_timer.cancel()
        self._track_window_timer = None
        if self._track_resume_timer is not None:
            self._track_resume_timer.cancel()
            self._track_resume_timer = None
        self.get_logger().info(
            f'{self.tunnel_slow_duration_sec}초 경과 — mission_track_following_node 종료, '
            f'{self.cruise_speed}m/s로 복귀, mission_parking_node warm standby 시작')
        self._kill_process('mission_track_following_node')
        self._start_process('mission_parking_node')
        self._phase = PHASE_PARKING_WAIT

    # ------------------------------------------------------------------ #
    # 6)/7) 주차
    # ------------------------------------------------------------------ #

    def parking_start_callback(self, msg: Bool):
        if self._phase != PHASE_PARKING_WAIT or self._parking_triggered or not msg.data:
            return
        self._parking_triggered = True
        self.get_logger().info(
            'parking_start 수신 — mission_inference_node 종료 '
            '(mission_parking_node는 warm standby 상태였다가 이 신호로 직접 활성화됨)')
        self._kill_process('mission_inference_node')
        self._phase = PHASE_PARKING_ACTIVE

    # ------------------------------------------------------------------ #
    # cmd_vel 중계
    # ------------------------------------------------------------------ #

    def cmd_callback(self, msg: Twist):
        phase = self._phase

        if phase in (PHASE_WAIT_LIGHT, PHASE_WAIT_ESTOP_READY):
            self.cmd_pub.publish(Twist())
            return

        if phase == PHASE_PARKING_ACTIVE:
            # mission_parking_node가 /cmd_vel을 직접 발행 중 — 경쟁을 피하려고
            # 이 노드는 더 이상 아무것도 publish하지 않는다.
            return

        if phase == PHASE_PEOPLE_ESTOP:
            if self._people_estop_blocked:
                self.cmd_pub.publish(Twist())
                return
            target = self.people_estop_slow_speed if self._people_estop_slow else self.cruise_speed
            self._publish_scaled(msg, target)
            return

        if phase == PHASE_TUNNEL:
            self._publish_scaled(msg, self.cruise_speed)
            return

        if phase == PHASE_TRACK_WINDOW:
            if self._track_blocked:
                self.cmd_pub.publish(Twist())
                return
            self._publish_scaled(msg, self.tunnel_slow_speed)
            return

        if phase == PHASE_PARKING_WAIT:
            self._publish_scaled(msg, self.cruise_speed)
            return

    def _publish_scaled(self, msg: Twist, target_speed: float):
        if abs(msg.linear.x) < 1e-6:
            # 원본 속도가 0(정지 신호)이면 비율 계산이 불가능하니 그대로 전달한다.
            self.cmd_pub.publish(msg)
            return

        scale = target_speed / msg.linear.x
        out = Twist()
        out.linear.x = target_speed
        out.angular.z = msg.angular.z * scale
        self.cmd_pub.publish(out)

    # ------------------------------------------------------------------ #
    # 프로세스 관리
    # ------------------------------------------------------------------ #

    def _start_process(self, process_name: str):
        subprocess.Popen(['ros2', 'run', 'jetracer_ros2', process_name])
        self.get_logger().info(f'{process_name} 시작')

    def _kill_process(self, process_name: str):
        """process_name 실행파일 경로를 가진 프로세스에 SIGINT를 보낸다.

        리눅스는 프로세스 이름(comm)을 15자로 자르기 때문에, mission_inference_node
        (22자)나 mission_traffic_light_node(26자) 같은 이름엔 pgrep -x(완전 일치)가
        전혀 안 먹힌다. 그래서 -f(전체 명령줄)로 찾되, 그냥 부분 문자열로 찾으면
        mission_traffic_light_node를 찾다가 debug_mission_traffic_light_node
        (이름에 그대로 포함됨)까지 같이 걸리므로, 앞에 '/'를 붙여 실행파일 경로의
        마지막 구성요소로 정확히 매칭한다.
        """
        pattern = f'/{process_name}([[:space:]]|$)'
        try:
            output = subprocess.check_output(['pgrep', '-f', pattern])
        except subprocess.CalledProcessError:
            self.get_logger().warn(f'{process_name} 프로세스를 못 찾았습니다 (이미 꺼져있는 듯).')
            return

        for pid_str in output.decode().split():
            pid = int(pid_str)
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGINT)
                self.get_logger().info(f'{process_name}(pid={pid})에 SIGINT 전송')
            except ProcessLookupError:
                pass


def main(args=None):
    argv = sys.argv if args is None else args

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
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
