import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String
from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega

# ================= [튜닝 설정값] =================
APPROACH_SPEED = 0.25   # 정지선 접근 속도 [m/s]
STOP_DIST_M    = 0.0    # 정지선 인식 기준 거리 [m]

# 정지선 도달 후 순서대로 실행할 동작: (단계명, 시간[초], 속도[m/s], 조향각[rad])
PARKING_STEPS = [
    ('settle',  0.5,  0.00,  0.00),   # 정지선 위 잠깐 정지 (0.0으로 바꾸면 멈춤 없음)
    ('forward', 2.1,  0.50,  0.00),   # a: 직진
    ('turn',    1.9,  0.50, -0.45),   # b: 우회전 (음수 = 오른쪽)
    ('enter',   0.1,  0.50,  0.00),   # c: 주차 칸 진입
]
# ==================================================


class SimpleParkingNode(Node):
    def __init__(self):
        super().__init__('parking_mission_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, 'parking_mission/state', 10)

        self.create_subscription(Bool, 'mission/stop_line_detected', self._det_cb, 10)
        self.create_subscription(Float32, 'mission/stop_line_distance', self._dist_cb, 10)

        self.detected = False
        self.distance = -1.0
        self.mode = 'APPROACH'      # APPROACH(접근) -> PARKING(주차) -> DONE(완료)
        self.step_idx = 0
        self.step_start = 0.0

        self.create_timer(0.05, self._loop)  # 20Hz 실행
        self.get_logger().info('단순 주차 노드 시작! 정지선까지 접근합니다.')

    def _det_cb(self, msg):
        self.detected = msg.data

    def _dist_cb(self, msg):
        self.distance = msg.data

    def _drive(self, speed, steer=0.0):
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = inner_angle_to_omega(float(steer), float(speed))
        self.cmd_pub.publish(cmd)

    def _loop(self):
        now = time.monotonic()

        # 1. 정지선까지 접근
        if self.mode == 'APPROACH':
            if self.detected and 0.0 <= self.distance <= STOP_DIST_M:
                self.mode = 'PARKING'
                self.step_idx = 0
                self.step_start = now
                self.get_logger().info(f'정지선 도달! -> {PARKING_STEPS[0][0]} 시작')
            else:
                self._drive(APPROACH_SPEED, 0.0)

        # 2. 리스트(PARKING_STEPS) 순서대로 주차 실행
        elif self.mode == 'PARKING':
            name, duration, speed, steer = PARKING_STEPS[self.step_idx]
            if now - self.step_start >= duration:
                self.step_idx += 1
                self.step_start = now
                if self.step_idx >= len(PARKING_STEPS):
                    self.mode = 'DONE'
                    self._drive(0.0)
                    self.state_pub.publish(String(data='DONE'))
                    self.get_logger().info('주차 완료!')
                    return
                self.get_logger().info(f'다음 단계 -> {PARKING_STEPS[self.step_idx][0]}')
                _, _, speed, steer = PARKING_STEPS[self.step_idx]

            self._drive(speed, steer)

        # 3. 완료 후 정지 유지
        elif self.mode == 'DONE':
            self._drive(0.0)
            self.state_pub.publish(String(data='DONE'))


def main():
    rclpy.init()
    node = SimpleParkingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._drive(0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
