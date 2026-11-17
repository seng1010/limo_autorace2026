#!/usr/bin/env python3
"""T자 주차 (M4) — 빈 칸을 라이다로 찾아 후진으로 넣는다.  [claude 브랜치 전용]

대회장 (도면 2026Track): 차로 오른쪽에 주차 칸 3개(폭 0.50 m, 깊이 0.60 m)가 붙어 있고 그 뒤에 3칸이 더 있다.
  6칸 중 5칸에 LIMO 가 서 있고, 차로에 붙은 3칸 중 빈 한 칸에 넣는다 (어느 칸인지는 당일).
  주차 시작 흰 선에서 첫 칸 왼쪽 끝까지 0.44 m, 칸 가운데는 선에서 0.71 / 1.23 / 1.75 m.

왜 후진인가
  이 차는 최대로 꺾어도 실제 회전 반경이 약 0.6 m 다 (대회장 bag 자이로로 잼). 차로 가운데에서 90도 돌면 차축이
  칸 안으로 0.34~0.40 m 들어간다. 앞으로 넣으면 앞 범퍼(차축 + 0.26 m)가 칸 끝 선(0.60 m)에 걸리고,
  뒤로 넣으면 차체가 0.14 ~ 0.46 m 에 놓여 앞뒤 14 cm 씩 남는다.

순서 (매니저가 흰 선 앞 완전 정지를 확인하고 mission/parking/start 를 주면 시작)
  1) SETTLE  : 서서 자이로 0점 + 라이다로 오른쪽 LIMO 들을 보고 빈 칸을 고른다
  2) FORWARD : 곧게 전진 (자이로로 방향 유지). 지나가며 빈 칸 위치를 계속 고친다. 빈 칸 가운데 + 회전 반경만큼 지나가서 정지
  3) ARC     : 오른쪽으로 꺾고 후진. 자이로 방향 + odom 거리로 원 위에 있는지 보며 조향을 고친다. 90도 돌면
  4) BACK    : 곧게 후진해서 깊이를 맞춘다
  5) DONE    : 정지, mission/parking/state = complete

기존 주차 노드(parking_line_detection)와 토픽이 같다. 매니저는 M4 에서 이 노드를 켠다 (mission_manager_node MISSION_PROCESSES).

터미널 줄:  [주차G] ...      값은 아래 상수. 실차에서 맞출 것은 '맞추기' 표시.
"""

import math
import time

import numpy as np

VERSION = 'claude-park1'

# ---- 차 ----
WHEELBASE, TRACK = 0.2, 0.172
MAX_INNER = 0.4887           # 하드웨어 최대 조향각 [rad]
K_STEER = 1.20               # 실제 회전 반경 = 명령 반경 x 이 값 (대회장 bag: 명령 0.51 m 에 실제 0.60~0.62 m). 맞추기:
                             #   주차할 때 덜 돌아서 칸 왼쪽 선에 붙으면 ↑, 너무 돌아 오른쪽에 붙으면 ↓
LIDAR_X = 0.184              # 뒤 차축 -> 라이다 [m]
FRONT, REAR, HALF_W = 0.26, 0.062, 0.11     # 뒤 차축 -> 앞 범퍼 / 뒤 범퍼, 차 반폭
YAW_ODOM_SCALE = 1.20        # 자이로(/imu)가 없을 때만: odom 방향은 실제의 이 배만큼 돈다

# ---- 주차장 (도면) ----
SLOT_PITCH = 0.52            # 칸 간격 (폭 0.50 + 선 0.02)
SLOT_X = (0.91, 1.43, 1.95)  # 시작 자세(앞바퀴 축이 흰 선 위 = 뒤 차축이 선 0.2 m 뒤)에서 칸 가운데까지 앞으로 [m]
SLOT_TOP = 0.26              # 차로 가운데 -> 칸 입구 선 [m]
LIMO_HALF = 0.11
BLOCK_SHIFT = 0.25           # 정지 위치가 틀려도: 칸 전체가 앞뒤로 이만큼까지 밀려 보이는 것은 받아들인다
DEFAULT_SLOT = 2             # 라이다로 못 고르면 이 칸 (1, 2, 3). 맞추기: 당일 칸을 알면 이 값을 그 칸으로 바꿔 두면 안전

# ---- 동작 ----
SETTLE_SEC = 1.0
V_FWD, V_REV = 0.15, 0.10
PARK_R = 0.62                # 후진 원의 반경 [m]. 실제 최소 반경(약 0.56~0.60)보다 조금 크게 잡아야 양쪽으로 고칠 여유가 있다
END_BIAS = 0.05              # 칸 가운데보다 이만큼 왼쪽(먼저 지나온 쪽)에 서도록. 안쪽 뒷바퀴가 칸 오른쪽 선을 스치면 ↑
Y_SHIFT = 0.0                # 전진하면서 시작 위치보다 왼쪽으로 이만큼 붙는다. 칸 오른쪽 선을 밟으면 ↑ (대신 차 앞 왼쪽이 차로 윗선에 가까워진다)
X_MAX = 2.65                 # 이보다 더 앞으로는 안 간다. 3번 칸은 차 앞이 차로 끝 선을 10~20 cm 넘는다 (그래야 반경이 나온다)
DEPTH = 0.40                 # 끝났을 때 뒤 차축이 칸 입구 선에서 들어간 깊이. 차체는 0.14 ~ 0.46 m 에 놓인다 (칸 깊이 0.60)
                             #   맞추기: 앞이 선에 걸리면 ↑, 뒤가 걸리면 ↓
ARC_DONE = math.radians(88.0)
K_E, K_H = 25.0, 8.0         # 원에서 벗어난 거리 / 방향 오차 되먹임
K_TH, K_Y = 2.5, 1.0         # 직진·직후진 방향 유지
FRONT_GUARD = 0.30           # 전진 중 정면 이 거리 안에 뭔가 있으면 거기서 멈추고 후진을 시작한다
PHASE_TIMEOUT = 25.0         # 한 단계가 이보다 길어지면 정지하고 끝낸다


def steer_for(kappa):
    """실제로 내고 싶은 곡률(왼쪽 +, 1/m) -> 안쪽 바퀴 조향각 [rad]."""
    k = abs(kappa) * K_STEER
    if k < 1e-3:
        return 0.0
    r = max(1.0 / k, WHEELBASE / math.tan(MAX_INNER) + TRACK / 2.0)
    return math.copysign(min(MAX_INNER, math.atan(WHEELBASE / (r - TRACK / 2.0))), kappa)


def omega_for(inner, v):
    """inner_angle_to_omega 와 같은 식 (utils.ackermann_utils)."""
    if abs(inner) < 1e-4 or v == 0.0:
        return 0.0
    r = WHEELBASE / math.tan(abs(inner)) + TRACK / 2.0
    c = math.copysign(math.atan(WHEELBASE / r), inner)
    return v * math.tan(c) / WHEELBASE


def pick_slot(px, py):
    """시작 좌표의 라이다 점(오른쪽 LIMO 들) -> (빈 칸 번호 1~3, 칸 가운데 x, 믿을 만한가).
    세 칸 중 두 칸에 LIMO 앞면이 보여야 한다. 칸 전체의 앞뒤 밀림(±BLOCK_SHIFT)도 같이 맞춘다."""
    m = (py > -0.85) & (py < -0.28) & (px > 0.15) & (px < 2.9)
    x = px[m]
    if x.size < 12:
        return None
    best = []
    for shift in np.arange(-BLOCK_SHIFT, BLOCK_SHIFT + 1e-6, 0.02):
        for empty in (1, 2, 3):
            score = 0.0
            for k in (1, 2, 3):
                d = np.abs(x - (SLOT_X[k - 1] + shift))
                if k == empty:
                    score -= 2.0 * np.count_nonzero(d < 0.19)
                else:
                    score += min(np.count_nonzero(d < LIMO_HALF + 0.03), 60)
                    score -= 0.5 * np.count_nonzero((d > LIMO_HALF + 0.05) & (d < 0.24))
            best.append((score, empty, shift))
    best.sort(reverse=True)
    score, empty, _ = best[0]
    same = [b for b in best if b[1] == empty and b[0] >= score - 2.0]
    shift = float(np.mean([b[2] for b in same]))
    other = max([b[0] for b in best if b[1] != empty], default=-1e9)
    return empty, SLOT_X[empty - 1] + shift, bool(score >= 10 and score - other >= 6)


class ParkPlanner:
    """ROS 없이 도는 본체. 좌표: 시작할 때 뒤 차축이 원점, 차 앞이 +x, 왼쪽이 +y."""

    def __init__(self, default_slot=DEFAULT_SLOT, log=print):
        self.log = log
        self.default_slot = default_slot
        self.state = 'IDLE'
        self.x = self.y = self.th = 0.0
        self.t0 = None
        self._odom = None
        self._gyro_t = None
        self._gyro_n, self._gyro_sum, self.bias = 0, 0.0, 0.0
        self._imu_seen = None
        self.pts = []                      # 시작 좌표의 라이다 점 모음
        self.front = None                  # 정면 통로의 가장 가까운 것 [m, 앞 범퍼에서]
        self.slot = None                   # (번호, 칸 가운데 x, 믿음)
        self.c = None                      # 후진 원: (cx, cy, R)
        self.done_msg = ''

    # ----- 센서 -----
    def feed_imu(self, wz, t):
        self._imu_seen = t
        if self.state == 'SETTLE':
            self._gyro_n += 1
            self._gyro_sum += wz
        elif self.state in ('FORWARD', 'PAUSE', 'ARC', 'BACK') and self._gyro_t is not None:
            dt = t - self._gyro_t
            if 0.0 < dt < 0.2:
                self.th += (wz - self.bias) * dt
        self._gyro_t = t

    def feed_odom(self, ox, oy, oyaw, v, t):
        if self._odom is not None and self.state in ('FORWARD', 'PAUSE', 'ARC', 'BACK'):
            px, py, pyaw = self._odom
            d = math.hypot(ox - px, oy - py)
            if d < 0.2:
                fwd = (ox - px) * math.cos(pyaw) + (oy - py) * math.sin(pyaw)
                ds = math.copysign(d, fwd if abs(fwd) > 1e-6 else v)
                if self._imu_seen is None or t - self._imu_seen > 0.5:        # 자이로 없음: odom 방향을 줄여서 쓴다
                    self.th += (((oyaw - pyaw + math.pi) % (2 * math.pi)) - math.pi) / YAW_ODOM_SCALE
                self.x += ds * math.cos(self.th)
                self.y += ds * math.sin(self.th)
        self._odom = (ox, oy, oyaw)

    def feed_scan(self, ranges, angle_min, angle_inc, range_min=0.15):
        r = np.asarray(ranges, dtype=np.float32)
        a = angle_min + np.arange(r.size) * angle_inc
        ok = np.isfinite(r) & (r >= max(range_min, 0.12)) & (r < 3.5)
        lx, ly = LIDAR_X + r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        corridor = (np.abs(ly) < HALF_W + 0.04) & (lx > FRONT)
        self.front = float(lx[corridor].min() - FRONT) if corridor.any() else None
        if self.state in ('SETTLE', 'FORWARD') and abs(self.th) < math.radians(12):
            c, s = math.cos(self.th), math.sin(self.th)
            self.pts.append(np.stack([self.x + lx * c - ly * s, self.y + lx * s + ly * c], 1))
            self.pts = self.pts[-40:]

    # ----- 한 번 -----
    def start(self, now):
        self.__init__(self.default_slot, self.log)
        self.state, self.t0 = 'SETTLE', now
        self.log(f'[주차G] {VERSION} 시작: 서서 빈 칸을 찾는다')

    def _set(self, state, now, text=''):
        self.state, self.t0 = state, now
        self.log(f'[주차G] -> {state} {text}')

    def _update_slot(self):
        if not self.pts:
            return
        p = np.concatenate(self.pts)
        got = pick_slot(p[:, 0], p[:, 1])
        if got is None:
            return
        if self.slot is None or got[2] or not self.slot[2]:
            if self.slot is None or got[0] != self.slot[0] or got[2] != self.slot[2]:
                self.log(f"[주차G] 빈 칸 {got[0]}번 (가운데까지 앞으로 {got[1]:.2f} m, {'확실' if got[2] else '불확실'})")
            self.slot = got

    def step(self, now):
        """반환: (속도 [m/s], 안쪽 바퀴 조향각 [rad]). 끝나면 state == 'DONE'."""
        st = self.state
        if st in ('IDLE', 'DONE'):
            return 0.0, 0.0
        if now - self.t0 > PHASE_TIMEOUT:
            self._set('DONE', now, f'({st} 단계가 너무 길어 정지)')
            return 0.0, 0.0
        if st == 'SETTLE':
            if now - self.t0 >= SETTLE_SEC:
                self.bias = self._gyro_sum / self._gyro_n if self._gyro_n >= 20 else 0.0
                self._update_slot()
                if self.slot is None:
                    self.slot = (self.default_slot, SLOT_X[self.default_slot - 1], False)
                    self.log(f'[주차G] 라이다로 빈 칸을 못 골랐다 -> 기본 {self.default_slot}번 칸')
                self._set('FORWARD', now, f"(자이로 {'있음' if self._gyro_n >= 20 else '없음 - odom 방향 사용'})")
            return 0.0, 0.0
        if st == 'FORWARD':
            if self.x < self.slot[1] - 0.25:
                self._update_slot()                      # 빈 칸을 지나치기 전까지 계속 고친다
            goal = min(self.slot[1] - END_BIAS + PARK_R, X_MAX)
            blocked = self.front is not None and self.front < FRONT_GUARD
            if self.x >= goal - 0.02 or blocked:
                self._set('PAUSE', now, f"(x {self.x:.2f} m{', 정면 막힘' if blocked else ''})")
                return 0.0, 0.0
            th_d = max(-0.10, min(0.10, K_Y * (Y_SHIFT - self.y)))
            v = V_FWD if goal - self.x > 0.15 else 0.10
            return v, steer_for(K_TH * (th_d - self.th))
        if st == 'PAUSE':
            if now - self.t0 >= 0.6:
                r = max(0.56, min(0.85, self.x - (self.slot[1] - END_BIAS)))
                self.c = (self.x + r * math.sin(self.th), self.y - r * math.cos(self.th), r)
                self.th_arc0 = self.th
                self._set('ARC', now, f'(반경 {r:.2f} m, 지금 방향 {math.degrees(self.th):+.0f}도)')
            return 0.0, 0.0
        if st == 'ARC':
            if self.th >= ARC_DONE:
                self._set('BACK', now, f'(x {self.x:.2f}, 깊이 {-self.y - SLOT_TOP:.2f} m)')
                return 0.0, 0.0
            cx, cy, r = self.c
            phi = math.atan2(self.y - cy, self.x - cx)
            er = math.hypot(self.x - cx, self.y - cy) - r
            eh = ((self.th - (phi - math.pi / 2) + math.pi) % (2 * math.pi)) - math.pi
            kappa = -(1.0 / r) - (K_E * er - K_H * eh)
            return -V_REV, steer_for(min(-0.2, kappa))
        if st == 'BACK':
            depth = -self.y - SLOT_TOP
            if depth >= DEPTH - 0.01:
                self.done_msg = (f'끝: 칸 가운데에서 옆으로 {self.x - self.slot[1]:+.2f} m, 깊이 {depth:.2f} m, '
                                 f'방향 {math.degrees(self.th):.0f}도 (자기 추정)')
                self._set('DONE', now, self.done_msg)
                return 0.0, 0.0
            return -V_REV, steer_for(max(-1.0, min(1.0, K_TH * (self.th - math.pi / 2))))
        return 0.0, 0.0


# ======================= ROS 노드 (매니저 연동) =======================
def managed_main(args=None):
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Bool, String

    class ClaudeParkingNode(Node):
        def __init__(self):
            super().__init__('claude_parking_node')
            self.declare_parameter('default_slot', DEFAULT_SLOT)
            slot = int(self.get_parameter('default_slot').value)
            self.plan = ParkPlanner(slot if slot in (1, 2, 3) else DEFAULT_SLOT, log=lambda s: self.get_logger().info(s))
            self._enabled = False
            self._enable_at = self._active_at = None
            self._active = ''
            self.cmd_pub = self.create_publisher(Twist, 'mission/parking/cmd_vel', 10)
            self.state_pub = self.create_publisher(String, 'mission/parking/state', 10)
            self.create_subscription(Bool, 'mission/parking/start', self._start_cb, 10)
            self.create_subscription(String, 'mission/active', self._active_cb, 10)
            self.create_subscription(Odometry, '/odom', self._odom_cb, 20)
            self.create_subscription(Imu, '/imu', self._imu_cb, qos_profile_sensor_data)
            self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
            self.create_timer(0.05, self._tick)
            self.get_logger().info(f'[주차G] {VERSION} 준비 (매니저의 시작 신호 대기)')

        @staticmethod
        def _sec(stamp):
            return stamp.sec + stamp.nanosec * 1e-9

        def _start_cb(self, msg):
            self._enabled = bool(msg.data)
            self._enable_at = time.monotonic()

        def _active_cb(self, msg):
            self._active = msg.data
            self._active_at = time.monotonic()

        def _odom_cb(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.plan.feed_odom(p.x, p.y, yaw, msg.twist.twist.linear.x, time.monotonic())

        def _imu_cb(self, msg):
            self.plan.feed_imu(msg.angular_velocity.z, time.monotonic())

        def _scan_cb(self, msg):
            self.plan.feed_scan(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min)

        def stop(self):
            self.cmd_pub.publish(Twist())

        def _tick(self):
            now = time.monotonic()
            allowed = (self._enabled and self._active == 'M4'
                       and self._enable_at is not None and now - self._enable_at <= 0.5
                       and self._active_at is not None and now - self._active_at <= 1.5)
            if self.plan.state == 'DONE':
                self.stop()
                self.state_pub.publish(String(data='complete'))
                return
            if not allowed:
                if self.plan.state != 'IDLE':
                    # 매니저가 멈추라고 했다 (비상 정지 등): 서고, 다시 허가가 오면 처음부터가 아니라 이어서 한다
                    self.plan.t0 = now if self.plan.t0 is None else self.plan.t0 + 0.05
                self.stop()
                return
            if self.plan.state == 'IDLE':
                self.plan.start(now)
            v, inner = self.plan.step(now)
            cmd = Twist()
            cmd.linear.x = float(v)
            cmd.angular.z = float(omega_for(inner, v))
            self.cmd_pub.publish(cmd)
            self.state_pub.publish(String(data='complete' if self.plan.state == 'DONE' else self.plan.state))

    rclpy.init(args=args)
    node = None
    try:
        node = ClaudeParkingNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            for _ in range(5):
                node.stop()
                time.sleep(0.02)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    managed_main()
