#!/usr/bin/env python3
"""T자 주차 (M4) — 빈 칸을 라이다로 찾아 넣는다. 기본은 전진 주차, PARK_MODE 한 줄로 후진 주차.  [claude 브랜치 전용]

대회장 (도면 2026Track): 차로 오른쪽에 주차 칸 3개(폭 0.50 m, 깊이 0.60 m)가 붙어 있고 그 뒤에 3칸이 더 있다.
  6칸 중 5칸에 LIMO 가 서 있고, 차로에 붙은 3칸 중 빈 한 칸에 넣는다 (어느 칸인지는 당일).
  주차 시작 흰 선에서 첫 칸 왼쪽 끝까지 0.44 m, 칸 가운데는 선에서 0.71 / 1.23 / 1.75 m.

전진 / 후진
  회전 반경: 조향 25도에서 실제 0.60~0.62 m (대회장 bag 자이로로 잼), 최대 조향(28도)은 0.55 m 쯤으로 추정.
  전진(기본): 다 돌았을 때 앞 범퍼가 칸 끝 선(0.60 m)의 몇 cm 앞에 선다. 여유가 작지만 라이다가 칸 안을 본다.
  후진: 차체가 칸 깊이 0.14 ~ 0.46 m 에 놓여 여유는 크지만 뒤를 못 본다. 3번 칸은 차 앞이 차로 끝 선을 넘는다.

순서 (매니저가 흰 선 앞 완전 정지를 확인하고 mission/parking/start 를 주면 시작)
  1) SETTLE  : 서서 자이로 0점을 잡는다
  2) FORWARD : 곧게 전진 (자이로로 방향 유지) 하면서 1번 칸부터 차례로 본다.
               LIMO 가 보이면 다음 칸. 라이다 광선이 칸 한가운데를 지나 더 뒤까지 가면 빈 칸 -> 그 칸으로.
               먼 칸은 앞 LIMO 에 가려 흰 선에서는 안 보일 수 있다: 그래서 서서 한 번에 정하지 않고, 그 칸 앞에 갈 때까지 본다.
               1, 2번에 차가 있으면 3번은 못 봐도 3번이다.
  전진 주차  3) ENTERF: 빈 칸 가운데의 반경만큼 앞에서 오른쪽으로 꺾어 들어가며 칸 가운데 선에 맞춘다.
                        들어가면서 실제 회전 반경을 자이로로 잰다. 칸 옆 선을 넘게 생겼거나 정면에 뭔가 닿을 듯하면
             4) KBACK : 왼쪽으로 꺾고 조금 물러났다가(차 머리가 오른쪽으로 더 돈다) 다시 ENTERF. 물러날 거리는 굴려 보고 고른다
  후진 주차  3) ARC   : 빈 칸을 반경만큼 지나가 서고, 오른쪽으로 꺾어 후진
             4) BACK  : 곧게 후진해서 깊이를 맞춘다
  5) DONE    : 정지, mission/parking/state = complete

기존 주차 노드(parking_line_detection)와 토픽이 같다. 매니저는 M4 에서 이 노드를 켠다 (mission_manager_node MISSION_PROCESSES).

터미널 줄:  [주차G] ...      값은 아래 상수. 실차에서 맞출 것은 '맞추기' 표시.
"""

import math
import time

import numpy as np

VERSION = 'claude-park3'

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
PARK_MODE = 'forward'        # 'forward' = 전진 주차 (기본) / 'reverse' = 후진 주차. 이 한 줄만 바꾸면 된다
# 전진 주차: 빈 칸 앞에서 오른쪽으로 최대한 꺾어 들어간다. 라이다가 칸 안(양옆 LIMO, 뒤 칸 LIMO)을 본다.
#   들어간 깊이는 회전 반경으로 정해진다: 다 돌았을 때 앞 범퍼 깊이 = 반경 - (차로 가운데에서 왼쪽으로 붙은 거리). 칸 깊이는 0.60 m
F_R = 0.66                   # 빈 칸 가운데의 이만큼 앞까지는 그 칸이 빈 것을 정해야 한다 [m] (R_EST0 보다 커야 한다)
F_BIAS = 0.04                # 칸 가운데보다 이만큼 오른쪽(+)에 서도록. 들어갈 때 차 오른쪽 뒤 모서리가 칸 왼쪽 선을 스치므로 조금 오른쪽을 노린다 (스치면 ↑)
F_Y_SHIFT = 0.05             # 전진하면서 시작 위치보다 왼쪽으로 이만큼 붙는다 (차 앞 왼쪽이 차로 윗선을 밟으면 ↓)
F_MAX_DEPTH = 0.45           # (옆 칸 차가 안 보일 때만 쓰는 기준. 보이면 앞 범퍼를 옆 차의 안쪽 끝에 맞춰 선다) 앞 모서리가 칸 입구 선에서 이 깊이에 닿으면 끝. 차 길이 0.32, 칸 깊이 0.60 이라 0.34 ~ 0.58 이면 다 들어간 것인데,
                             #   시작할 때 차로 안에서 좌우로 치우친 만큼(±6 cm) 실제 깊이가 달라져서 가운데를 노린다. 맞추기: 앞이 끝 선에 걸리면 ↓, 뒤가 입구 선에 걸리면 ↑
F_MIN_DEPTH, F_HARD_MAX = 0.36, 0.56      # 옆 차와 나란히 맞출 때도 추정 깊이가 이 사이일 때만 (옆 차가 이상하게 서 있을 때 대비)
F_FRONT_STOP = 0.12          # 정면(뒤 칸 LIMO 등)이 앞 범퍼에서 이 거리 안이면 거기서 끝낸다
K_L, K_TH2 = 4.0, 4.0        # 칸 가운데 선 따라가기: 옆으로 벗어난 거리 -> 방향, 방향 오차 -> 조향
R_EST0 = 0.62                # 최대 조향의 실제 회전 반경 처음 추정. 일부러 조금 크게 잡는다 (일찍 꺾기 시작하고, 들어가면서 자이로로 다시 재서 맞춘다)
R_MODEL_MIN = (WHEELBASE / math.tan(MAX_INNER) + TRACK / 2.0) * K_STEER      # 최대 조향에서 모델이 믿는 실제 반경 (0.555)
LINE_TOL = 0.02              # 칸 옆 선에서 이만큼 안쪽을 한계로 본다 (자기 위치 추정 오차 몫)
K_TURNS_MAX = 4              # 꺾어 물러나기(왼쪽으로 꺾고 후진 -> 오른쪽으로 꺾고 전진) 최대 횟수. 조향각이 모자라 한 번에 못 들어갈 때 쓴다
K_BACK_MAX = 0.36            # 한 번에 물러나는 최대 거리 [m]
LANE_TOP = 0.25              # 차로 가운데 -> 차로 윗선(왼쪽 선). 물러날 때 차체가 이 선을 넘지 않게 한다
# 후진 주차 (PARK_MODE = 'reverse'): 뒤는 라이다가 못 본다. 자이로 + odom 만으로 들어간다
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


def corners(x, y, th):
    """차체 네 모서리 (앞왼, 앞오, 뒤오, 뒤왼)."""
    c, s = math.cos(th), math.sin(th)
    return [(x + a * c - b * s, y + a * s + b * c) for a, b in ((FRONT, HALF_W), (FRONT, -HALF_W), (-REAR, -HALF_W), (-REAR, HALF_W))]


def body_pts(x, y, th):
    """차체 둘레 점 (모서리 넷 + 앞면 가운데 + 양옆 가운데)."""
    c, s = math.cos(th), math.sin(th)
    return [(x + a * c - b * s, y + a * s + b * c) for a, b in ((FRONT, HALF_W), (FRONT, -HALF_W), (-REAR, -HALF_W), (-REAR, HALF_W),
                                                               (FRONT, 0.0), (0.10, HALF_W), (0.10, -HALF_W))]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


MOVING = ('FORWARD', 'PASS', 'PAUSE', 'ARC', 'BACK', 'ENTERF', 'KPLAN', 'KBACK', 'RETURN')


class ParkPlanner:
    """ROS 없이 도는 본체. 좌표: 시작할 때 뒤 차축이 원점, 차 앞이 +x, 왼쪽이 +y. 칸은 오른쪽(-y)."""

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
        self.scans = []                    # 빈 칸 판단용: (라이다 위치, 광선 끝점들, 맞은 광선인가) - 시작 좌표
        self.cloud = []                    # 본 장애물 점 (시작 좌표). 후진·전진 계획의 여유 확인용
        self.live = (np.zeros(0), np.zeros(0))   # 지금 스캔의 점 (차 좌표)
        self.front = None                  # 정면 통로의 가장 가까운 것 [m, 앞 범퍼에서]
        self.shift = 0.0                   # 칸 전체가 도면 위치에서 앞뒤로 밀린 양
        self.k = 1                         # 지금 살펴보는 칸
        self.seen = {}                     # 칸 번호 -> 'occ' / 'empty'
        self.ev = {}                       # 칸 번호 -> (스캔당 맞은 점 수, 스캔당 칸을 지나간 광선 수)
        self.ret = None                    # 지나친 칸으로 되돌아가기: (칸 번호, 목표 x)
        self.slot = None                   # 정한 칸: (번호, 칸 가운데 x, 직접 확인했나)
        self.wait_t = None
        self.c = None                      # 후진 원: (cx, cy, R)
        self.r_est = R_EST0                # 최대 조향의 실제 회전 반경 (들어가면서 잰다)
        self._sat = None                   # 반경 재기: 바로 전 (거리, 방향, 명령 곡률)
        self._turn = self._acc_s = self._acc_th = 0.0
        self._ign = 0.0
        self.odo = 0.0
        self.kturns = 0
        self.kb = None                     # 꺾어 물러나기: (목표 거리, 시작 odo, 곡률)
        self.kp = None
        self.done_msg = ''

    # ----- 센서 -----
    def feed_imu(self, wz, t, seen=None):
        """t = 메시지 시각 (적분용), seen = 받은 시각 (자이로가 살아 있는지 보는 용, odom 과 같은 시계)."""
        self._imu_seen = t if seen is None else seen
        if self.state == 'SETTLE':
            self._gyro_n += 1
            self._gyro_sum += wz
        elif self.state in MOVING and self._gyro_t is not None:
            dt = t - self._gyro_t
            if 0.0 < dt < 0.5:
                self.th += (wz - self.bias) * dt
        self._gyro_t = t

    def feed_odom(self, ox, oy, oyaw, v, t):
        if self._odom is not None and self.state in MOVING:
            px, py, pyaw = self._odom
            d = math.hypot(ox - px, oy - py)
            if d < 0.2:
                fwd = (ox - px) * math.cos(pyaw) + (oy - py) * math.sin(pyaw)
                ds = math.copysign(d, fwd if abs(fwd) > 1e-6 else v)
                if self._imu_seen is None or t - self._imu_seen > 0.5:        # 자이로 없음: odom 방향을 줄여서 쓴다
                    self.th += wrap(oyaw - pyaw) / YAW_ODOM_SCALE
                self.x += ds * math.cos(self.th)
                self.y += ds * math.sin(self.th)
                self.odo += abs(ds)
        self._odom = (ox, oy, oyaw)

    def feed_scan(self, ranges, angle_min, angle_inc, range_min=0.15):
        r = np.asarray(ranges, dtype=np.float32)
        a = angle_min + np.arange(r.size) * angle_inc
        far = ~np.isfinite(r) | (r >= 3.5)                     # 돌아온 것 없음 = 그 방향은 3.5 m 까지 비어 있다
        hit = ~far & (r >= max(range_min, 0.12))
        use = hit | far
        rr = np.where(far, 3.5, r)[use]
        lx, ly, h = LIDAR_X + rr * np.cos(a[use]), rr * np.sin(a[use]), hit[use]
        self.live = (lx[h], ly[h])
        corridor = h & (np.abs(ly) < HALF_W + 0.04) & (lx > FRONT)
        self.front = float(lx[corridor].min() - FRONT) if corridor.any() else None
        c, s = math.cos(self.th), math.sin(self.th)
        w = np.stack([self.x + lx * c - ly * s, self.y + lx * s + ly * c], 1)
        if self.state in ('SETTLE', 'FORWARD') and abs(self.th) < math.radians(12):
            self.scans.append(((self.x + LIDAR_X * c, self.y + LIDAR_X * s), w, h))
            self.scans = self.scans[-40:]
        if self.state != 'IDLE':
            self.cloud.append(w[h & (rr < 1.5)])
            self.cloud = self.cloud[-50:]

    # ----- 빈 칸 판단 -----
    def _update_shift(self):
        """LIMO 앞면 덩어리의 가운데로 칸 전체의 앞뒤 밀림을 잰다 (정지 위치가 틀려도 되게)."""
        if not self.scans:
            return
        p = np.concatenate([w[h] for _, w, h in self.scans[-15:]])
        x = np.sort(p[(p[:, 1] > -0.85) & (p[:, 1] < -0.28) & (p[:, 0] > 0.15) & (p[:, 0] < 2.9), 0])
        if x.size < 5:
            return
        res = []
        for seg in np.split(x, np.nonzero(np.diff(x) > 0.12)[0] + 1):
            if seg.size >= 5 and 0.15 <= seg[-1] - seg[0] <= 0.32:
                mid = (seg[0] + seg[-1]) / 2.0
                d = mid - np.array(SLOT_X)
                j = int(np.argmin(np.abs(d)))
                if abs(d[j]) <= BLOCK_SHIFT:
                    res.append(d[j])
        if res:
            self.shift = float(np.median(res))

    def slot_state(self, k):
        """k번 칸: 'occ' (LIMO 가 보임) / 'empty' (광선이 칸 한가운데를 지나 더 뒤까지 감) / 'unknown' (가려서 안 보임)."""
        c = SLOT_X[k - 1] + self.shift
        hits = free = 0
        for (ox, oy), w, h in self.scans[-25:]:
            px, py = w[:, 0], w[:, 1]
            hits += int(np.count_nonzero(h & (np.abs(px - c) < 0.15) & (py > -0.82) & (py < -0.30)))
            thru = (py < -0.64) & (oy > -0.45)
            if thru.any():
                xc = ox + (px[thru] - ox) * (-0.56 - oy) / (py[thru] - oy)
                free += int(np.count_nonzero(np.abs(xc - c) < 0.06))
        n = max(1, len(self.scans[-25:]))
        self.ev[k] = (hits / n, free / n)
        if hits >= 1.2 * n and free < 0.8 * n:
            return 'occ', hits
        if free >= 0.8 * n and hits < 0.4 * n:
            return 'empty', hits
        return 'unknown', hits

    def _clear(self, pts_xy, margin):
        """차체 점들이 본 장애물 점에서 margin 이상 떨어져 있나."""
        if not self.cloud:
            return True
        c = np.concatenate(self.cloud[-25:])
        if c.shape[0] > 500:
            c = c[:: c.shape[0] // 500 + 1]
        q = np.asarray(pts_xy)
        d2 = ((q[:, None, :] - c[None, :, :]) ** 2).sum(2)
        return bool(d2.min() >= margin * margin) if d2.size else True

    # ----- 한 번 -----
    def start(self, now):
        self.__init__(self.default_slot, self.log)
        self.state, self.t0 = 'SETTLE', now
        self.log(f"[주차G] {VERSION} 시작 ({'전진' if PARK_MODE == 'forward' else '후진'} 주차): 서서 빈 칸을 찾는다")

    def _set(self, state, now, text=''):
        self.state, self.t0 = state, now
        self.log(f'[주차G] -> {state} {text}')

    def _commit(self, k, sure, why, now):
        c = SLOT_X[k - 1] + self.shift
        self.slot = (k, c, sure)
        self.log(f'[주차G] 빈 칸 {k}번으로 정함: {why} (칸 가운데까지 앞으로 {c:.2f} m, 칸 밀림 {self.shift:+.2f} m)')
        if PARK_MODE == 'forward':
            self._sat = None
            self._set('ENTERF', now, f'(x {self.x:.2f} m, 옆 {self.y:+.2f} m)')
        else:
            self._set('PASS', now, '(빈 칸을 지나가서 선다)')

    def _forward(self, now):
        """곧게 가면서 1번 칸부터 차례로 본다. 빈 칸이면 그 칸으로, 차 있으면 다음 칸."""
        self._update_shift()
        while True:
            k = self.k
            c = SLOT_X[k - 1] + self.shift
            dec = (c + F_BIAS - F_R - 0.06) if PARK_MODE == 'forward' else (c - 0.15)     # 이 지점까지는 정해야 한다
            st, hits = self.slot_state(k)
            if st == 'occ' and k < 3:
                self.seen[k] = 'occ'
                self.k += 1
                self.wait_t = None
                continue
            break
        last = k == 3 and self.seen.get(1) == 'occ' and self.seen.get(2) == 'occ'
        if st == 'occ' and k == 3:
            if self.x >= dec:                       # 세 칸 다 차 보인다 = 어느 칸을 잘못 봤다. 빈 증거가 가장 많은 칸으로
                kb = max((1, 2, 3), key=lambda j: self.ev.get(j, (9, 0))[1] - self.ev.get(j, (9, 0))[0])
                cb = SLOT_X[kb - 1] + self.shift
                back = (cb + F_BIAS - F_R - 0.06) if PARK_MODE == 'forward' else (cb - END_BIAS + PARK_R)
                self.k = kb
                if kb < 3 and self.x > back + 0.03 and PARK_MODE == 'forward':
                    self.ret = (kb, back)
                    self._set('RETURN', now, f'(세 칸 모두 차 있는 것처럼 보임 -> 빈 증거가 가장 많은 {kb}번 칸 앞으로 곧게 되돌아감)')
                    return 0.0, 0.0
                self._commit(kb, False, '세 칸 모두 차 있는 것처럼 보임 -> 빈 증거가 가장 많은 칸', now)
                return self.step(now)
        elif self.x >= dec:
            if st == 'empty':
                self._commit(k, True, '빈 것을 직접 봄', now)
                return self.step(now)
            if last:
                self._commit(3, False, '1, 2번 칸에 차가 있음 (3번은 가려서 직접 못 봄)', now)
                return self.step(now)
            if self.wait_t is None:
                self.wait_t = now
            if now - self.wait_t < 1.2:
                return 0.0, 0.0                     # 서서 더 본다
            if not self.scans or sum(int(h.sum()) for _, _, h in self.scans[-10:]) < 20:
                kk = self.default_slot if SLOT_X[self.default_slot - 1] + self.shift >= c - 1e-6 else k
                self.k = kk
                self._commit(kk, False, '라이다에 아무것도 안 보임 -> 기본 칸', now)
                return self.step(now)
            self.seen[k] = 'occ'                    # 빈 것을 확인 못 한 칸에는 안 들어간다
            self.log(f'[주차G] {k}번 칸이 빈 것을 확인 못 함 -> 다음 칸')
            self.k, self.wait_t = min(3, k + 1), None
            if k == 3:
                self._commit(3, False, '확인 못 했지만 마지막 칸', now)
                return self.step(now)
            return 0.0, 0.0
        blocked = self.front is not None and self.front < FRONT_GUARD
        if blocked:
            self._commit(k, False, '정면이 막혀 더 못 감', now)
            return self.step(now)
        ysh = F_Y_SHIFT if PARK_MODE == 'forward' else Y_SHIFT
        th_d = max(-0.10, min(0.10, K_Y * (ysh - self.y)))
        v = V_FWD if (dec - self.x > 0.15 or st == 'empty') else 0.10
        return v, steer_for(K_TH * (th_d - self.th))

    # ----- 전진 주차 -----
    def _enter_kappa(self, x, th, xg, r):
        """전진 진입 조향 -> (곡률, 최대 조향인가). r = 최대 조향의 실제 회전 반경.
        지금 방향에서 최대로 꺾어 돌면 칸 가운데 선에 딱 닿는 지점까지는 곧게 가고, 거기서부터 최대로 꺾는다.
        (반경이 생각보다 작으면 저절로 늦게 꺾고, 크면 선을 지나친다 -> 그건 꺾어 물러나기가 고친다)
        거의 다 돌면(75도) 칸 가운데 선을 따라가는 미세 조향으로 바꾼다."""
        alpha = -th
        if alpha < math.radians(75.0):
            e = (xg - x) - r * (1.0 - math.sin(max(0.0, alpha)))
            f = 1.0 if e <= 0.0 else max(0.0, 1.0 - e / 0.04)
            return -f / r, f >= 0.97
        th_d = -math.pi / 2 + max(-0.6, min(0.6, K_L * (xg - x)))
        k = max(-1.0 / r, min(1.0 / r, K_TH2 * wrap(th_d - th)))
        return k, abs(k) >= 0.97 / r

    def _lat_out(self, x, y, th, tol):
        """칸 안(입구 선 아래)에 들어간 모서리가 칸 옆 선을 넘는 양 [m] (안 넘으면 0 이하)."""
        worst = -9.0
        for cx, cy in corners(x, y, th):
            if -cy - SLOT_TOP > 0.005:
                worst = max(worst, abs(cx - self.slot[1]) - (SLOT_PITCH / 2 - 0.01 - tol))
        return worst

    def _front_depth(self, x, y, th):
        return max(-cy for cx, cy in corners(x, y, th)[:2]) - SLOT_TOP

    def _roll_in(self, x, y, th, r, xg):
        """지금 자세에서 전진 진입 규칙대로 굴려 본다 -> (잘 끝나나, 끝 자세, 어긋난 정도)."""
        for i in range(160):
            if i % 3 == 0 and not self._clear(body_pts(x, y, th), 0.03):
                return False, (x, y, th), 1.5 + abs(x - xg)             # 본 장애물(옆 칸 차)에 닿는다
            if self._lat_out(x, y, th, LINE_TOL) > 0.0:
                return False, (x, y, th), 1.0 + abs(x - xg)
            if self._front_depth(x, y, th) >= F_MAX_DEPTH:
                err = abs(wrap(th + math.pi / 2))
                return err <= math.radians(6.0) and abs(x - xg) <= 0.05, (x, y, th), err + 2.0 * abs(x - xg)
            k = self._enter_kappa(x, th, xg, r)[0]
            x, y, th = x + 0.01 * math.cos(th), y + 0.01 * math.sin(th), th + 0.01 * k
        return False, (x, y, th), 2.0

    def _plan_back(self):
        """어느 쪽으로 꺾고 얼마나 물러나면 다시 들어갈 때 맞는지 굴려 보고 고른다 -> (거리, 물러날 때 곡률, 어긋남, 맞나).
        왼쪽으로 꺾고 물러나면 차 머리가 오른쪽으로 더 돈다 (덜 돌았을 때). 오른쪽으로 꺾고 물러나면 온 길을 되짚는다 (너무 일찍 꺾었을 때)."""
        kmax, xg = 1.0 / self.r_est, self.slot[1] + F_BIAS
        best = None
        for kr in (kmax, 0.0, -kmax):
            x, y, th, d = self.x, self.y, self.th, 0.0
            while d < K_BACK_MAX:
                x, y, th = x - 0.01 * math.cos(th), y - 0.01 * math.sin(th), th - 0.01 * kr
                d += 0.01
                cs = corners(x, y, th)
                if max(cy for _, cy in cs) > LANE_TOP - 0.015 or not self._clear(cs, 0.04):
                    break                               # 차로 윗선이나 본 장애물에 닿는다: 여기까지만
                if d >= 0.06 and round(d * 100) % 3 == 0:
                    ok, _, cost = self._roll_in(x, y, th, self.r_est, xg)
                    cost += 0.2 * d + (0.0 if ok else 1.0)
                    if best is None or cost < best[2] - 1e-6:
                        best = (d, kr, cost, ok)
                    if ok:
                        break
        return best if best is not None else (0.06, kmax, 9.0, False)

    def _flush(self):
        """옆 칸 LIMO 의 안쪽 끝이 내 앞 범퍼보다 얼마나 더 앞에 있나 [m] (양옆 평균). 옆 차가 안 보이면 None.
        옆 차와 나란히(앞 범퍼를 옆 차의 안쪽 끝에 맞춰) 서면, 옆 차가 칸 안에 있는 한 나도 칸 안이다.
        내 위치 추정(시작할 때 차로 안에서 치우친 정도, 시작 방향 오차)에 기대지 않는 깊이 기준."""
        lx, ly = self.live
        ends = []
        for side in (1.0, -1.0):
            m = (side * ly > 0.24) & (side * ly < 0.60) & (lx > -0.05) & (lx < 1.0)
            if np.count_nonzero(m) < 4:
                continue
            xs = np.sort(lx[m])
            if xs[0] > FRONT + 0.12:
                continue                                    # 옆 칸이 아니라 뒤 줄 차다
            cut = np.nonzero(np.diff(xs) > 0.12)[0]
            ends.append(float(xs[cut[0]] if cut.size else xs[-1]))
        return (sum(ends) / len(ends) - FRONT) if ends else None

    def _enter(self, now):
        x, y, th = self.x, self.y, self.th
        xg = self.slot[1] + F_BIAS
        fd = self._front_depth(x, y, th)
        ang = abs(wrap(th + math.pi / 2))
        lx, ly = self.live
        touch = bool(np.any((lx > 0.10) & (lx < FRONT + 0.05) & (np.abs(ly) < HALF_W + 0.02))) if lx.size else False
        near = self.front is not None and self.front < F_FRONT_STOP and ang <= math.radians(25.0)   # 거의 다 돌았을 때의 정면 = 뒤 줄 차
        flush = self._flush() if ang <= math.radians(20.0) else None
        if flush is None:
            deep = fd >= F_MAX_DEPTH                    # 옆 차가 안 보이면 내 위치 추정으로
        else:
            deep = (flush <= 0.01 and fd >= F_MIN_DEPTH) or fd >= F_HARD_MAX
        if deep or (near and fd > 0.25):
            good = ang <= math.radians(6.0) and abs(x - xg) <= 0.06 and self._lat_out(x, y, th, 0.0) <= 0.0
            if good or self.kturns >= K_TURNS_MAX:
                how = '옆 차와 나란히' if flush is not None and flush <= 0.01 else ('정면이 가까움' if near else '추정 깊이로')
                self.done_msg = (f'끝({how}): 칸 가운데에서 옆으로 {x - self.slot[1]:+.2f} m, 앞 모서리 깊이 {fd:.2f} m, 방향 {math.degrees(-th):.0f}도 '
                                 f"(자기 추정), 꺾어 물러나기 {self.kturns}번{'' if good else ', 자세가 덜 맞음'}")
                self._set('DONE', now, self.done_msg)
                return 0.0, 0.0
            return self._start_back(now, '깊이는 다 들어갔는데 방향·옆이 안 맞음')
        # 조금 앞(6 cm)에서 칸 옆 선을 넘거나, 지금 정면에 뭔가 닿을 듯하면: 꺾어 물러났다가 다시
        k, full = self._enter_kappa(x, th, xg, self.r_est)
        x2, y2, th2 = x + 0.06 * math.cos(th + 0.03 * k), y + 0.06 * math.sin(th + 0.03 * k), th + 0.06 * k
        if touch:                                       # 장애물은 무조건: 물러나서 다시, 다 썼으면 정지
            if self.kturns >= K_TURNS_MAX:
                self._set('DONE', now, '(정면이 막힘: 더 못 고치고 정지)')
                return 0.0, 0.0
            return self._start_back(now, '정면에 닿을 듯함')
        if self._lat_out(x2, y2, th2, LINE_TOL) > 0.0 and self.kturns < K_TURNS_MAX and self.odo >= self._ign:
            return self._start_back(now, '이대로 가면 칸 옆 선을 넘음', soft=True)
        # 최대 조향으로 도는 동안 실제 회전 반경을 잰다 (다음 계획에 쓴다)
        if self._sat is not None:
            o0, th0, kprev = self._sat
            keff = min(abs(kprev), 1.0 / R_MODEL_MIN)          # 명령한 곡률 (모델이 낼 수 있다고 믿는 만큼)
            if keff > 0.5:
                self._turn = self._turn + (self.odo - o0)
                if self._turn > 0.05:                              # 처음 5 cm 는 조향이 따라오는 중
                    self._acc_s += keff * (self.odo - o0)
                    self._acc_th += abs(th - th0)
                    if self._acc_th > math.radians(8.0):
                        self.r_est = max(0.45, min(0.85, R_MODEL_MIN * self._acc_s / self._acc_th))
        kcmd = math.copysign(3.0, k) if full else k * self.r_est / R_MODEL_MIN      # 덜 꺾을 때도 잰 비율만큼 고쳐서 명령
        self._sat = (self.odo, th, kcmd)
        return 0.11, steer_for(kcmd)

    def _start_back(self, now, why, soft=False):
        """먼저 선다. 선 다음(KPLAN)에 어떻게 물러날지 계산한다 - 계산하는 동안 차가 움직이면 안 된다."""
        self.kp = (why, soft)
        self._set('KPLAN', now, f'({why})')
        return 0.0, 0.0

    def _kplan(self, now):
        why, soft = self.kp
        d, kr, cost, ok = self._plan_back()
        if soft and not ok:
            self._ign = self.odo + 0.10                 # 고칠 방법이 없으면 선을 조금 밟더라도 들어간다 (입구에 서 버리는 것보다 낫다)
            self._set('ENTERF', now, '(물러나도 못 고침 -> 그대로 들어감)')
            return 0.0, 0.0
        self.kturns += 1
        self.kb = (d, self.odo, kr)
        how = '왼쪽으로 꺾고' if kr > 0 else ('오른쪽으로 꺾고' if kr < 0 else '곧게')
        self._set('KBACK', now, f"({how} {d:.2f} m 물러남, 잰 회전 반경 {self.r_est:.2f} m{'' if ok else ', 한 번으로는 다 못 맞춤'})")
        return 0.0, 0.0

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
                self._set('FORWARD', now, f"(자이로 {'있음' if self._gyro_n >= 20 else '없음 - odom 방향 사용'})")
            return 0.0, 0.0
        if st == 'FORWARD':
            return self._forward(now)
        if st == 'ENTERF':
            return self._enter(now)
        if st == 'RETURN':
            kb, back = self.ret
            if now - self.t0 < 0.4:
                return 0.0, 0.0
            if self.x <= back + 0.01:
                self._commit(kb, False, '되돌아와서 다시 들어감', now)
                return 0.0, 0.0
            return -V_REV, steer_for(max(-1.0, min(1.0, K_TH * self.th)))
        if st == 'KPLAN':
            return (0.0, 0.0) if now - self.t0 < 0.4 else self._kplan(now)
        if st == 'KBACK':
            d, o0, kr = self.kb
            if now - self.t0 < 0.2:
                return 0.0, 0.0
            cs = corners(self.x, self.y, self.th)
            if self.odo - o0 >= d or max(cy for _, cy in cs) > LANE_TOP - 0.005:
                self._sat = None
                self._set('ENTERF', now, '(다시 들어감)')
                return 0.0, 0.0
            return -V_REV, steer_for(math.copysign(3.0, kr) if kr else 0.0)
        if st == 'PASS':
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
                self._set('ARC', now, f'(반경 {r:.2f} m, 지금 방향 {math.degrees(self.th):+.0f}도)')
            return 0.0, 0.0
        if st == 'ARC':
            if self.th >= ARC_DONE:
                self._set('BACK', now, f'(x {self.x:.2f}, 깊이 {-self.y - SLOT_TOP:.2f} m)')
                return 0.0, 0.0
            cx, cy, r = self.c
            phi = math.atan2(self.y - cy, self.x - cx)
            er = math.hypot(self.x - cx, self.y - cy) - r
            eh = wrap(self.th - (phi - math.pi / 2))
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
            stamp = self._sec(msg.header.stamp)
            now = time.monotonic()
            self.plan.feed_imu(msg.angular_velocity.z, stamp if stamp > 0.0 else now, seen=now)

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
