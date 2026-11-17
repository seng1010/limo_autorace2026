#!/usr/bin/env python3
"""3차선 장애물(M7) + 갈림길(M2) 한 파일 — '위치를 안 세는' 방식.

생각
  - 몇 번 차로인지, 시작선에서 몇 m 인지 세지 않는다. 매 프레임 '지금 차에서 보이는 것'만으로 길을 고른다.
  - 라이다 점(정차 LIMO) + 카메라로 본 노란 선(= 벽)을 차 둘레 2 cm 격자에 찍는다.
  - 도로와 나란한 줄 15개(지금 자리에서 옆으로 -0.75 ~ +0.75 m)마다 실제 조향 규칙으로 경로를 미리 굴려
    '차체가 안 닿고 갈 수 있는 거리'를 잰다. 가장 멀리 가는 줄들 중 지금 가던 줄(갈림길이면 표지판 쪽)에 가까운 줄로 간다.
  - 갈림길: YOLO 표지판 상자가 보이면 그 방향·거리(라이다로 보정)로 섬 위치를 잡는다.
      표지판을 아직 모르면 섬 READ_STOP 앞에서 서서 읽고, 알면 반대쪽 입구를 벽으로 막고 표지판 쪽으로 간다.
      갈림길 차로에 들어가면 기본 주행(simple_lane 노란 차선 주행)에 넘기고, 섬 뒤 합류가 끝날 때까지 코너 고정 조향만 꺼 둔다.
  - odom 은 '방금 옆으로 얼마나 움직였나'(목표 줄을 이어 가는 용도)와 지나친 점 기억에만 쓴다.

쓰는 법 (simple_lane.py, three_lane.py 와 같은 폴더에 이 파일을 둔다. 두 파일은 고치지 않는다)
  python3 lane3_gap.py drive lane3 rec            3차선 입구(시작선 0.3~1 m 앞, 가운데)에 놓고 단독 시험
  python3 lane3_gap.py drive lane3 goal_a rec     표지판 대신 갈림길 방향 고정 (goal_b = 오른쪽)
  python3 lane3_gap.py drive --topic inference/cmd_vel     매니저와 함께 (simple_lane 대신 이걸 켠다. mission/active = M7 에서 켜짐)
  인자는 simple_lane.py 와 같다. 이 파일은 simple_lane 을 불러서 3차선 부분만 바꿔 끼운다.

터미널 줄:  3차선G <상태> 목표 +0.32 m, 갈 수 있는 거리 1.8 (좌 2.2 중 0.9 우 0.4) 표지판 A 섬 1.6 m | 사유
"""

import math
import sys
import time
import types
from collections import deque

import cv2
import numpy as np

try:
    import three_lane as T               # 같은 폴더의 파일 (LineFitter, SignVote 만 빌려 쓴다)
except ImportError:
    from jetracer_ros2 import three_lane as T

# ---------------------------------------------------------------------------
# 실차에서 맞출 값
# ---------------------------------------------------------------------------
V_KEEP = 0.20            # 곧게 갈 때 속도 [m/s]
V_CHANGE = 0.16          # 옆으로 옮기는 중
V_SLOW = 0.10            # 앞이 가깝거나 급하게 꺾을 때
M_OBS_COMFORT = 0.10     # 장애물(라이다)과 차체 사이 여유: 보통 [m]
M_OBS = 0.05             # 보통 여유로 MIN_GO 도 못 가면 이 여유로
M_LINE = 0.03            # 노란 선과 차체 사이 여유
MIN_GO = 0.90            # 이만큼도 못 가는 길뿐이면 여유를 줄이고, 그래도 없으면 급하게 꺾는 경로를 본다
STOP_FREE = 0.30         # 고른 길이 이만큼도 안 되면 정지
GUARD_DIST = 0.28        # 지금 조향대로 이 거리 안에 장애물이 걸리면 무조건 정지
TIE = 0.30               # 가장 멀리 가는 줄과 이 차이 안이면 같은 것으로 보고 가까운 줄을 고른다
LIDAR_AHEAD = 0.0        # 라이다가 카메라 바닥 투영점보다 앞에 있는 거리 [m] (안 쟀음. 재면 넣기)
LIDAR_MAX = 3.3
# 표지판
SIGN_K = 56.0            # 표지판까지 거리 [m] = SIGN_K / 상자 높이 [px] (33 px ≈ 1.7 m, 41 px ≈ 1.3 m)
YOLO_IMG_W = 640.0       # YOLO 가 보는 영상 가로 [px] (다르면 고칠 것: 표지판 방향 계산에 쓴다)
BEAR_K = 1.113           # (상자 가운데 가로 비율 - 0.5) * BEAR_K = tan(방향). 아래 고친 카메라 보정값에서 계산
SIGN_NEED_N = 3          # 표지판 상자가 최근 SIGN_NEED_SEC 안에 이만큼, 비슷한 자리에서 보여야 섬으로 믿는다
SIGN_NEED_SEC = 1.5      # (11/17 실차: 상자 하나를 한 번 표지판으로 잘못 본 것을 섬으로 믿고 그쪽으로 갔다)
# 카메라 바닥 보정 (three_lane.BEV_GROUND 대신 이 파일 안에서만 쓴다. three_lane.py 는 그대로).
#   11/17 실차 기록: 기존 보정으로는 나란한 차로 선이 멀수록 벌어져 보이고(1.3 m 에서 1.42배), 도로 방향이 7~15도 틀렸다.
#   차로 선이 모이는 점(영상 세로 122 px)으로 다시 계산: 먼 줄(영상 세로 46 %)은 1.326 m / +-0.24 가 아니라 1.157 m / +-0.169.
#   옆 거리는 차로 폭(0.535)에 맞춰 1.065배. USE_BEV_FIX = False 면 기존 보정 + 기존 보정용 줄임(FAR_SHRINK)
USE_BEV_FIX = True
BEV_LAT = 1.065
BEV_GROUND_FIX = [-0.1686 * BEV_LAT, 1.157, 0.1686 * BEV_LAT, 1.157, 0.22 * BEV_LAT, 0.421, -0.22 * BEV_LAT, 0.421]
SIGN_SEE_H = 22.0        # 이 높이 이상 보일 때부터 섬 위치로 쓴다 (약 2.5 m)
GUESS_MIN_H = 28.0       # 확정 못 했을 때 '많이 읽힌 쪽'으로 세는 최소 높이
READ_STOP = 1.50         # 표지판을 아직 모르면 섬 이 거리 앞에서 서서 읽는다
READ_WAIT_SEC = 3.0      # 서서 읽는 최대 시간. 지나면 많이 읽힌 쪽, 그것도 없으면 DEFAULT_SIDE
DEFAULT_SIDE = 2         # 0 = A(왼쪽), 2 = B(오른쪽)
FORK_OFFSET = 0.535      # 섬 가운데에서 갈림길 차로 가운데까지 옆 거리
FORK_WAIT_SEC = 4.0      # 표지판 쪽이 막혀 이만큼 서 있으면 반대쪽 벽을 치운다 (장애물 회피가 우선, 완주가 목표)
HANDOVER_IN = 0.45       # 섬 앞끝을 이만큼 지나 갈림길 차로 안이고, 기본 주행이 양쪽 노란 선을 잡으면 기본 주행에 넘긴다
HANDOVER_MAX = 1.00      # 양쪽 노란 선을 못 잡아도 섬 앞끝 + 이 거리에서는 넘긴다 (섬 뒤 합류의 꺾인 선은 기본 주행이 따라간다)
EXIT_AFTER = 2.60        # 섬 앞끝 + 이 거리까지는 기본 주행의 코너 고정 조향(latch)을 꺼 둔다 (도면: 합류 끝 2.29.
                         # 켜 두면 합류의 45도 선을 90도 코너로 보고 길을 벗어났다 - 11/15 실차)
# 막혔을 때
STUCK_SEC = 1.5
REVERSE_M = 0.50
REVERSE_MAX = 3
V_REVERSE = 0.10
REVERSE_RESUME_FREE = 0.50   # 정지 한계 30 cm보다 여유 있는 전진 경로가 생기면 후진 조기 종료
REVERSE_RESUME_FRAMES = 3
# 켜고 끄기
ARM_AFTER_M = 1.4        # 매니저: M7 시작 후 이만큼 간 뒤부터 (그 전은 코너)
ENTRY_FRAMES = 3         # 흰 점선이 연속 이만큼 보이면 시작
ENTRY_WAIT_M = 1.2       # 그래도 못 보면 ARM_AFTER_M + 이 거리에서 그냥 시작
MAX_ON_M = 9.0           # 이만큼 넘게 가면 끝낸다
NO_SIGN_EXIT_M = 3.0     # 표지판 상자를 한 번도 못 봤을 때(YOLO 꺼짐): 이만큼 간 뒤 양쪽 노란 선이 계속 보이면 갈림길 차로로 본다
NO_SIGN_FORK_M = 2.2     # 그때 latch 를 꺼 두는 거리

# ---------------------------------------------------------------------------
# 거의 안 바꾸는 값
# ---------------------------------------------------------------------------
CAM = T.CAMERA_FORWARD_OFFSET          # 뒤 차축 -> 카메라 바닥 투영점 0.184
WB, TRACK, MAX_STEER = T.WHEELBASE, T.TRACK, T.MAX_STEER
ROLL_STEER = math.radians(23.0)        # 미리 굴릴 때 최대 조향 (실제 25도보다 작게)
STEP = 0.04                            # 미리 굴리는 한 걸음 [m]
N_STEP = 70                            # 2.8 m (다음 줄 장애물까지 보고 고른다)
LAG_ALPHA = 1.0 - math.exp(-STEP / 0.08)   # 조향이 먹는 데 0.08 m
CAND = np.arange(-7, 8) * 0.107        # 후보 줄 (5칸 = 차로 하나 0.535)
HMAX = math.radians(30.0)              # 옮길 때 도로와 이 각도 넘게 틀지 않는다
HMAX_TIGHT = math.radians(45.0)        # 급할 때
LOOK_KEEP, LOOK_CHANGE, LOOK_TIGHT = 0.55, 0.42, 0.34
BODY_R = 0.115                         # 차체(0.32 x 0.22, 뒤 차축 뒤 0.06 ~ 앞 0.26)를 덮는 원 5개의 반지름
BODY_OFF = (-0.03, 0.035, 0.10, 0.165, 0.23)   # 원 가운데: 뒤 차축 앞 [m]
NB = len(BODY_OFF)
RES = 0.02
GX0, GX1, GY0, GY1 = -0.80, 3.40, -1.40, 1.40
NX, NY = int((GX1 - GX0) / RES), int((GY1 - GY0) / RES)
MEM_STEP, MEM_KEEP = 0.05, 26          # 지나친 점 기억: 5 cm 마다, 1.3 m 치
TH_GAIN, TH_ACCEPT = 0.30, math.radians(20.0)
HINT_COST = 0.10                       # 차로 가운데(보이는 선으로 짐작)가 아닌 줄에 붙는 벌점 [m]
KEEP_BONUS = 0.20                      # 가던 줄을 유지하는 쪽으로 주는 덤 [m]
SIDE_BIAS = 0.04                       # 똑같으면 표지판 쪽(모르면 DEFAULT_SIDE)
LANE_HALF = 0.2675


def make_fitter():
    """three_lane.LineFitter 를 만들되, 카메라 바닥 보정만 고친 값으로 바꾼다."""
    fit = T.LineFitter()
    if USE_BEV_FIX:
        src = np.float32(T.BEV_SRC).reshape(4, 2)
        src[:, 0] *= T.IMG_W - 1
        src[:, 1] *= T.IMG_H - 1
        i2g = cv2.getPerspectiveTransform(src, np.float32(BEV_GROUND_FIX).reshape(4, 2))
        g2b = np.array([[1 / T.RES, 0, T.XW / T.RES], [0, -1 / T.RES, T.F1 / T.RES], [0, 0, 1.0]])
        fit.i2b = g2b @ i2g
    return fit


def _shrink(u):
    return u if USE_BEV_FIX else T._shrink(u)


def _pursuit(delta, th_rel, look):
    """목표 줄(카메라 투영점에서 왼쪽으로 delta m, 도로 방향은 차 정면에서 왼쪽 th_rel)을 따라가는 조향각. 배열 가능."""
    delta, th_rel = np.asarray(delta, float), np.asarray(th_rel, float)
    forward = look * np.cos(th_rel) - delta * np.sin(th_rel) + CAM
    left = look * np.sin(th_rel) + delta * np.cos(th_rel)
    curv = 2.0 * np.abs(left) / np.maximum(forward ** 2 + left ** 2, 1e-9)
    inner = np.maximum(1e-6, 1.0 / np.maximum(curv, 1e-9) - TRACK / 2.0)
    ang = np.minimum(np.arctan(WB / inner), MAX_STEER)
    ang = np.where(forward <= 0.05, MAX_STEER, ang)
    return np.sign(left) * ang


def _limit_heading(steer, th_rel, hmax):
    heading = -np.asarray(th_rel, float)                 # 차가 도로보다 왼쪽을 보면 +
    over = np.abs(heading) - hmax
    back = np.minimum(MAX_STEER, math.radians(4.0) + 2.0 * np.maximum(over, 0.0))
    lim = np.where(heading > 0, np.minimum(steer, -back), np.maximum(steer, back))
    return np.where(over > 0, lim, steer)


def _radius(steer):
    return WB / np.tan(np.maximum(np.abs(steer), 1e-4)) + TRACK / 2.0


class GapCore:
    """켜진 뒤 한 프레임씩. step_obs(obs, pose, scan, goal, now) -> dict(v, steer, state, reason, note)."""

    def __init__(self):
        self.psi = 0.0               # 도로 기준 차 방향 (왼쪽 +)
        self.s = self.lat = 0.0      # 뒤 차축: 켠 뒤 도로 방향으로 간 거리 / 옆으로 움직인 거리 (odom, 짧게만 믿는다)
        self.pose = None
        self.travel = 0.0
        self.tgt = None              # 가고 있는 줄의 옆 위치 (self.lat 과 같은 기준)
        self.steer = 0.0
        self.v = 0.0
        self.t_prev = None
        self.mem = deque(maxlen=MEM_KEEP)   # [(장애물 점 world, 노란 점 world)]
        self.mem_at = -1.0
        # 갈림길
        self.isl = None              # (섬 앞끝 s, 섬 가운데 lat)
        self._isl_est = deque(maxlen=7)
        self.sign_obs = None         # (방향[왼쪽 +], 높이, 받은 시각)
        self.guess = {0: 0, 2: 0}
        self.goal = None
        self.read_since = None
        self.read_done = False
        self.wall_off = False
        self.block_since = None
        self.side_taken = None
        # 막힘
        self.stop_since = None
        self.rev_left = 0.0
        self.resume_hits = 0
        self.resume_target = None
        self.resume_stamp = None
        self.reverse_settle_until = 0.0
        self.rev_count = 0
        self.back_room = 0.4
        self.s_stuck = -9.0
        self.message = ''
        self.give_up = False
        self.sign_txt = ''
        self.clr = (9.0, 9.0)
        self.free3 = (0.0, 0.0, 0.0)

    # ----- 자리 -----
    @property
    def sP(self):
        return self.s + CAM * math.cos(self.psi)

    @property
    def latP(self):
        return self.lat + CAM * math.sin(self.psi)

    def start(self, psi, pose):
        self.psi, self.pose = float(psi), pose
        self.s, self.lat = -CAM * math.cos(psi), -CAM * math.sin(psi)
        self.tgt = 0.0

    def _advance(self, pose, now):
        dt = 0.05 if self.t_prev is None else min(0.3, max(0.0, now - self.t_prev))
        self.t_prev = now
        if pose is not None and self.pose is not None:
            dx, dy, dyaw = pose[0] - self.pose[0], pose[1] - self.pose[1], pose[2] - self.pose[2]
            if math.hypot(dx, dy) > 0.5 or abs(dyaw) > 1.0:          # odom 튐
                dx = dy = dyaw = 0.0
            mid = self.pose[2] + dyaw / 2.0
            ds = dx * math.cos(mid) + dy * math.sin(mid)
        else:                                                         # odom 없음: 명령으로 짐작
            ds = self.v * dt
            dyaw = 0.0 if abs(self.steer) < 1e-3 else math.copysign(ds / float(_radius(self.steer)), self.steer)
        if pose is not None:
            self.pose = pose
        pm = self.psi + dyaw / 2.0
        self.s += ds * math.cos(pm)
        self.lat += ds * math.sin(pm)
        self.psi += dyaw
        self.travel += abs(ds)
        if ds < 0:
            self.rev_left -= abs(ds)
            self.back_room = max(0.0, self.back_room + ds)
        else:
            self.back_room = min(1.0, self.back_room + ds)

    def _camera_heading(self, obs):
        if not obs or obs.get('n', 0) < 300:
            return
        d = -obs['th'] - self.psi
        if abs(d) <= TH_ACCEPT:
            self.psi += TH_GAIN * d

    # ----- 좌표 바꾸기 (차 좌표: 카메라 바닥 투영점이 원점, x 앞, y 왼쪽) -----
    def _road_to_car(self, s, lat):
        a, l = np.asarray(s, float) - self.sP, np.asarray(lat, float) - self.latP
        c, sn = math.cos(self.psi), math.sin(self.psi)
        return a * c + l * sn, -a * sn + l * c

    def _car_to_road(self, x, y):
        c, sn = math.cos(self.psi), math.sin(self.psi)
        return self.sP + x * c - y * sn, self.latP + x * sn + y * c

    # ----- 보이는 것 모으기 -----
    def _lidar_points(self, scan):
        if scan is None:
            return np.zeros(0), np.zeros(0), None
        r = np.asarray(scan[0], dtype=np.float32)
        a = scan[1] + np.arange(r.size) * scan[2]
        ok = np.isfinite(r) & (r >= max(scan[3], 0.12)) & (r <= LIDAR_MAX)
        return r[ok] * np.cos(a[ok]) + LIDAR_AHEAD, r[ok] * np.sin(a[ok]), (r[ok], a[ok])

    def _yellow_points(self, obs):
        """카메라 노란 점 + 화면 아래부터 보이는 노란 선은 차 옆·뒤까지 이어진 것으로 (카메라는 0.42 m 앞부터만 본다)."""
        z = np.zeros(0)
        if not obs:
            return z, z, z, z
        xs, ys = [], []
        rx, ry = z, z
        yx = obs.get('yellow_xy')
        if yx is not None and len(yx):
            lat = -yx[:, 0]
            if not USE_BEV_FIX:                               # 기존 보정은 먼 선이 벌어져 보인다
                far = np.abs(lat) > T.FAR_U
                lat = np.where(far, np.sign(lat) * (T.FAR_U + (np.abs(lat) - T.FAR_U) / T.FAR_SHRINK), lat)
            rx, ry = yx[:, 1], lat
        th = obs['th']
        ystart, short = obs.get('ystart') or {}, obs.get('short', ())
        for u, n, ratio in obs['lines']:
            if ratio < 0.7 or n < 50 or u in short or ystart.get(u, 9.9) > 0.60:
                continue
            us = _shrink(u)
            t = np.arange(-0.40, ystart[u], 0.04)
            xs.append(t * math.cos(th) - us * math.sin(th))
            ys.append(t * math.sin(th) + us * math.cos(th))
        if not xs:
            return rx, ry, z, z
        return rx, ry, np.concatenate(xs), np.concatenate(ys)

    def _lane_hints(self, obs):
        """보이는 선으로 짐작한 차로 가운데들 (지금 자리 기준 옆 거리). 벌점에만 쓴다 (틀려도 안전과 무관)."""
        if not obs:
            return []
        hints = []
        for u, n, ratio in obs['lines']:
            if n < 40:
                continue
            us = _shrink(u)
            if ratio <= 0.3 and abs(us) < 0.9:
                hints += [us - LANE_HALF, us + LANE_HALF]
            elif ratio >= 0.7:
                hints.append(us - math.copysign(LANE_HALF, us))
        return hints

    def _sign_update(self, lidar, now):
        """YOLO 표지판 상자 -> 섬 앞끝 위치. 거리는 그 방향 라이다 값이 상자 높이로 본 거리와 비슷하면 라이다 값을 쓴다."""
        if self.sign_obs is None or now - self.sign_obs[2] > 0.4:
            return
        bearing, h, _ = self.sign_obs
        self.sign_obs = None
        self.sign_txt = f' sg{math.degrees(bearing):+.0f}/{h:.0f}'
        if h < SIGN_SEE_H:
            return
        dist = SIGN_K / h
        if lidar is not None:
            r, a = lidar
            near = r[np.abs(a - bearing) < math.radians(5.0)]
            near = near[(near > 0.65 * dist) & (near < 1.35 * dist)]
            if near.size >= 2:
                dist = float(np.sort(near)[1]) + LIDAR_AHEAD
        if dist > 3.2:
            return
        s, lat = self._car_to_road(dist * math.cos(bearing), dist * math.sin(bearing))
        self._isl_est.append((float(s), float(lat), now))
        est = np.array([e for e in self._isl_est if now - e[2] <= SIGN_NEED_SEC])
        if len(est) < SIGN_NEED_N or np.ptp(est[:, 1]) > 0.30 or np.ptp(est[:, 0]) > 0.60:
            return                                            # 한두 번 보인 것, 자리가 흩어진 것은 표지판이 아니다
        new = (float(np.median(est[:, 0])), float(np.median(est[:, 1])))
        if self.isl is None:
            self.message = f'[3차선G] 표지판(섬) 보임: 앞 {new[0] - self.sP:.2f} m, 옆 {new[1] - self.latP:+.2f} m'
        if self.isl is None or new[0] - self.sP > 0.5:        # 아주 가까우면 상자가 잘려 틀리므로 그 전 값을 쓴다
            self.isl = new

    # ----- 격자 -----
    @staticmethod
    def _grid(x, y):
        occ = np.full((NY, NX), 255, np.uint8)
        if len(x):
            ix = ((np.asarray(x) - GX0) / RES).astype(int)
            iy = ((np.asarray(y) - GY0) / RES).astype(int)
            ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
            occ[iy[ok], ix[ok]] = 0
            if ok.any():
                return cv2.distanceTransform(occ, cv2.DIST_L2, 3) * RES
        return np.full((NY, NX), 9.0, np.float32)

    @staticmethod
    def _look(grid, x, y):
        ix = ((x - GX0) / RES).astype(int)
        iy = ((y - GY0) / RES).astype(int)
        ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
        out = np.full(x.shape, 9.0)
        out[ok] = grid[iy[ok], ix[ok]]
        return out

    def _rollout(self, d, look, hmax):
        """후보 줄 d(배열)마다 경로를 굴려 차체 원들의 가운데를 돌려준다: (K, N_STEP, 원 개수) x, y."""
        k = len(d)
        ax, ay, h = np.full(k, -CAM), np.zeros(k), np.zeros(k)
        st = np.full(k, float(np.clip(self.steer, -ROLL_STEER, ROLL_STEER)))
        n0, n1 = math.sin(self.psi), math.cos(self.psi)       # 도로 왼쪽 방향 (차 좌표)
        cx, cy = np.zeros((k, N_STEP, NB)), np.zeros((k, N_STEP, NB))
        for i in range(N_STEP):
            px, py = ax + CAM * np.cos(h), ay + CAM * np.sin(h)
            th_rel = -(h + self.psi)
            cmd = _limit_heading(_pursuit(d - (px * n0 + py * n1), th_rel, look), th_rel, hmax)
            st += (np.clip(cmd, -ROLL_STEER, ROLL_STEER) - st) * LAG_ALPHA
            dh = STEP / _radius(st) * np.sign(st)
            ax += STEP * np.cos(h + dh / 2.0)
            ay += STEP * np.sin(h + dh / 2.0)
            h += dh
            for j, off in enumerate(BODY_OFF):
                cx[:, i, j], cy[:, i, j] = ax + off * np.cos(h), ay + off * np.sin(h)
        return cx, cy

    def _free(self, g_obs, g_line, cx, cy, m_obs, g0):
        """경로마다 여유를 지키며 갈 수 있는 거리. 지금 이미 여유 안쪽이면(g0 < 0) 더 나빠지지 않는 경로는 허용."""
        g = np.minimum(self._look(g_obs, cx, cy) - BODY_R - m_obs, self._look(g_line, cx, cy) - BODY_R - M_LINE).min(2)
        bad = g < min(0.0, g0 - 0.02)
        first = np.where(bad.any(1), bad.argmax(1), N_STEP)
        return first * STEP

    # ----- 한 프레임 -----
    def step_obs(self, obs, pose, scan, goal, now):
        self._advance(pose, now)
        self._camera_heading(obs)
        lx, ly, lidar = self._lidar_points(scan)
        yrx, yry, yex, yey = self._yellow_points(obs)
        yx, yy = np.concatenate([yrx, yex]), np.concatenate([yry, yey])
        self._sign_update(lidar, now)
        if goal is not None:
            self.goal = goal
        side = 0.0 if self.goal is None else (1.0 if self.goal == 0 else -1.0)
        to_isl = None if self.isl is None else self.isl[0] - self.sP
        isl_rel = None if self.isl is None else self.isl[1] - self.latP

        # 지나친 점 기억 (라이다는 뒤를, 카메라는 0.42 m 안을 못 본다).
        # odom 방향은 꺾을 때 실제보다 훨씬 많이 돈다 (11/17 실차: 실제 28도에 49도) -> odom 좌표가 아니라
        # 카메라로 고친 방향(psi)으로 센 도로 좌표(s, lat)에 기억한다
        if self.travel - self.mem_at >= MEM_STEP:
            self.mem_at = self.travel
            near_o, near_y = lx < 0.9, yrx < 0.9
            self.mem.append((np.stack(self._car_to_road(lx[near_o], ly[near_o]), 1),
                             np.stack(self._car_to_road(yrx[near_y], yry[near_y]), 1)))
        if self.mem:
            for kind in (0, 1):
                w = np.concatenate([m[kind] for m in self.mem])
                if not len(w):
                    continue
                mx_, my_ = self._road_to_car(w[:, 0], w[:, 1])
                keep = mx_ < (0.15 if kind == 0 else 0.45)
                # 차체가 지금 있는 자리의 점은 있을 수 없다 (기억이 밀린 것) -> 버린다
                keep &= ~((mx_ > -CAM - 0.08) & (mx_ < -CAM + 0.28) & (np.abs(my_) < 0.13))
                if kind == 0 and scan is not None:
                    # 지금 라이다가 그 자리 너머를 보고 있으면 거기엔 아무것도 없다 -> 버린다
                    r = np.asarray(scan[0], dtype=np.float32)
                    idx = np.round((np.arctan2(my_, mx_ - LIDAR_AHEAD) - scan[1]) / scan[2]).astype(int)
                    ok = (idx >= 0) & (idx < r.size)
                    live = np.where(ok, r[np.clip(idx, 0, r.size - 1)], 0.0)
                    keep &= ~(ok & (live > np.hypot(mx_ - LIDAR_AHEAD, my_) + 0.15))
                if kind == 0:
                    lx, ly = np.concatenate([lx, mx_[keep]]), np.concatenate([ly, my_[keep]])
                else:
                    yx, yy = np.concatenate([yx, mx_[keep]]), np.concatenate([yy, my_[keep]])

        # 갈림길: 섬 둘레 처리
        reason = ''
        pref = None
        if self.isl is not None:
            if self.goal is None and to_isl > -0.2:
                # 표지판을 읽을 때까지는 섬·표지판 기둥을 못 본 척하고 가운데로 다가간다 (READ_STOP 에서 선다)
                for arr in ('l', 'y'):
                    px, py = (lx, ly) if arr == 'l' else (yx, yy)
                    al, la = self._car_to_road(px, py)
                    if arr == 'l':                                # 라이다는 표지판 기둥 둘레만 (LIMO·상자는 남는다)
                        keep = np.hypot(al - self.isl[0] - 0.05, la - self.isl[1]) > 0.22
                    else:
                        keep = ~((al - self.sP >= to_isl - 0.35) & (np.abs(la - self.isl[1]) < 0.45))
                    if arr == 'l':
                        lx, ly = px[keep], py[keep]
                    else:
                        yx, yy = px[keep], py[keep]
                pref, reason = isl_rel, '가운데에서 표지판 읽는 중'
            elif self.goal is not None:
                s0, l0 = self.isl
                if to_isl > -0.3:
                    # 섬 앞쪽 0.8 m 를 벽으로 (노란 테두리를 못 봐도 섬으로 들어가지 않게, 표지판 위치 오차를 봐서 실제보다 좁게)
                    ss = np.arange(s0, s0 + 0.80, 0.04)
                    vs = [ss, ss, np.full(11, s0)]
                    vl = [np.full(ss.size, l0 + 0.20), np.full(ss.size, l0 - 0.20), np.linspace(l0 - 0.20, l0 + 0.20, 11)]
                    if not self.wall_off:                         # 반대쪽 입구를 막는다
                        wl = np.arange(0.0, 0.95, 0.04)
                        vs.append(np.full(wl.size, s0 + 0.10))
                        vl.append(l0 - side * wl)
                    vx, vy = self._road_to_car(np.concatenate(vs), np.concatenate(vl))
                    yx, yy = np.concatenate([yx, vx]), np.concatenate([yy, vy])     # 노란 선과 같은 여유(M_LINE)로
                if to_isl > -0.3:
                    pref, reason = isl_rel + side * FORK_OFFSET, '표지판 쪽으로'
                else:
                    reason = '갈림길 차로'                        # 들어온 뒤에는 가던 줄 그대로 (노란 선 사이)
        if pref is None:
            pref = 0.0 if self.tgt is None else self.tgt - self.latP

        g_obs, g_line = self._grid(lx, ly), self._grid(yx, yy)
        c0x, c0y = np.array(BODY_OFF) - CAM, np.zeros(NB)
        o0 = float(self._look(g_obs, c0x, c0y).min()) - BODY_R
        l0c = float(self._look(g_line, c0x, c0y).min()) - BODY_R - M_LINE
        self.clr = (o0, l0c + M_LINE)                             # 지금 차체 옆 여유: 장애물 / 노란 선

        # 후진 중에도 아래의 동일한 전진 경로·충돌검사를 수행한다.
        # 센서 단절 시 후진 명령을 유지하거나 빈 지도를 탈출 경로로 믿지 않는다.
        if self.rev_left > 0.0 and scan is None:
            self.resume_hits = 0
            self.v, self.steer = 0.0, 0.0
            return self._out('BLOCKED', '후진 중 라이다 끊김', 0.0, 0.0, to_isl)
        if now < self.reverse_settle_until:
            self.v, self.steer = 0.0, 0.0
            return self._out('REVERSE_STOP', '후진 종료 후 방향 전환 대기', 0.0, 0.0, to_isl)
        prior_target = self.tgt

        # 후보 줄마다 갈 수 있는 거리
        prev = pref if self.tgt is None else self.tgt - self.latP
        d = np.append(CAND, [pref, prev])          # 끝에서 둘째 = 가고 싶은 줄, 맨 끝 = 지금 가던 줄
        hints = self._lane_hints(obs)
        tight = False
        look = LOOK_KEEP if abs(pref) < 0.08 else LOOK_CHANGE
        cx, cy = self._rollout(d, look, HMAX)
        free = self._free(g_obs, g_line, cx, cy, M_OBS_COMFORT, min(o0 - M_OBS_COMFORT, l0c))
        level = '보통'
        if free.max() < MIN_GO:
            free = self._free(g_obs, g_line, cx, cy, M_OBS, min(o0 - M_OBS, l0c))
            level = '좁게'
            if free.max() < MIN_GO:
                cx2, cy2 = self._rollout(d, LOOK_TIGHT, HMAX_TIGHT)
                free2 = self._free(g_obs, g_line, cx2, cy2, M_OBS, min(o0 - M_OBS, l0c))
                if free2.max() > free.max() + 0.10:
                    free, tight, level = free2, True, '급하게'
        best = float(free.max())
        ok = free >= min(N_STEP * STEP - 0.05, best - TIE)
        cost = np.abs(d - pref)
        if hints:
            cost = cost + HINT_COST * (np.min(np.abs(d[:, None] - np.array(hints)[None, :]), 1) > 0.07)
        bias = side if side else (1.0 if DEFAULT_SIDE == 0 else -1.0)
        cost = cost + SIDE_BIAS * (d * bias < -0.05)
        cost[-2] = 0.0                                            # 가고 싶은 줄(가던 줄 또는 표지판 쪽 줄)이 갈 만하면 그 줄
        cost[-1] = abs(prev - pref)
        k = int(np.argmin(np.where(ok, cost, 1e9)))
        if ok[-1] and cost[-1] <= cost[k] + KEEP_BONUS:           # 가던 줄이 아직 갈 만하면 조금 손해여도 안 바꾼다 (프레임마다 왔다 갔다 방지)
            k = len(d) - 1
        dk, fk = float(d[k]), float(free[k])
        self.tgt = self.latP + dk
        self.free3 = (float(free[10:15].max()), float(free[5:10].max()), float(free[0:5].max()))

        # 조향·속도
        th_rel = -self.psi
        hmax = HMAX_TIGHT if tight else HMAX
        look = LOOK_TIGHT if tight else (LOOK_KEEP if abs(dk) < 0.08 else LOOK_CHANGE)
        cmd = float(_limit_heading(_pursuit(dk, th_rel, look), th_rel, hmax))
        self.steer += 0.6 * (cmd - self.steer)
        v = V_KEEP if (abs(dk) < 0.10 and fk > 1.2) else V_CHANGE
        if tight or fk < 0.8:
            v = V_SLOW
        state = 'KEEP' if abs(dk) < 0.10 else 'CHANGE'
        # 정상 전진과 후진 중 재탐색이 같은 실제 조향 전방 검사를 공유한다.
        stop = ''
        if fk < STOP_FREE:
            stop = f'갈 길 없음 ({level}, 차체 옆 여유 장애물 {min(o0, 9):.2f} / 노란 선 {min(l0c + M_LINE, 9):.2f} m)'
        else:
            gx, gy = self._arc(self.steer)
            g = (self._look(g_obs, gx, gy) - BODY_R).min(1)
            if (g < min(0.02, o0 - 0.02)).any():
                stop = f'정면 {GUARD_DIST:.2f} m 안에 장애물'
        if self.rev_left > 0.0:
            target = self.tgt
            stable = (not stop and fk >= REVERSE_RESUME_FREE
                      and (self.resume_target is None or abs(target-self.resume_target) <= .15))
            continuous = (self.resume_stamp is not None and 0 < now-self.resume_stamp <= .25)
            self.resume_hits = (self.resume_hits+1 if continuous else 1) if stable else 0
            self.resume_target, self.resume_stamp = target, now
            bx = np.concatenate([np.array(BODY_OFF) - CAM - b for b in (0.06, 0.12, 0.18)])
            behind = float(self._look(g_obs, bx, np.zeros(bx.size)).min()) - BODY_R
            if self.resume_hits >= REVERSE_RESUME_FRAMES or behind < .03 or self.back_room <= .02:
                why = '전진 경로 확보' if self.resume_hits >= REVERSE_RESUME_FRAMES else '후방 여유 부족'
                self.rev_left = 0.0
                self.resume_hits = 0
                self.reverse_settle_until = now + .15
                self.v, self.steer = 0.0, 0.0
                self.message = f'[3차선G] {why} -> 후진 종료, 정지 후 새 프레임으로 전진 재평가'
                return self._out('REVERSE_STOP', why, dk, fk, to_isl)
            self.tgt = prior_target  # 시험한 전진 목표를 후진 중의 확정 목표로 저장하지 않음
            self.v, self.steer = -V_REVERSE, 0.0
            return self._out('REVERSE', f'후진 {self.rev_left:.2f} m 남음 / 전진 후보 {fk:.2f} m', dk, fk, to_isl)
        # 섬 앞에서 서서 읽기
        if self.isl is not None and self.goal is None and to_isl <= READ_STOP:
            if self.read_since is None:
                self.read_since = now
            if now - self.read_since >= READ_WAIT_SEC:
                a, b = self.guess[0], self.guess[2]
                self.goal = 0 if a > b else (2 if b > a else DEFAULT_SIDE)
                self.message = (f"[3차선G] 표지판 확정 못 함 -> {'A(왼쪽)' if self.goal == 0 else 'B(오른쪽)'} "
                                f"(읽힌 횟수 A {a} / B {b})")
            self.v = 0.0
            return self._out('READ', f'섬 {to_isl:.2f} m 앞에서 서서 표지판 읽는 중', dk, fk, to_isl)
        if stop:
            self.v = 0.0
            if self.stop_since is None:
                self.stop_since = now
            if to_isl is not None and to_isl < -HANDOVER_IN:
                self.give_up = True                               # 갈림길 차로 안에서 막힘(꺾인 노란 선) -> 후진하지 않고 기본 주행에 넘긴다
            if self.goal is not None and self.isl is not None and not self.wall_off and -0.3 < to_isl < 2.0:
                if self.block_since is None:
                    self.block_since = now
                if now - self.block_since >= FORK_WAIT_SEC:
                    self.wall_off = True
                    self.message = '[3차선G] 표지판 쪽이 계속 막힘 -> 반대쪽도 허용'
            if now - self.stop_since >= STUCK_SEC and self.rev_count < REVERSE_MAX and self.back_room > 0.15:
                if self.sP > self.s_stuck + 0.8:
                    self.rev_count = 0
                self.s_stuck = max(self.s_stuck, self.sP)
                self.rev_count += 1
                self.rev_left = min(REVERSE_M, self.back_room)
                self.resume_hits = 0
                self.resume_target = self.resume_stamp = None
                self.stop_since = None
                self.tgt = None
                self.message = f'[3차선G] 막힘 -> 후진 {self.rev_left:.2f} m ({self.rev_count}번째)'
            return self._out('BLOCKED', stop, dk, fk, to_isl)
        self.stop_since = self.block_since = None
        self.v = v
        if self.isl is not None and self.side_taken is None and to_isl < -0.5:
            self.side_taken = 0 if self.latP > self.isl[1] else 2
        return self._out(state, f'{reason} {level}'.strip(), dk, fk, to_isl)

    def _arc(self, steer):
        """지금 조향대로 GUARD_DIST 까지 갈 때 차체 원 가운데들: (걸음, 원 개수)."""
        n = int(GUARD_DIST / STEP)
        ax, ay, h = -CAM, 0.0, 0.0
        dh = 0.0 if abs(steer) < 1e-3 else math.copysign(STEP / float(_radius(steer)), steer)
        gx, gy = np.zeros((n, NB)), np.zeros((n, NB))
        for i in range(n):
            ax, ay, h = ax + STEP * math.cos(h + dh / 2), ay + STEP * math.sin(h + dh / 2), h + dh
            for j, off in enumerate(BODY_OFF):
                gx[i, j], gy[i, j] = ax + off * math.cos(h), ay + off * math.sin(h)
        return gx, gy

    def _out(self, state, why, dk, fk, to_isl):
        g = '-' if self.goal is None else ('A' if self.goal == 0 else 'B')
        f3 = self.free3
        isl = '' if to_isl is None else f' 섬 {to_isl:.1f} m'
        reason = (f'3차선G {state} 목표 {dk:+.2f} m, 갈 수 있는 거리 {fk:.1f} (좌 {f3[0]:.1f} 중 {f3[1]:.1f} 우 {f3[2]:.1f}) '
                  f'표지판 {g}{isl}' + (f' | {why}' if why else ''))
        note = (f'L3G {state} d{dk:+.2f} free{fk:.1f} f3{f3[0]:.1f}/{f3[1]:.1f}/{f3[2]:.1f} psi{math.degrees(self.psi):+.0f} '
                f's{self.sP:.2f} lat{self.latP:+.2f} goal{g}'
                + ('' if to_isl is None else f' isl{to_isl:.2f}/{self.isl[1] - self.latP:+.2f}')
                + f' rev{self.rev_count} clr{min(self.clr[0], 9):.2f}/{min(self.clr[1], 9):.2f}' + self.sign_txt)
        self.sign_txt = ''
        return dict(v=self.v, steer=self.steer, state='LANE3', reason=reason, note=note)


class GapSupervisor:
    """simple_lane.py 가 three_lane.Lane3Supervisor 대신 쓴다 (같은 함수 이름). off -> armed -> on -> done."""

    def __init__(self, forced=False, goal=None):
        self.fitter = make_fitter()
        self.core = GapCore()
        self.forced = forced
        self.mode = 'armed' if forced else 'off'
        self.fixed_goal = goal
        self.vote = T.SignVote()
        self.arm_travel = 0.0
        self._pose = None
        self._white = self._both = 0
        self._fork_end = 0.0
        self.message = ''
        self.side_taken = None
        self.route_complete = False
        self._fork_observed = False

    def _reset(self):
        self.route_complete = self._fork_observed = False
        self.core = GapCore()
        self.vote.reset()
        self.arm_travel = 0.0
        self._white = self._both = 0

    def set_mission(self, name):
        if self.forced:
            return
        if name == 'M7':
            if self.mode == 'off':
                self._reset()
                self.mode = 'armed'
        elif name != 'M2':
            self.mode = 'off'

    def see_signs(self, detections, now=None):
        if self.mode not in ('armed', 'on'):
            return
        signs = []
        for det in detections:
            label = str(det.get('label', '')).lower().replace('sign_', '').strip()
            if label in ('a', 'b') and len(det.get('bbox', ())) == 4:
                signs.append((det.get('confidence', 0.0), label, [float(v) for v in det['bbox']]))
        if not signs:
            return
        _, label, box = max(signs)
        height = box[3] - box[1]
        frac = (box[0] + box[2]) / 2.0 / YOLO_IMG_W
        self.core.sign_obs = (-math.atan((frac - 0.5) * BEAR_K), height, time.monotonic() if now is None else now)
        if self.fixed_goal is not None:
            return
        if height >= GUESS_MIN_H:
            self.core.guess[0 if label == 'a' else 2] += 1
        decided = self.vote.add(label, height)
        if decided is not None:
            self.core.goal = decided
            self.message = f"[3차선G] 표지판 {'A(왼쪽)' if decided == 0 else 'B(오른쪽)'} 확정 (상자 높이 {height:.0f} px)"

    @property
    def latch_off(self):
        """True 면 기본 주행의 코너 고정 조향을 끈다 (3차선 입구 사선, 섬 뒤 합류의 꺾인 선을 코너로 보지 않게)."""
        return self.mode == 'fork' or (self.mode in ('armed', 'on') and (self.forced or self.arm_travel >= ARM_AFTER_M))

    def step(self, image, pose, scan, both_yellow, tune=None):
        if self.mode in ('armed', 'on') and (self.forced or self.arm_travel >= ARM_AFTER_M - 0.05):
            prior = -self.core.psi if self.mode == 'on' else None
            obs = self.fitter.fit(image, prior, tune)
        else:
            obs = None
        return self.step_obs(obs, pose, scan, both_yellow, time.monotonic())

    def step_obs(self, obs, pose, scan, both_yellow=False, now=0.0):
        """반환: None (기본 주행 명령을 그대로 씀) 또는 dict(v, steer, state, reason, note)."""
        if pose is not None and self._pose is not None:
            step = math.hypot(pose[0] - self._pose[0], pose[1] - self._pose[1])
            if step < 0.5:
                self.arm_travel += step
        if pose is not None:
            self._pose = pose
        if self.mode in ('off', 'done'):
            return None
        if self.mode == 'fork':                 # 갈림길 차로 ~ 합류: 기본 주행이 몰고, latch 만 꺼 둔다
            self.core._advance(pose, now)
            if self.core.sP >= self._fork_end:
                self.route_complete = self._fork_observed
                self.mode, self.message = 'done', '[3차선G] 합류 끝 -> 기본 주행 (코너 조향 다시 켬)'
            return None
        if self.mode == 'armed':
            if not self.forced:
                if self.arm_travel < ARM_AFTER_M:
                    return None
                white = bool(obs) and any(r <= 0.3 and n >= 40 and abs(u) < 0.6 for u, n, r in obs['lines'])
                self._white = self._white + 1 if white else 0
                if self._white < ENTRY_FRAMES and self.arm_travel < ARM_AFTER_M + ENTRY_WAIT_M:
                    return None
            elif scan is None:
                return dict(v=0.0, steer=0.0, state='LANE3', reason='3차선G: 라이다(/scan) 기다리는 중', note='L3G wait scan')
            self.mode = 'on'
            self.core.start(-obs['th'] if obs else 0.0, pose)
            self.message = f"[3차선G] 시작 (도로 방향 {math.degrees(-self.core.psi):+.0f}도)"
        core = self.core
        if scan is None and core.v > 0.0:
            core.v = 0.0
            return dict(v=0.0, steer=core.steer, state='LANE3', reason='3차선G: 라이다가 끊김 -> 정지', note='L3G no scan')
        out = core.step_obs(obs, pose, scan, self.fixed_goal, now)
        if core.message and not self.message:
            self.message, core.message = core.message, ''
        self.side_taken = core.side_taken
        done = ''
        if core.isl is not None:
            past = core.sP - core.isl[0]
            self._both = self._both + 1 if (both_yellow and past >= HANDOVER_IN and core.rev_left <= 0.0) else 0
            if self._both >= 3:
                done = '갈림길 차로 진입 (양쪽 노란 선)'
            elif past >= HANDOVER_MAX:
                done = '갈림길 차로 진입'
            elif core.give_up:
                done = '갈림길 차로 안에서 길이 안 나옴'
            self._fork_end = core.isl[0] + EXIT_AFTER
        else:
            self._both = self._both + 1 if (both_yellow and core.travel >= NO_SIGN_EXIT_M) else 0
            if self._both >= 8:
                done = '양쪽 노란 선이 계속 보임 (표지판 상자는 못 봄)'
            self._fork_end = core.sP + NO_SIGN_FORK_M
        if core.travel >= MAX_ON_M:
            done, self._fork_end = '최대 거리', core.sP
        if done:
            self._fork_observed = (core.isl is not None and not core.give_up
                                   and core.sP-core.isl[0] >= HANDOVER_IN
                                   and core.travel < MAX_ON_M)
            self.mode = 'fork'
            side = core.side_taken if core.isl is None else (0 if core.latP > core.isl[1] else 2)
            self.side_taken = side
            self.message = (f"[3차선G] {done}: {({0: 'A(왼쪽)', 2: 'B(오른쪽)', None: '?'})[side]} -> 기본 주행에 넘김 "
                            f'(합류 끝까지 코너 조향 끔)')
            return None
        return out


def main():
    try:
        import simple_lane as SL                      # 같은 폴더의 simple_lane.py (고치지 않는다)
    except ImportError:
        from jetracer_ros2 import simple_lane as SL
    SL._three_lane = types.SimpleNamespace(Lane3Supervisor=GapSupervisor)
    if 'l3simple' in sys.argv:
        sys.argv.remove('l3simple')
    print('[3차선G] lane3_gap.py: 3차선 + 갈림길을 위치 안 세는 방식으로 돌립니다 (simple_lane 은 그대로)')
    SL.main()


if __name__ == '__main__':
    main()
