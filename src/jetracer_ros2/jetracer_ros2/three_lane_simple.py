"""3차선(M7) 단순 버전: 차로 위치는 odom 으로, 장애물은 라이다 지도로, 카메라는 도로 방향과 노란 선 벽에만 쓴다.

three_lane.py 와 다른 점
  - 카메라로 '몇 번 차로인지'를 매 프레임 맞추지 않는다 (실차 실패 대부분이 여기서 나왔다).
    진입할 때 한 번만 위치를 잡고 (도면상 진입 차로 = 가운데 차로), 그 뒤로는 odom 으로 옆 위치를 센다.
  - 카메라는 도로 방향(th)만 천천히 고친다. 모든 선이 도로와 나란해서, 어느 선이 어느 차로 선인지 몰라도 방향은 맞는다.
  - 라이다 점을 도로 좌표 지도(5 cm 칸)에 쌓는다. 장애물(정차 LIMO)은 움직이지 않으므로, 지나쳐서 라이다에
    안 보이는 옆·뒤 장애물도 기억해 차로를 옮길 때 쓴다.
  - 판단 규칙이 짧다: 내 차로 앞이 막히면 옆 차로 중 더 멀리 빈 쪽으로 한 칸. 갈 데가 없으면 서고, 오래 서 있으면 조금 후진.

simple_lane.py 에서 Lane3Supervisor 대신 쓴다 (실행 인자 l3simple). 같은 함수 이름(set_mission, see_signs, step ...)을 가진다.

좌표 (three_lane.py 와 같음)
  s  : 도로 방향 거리 [m], M7 흰 시작선이 0 (진입할 때 흰 점선 시작까지의 거리로 맞춘다)
  c  : 도로 가운데에서 카메라 바닥 투영점까지 옆 거리, 왼쪽 + [m]
  th : 도로 방향이 차 정면에서 왼쪽으로 몇 rad (차가 도로보다 오른쪽을 보면 +). psi = -th 는 도로 기준 차 방향
  차로 0 = 왼쪽(A), 1 = 가운데, 2 = 오른쪽(B)
"""

import math
import time
from typing import Optional

import numpy as np

try:
    from jetracer_ros2 import three_lane as T
except ImportError:                     # 파일만 복사해서 python3 simple_lane.py 로 돌릴 때
    import three_lane as T

# ---------------------------------------------------------------------------
# 실차에서 맞출 값 (줄자로 재거나 기록을 보고 조정)
# ---------------------------------------------------------------------------
V_KEEP = 0.22            # 차로 따라갈 때 속도 [m/s]
V_CHANGE = 0.18          # 차로 옮길 때 속도
V_SLOW = 0.10            # 앞이 막혔는데 옮길 데가 없을 때 다가가는 속도
BLOCK_DIST = 1.30        # 내 차로 앞 장애물이 이 안이면(라이다 기준) 옮길 차로를 찾는다
MIN_START = 0.55         # 앞 장애물이 이보다 가까우면 옮기기 시작하지 않는다 (꺾다가 앞 장애물 모서리에 닿는다)
GAIN = 0.45              # 옮길 차로는 지금 차로보다 이만큼 더 멀리 비어 있어야 한다
STOP_DIST = 0.35         # 갈 데가 없으면 앞 장애물 이 거리에서 정지
# 도면 (2026Track, 도로 방향 거리). s 는 M7 흰 시작선을 0 으로 잰다
START_TO_DASH = 0.30     # M7 시작선 -> 흰 점선 시작
START_TO_ISLAND = 4.36   # M7 시작선 -> 섬 앞끝 (카메라로 섬을 보면 그 값을 쓴다)
# 섬 뒤 합류 (도면): 섬은 앞끝에서 1.74 m 에서 끝나고, 바깥 선이 앞끝 + 1.79 ~ 2.29 m 에서 45도로 꺾여
# 차로 하나(가운데 차로 자리, 폭 0.53)로 합쳐진다. 기본 주행은 이 45도를 90도 코너로 보고 60도를 돌아 길을 벗어났다 (11/15 실차)
ISLAND_LEN = 1.74
MERGE_TURN = 1.33        # 섬 앞끝 + 이 거리부터 가운데(합류 차로)로 옮긴다 (시뮬레이션: 늦으면 꺾인 바깥 선, 이르면 섬 끝에 닿음)
MERGE_LOOK = 0.50        # 합류할 때 목표점 거리
MERGE_HEADING = math.radians(40.0)
MERGE_END = 2.29         # 섬 앞끝 + 이 거리에서 합류 끝
CHANGE_FROM_S = 0.20     # 이 위치(카메라 투영점) 전에는 차로를 옮기지 않는다. 시작선까지는 1차선이고, 도로는 시작선 뒤
                         # 0.05~0.57 m 에서 45도로 벌어진다 (도면). 일찍 옮기면 1차선 노란 선을 밟는다 (11/15 실차)
LEAVE_DIST = 1.45        # 가운데 차로에서 섬이 이 안으로 오면 옆 차로로 빠진다
READ_SLOW_BELOW = 2.0    # 표지판을 아직 모르고 가운데 차로에 있으면, 섬이 이 안일 때 천천히 (표지판은 약 1.7 m 안에서만 맞게 읽힌다)
V_READ = 0.08
TRANSIT_NEED = 1.00      # 두 칸 건너기: 가운데 차로가 이만큼 비어 있어야 한다
CENTER_NEED = 2.00       # 바깥 차로에서 가운데로 들어가려면 가운데 차로가 이만큼 비어 있어야 한다 (섬 포함)
DEFAULT_SIDE = 2         # 표지판을 끝까지 못 읽었을 때 갈 쪽 (2 = 오른쪽 = B)
CAM_FROM_ODOM = 0.184    # 카메라 바닥 투영점이 odom 기준점보다 앞에 있는 거리 [m] (odom 기준점 = 뒤 차축으로 가정)
LIDAR_AHEAD = 0.0        # 라이다가 카메라 바닥 투영점보다 앞에 있는 거리 [m]. three_lane.py 는 0 으로 가정 -> 실측해서 넣기

# ---------------------------------------------------------------------------
# 거의 안 바꾸는 값
# ---------------------------------------------------------------------------
LANE_CENTER = T.LANE_CENTER          # (0.535, 0.0, -0.535)
LANE_BAND = 0.20                     # 차로 가운데에서 이 안의 점은 그 차로의 장애물 (LIMO 반폭 0.11 + 놓인 위치·위치 추정 오차)
ROAD_HALF = 0.83
CELL = 0.05                          # 지도 칸 [m]
OCC_HITS = 2                         # 이만큼 찍힌 칸만 장애물 (라이다 잡음 1번은 무시)
SCAN_MIN, SCAN_MAX = 0.12, 3.3       # 쓰는 라이다 거리 (멀리 봐야 다음 줄 장애물까지 보고 차로를 고른다)
MAP_MAX = 2.0                        # 이 안의 점만 지도에 남긴다. 먼 점은 방향 오차로 옆으로 번져서 이번 프레임에만 쓴다
CAR_REAR = 0.25 + 0.0                # 카메라 투영점에서 차 뒤끝까지 (뒤 차축 0.184 + 뒤 0.06)
SIDE_BACK = 0.10                     # 옆 차로 확인: 차 뒤끝보다 이만큼 더 뒤부터
SIDE_FRONT = 0.30                    # 옆 차로 확인: 내 앞 장애물보다 이만큼 더 앞까지
SIDE_PASSED = 0.10                   # 옆 차로 장애물의 앞끝이 카메라 투영점 앞 이 안이면 '옆에 있음' (차 옆면과의 간격만 본다)
SIDE_GAP = 0.15                      # 옆에 있는 장애물과 차 옆면 사이 최소 간격
CAR_HALF_WIDTH = 0.11
NO_OBSTACLE = 3.0                    # 빈 거리가 이 이상이면 '아무것도 없음'으로 본다
TIE = 0.30                           # 두 차로 빈 거리가 이 안이면 표지판 쪽(모르면 기본 쪽)을 고른다
LOOKAHEAD_KEEP = 0.55
LOOKAHEAD_CHANGE = 0.40
HEADING_MAX = math.radians(40.0)     # 옮길 때 도로와 이 이상 틀지 않는다
ARRIVE_TOL = 0.08                    # 목표 차로 가운데에서 이 안 + 도로와 나란하면 옮기기 끝
ABORT_BACK = 0.15                    # 옮기기 시작 직후 그 차로가 막히면, 아직 이 안이면 원래 차로로 돌아간다
PREF_FRAMES = 5                      # '가고 싶은 차로'로 옮기는 조건이 연속 이만큼 맞아야 옮긴다
TH_GAIN = 0.15                       # 카메라 도로 방향으로 고치는 비율 (한 프레임)
TH_ACCEPT = math.radians(6.0)        # odom 으로 예상한 방향과 이 이상 다르면 카메라 방향을 안 쓴다
TH_MIN_POINTS = 300
LINE_GAIN = 0.25                     # 차로 선으로 옆 위치를 고치는 비율 (한 프레임)
LINE_GATE = 0.20                     # 선으로 구한 위치가 지금 위치와 이 안일 때만 쓴다 (선 간격 0.27 의 절반 남짓: 차로를 통째로 옮기지 않는다)
LINE_TH_MAX = math.radians(45.0)     # 도로와 이보다 크게 틀어져 있으면 선으로 고치지 않는다
GUARD_DIST = 0.30                    # 정면 통로에 물체가 이 안이면 무조건 정지
STUCK_SEC = 1.5                      # 이만큼 서 있으면 후진
REVERSE_M = 0.40                     # 한 번에 후진하는 거리
REVERSE_MAX_COUNT = 3
V_REVERSE = 0.10
ISLAND_CONFIRM = 3                   # 카메라로 섬을 이만큼 연속 비슷한 위치에서 봐야 믿는다
ISLAND_MIN_VIEW = 0.55               # 섬 앞끝이 이보다 가까우면 카메라 값을 새로 받지 않는다 (카메라 사각)
ARM_AFTER_M = 1.4                    # M7 시작 후 이만큼 간 뒤부터 진입을 찾는다 (그 전은 코너)
ENTRY_FRAMES = 3
ENTRY_TH_MAX = math.radians(8.0)     # 진입 방향도 이 안으로 자른다 (기본 주행이 1차선을 곧게 따라오는 중)
ENTRY_C_MAX = 0.10                   # 진입 차로는 가운데 차로 (도면) -> 카메라가 다른 값을 줘도 이 안으로 자른다
EXIT_FRAMES = 4
MAX_ON_M = 9.0                       # 이만큼 넘게 가면 끝낸다
LOST_FRAMES = 30


class RoadMap:
    """라이다 점을 도로 좌표 칸에 센다. 정차 LIMO 는 움직이지 않으므로 지운 적 없이 쌓는다."""

    def __init__(self):
        self.hits = {}
        self.virtual = []          # [(s_from, s_end, lat_lo, lat_hi)] 섬처럼 라이다에 안 잡히는 막힌 곳
        self.transient = (np.zeros(0), np.zeros(0))    # 이번 프레임의 먼 점 (지도에 안 남김)

    def add(self, s, lat):
        for a, b in zip(np.floor(s / CELL).astype(int), np.floor(lat / CELL).astype(int)):
            k = (int(a), int(b))
            self.hits[k] = self.hits.get(k, 0) + 1

    def occupied(self):
        """(s, lat) 배열: 장애물로 보는 칸의 가운데."""
        cells = [k for k, n in self.hits.items() if n >= OCC_HITS]
        if not cells:
            return np.zeros(0), np.zeros(0)
        arr = np.asarray(cells, dtype=float)
        return (arr[:, 0] + 0.5) * CELL, (arr[:, 1] + 0.5) * CELL

    def first_in_lane(self, lane, s_from, s_to=1e9):
        """차로 lane 에서 s_from ~ s_to 사이 가장 가까운 장애물의 s. 없으면 None."""
        s, lat = self._s, self._lat
        center = LANE_CENTER[lane]
        best = None
        if s.size:
            sel = (np.abs(lat - center) <= LANE_BAND) & (s >= s_from) & (s <= s_to)
            if sel.any():
                best = float(s[sel].min())
        for v_from, v_end, lo, hi in self.virtual:
            if lo <= center <= hi and v_from <= s_to and v_end >= s_from:
                v = max(v_from, s_from)
                best = v if best is None else min(best, v)
        return best

    def refresh(self):
        s, lat = self.occupied()
        self._s, self._lat = np.concatenate([s, self.transient[0]]), np.concatenate([lat, self.transient[1]])


class SimpleThreeLane:
    """진입 뒤 한 프레임씩 호출. step_obs(obs, pose, scan, goal, now)."""

    def __init__(self):
        self.map = RoadMap()
        self.map.refresh()
        self.s = 0.0                 # 카메라 투영점의 도로 방향 위치
        self.c = 0.0
        self.th = 0.0
        self.pose = None
        self.travel = 0.0
        self.lane = 1                # 지금 따라가는(또는 옮겨 가는) 목표 차로
        self.changing = False
        self.change_from = 1
        self.state = 'KEEP'
        self.reason = ''
        self.island_s = None         # 카메라로 확인한 섬 앞끝 위치
        self._isl_seen = []
        self._pref_count = 0
        self.stop_since = None
        self.reversing = False
        self.reverse_left = 0.0
        self.reverse_count = 0
        self.lane_run = 0.6          # 지금 차로에서 곧게 온 거리 (후진해도 되는 거리). 진입 전 직선을 조금 인정
        self.ahead = [NO_OBSTACLE] * 3

    # ----- 위치 -----
    def start(self, c, th, pose, s=START_TO_DASH):
        self.c, self.th, self.pose, self.s = float(c), float(th), pose, float(s)
        self.lane = int(np.argmin([abs(self.c - x) for x in LANE_CENTER]))

    def _advance(self, pose):
        if pose is None:
            return
        if self.pose is None:
            self.pose = pose
            return
        ds = math.hypot(pose[0] - self.pose[0], pose[1] - self.pose[1])
        forward = ((pose[0]-self.pose[0])*math.cos(self.pose[2])
                   + (pose[1]-self.pose[1])*math.sin(self.pose[2]))
        dyaw = pose[2] - self.pose[2]
        self.pose = pose
        if ds > 0.5 or abs(dyaw) > 1.0:           # odom 튐
            return
        if forward < 0:                            # actual motion, including coast after a stop
            ds = -ds
            self.reverse_left -= abs(ds)
        # odom 기준점(뒤 차축)을 옮긴 뒤 카메라 투영점을 다시 구한다. 차가 틀어지면 카메라는 뒤 차축보다 옆으로 더 움직인다
        psi0 = -self.th
        psi_mid = psi0 + dyaw / 2.0
        so = self.s - CAM_FROM_ODOM * math.cos(psi0) + ds * math.cos(psi_mid)
        yo = self.c - CAM_FROM_ODOM * math.sin(psi0) + ds * math.sin(psi_mid)
        self.th -= dyaw
        psi1 = -self.th
        self.s = so + CAM_FROM_ODOM * math.cos(psi1)
        self.c = yo + CAM_FROM_ODOM * math.sin(psi1)
        self.travel += ds
        self.lane_run = max(0.0, self.lane_run + ds)

    def _camera_heading(self, obs):
        if not obs or obs.get('n', 0) < TH_MIN_POINTS:
            return
        d = obs['th'] - self.th
        if abs(d) <= TH_ACCEPT:
            self.th += TH_GAIN * d

    def _camera_lines(self, obs):
        """보이는 차로 선(흰 점선 +-0.27, 바깥 노란 선 +-0.80, 섬 옆이면 섬 테두리 +-0.27)으로 옆 위치를 조금씩 고친다.
        선마다 '지금 위치에서 가장 가까운' 자리에만 맞추고, 그 자리가 LINE_GATE 넘게 떨어지면 버린다
        -> 어느 선이 어느 차로 선인지 헷갈려 한 차로씩 튀는 일(three_lane.py 의 실패)은 생기지 않는다.
        odom 방향은 차로를 바꾸며 크게 틀 때 수십 cm 씩 밀린다 (11/15 실차: 오른쪽 -> 가운데로 옮기는 동안 0.5 m)."""
        if not obs or abs(self.th) > LINE_TH_MAX:
            return False
        short = obs.get('short', ())
        ystart = obs.get('ystart') or {}
        beside_island = self.s > self.island_at - 0.30
        cands = []
        for u, n, ratio in obs['lines']:
            if n < 40 or abs(u) > 1.0 or u in short:
                continue
            if ratio <= 0.3:
                spots = T.WHITE_POS
            elif ratio >= 0.7:
                if ystart.get(u, 0.0) > 0.60:      # 앞에서 시작하는 노란 선 (섬 테두리 앞끝)
                    continue
                spots = T.YELLOW_POS + (T.WHITE_POS if beside_island else ())
            else:
                continue
            us = T._shrink(u)
            best = min((p - us for p in spots), key=lambda cc: abs(cc - self.c))
            if abs(best - self.c) <= LINE_GATE:
                cands.append((best, min(n, 300) * (1.0 if abs(u) <= 0.45 else 0.4)))
        if not cands:
            return False
        cands.sort()
        half, acc = sum(w for _, w in cands) / 2.0, 0.0
        for cc, w in cands:                        # 가중 중앙값 (선 하나가 틀려도 끌려가지 않게)
            acc += w
            if acc >= half:
                break
        self.c += LINE_GAIN * (cc - self.c)
        return True

    def _camera_island(self, obs):
        if not obs or obs.get('yellow_xy') is None:
            return
        d = T.island_distance(obs['yellow_xy'], self.c, self.th)
        if d is None or d < ISLAND_MIN_VIEW:
            return
        pos = self.s + d
        self._isl_seen = [p for p in self._isl_seen[-(ISLAND_CONFIRM - 1):] if abs(p - pos) < 0.30] + [pos]
        if len(self._isl_seen) >= ISLAND_CONFIRM:
            est = float(np.median(self._isl_seen))
            self.island_s = est if self.island_s is None else min(self.island_s, est)

    @property
    def island_at(self):
        """섬 앞끝 s. 카메라로 봤으면 그 값, 아니면 도면 값 (카메라 값이 도면보다 0.5 m 넘게 멀면 도면 값)."""
        if self.island_s is None:
            return START_TO_ISLAND
        return min(self.island_s, START_TO_ISLAND + 0.5)

    def _scan_to_map(self, scan):
        r = np.asarray(scan[0], dtype=np.float32)
        a = scan[1] + np.arange(r.size) * scan[2]
        ok = np.isfinite(r) & (r >= max(scan[3], SCAN_MIN)) & (r <= SCAN_MAX)
        x, y = r[ok] * np.cos(a[ok]) + LIDAR_AHEAD, r[ok] * np.sin(a[ok])
        psi = -self.th
        s = self.s + x * math.cos(psi) - y * math.sin(psi)
        lat = self.c + x * math.sin(psi) + y * math.cos(psi)
        keep = np.abs(lat) <= ROAD_HALF
        # 섬 위(갈림길 표지판 받침대)는 지도에 넣지 않는다. 섬은 따로 막아 둔다. 받침대 끝이 옆 차로 칸에 걸려
        # 옆 차로가 막혔다고 보고 섬 앞에서 서고 후진을 되풀이했다 (11/16 실차: 표지판이 섬 앞끝에 있음)
        isl0 = self.island_at
        keep &= ~((s > isl0 - 0.30) & (s < isl0 + ISLAND_LEN + 0.10) & (np.abs(lat) < T.ISLAND_HALF))
        near = keep & (r[ok] <= MAP_MAX)
        self.map.add(s[near], lat[near])
        far = keep & ~near
        self.map.transient = (s[far], lat[far])

    # ----- 판단 -----
    def _free(self, lane):
        """차로 lane 에서 카메라 투영점부터 앞 장애물까지 거리 (없으면 NO_OBSTACLE)."""
        hit = self.map.first_in_lane(lane, self.s + 0.02)
        return NO_OBSTACLE if hit is None else min(NO_OBSTACLE, hit - self.s)

    def _side_clear(self, lane, upto):
        """차로 lane 으로 지금 꺾어도 되는지 (지도 기준: 지나친 장애물도 기억).
        앞(카메라 투영점 + SIDE_PASSED ~ upto)에는 아무것도 없어야 한다. 옆~뒤(차 뒤끝 ~ SIDE_PASSED)의 장애물은
        차 옆면과 SIDE_GAP 이상 떨어져 있으면 괜찮다: 앞바퀴 조향 차는 꺾을 때 뒤쪽이 옆으로 거의 안 움직인다."""
        if self.map.first_in_lane(lane, self.s + SIDE_PASSED, self.s + upto) is not None:
            return False
        s, lat = self.map._s, self.map._lat
        if s.size:
            sel = ((np.abs(lat - LANE_CENTER[lane]) <= LANE_BAND) & (s >= self.s - CAR_REAR - SIDE_BACK)
                   & (s < self.s + SIDE_PASSED))
            if sel.any() and float((np.abs(lat[sel] - self.c) - CAR_HALF_WIDTH).min()) < SIDE_GAP:
                return False
        return True

    def _goal_order(self, goal):
        side = goal if goal in (0, 2) else DEFAULT_SIDE
        return lambda j: (0 if j == side else 1)

    def _pick(self, cur, a, goal, exclude=()):
        """막힌 차로 cur(앞 a m) 대신 옮길 이웃 차로. 없으면 None."""
        cands = []
        for j in (cur - 1, cur + 1):
            if not 0 <= j <= 2 or j in exclude:
                continue
            if self.ahead[j] < a + GAIN or not self._side_clear(j, a + SIDE_FRONT):
                continue
            cands.append(j)
        if not cands and cur != 1 and 1 not in exclude and a >= TRANSIT_NEED:
            far = 2 - cur                    # 옆(가운데)은 곧 막히지만 반대편 바깥 차로가 비었다 -> 한 번에 두 칸
            if (self.ahead[far] >= a + GAIN and self.ahead[1] >= TRANSIT_NEED
                    and self._side_clear(1, TRANSIT_NEED) and self._side_clear(far, a + SIDE_FRONT)):
                cands = [far]
        if not cands:
            return None
        best = max(self.ahead[j] for j in cands)
        cands = [j for j in cands if self.ahead[j] >= best - TIE]
        cands.sort(key=self._goal_order(goal))
        return cands[0]

    def _begin(self, j, why):
        self.change_from, self.lane, self.changing = self.lane, j, True
        self._pref_count = 0
        self.lane_run = 0.0
        self.state, self.reason = 'CHANGE', why

    def _decide(self, goal, front):
        """반환: 속도."""
        self.ahead = [self._free(j) for j in range(3)]
        cur = self.lane
        a = self.ahead[cur]
        to_island = self.island_at - self.s
        if self.s >= self.island_at + MERGE_TURN:
            # 섬 끝: 바깥 차로에서 가운데(합류 차로)로. 갈림길은 이미 지났고 이 구간에 장애물은 없다
            if cur != 1:
                self._begin(1, '합류 -> 가운데')
            elif self.changing and abs(self.c) <= ARRIVE_TOL and abs(self.th) < math.radians(10.0):
                self.changing = False
            # 여기서는 지도의 '섬' 막힘을 보지 않는다 (섬 끝을 비스듬히 지나간다). 정면 물체는 정면 정지(GUARD)가 막는다
            self.state = 'CHANGE' if self.changing else 'KEEP'
            self.reason = '합류'
            return V_CHANGE
        if self.s < CHANGE_FROM_S and not self.changing:
            # 아직 1차선 (도로가 벌어지기 전): 가운데 차로를 따라가기만 한다
            self._pref_count = 0
            if a < STOP_DIST:
                self.state, self.reason = 'STOP', f'3차선 입구 전, 앞 {a:.2f} m'
                return 0.0
            self.state, self.reason = 'KEEP', f'3차선 입구 전 (시작선 {self.s:+.2f} m)'
            return V_KEEP if a >= BLOCK_DIST else V_SLOW
        if self.changing:
            if abs(self.c - LANE_CENTER[cur]) <= ARRIVE_TOL and abs(self.th) < math.radians(10.0):
                self.changing = False
            elif (abs(self.c - LANE_CENTER[self.change_from]) < ABORT_BACK
                    and not self._side_clear(cur, min(a, BLOCK_DIST) + SIDE_FRONT) and a < BLOCK_DIST):
                # 막 옮기기 시작했는데 그 차로에 장애물이 새로 보였다 -> 원래 차로로
                self.lane, self.changing = self.change_from, False
                self.state, self.reason = 'KEEP', f'차로 {cur} 이 막힘 -> 차로 {self.lane} 로 돌아감'
                return V_SLOW
            else:
                if a < STOP_DIST:
                    self.state, self.reason = 'STOP', f'옮기는 중 앞 {a:.2f} m'
                    return 0.0
                self.state, self.reason = 'CHANGE', f'차로 {self.change_from} -> {cur}'
                return V_CHANGE
        # 1) 가운데 차로에서 섬이 가까우면 옆으로 빠진다 (가운데 차로는 섬에서 끝난다)
        if cur == 1 and to_island < LEAVE_DIST:
            j = self._pick(1, min(a, max(to_island, 0.0)), goal)
            if j is not None:
                self._begin(j, f'섬 {to_island:.1f} m -> 차로 {j}')
                return V_CHANGE
        # 2) 내 차로가 막힘
        if a < BLOCK_DIST:
            if a >= MIN_START:
                j = self._pick(cur, a, goal)
                if j is not None:
                    self._begin(j, f'차로 {cur} 앞 {a:.2f} m 막힘 -> 차로 {j}')
                    return V_CHANGE
            if a < STOP_DIST:
                self.state, self.reason = 'STOP', f'앞 {a:.2f} m, 옮길 차로 없음'
                return 0.0
            self.state, self.reason = 'BLOCKED', f'앞 {a:.2f} m, 옮길 차로 없음 -> 서행'
            return V_SLOW
        # 3) 막히지 않았으면 가고 싶은 차로로 (표지판 쪽, 모르면 가운데에서 표지판 읽기)
        if goal in (0, 2):
            pref = goal
        else:
            pref = 1 if to_island > LEAVE_DIST + 0.3 else cur
        if pref != cur:
            j = cur + (1 if pref > cur else -1)
            need = CENTER_NEED if j == 1 else min(NO_OBSTACLE, BLOCK_DIST + GAIN)
            if j == 1:
                ok = to_island >= CENTER_NEED and self.ahead[1] >= need and self._side_clear(1, need)
            else:
                ok = self.ahead[j] >= need and self._side_clear(j, need)
            self._pref_count = self._pref_count + 1 if ok else 0
            if self._pref_count >= PREF_FRAMES:
                self._begin(j, ('표지판 쪽' if goal in (0, 2) else '가운데에서 표지판 읽기') + f' -> 차로 {j}')
                return V_CHANGE
        else:
            self._pref_count = 0
        self.state, self.reason = 'KEEP', ''
        if cur == 1 and goal not in (0, 2) and to_island < READ_SLOW_BELOW:
            self.reason = f'섬 {to_island:.1f} m: 표지판 읽는 중'
            return V_READ
        return V_KEEP

    def _recover(self, speed, now):
        """서 있으면 후진. 반환: (속도, 후진 중인지)."""
        if self.reversing:
            if self.reverse_left <= 0.0:
                self.reversing = False
                self.stop_since = None
            else:
                return -V_REVERSE, True
        if speed > 0.0:
            self.stop_since = None
            return speed, False
        if self.stop_since is None:
            self.stop_since = now
        if now - self.stop_since < STUCK_SEC or self.reverse_count >= REVERSE_MAX_COUNT:
            return 0.0, False
        room = min(REVERSE_M, self.lane_run)
        behind = self.map.first_in_lane(self.lane, self.s - CAR_REAR - room - 0.05, self.s - CAR_REAR)
        if behind is not None:
            room = min(room, self.s - CAR_REAR - behind - 0.05)
        if room < 0.10:
            return 0.0, False
        self.reversing, self.reverse_left = True, room
        self.reverse_count += 1
        self.lane_run = 0.0
        return -V_REVERSE, True

    def step_obs(self, obs, pose, scan, goal=None, now=None):
        """반환: dict(v, steer, state, reason, wall_text)."""
        now = time.monotonic() if now is None else now
        self._advance(pose)
        self._camera_heading(obs)
        self._camera_lines(obs)
        self._camera_island(obs)
        front = T.LIDAR_MAX
        if scan is not None:
            self._scan_to_map(scan)
            front = T.front_gap(scan[0], scan[1], scan[2], scan[3])
        self.map.virtual = [(self.island_at, self.island_at + ISLAND_LEN, -T.ISLAND_HALF + 0.05, T.ISLAND_HALF - 0.05)]
        self.map.refresh()
        if scan is None:
            self.state, self.reason = 'NOSCAN', '라이다 없음 -> 정지'
            return dict(v=0.0, steer=0.0, state=self.state, reason=self.reason, wall_text='')
        else:
            speed = self._decide(goal, front)
        merging = self.s >= self.island_at + MERGE_TURN
        look = MERGE_LOOK if merging else (LOOKAHEAD_CHANGE if self.changing else LOOKAHEAD_KEEP)
        steer = T.steer_to_lane(self.c, self.th, self.lane, look)
        steer = T.limit_heading(steer, self.th, MERGE_HEADING if merging else HEADING_MAX)
        if front < GUARD_DIST and speed > 0.0:
            speed, self.state, self.reason = 0.0, 'STOP', f'정면 {front:.2f} m 에 물체'
        speed, rev = self._recover(speed, now)
        wall_text = ''
        if rev:
            steer = 0.0
            self.state, self.reason = 'REVERSE', f'막혀서 후진 (남은 {self.reverse_left:.2f} m)'
        else:
            if obs:
                cross = ()
                if self.changing:      # 섬 앞에서 빠질 때 넘어가는 섬 테두리는 벽이 아니다 (three_lane 과 같음)
                    way = LANE_CENTER[self.lane] - self.c
                    cross = tuple(u for u, n, ratio in obs['lines']
                                  if ratio >= 0.75 and u * way > 0.0 and abs(self.c + T._shrink(u)) < T.ISLAND_INSIDE)
                wall = T.yellow_wall(obs['lines'], obs['th'], obs.get('ystart'), skip=cross)
                if wall is not None:
                    steer, wall_text = wall
        return dict(v=speed, steer=steer, state=self.state, reason=self.reason, wall_text=wall_text)


def dash_start_ahead(fitter, obs):
    """방금 fit 한 이미지에서 흰 점선이 카메라 투영점 앞 몇 m 에서 시작하는지. 화면 아래 끝부터 보이면(이미 점선 안) None."""
    if not obs or fitter.white is None:
        return None
    th = obs['th']
    us = [u for u, n, ratio in obs['lines'] if ratio <= 0.3 and n >= 50 and abs(u) < 0.5]
    if not us:
        return None
    py, px = np.nonzero(fitter.white)
    if px.size == 0:
        return None
    x = -T.XW + (px + 0.5) * T.RES
    f = T.F1 - (py + 0.5) * T.RES
    u = -x * math.cos(th) - f * math.sin(th)
    along = -x * math.sin(th) + f * math.cos(th)
    sel = np.zeros(u.size, bool)
    for line_u in us:
        sel |= np.abs(u - line_u) < 0.04
    if sel.sum() < 10:
        return None
    d = float(np.percentile(along[sel], 5))
    return d if d >= 0.55 else None


class Lane3SimpleSupervisor:
    """Lane3Supervisor 와 같은 모양. mode: off -> armed -> on -> done."""

    def __init__(self, forced=False, goal=None):
        self.fitter = T.LineFitter()
        self._look = T.ThreeLane()          # 진입 판단(look)에만 쓴다
        self.core = self.make_core()
        self.forced = forced
        self.mode = 'armed' if forced else 'off'
        self.sign_goal = goal
        self.fixed_goal = goal is not None
        self.vote = T.SignVote()
        self.arm_travel = 0.0
        self._pose = None
        self._white = self._both = self._lost = 0
        self._seen = None
        self.steer = 0.0
        self.side_taken = None
        self.last = None                    # see_signs 가 쓰는 .ok / .c
        self.message = ''

    smooth_output = True

    def make_core(self):
        return SimpleThreeLane()

    def set_mission(self, name):
        if self.forced:
            return
        if name == 'M7':
            if self.mode == 'off':
                self.mode, self.arm_travel = 'armed', 0.0
                self.core = self.make_core()
                self._white = self._both = self._lost = 0
                self._seen = None
                self.vote.reset()
                if not self.fixed_goal:
                    self.sign_goal = None
        elif name != 'M2':
            self.mode = 'off'

    def set_sign(self, event):
        if event == 'sign_a':
            self.sign_goal = 0
        elif event == 'sign_b':
            self.sign_goal = 2

    def see_signs(self, detections):
        if self.fixed_goal or self.mode not in ('armed', 'on'):
            return
        detections = [dict(d, label={'sign_a': 'a', 'sign_b': 'b'}.get(d.get('label'), d.get('label'))) for d in detections]
        signs = [d for d in detections if d.get('label') in ('a', 'b') and len(d.get('bbox', ())) == 4]
        if not signs:
            return
        if self.mode == 'on' and abs(self.core.c) > T.SIGN_CENTER_HALF:
            return                          # 옆 차로에서 비스듬히 읽은 값은 틀린다
        best = max(signs, key=lambda d: d.get('confidence', 0.0))
        height = float(best['bbox'][3]) - float(best['bbox'][1])
        decided = self.vote.add(best['label'], height)
        if decided is not None:
            self.sign_goal = decided
            self.message = f"[3차선S] 표지판 {'A(왼쪽)' if decided == 0 else 'B(오른쪽)'} 확정 (상자 높이 {height:.0f} px)"

    @property
    def latch_off(self):
        return self.mode in ('armed', 'on') and (self.forced or self.arm_travel >= ARM_AFTER_M)

    def _try_enter(self, image, pose, tune):
        seen, white = self._look.look(image, tune)
        if seen is None or not white or pose is None:
            self._white, self._seen = 0, None
            return False
        if self._seen is not None:
            c0, th0, pose0 = self._seen
            ds = math.hypot(pose[0] - pose0[0], pose[1] - pose0[1])
            th_exp = th0 - (pose[2] - pose0[2])
            same = (abs(seen[1] - th_exp) <= T.ACQUIRE_TH_TOL
                    and abs(seen[0] - (c0 - ds * math.sin(th0))) <= T.ACQUIRE_C_TOL)
            self._white = self._white + 1 if same else 1
        else:
            self._white = 1
        self._seen = (seen[0], seen[1], pose)
        if self._white < ENTRY_FRAMES:
            return False
        # 진입은 늘 1차선(= 가운데 차로)을 곧게 따라오다 한다. 카메라 한 장 값은 입구의 45도 벌어지는 선 때문에 크게 틀릴 수 있어
        # (11/15 실차: 가운데 직진 중인데 c -0.52, 방향 +40도로 잡고 왼쪽으로 74도 꺾음) 둘 다 좁게 자른다
        c = float(np.clip(seen[0], -ENTRY_C_MAX, ENTRY_C_MAX))
        th0 = float(np.clip(seen[1], -ENTRY_TH_MAX, ENTRY_TH_MAX))
        # 시작선 기준 위치: 흰 점선 시작이 앞 d m 에 보이면 지금 s = 0.30 - d. 이미 점선 안이면 점선 시작으로 본다
        d = dash_start_ahead(self._look.fitter, self._look.fitter.fit(image, th0, tune))
        s0 = START_TO_DASH - d if d is not None else START_TO_DASH
        self.core = self.make_core()
        self.core.start(c, th0, pose, s0)
        self.mode = 'on'
        self.message = (f'[3차선S] 시작: 차로 {self.core.lane} (가운데에서 {c * 100:+.0f} cm, '
                        f'도로 방향 {math.degrees(th0):+.0f}도(카메라 {math.degrees(seen[1]):+.0f}), 시작선에서 {s0:+.2f} m'
                        + (')' if d is not None else ', 점선 시작을 못 봐서 점선 시작으로 가정)'))
        return True

    def step(self, image, pose, scan, both_yellow, tune=None, now=None):
        if pose is not None and self._pose is not None:
            d = math.hypot(pose[0] - self._pose[0], pose[1] - self._pose[1])
            if d < 0.5:
                self.arm_travel += d
        if pose is not None:
            self._pose = pose
        if self.mode in ('off', 'done'):
            return None
        if self.mode == 'armed':
            if not self.forced and self.arm_travel < ARM_AFTER_M:
                return None
            if not self._try_enter(image, pose, tune):
                return None
        prior = self.core.th
        obs = self.fitter.fit(image, prior, tune)
        return self.step_obs(obs, pose, scan, both_yellow, now)

    def step_obs(self, obs, pose, scan, both_yellow, now=None):
        core = self.core
        if pose is None:
            self._lost += 1
            if hasattr(core, 'diagnostics'):
                core.diagnostics.update(state='NO_ODOM',reason='odometry unavailable; M7 retained')
            return dict(v=0.0, steer=self.steer, state='LANE3', reason='3차선S: odom 없음 -> 정지', note='L3S noodom')
        self._lost = 0
        out = core.step_obs(obs, pose, scan, self.sign_goal, now)
        self.last = type('L', (), dict(ok=True, c=core.c))()
        if core.s < core.island_at + MERGE_TURN and core.lane in (0, 2):
            self.side_taken = core.lane                 # 갈림길에서 고른 쪽 (기록용)
        merged = (out['v'] > 0 and core.s >= core.island_at + MERGE_END + 0.20 and core.lane == 1
                  and abs(core.c) < 0.12 and abs(core.th) < math.radians(15.0))
        self._both = self._both + 1 if merged else 0
        if self._both >= EXIT_FRAMES:
            self.mode = 'done'
            side = {0: 'A(왼쪽)', 2: 'B(오른쪽)'}.get(self.side_taken, '?')
            self.message = f"[3차선S] 끝: 갈림길 {side} 지나 합류 완료 -> 기본 주행"
            return None
        if core.travel >= MAX_ON_M:
            return dict(v=0.0, steer=0.0, state='LANE3', reason='M7 exit not confirmed', note='L3S exit_unconfirmed')
        if out['wall_text'] or not self.smooth_output:
            self.steer = out['steer']
        else:
            self.steer += 0.7 * (out['steer'] - self.steer)
        g = self.sign_goal
        goal_text = '-' if g is None else ('A' if g == 0 else 'B')
        free = '/'.join(f'{x:.1f}' for x in core.ahead)
        reason = (f"3차선S {out['state']} 차로 {core.lane} 빈거리 {free} 목표 {goal_text}"
                  + (f" | {out['reason']}" if out['reason'] else '') + (f" | {out['wall_text']}" if out['wall_text'] else ''))
        note = (f"L3S {out['state']} tgt{core.lane} c{core.c:+.2f} th{math.degrees(core.th):+.0f} s{core.s:.2f} "
                f"free{free} goal{goal_text} isl{core.island_at:.1f}" + ('' if core.island_s is None else 'cam')
                + (' WALL' if out['wall_text'] else ''))
        return dict(v=out['v'], steer=self.steer, state='LANE3', reason=reason, note=note)
