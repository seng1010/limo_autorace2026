"""3차선 구간 주행: 차로 선(흰 점선 + 노란 실선)으로 내 위치를 잡고, 라이다로 빈 차로를 골라 옮겨 간다.
ROS 없음 (simple_lane.py 가 불러 쓴다. 기록(bag)으로 오프라인 확인 가능).

좌표
  th : 도로 방향이 차 정면에서 왼쪽으로 몇 rad 인지 (차가 도로보다 오른쪽을 보면 +)
  c  : 도로 가운데에서 차(카메라 바닥 투영점)까지 거리, 왼쪽 + [m]
  차로 번호 0 = 왼쪽(갈림길 A), 1 = 가운데, 2 = 오른쪽(갈림길 B)

흐름 (한 프레임)
  1. LineFitter.fit(image)  -> 도로 방향 th, 선들 (차에서 왼쪽으로 u m, 노란색 여부)
  2. RoadState.update(...)  -> c, th (선이 안 보이는 동안은 odom 으로 이어 감)
  3. lane_free(scan, ...)   -> 차로별로 앞이 비어 있는 거리
  4. Planner.update(...)    -> 목표 차로, 속도
  5. steer_to_lane(...)     -> 조향각

11/15 03:14 기록(bag_3lane, bag_3lane2)으로 1~4 확인. 닫힌 고리(실제 주행)는 시뮬레이션만.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMG_W, IMG_H = 640, 480
# simple_lane.py 와 같은 카메라 보정값
BEV_SRC = [0.3605, 0.46, 0.6395, 0.46, 1.00, 0.82, 0.00, 0.82]
BEV_GROUND = [-0.24, 1.326, 0.24, 1.326, 0.22, 0.421, -0.22, 0.421]
CAMERA_FORWARD_OFFSET = 0.184
WHEELBASE = 0.20
TRACK = 0.172

# 넓은 BEV (옆 차로 선까지 보이게)
XW, F0, F1, RES = 1.0, 0.40, 2.20, 0.01
FIT_FMAX = 1.9

# 도로 모양 [m]. 선 위치(도로 가운데 기준, 왼쪽 +). 11/15 기록: 가운데 차로 흰 선 +-0.29, 바깥 차로 폭 0.52
WHITE_POS = (0.27, -0.27)
YELLOW_POS = (0.80, -0.80)
LANE_CENTER = (0.535, 0.0, -0.535)          # 차로 0, 1, 2
ISLAND_INSIDE = 0.55                        # 도로 가운데에서 이 안쪽에 보이는 노란 선은 갈림길 섬 테두리로 본다
ISLAND_HALF = 0.36                          # 섬(가운데 차로 끝) 폭의 절반 + 여유
ISLAND_MIN_POINTS = 25
FAR_U = 0.35                                # 이보다 먼 선은 BEV 가 벌어져 보여서 줄여 쓴다
FAR_SHRINK = 1.25

MAX_STEER = math.radians(25.0)
LOOKAHEAD = 0.55                            # 목표점: 카메라 앞 이 거리(도로 방향) [m]
LOOKAHEAD_CHANGE = 0.42                     # 차로를 옮기는 동안은 더 가까운 점을 보고 빨리 꺾는다

# 라이다
LIDAR_MAX = 2.5
LANE_HALF = 0.27
ROAD_HALF = 0.83                            # 이보다 바깥의 점은 도로 밖 (벽, 사람, 다른 구간)

# 판단
BLOCK_DIST = 1.35                           # 내 차로 앞 장애물이 이보다 가까우면 옮길 차로를 찾는다
MIN_START = 0.80                            # 앞 장애물이 이보다 가까우면 옮기기 시작하지 않는다 (옮기는 데 약 0.7 m 필요)
TRANSIT_CLEAR = 1.10                        # 두 칸 옮길 때 가로지르는 가운데 차로에 필요한 빈 거리
FAR_CLEAR = 1.90                            # 두 칸 옮길 때 도착 차로에 필요한 빈 거리
SIDE_CLEAR = 1.05                           # 옮길 차로는 옆~앞 이 거리까지 비어 있어야 한다
SIDE_BEHIND = -0.10                         # 옆 차로 확인을 시작하는 위치 (라이다보다 뒤 -). 지나친 장애물은 곧 시야(좌우 110도)에서 사라진다
GAIN_MIN = 0.45                             # 옮길 차로가 내 차로보다 이만큼은 더 비어 있어야 한다
STOP_DIST = 0.38                            # 갈 곳이 없으면 장애물 앞 이 거리에서 정지
GOAL_CLEAR = 2.40                           # 갈림길 방향 차로로 미리 옮기는 건 그 차로에 장애물이 안 보일 때만
ARRIVE_TOL = 0.10                           # 목표 차로 가운데에서 이 안이면 옮기기 끝
V_KEEP, V_CHANGE, V_SLOW = 0.22, 0.20, 0.12


class LineFitter:
    """이미지 -> 선 (도로 방향 + 선마다 차에서 왼쪽 거리 u, 점 개수, 노란 비율)."""

    def __init__(self):
        src = np.float32(BEV_SRC).reshape(4, 2)
        src[:, 0] *= IMG_W - 1
        src[:, 1] *= IMG_H - 1
        i2g = cv2.getPerspectiveTransform(src, np.float32(BEV_GROUND).reshape(4, 2))
        self.w, self.h = int(2 * XW / RES), int((F1 - F0) / RES)
        g2b = np.array([[1 / RES, 0, XW / RES], [0, -1 / RES, F1 / RES], [0, 0, 1.0]])
        self.i2b = g2b @ i2g
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        self._rng = np.random.default_rng(0)
        self.yellow = self.white = None

    def _thin(self, mask):
        # 15 cm 보다 굵은 덩어리(밝은 바닥, 상자)는 선이 아니다
        fat = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel), self._kernel)
        return cv2.bitwise_and(mask, cv2.bitwise_not(fat))

    def masks(self, image, tune=None):
        tune = tune or {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (tune.get('h_min', 15), max(60, tune.get('s_min', 40)), max(70, tune.get('v_min', 60))),
                             (tune.get('h_max', 40), 255, 255))
        white = cv2.inRange(hsv, (0, 0, 150), (179, 60, 255))
        warp = lambda m: cv2.warpPerspective(m, self.i2b, (self.w, self.h), flags=cv2.INTER_NEAREST)  # noqa: E731
        self.yellow, self.white = self._thin(warp(yellow)), self._thin(warp(white))
        return self.yellow, self.white

    def fit(self, image, th_prior=None, tune=None):
        """반환: None 또는 dict(th, lines=[(u, n, yellow_ratio)], score)."""
        yellow, white = self.masks(image, tune)
        pts = []
        for mask, is_yellow in ((yellow, 1.0), (white, 0.0)):
            py, px = np.nonzero(mask)
            if len(px) == 0:
                continue
            x = -XW + (px + 0.5) * RES
            f = F1 - (py + 0.5) * RES
            keep = f < FIT_FMAX
            pts.append(np.stack([x[keep], f[keep], np.full(int(keep.sum()), is_yellow)], 1))
        if not pts:
            return None
        P = np.concatenate(pts)
        if len(P) < 120:
            return None
        if len(P) > 3000:
            P = P[self._rng.choice(len(P), 3000, replace=False)]
        if th_prior is None:
            angles = np.arange(-math.radians(50), math.radians(50) + 1e-6, math.radians(2.0))
        else:
            angles = np.arange(th_prior - math.radians(16), th_prior + math.radians(16) + 1e-6, math.radians(2.0))
        bins = np.arange(-1.6, 1.6001, 0.02)
        best = None
        for th in angles:
            u = -P[:, 0] * math.cos(th) - P[:, 1] * math.sin(th)
            hist, _ = np.histogram(u, bins=bins)
            score = float((hist.astype(np.float64) ** 2).sum())
            if best is None or score > best[0]:
                best = (score, th)
        # 1도 단위로 한 번 더
        for th in (best[1] - math.radians(1.0), best[1] + math.radians(1.0)):
            u = -P[:, 0] * math.cos(th) - P[:, 1] * math.sin(th)
            hist, _ = np.histogram(u, bins=bins)
            score = float((hist.astype(np.float64) ** 2).sum())
            if score > best[0]:
                best = (score, th)
        score, th = best
        u = -P[:, 0] * math.cos(th) - P[:, 1] * math.sin(th)
        hist, _ = np.histogram(u, bins=bins)
        smooth = np.convolve(hist, [1, 2, 1], 'same')
        centers = bins[:-1] + 0.01
        lines = []
        floor = max(30.0 * len(P) / 3000.0 + 10.0, 0.08 * smooth.max())
        for i in np.argsort(smooth)[::-1]:
            if smooth[i] < floor:
                break
            if any(abs(centers[i] - line[0]) < 0.20 for line in lines):
                continue
            sel = np.abs(u - centers[i]) < 0.04
            if sel.sum() < 12:
                continue
            lines.append((float(u[sel].mean()), int(sel.sum()), float(P[sel, 2].mean())))
        if not lines:
            return None
        return dict(th=float(th), lines=sorted(lines), score=score / float(len(P)) ** 2, n=len(P),
                    yellow_xy=P[P[:, 2] > 0.5, :2])


def _shrink(u):
    if abs(u) <= FAR_U:
        return u
    return math.copysign(FAR_U + (abs(u) - FAR_U) / FAR_SHRINK, u)


def _match_score(c, lines, c_prior=None):
    """c_prior 가 있으면, 그 위치에서 봤을 때 도로 안쪽(|위치| < ISLAND_INSIDE)에 있는 노란 선은 쓰지 않는다.
    갈림길 섬 테두리(노란색)를 바깥 노란 선으로 착각해 위치가 반대편으로 넘어가는 것을 막는다 (11/15 3번 주행)."""
    total = 0.0
    for u, n, yellow_ratio in lines:
        if c_prior is not None and yellow_ratio >= 0.7 and abs(c_prior + _shrink(u)) < ISLAND_INSIDE:
            continue
        pos = c + _shrink(u)
        if yellow_ratio >= 0.7:
            expected = YELLOW_POS
        elif yellow_ratio <= 0.3:
            expected = WHITE_POS
        else:
            expected = YELLOW_POS + WHITE_POS
        d = min(abs(pos - e) for e in expected)
        weight = min(n, 300) * (1.0 if abs(u) <= 0.45 else 0.4)
        total += weight * max(0.0, 1.0 - (d / 0.15) ** 2)
    return total


class RoadState:
    """도로 안에서의 내 위치 (c, th). 선이 보이면 맞추고, 안 보이면 odom 으로 이어 간다."""

    def __init__(self):
        self.c: Optional[float] = None
        self.th = 0.0
        self.blind_m = 0.0          # 선으로 맞추지 못한 채 간 거리
        self.fixed = False          # 이번 프레임에 선으로 맞췄는지

    @property
    def lane(self) -> Optional[int]:
        if self.c is None:
            return None
        return int(np.argmin([abs(self.c - x) for x in LANE_CENTER]))

    def predict(self, ds: float, dyaw: float) -> None:
        self.th -= dyaw
        if self.c is not None:
            self.c -= ds * math.sin(self.th)
            self.blind_m += abs(ds)

    def update(self, obs: Optional[dict]) -> None:
        self.fixed = False
        if obs is None:
            return
        lines = obs['lines']
        if self.c is None or self.blind_m > 1.2:
            candidates = np.arange(-0.80, 0.8001, 0.02)
            prior = None
        else:
            window = 0.16 + 0.25 * min(self.blind_m, 0.6)
            candidates = np.arange(self.c - window, self.c + window + 1e-6, 0.02)
            prior = self.c
        scores = np.array([_match_score(float(c), lines, prior) for c in candidates])
        k = int(np.argmax(scores))
        if scores[k] < 40.0:
            return
        # 가장 잘 맞는 c 주변을 가중 평균
        near = np.abs(candidates - candidates[k]) <= 0.041
        c = float((candidates[near] * scores[near]).sum() / scores[near].sum())
        if self.c is None or self.blind_m > 1.2:
            self.c = c
        else:
            self.c += 0.6 * (c - self.c)
        self.th += 0.7 * (obs['th'] - self.th)
        self.blind_m = 0.0
        self.fixed = True


def island_distance(yellow_xy, c, th):
    """갈림길 섬(가운데 차로 끝의 노란 테두리)까지 도로 방향 거리 [m]. 안 보이면 None.
    yellow_xy: 노란 선 점들 (오른쪽, 앞) [m], 카메라 바닥 투영점 기준."""
    if yellow_xy is None or len(yellow_xy) == 0:
        return None
    x, f = yellow_xy[:, 0], yellow_xy[:, 1]
    lateral = c + (-x * math.cos(th) - f * math.sin(th))
    along = -x * math.sin(th) + f * math.cos(th)
    inside = (np.abs(lateral) <= ISLAND_HALF) & (along > 0.0)
    if int(inside.sum()) < ISLAND_MIN_POINTS:
        return None
    return float(np.percentile(along[inside], 10))


def lane_free(ranges, angle_min, angle_increment, range_min, c, th):
    """라이다 한 바퀴 -> 차로 0,1,2 별 (앞으로 비어 있는 거리, 옆~앞 구간에서 가장 가까운 거리).
    거리는 도로 방향으로 잰다. 없으면 LIDAR_MAX."""
    r = np.asarray(ranges, dtype=np.float32)
    a = angle_min + np.arange(r.size) * angle_increment
    ok = np.isfinite(r) & (r >= max(range_min, 0.05)) & (r <= LIDAR_MAX * 1.2)
    x, y = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])          # 앞, 왼쪽
    along = x * math.cos(th) + y * math.sin(th)
    lateral = c + (-x * math.sin(th) + y * math.cos(th))
    ahead, side = [], []
    for center in LANE_CENTER:
        in_lane = (np.abs(lateral - center) <= LANE_HALF) & (np.abs(lateral) <= ROAD_HALF)
        d = np.sort(along[in_lane & (along > 0.10) & (along < LIDAR_MAX)])
        ahead.append(float(d[1]) if d.size >= 2 else LIDAR_MAX)       # 점 2개 이상일 때만 (잡음 1개는 무시)
        d = np.sort(along[in_lane & (along > SIDE_BEHIND) & (along < LIDAR_MAX)])
        side.append(float(d[1]) if d.size >= 2 else LIDAR_MAX)
    return ahead, side


@dataclass
class Plan:
    target: int = 1
    speed: float = V_KEEP
    state: str = 'KEEP'              # KEEP / CHANGE / BLOCKED / STOP
    reason: str = ''


class Planner:
    def __init__(self):
        self.target: Optional[int] = None
        self.changing = False

    def _escape(self, cur, c, ahead, side, goal):
        """막힌 차로 cur 대신 갈 차로. 없으면 None."""
        in_cur = abs(c - LANE_CENTER[cur]) < LANE_HALF
        if in_cur and ahead[cur] < MIN_START:
            return None                      # 너무 가까워서 지금 꺾으면 앞 장애물을 스친다
        need = ahead[cur] + GAIN_MIN
        near = [j for j in (cur - 1, cur + 1)
                if 0 <= j <= 2 and side[j] >= SIDE_CLEAR and ahead[j] >= need]
        if near:
            if goal is not None:
                near.sort(key=lambda j: (abs(j - goal), -ahead[j]))
            else:
                near.sort(key=lambda j: -ahead[j])
            return near[0]
        far = 2 - cur
        if cur != 1 and side[1] >= TRANSIT_CLEAR and side[far] >= FAR_CLEAR \
                and ahead[far] >= max(ahead[cur], ahead[1]) + GAIN_MIN:
            return far                       # 가운데 차로를 가로질러 두 칸 이동
        return None

    def update(self, lane: int, c: float, ahead: List[float], side: List[float],
               goal: Optional[int] = None) -> Plan:
        if self.target is None:
            self.target = lane
        if self.changing and abs(c - LANE_CENTER[self.target]) <= ARRIVE_TOL:
            self.changing = False
        cur = self.target
        if ahead[cur] >= BLOCK_DIST:
            if self.changing:
                return Plan(cur, V_CHANGE, 'CHANGE', f'차로 {lane} -> {cur}')
            if goal is not None and goal != cur:
                j = cur + (1 if goal > cur else -1)
                if side[j] >= GOAL_CLEAR and ahead[j] >= GOAL_CLEAR:
                    self.target, self.changing = j, True
                    return Plan(j, V_CHANGE, 'CHANGE', f'갈림길 방향 -> 차로 {j}')
            return Plan(cur, V_KEEP, 'KEEP', '')
        # cur 가 막힘 (가는 중이든 옮기는 중이든)
        best = self._escape(cur, c, ahead, side, goal)
        if best is not None:
            self.target, self.changing = best, True
            return Plan(best, V_CHANGE, 'CHANGE', f'차로 {cur} 앞 {ahead[cur]:.2f} m 막힘 -> 차로 {best}')
        if ahead[cur] < STOP_DIST and abs(c - LANE_CENTER[cur]) < LANE_HALF:
            return Plan(cur, 0.0, 'STOP', f'앞 {ahead[cur]:.2f} m, 옮길 차로 없음')
        return Plan(cur, V_SLOW, 'BLOCKED', f'차로 {cur} 앞 {ahead[cur]:.2f} m, 옮길 차로 없음')


def steer_to_lane(c, th, target, lookahead=LOOKAHEAD):
    """목표 차로 가운데를 따라가는 조향각 (왼쪽 +)."""
    delta = LANE_CENTER[target] - c                         # 목표 차로 가운데가 왼쪽으로 얼마나
    forward = lookahead * math.cos(th) - delta * math.sin(th)
    left = lookahead * math.sin(th) + delta * math.cos(th)
    forward_rear = forward + CAMERA_FORWARD_OFFSET
    if forward_rear <= 0.05:
        return math.copysign(MAX_STEER, left)
    if abs(left) < 1e-6:
        return 0.0
    curvature = 2.0 * abs(left) / (forward_rear ** 2 + left ** 2)
    inside_radius = max(1e-6, 1.0 / curvature - TRACK / 2.0)
    return math.copysign(min(math.atan(WHEELBASE / inside_radius), MAX_STEER), left)


@dataclass
class ThreeLaneOutput:
    ok: bool = False                 # False 면 위치를 모름 (호출한 쪽이 기존 주행으로)
    speed: float = 0.0
    steer: float = 0.0               # 왼쪽 +
    state: str = ''
    reason: str = ''
    lane: Optional[int] = None
    target: Optional[int] = None
    c: float = 0.0
    th: float = 0.0
    ahead: List[float] = field(default_factory=list)
    lines: List[Tuple[float, int, float]] = field(default_factory=list)
    white_seen: bool = False         # 가까운 흰 점선이 보임 (3차선 시작 판단용)
    island: Optional[float] = None   # 갈림길 섬까지 거리 (보일 때)


class ThreeLane:
    """한 프레임씩 호출. scan 은 가장 최근 라이다 (ranges, angle_min, angle_increment, range_min) 또는 None."""

    def __init__(self):
        self.fitter = LineFitter()
        self.road = RoadState()
        self.planner = Planner()
        self._pose = None
        self.travel = 0.0            # 이 모드에서 간 거리

    def reset(self):
        self.road = RoadState()
        self.planner = Planner()
        self._pose = None
        self.travel = 0.0

    def step(self, image, pose, scan, goal=None, tune=None) -> ThreeLaneOutput:
        self._advance(pose)
        obs = self.fitter.fit(image, self.road.th if self.road.c is not None else None, tune)
        return self._decide(obs, scan, goal)

    def step_obs(self, obs, pose, scan, goal=None) -> ThreeLaneOutput:
        """선 검출 결과를 직접 넣는 버전 (시뮬레이션·테스트용)."""
        self._advance(pose)
        return self._decide(obs, scan, goal)

    def _advance(self, pose) -> None:
        ds = dyaw = 0.0
        if pose is not None and self._pose is not None:
            ds = math.hypot(pose[0] - self._pose[0], pose[1] - self._pose[1])
            dyaw = pose[2] - self._pose[2]
            if ds > 0.5 or abs(dyaw) > 1.0:          # odom 튐
                ds = dyaw = 0.0
        if pose is not None:
            self._pose = pose
        self.travel += ds
        self.road.predict(ds, dyaw)

    def _decide(self, obs, scan, goal) -> ThreeLaneOutput:
        self.road.update(obs)
        out = ThreeLaneOutput(lines=obs['lines'] if obs else [])
        out.white_seen = bool(obs) and any(
            ratio <= 0.3 and n >= 50 and abs(u) < 0.50 for u, n, ratio in obs['lines'])
        if self.road.c is None or self.road.blind_m > 1.0:
            out.state, out.reason = 'LOST', '차로 선으로 위치를 못 잡음'
            return out
        lane = self.road.lane
        if scan is not None:
            ahead, side = lane_free(scan[0], scan[1], scan[2], scan[3], self.road.c, self.road.th)
        else:
            ahead, side = [LIDAR_MAX] * 3, [LIDAR_MAX] * 3
        island = island_distance(obs.get('yellow_xy') if obs else None, self.road.c, self.road.th)
        out.island = island
        if island is not None:                   # 가운데 차로는 섬에서 끝난다 -> 장애물로 취급
            ahead[1], side[1] = min(ahead[1], island), min(side[1], island)
        plan = self.planner.update(lane, self.road.c, ahead, side, goal)
        out.ok = True
        out.speed, out.state, out.reason = plan.speed, plan.state, plan.reason
        out.steer = steer_to_lane(self.road.c, self.road.th, plan.target,
                                  LOOKAHEAD_CHANGE if plan.state == 'CHANGE' else LOOKAHEAD)
        out.lane, out.target, out.c, out.th, out.ahead = lane, plan.target, self.road.c, self.road.th, ahead
        if scan is None:
            out.reason = (out.reason + ' 라이다 없음').strip()
        return out


# ---------------------------------------------------------------------------
# simple_lane.py 와 잇는 부분: 언제 3차선 주행을 켜고 끌지
# ---------------------------------------------------------------------------
ARM_AFTER_M = 1.4          # M7 시작(횡단보도 출발) 후 이만큼 간 뒤부터 3차선 선을 찾는다 (그 전은 코너)
ENTRY_FRAMES = 3           # 가까운 흰 점선이 연속 이만큼 보이면 3차선 주행 시작
EXIT_FRAMES = 4            # 바깥 차로에서 양쪽 노란 선(갈림길 차로)이 연속 이만큼 보이면 기본 주행으로 넘김
DEFAULT_GOAL_AFTER_M = 1.8 # 표지판을 못 읽은 채 이만큼 가면 갈 쪽을 스스로 정한다
DEFAULT_SIDE = 2           # 그때 가운데 차로에 있으면 갈 쪽 (2 = 오른쪽 = B)
MAX_ON_M = 6.5             # 3차선 주행을 이 거리 넘게 하고 있으면 끝낸다 (도면상 점선 구간 약 3.7 m)
LOST_FRAMES = 30           # 위치를 못 잡은 채 이만큼 지나면 끝낸다
SIDE_TOL = 0.13


class Lane3Supervisor:
    """mode: off(안 씀) -> armed(3차선 선을 기다림) -> on(3차선 주행) -> done(기본 주행으로 넘김)."""

    def __init__(self, forced=False, goal=None):
        self.tl = ThreeLane()
        self.forced = forced
        self.mode = 'armed' if forced else 'off'
        self.sign_goal = goal             # 0 = A(왼쪽), 2 = B(오른쪽), None = 아직 모름
        self.arm_travel = 0.0
        self._pose = None
        self._white = self._both = self._lost = 0
        self.steer = 0.0
        self.last = None                  # 마지막 ThreeLaneOutput (화면·기록용)
        self.message = ''                 # 상태가 바뀔 때 한 번 찍을 글

    def set_mission(self, name):
        if self.forced:
            return
        if name == 'M7':
            if self.mode == 'off':
                self.mode, self.arm_travel = 'armed', 0.0
                self.tl.reset()
                self._white = self._both = self._lost = 0
        elif name != 'M2':
            self.mode = 'off'

    def set_sign(self, event):
        if event == 'sign_a':
            self.sign_goal = 0
        elif event == 'sign_b':
            self.sign_goal = 2

    @property
    def latch_off(self):
        """True 면 기본 주행의 코너 고정 조향을 끈다 (3차선 입구의 사선을 코너로 잘못 보지 않게)."""
        return self.mode in ('armed', 'on') and (self.forced or self.arm_travel >= ARM_AFTER_M)

    def step(self, image, pose, scan, both_yellow, tune=None):
        """반환: None (기본 주행 명령을 그대로 씀) 또는 dict(v, steer, state, reason, note)."""
        if pose is not None and self._pose is not None:
            step = math.hypot(pose[0] - self._pose[0], pose[1] - self._pose[1])
            if step < 0.5:
                self.arm_travel += step
        if pose is not None:
            self._pose = pose
        if self.mode in ('off', 'done'):
            return None
        if self.mode == 'armed':
            if not self.forced and self.arm_travel < ARM_AFTER_M:
                return None
            out = self.tl.step(image, pose, scan, None, tune)
            self.last = out
            self._white = self._white + 1 if (out.ok and out.white_seen) else 0
            if self._white < ENTRY_FRAMES:
                return None
            self.mode = 'on'
            self.tl.planner = Planner()
            self.tl.travel = 0.0
            self._both = self._lost = 0
            self.steer = 0.0
            self.message = f'[3차선] 시작: 차로 {out.lane} (도로 가운데에서 {out.c * 100:+.0f} cm)'
        goal = self.sign_goal
        if goal is None and self.tl.travel >= DEFAULT_GOAL_AFTER_M:
            lane = self.tl.road.lane
            goal = lane if lane in (0, 2) else DEFAULT_SIDE
        out = self.tl.step(image, pose, scan, goal, tune)
        self.last = out
        if not out.ok:
            self._lost += 1
            if self._lost >= LOST_FRAMES:
                self.mode, self.message = 'done', '[3차선] 끝: 차로 선을 못 찾음 -> 기본 주행'
                return None
            return dict(v=V_SLOW, steer=self.steer, state='LANE3', reason='3차선: 선을 못 찾아 직전 조향 유지',
                        note='L3 lost')
        self._lost = 0
        on_side = (out.target in (0, 2) and out.state == 'KEEP'
                   and abs(out.c - LANE_CENTER[out.target]) < SIDE_TOL)
        # 갈림길 섬: 바깥 차로에서 도로 안쪽 이웃 선이 흰 점선이 아니라 노란 선(섬 테두리)으로 보인다
        inward = 1.0 if out.target == 2 else -1.0
        island = on_side and any(ratio >= 0.7 and n >= 40 and 0.12 < inward * u < 0.50
                                 for u, n, ratio in out.lines)
        self._both = self._both + 1 if (on_side and (both_yellow or island)) else 0
        if self._both >= EXIT_FRAMES:
            self.mode = 'done'
            self.message = f"[3차선] 끝: 갈림길 {'A(왼쪽)' if out.target == 0 else 'B(오른쪽)'} 차로 진입 -> 기본 주행"
            return None
        if self.tl.travel >= MAX_ON_M:
            self.mode, self.message = 'done', '[3차선] 끝: 최대 거리 -> 기본 주행'
            return None
        self.steer += 0.6 * (out.steer - self.steer)
        goal_text = '-' if goal is None else ('A' if goal == 0 else ('B' if goal == 2 else str(goal)))
        free = '/'.join(f'{a:.1f}' for a in out.ahead)
        reason = (f"3차선 {out.state} 차로 {out.lane}->{out.target} 빈거리 {free} 목표 {goal_text}"
                  + (f' | {out.reason}' if out.reason else ''))
        note = (f"L3 {out.state} lane{out.lane} tgt{out.target} c{out.c:+.2f} th{math.degrees(out.th):+.0f} "
                f"free{free} goal{goal_text} s{self.tl.travel:.2f}"
                + ('' if out.island is None else f' isl{out.island:.1f}') + ('' if self.tl.road.fixed else ' nofix'))
        return dict(v=out.speed, steer=self.steer, state='LANE3', reason=reason, note=note)
