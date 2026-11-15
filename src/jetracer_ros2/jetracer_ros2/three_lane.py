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
from collections import deque
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
FIX_MIN_SCORE = 60.0                        # 선으로 위치를 맞췄다고 볼 최소 점수 (흰 조각 하나로는 못 넘는다)
EDGE_POS = 0.88                             # 바깥 흰 실선의 위치 (바깥 노란 선 0.80 ~ 매트 가장자리 0.97 의 가운데)
EDGE_TOL = 0.35
EDGE_MIN_POINTS = 100
TRACK_C_STEP = 0.03                         # 선으로 위치를 고칠 때 한 프레임에 옮기는 최대 폭 [m] (차 움직임은 odom 으로 이미 반영)
TRACK_C_STEP_BLIND = 0.30                   # 선을 못 본 채 간 거리 1 m 당 더 허용하는 폭
TRACK_TH_STEP = math.radians(3.0)           # 도로 방향을 고칠 때 한 프레임 최대 각
ACQUIRE_MARGIN = 1.10                       # 처음 위치를 잡을 때: 가장 잘 맞는 위치가 다른 차로 위치보다 이 배수 이상 잘 맞아야 한다
ACQUIRE_C_TOL = 0.12                        # 처음 위치를 잡을 때 연속 프레임끼리 허용하는 차이
ACQUIRE_TH_TOL = math.radians(12.0)
YELLOW_MIN_SPAN = 0.60                      # 바깥 노란 실선은 길게 보인다. 이보다 짧은 노란 조각은 위치 맞추기에 쓰지 않는다
SOLID_MIN_SPAN = 0.60                       # 흰 선이 이 길이 이상 보이는데
SOLID_MAX_GAP = 0.10                        # 끊긴 곳이 이보다 짧으면 실선 (점선은 0.14~0.22 m 씩 끊김, 실선은 0.05 이하)
HEADING_MAX = math.radians(42.0)            # 차로를 옮길 때 도로 방향에서 이 각도 넘게 틀지 않는다
WALL_MARGIN = 0.17                          # 노란 선과 차 가운데 사이 최소 거리 (차 반폭 0.11 + 여유)
WALL_RADIUS = 0.62                          # 나란해지는 데 쓰는 회전 반경 (실측 최소 0.50~0.53 에 여유)
WALL_AHEAD_MARGIN = 0.30                    # 노란 선이 '내가 그 선 자리를 넘는 지점'보다 이만큼 더 앞에서 시작하면 벽이 아니다 (갈림길 섬 테두리)
WALL_BESIDE_MAX = 0.90                      # 나란히 갈 때: 노란 선이 이보다 앞에서 시작하면 아직 옆에 없는 선
WALL_LAG_M = 0.08                           # 조향이 먹기 전에 가는 거리
ISLAND_INSIDE = 0.55                        # 도로 가운데에서 이 안쪽에 보이는 노란 선은 갈림길 섬 테두리로 본다
ISLAND_HALF = 0.36                          # 섬(가운데 차로 끝) 폭의 절반 + 여유
ISLAND_MIN_POINTS = 25
GLARE_MARGIN_PX = 45                        # 흰 덩어리 둘레로 같이 지우는 폭 [px = cm] (양쪽 22 cm)
FAR_U = 0.35                                # 이보다 먼 선은 BEV 가 벌어져 보여서 줄여 쓴다
FAR_SHRINK = 1.25

MAX_STEER = math.radians(25.0)
LOOKAHEAD = 0.55                            # 목표점: 카메라 앞 이 거리(도로 방향) [m]
LOOKAHEAD_CHANGE = 0.42                     # 차로를 옮기는 동안은 더 가까운 점을 보고 빨리 꺾는다
HEADING_TIGHT = math.radians(75.0)          # 급하게 옮길 때 허용하는 각도. 45도를 넘으면 차로 선이 안 보여 그동안은 odom 으로 간다
                                            # (규정: 정차 LIMO 접촉·차로 이탈은 1점 감점, 멈춰서 못 가면 미션 실패 -> 서는 쪽이 더 손해)
LOOKAHEAD_TIGHT = 0.30                      # 앞 장애물이 가까울 때(TIGHT_BELOW 안)는 더 급하게
TIGHT_BELOW = 1.00

# 라이다
LIDAR_MAX = 2.5
LANE_HALF = 0.27
ROAD_HALF = 0.83                            # 이보다 바깥의 점은 도로 밖 (벽, 사람, 다른 구간)

# 판단
BLOCK_DIST = 1.35                           # 내 차로 앞 장애물이 이보다 가까우면 옮길 차로를 찾는다
MIN_START = 0.60                            # 앞 장애물이 이보다 가까우면 옮기기 시작하지 않는다 (급하게 꺾어도 0.6 m 는 필요)
TRANSIT_CLEAR = 1.40                        # 두 칸 옮길 때 가로지르는 가운데 차로에 필요한 빈 거리
FAR_CLEAR = 2.20                            # 두 칸 옮길 때 도착 차로에 필요한 빈 거리
SIDE_CLEAR = 1.05                           # 옮길 차로는 옆~앞 이 거리까지 비어 있어야 한다
SIDE_BEHIND = -0.10                         # 옆 차로 확인을 시작하는 위치 (라이다보다 뒤 -). 지나친 장애물은 곧 시야(좌우 110도)에서 사라진다
SIDE_PASSED = 0.10                          # 옆 장애물의 끝이 라이다 앞 이 거리 안까지 왔고
SIDE_GAP = 0.18                             # 차 옆면과 이만큼 떨어져 있으면 지나친 것으로 본다 (꺾는 동안 차 뒤쪽이 닿지 않는다)
CAR_HALF_WIDTH = 0.11
GAIN_MIN = 0.45                             # 옮길 차로가 내 차로보다 이만큼은 더 비어 있어야 한다
# 경로 미리 굴려 보기 (path_clearance): 지금 자세에서 그 차로로 실제 조향 규칙대로 갔을 때 라이다 점과의 최소 여유
PATH_MARGIN = 0.03                          # 급할 때(앞이 URGENT_BELOW 안)는 이 여유만 나와도 간다
PATH_GOOD = 0.12                            # 급하지 않으면 이만큼 여유가 나올 때까지 기다린다 (조금 더 가면 더 넓은 길이 나오는 경우가 많다)
URGENT_BELOW = 9.9                          # 11/15 실차에서 맞춘 값: 앞 장애물 거리와 상관없이, 경로 여유가 PATH_MARGIN 만 나오면 바로 옮긴다 (원래 0.80)
ISLAND_URGENT = 2.50                        # 표지판 쪽으로 건널 때: 섬이 이 안이면 급한 것으로 본다
PATH_STEP = 0.03
PATH_MAX_M = 2.2
PATH_LAG_M = 0.08                           # 조향이 먹는 데 걸리는 거리 (실제보다 조금 느리게 잡는다)
PATH_STEER = math.radians(23.0)             # 미리 굴릴 때 쓰는 최대 조향 (실제 25도보다 작게)
CAR_BODY_CENTER = 0.10                      # 차체 가운데: 뒤 차축 앞 [m]
CAR_BODY_HALF_LEN = 0.16
PATH_ROAD_HALF = 0.80                       # 경로 중 차체가 이 밖(바깥 노란 선)으로 나가면 안 된다
MEMORY_BEHIND_M = 1.00                      # 지나친 라이다 점을 이 거리까지 기억한다 (라이다는 뒤를 못 본다)
CLOSE_GAIN = 0.15                           # 앞이 가까울 때(CLOSE_BELOW 안)는 옆 차로가 이만큼만 더 비어 있어도 옮긴다
CLOSE_BELOW = 0.90
BLIND_MAX = 1.6                             # 선 없이 odom 으로만 갈 수 있는 거리 (급하게 옮기는 동안 선이 안 보인다)
MEMORY_STEP_M = 0.03                        # 이만큼 갈 때마다 한 번 기억한다
REVERSE_MARGIN = 0.05                       # 후진할 때 기억한 점들과 차체 사이에 남길 여유
REVERSE_SEEN_MAX = 0.60                     # 기억한 점으로 확인하고 후진하는 최대 거리
REVERSE_RETRY = 0.40                        # 옮기다가 정면이 막혀 섰을 때 물러나는 거리 (같은 차로로 다시 꺾어 들어간다)
REVERSE_FORGET_M = 0.60                     # 후진한 뒤 이만큼 나아갔으면 후진 횟수를 다시 센다
ISLAND_NEAR_VIEW = 0.50                     # 섬 앞끝이 이보다 가까우면 카메라에 안 보인다 (옆 테두리만 보인다)
ISLAND_KEEP_M = 0.50                        # 섬이 잠깐 안 보여도 이 거리 동안은 기억한 위치를 쓴다
# 섬 앞에서 표지판 읽기 (11/15 실차: 오른쪽 차로에서 비스듬히 본 A 표지판을 B 로 두 번 확정 -> 옆 차로에서 읽은 값은 못 믿는다)
WAIT_START_MIN = 1.30                       # 섬이 이보다 가까우면 옆 차로에서 가운데로 들어가기 시작하지 않는다
WAIT_MARGIN = 0.10                          # 가운데에서 기다리는 동안 양쪽 차로로 가는 경로에 이만큼 여유가 남아 있어야 한다. 모자라면 그때 정한다
WAIT_SLOW_BELOW = 1.60                      # 섬이 이 안이면 기다리는 동안 천천히 간다 (표지판을 더 여러 번 읽는다)
ISLAND_POST_TOL = 0.20                      # 가운데 차로에서 라이다에 보이는 것이 섬(표지판 기둥)인지 장애물인지 가르는 여유
HOLD_FRAMES = 12                            # 막혀서 옮길 차로를 고른 뒤 이만큼(약 0.5초)은, 그 경로가 나오는 한 다른 차로로 다시 바꾸지 않는다
EXIT_HOLD = 0.90                            # 표지판 쪽이 아니거나 아직 모르면, 섬이 이 거리 안으로 올 때까지 3차선 주행을 끝내지 않는다
GUARD_HALF = 0.15                           # 정면 통로 반폭 (차 반폭 0.11 + 여유)
GUARD_DIST = 0.30                           # 정면 통로 안 이 거리(라이다 기준)에 물체가 있으면 위치 추정과 상관없이 선다
STOP_DIST = 0.38                            # 갈 곳이 없으면 장애물 앞 이 거리에서 정지
NO_OBSTACLE = LIDAR_MAX - 0.05              # 빈 거리가 이 이상이면 그 차로에 장애물이 안 보이는 것
GOAL_CLEAR_FRAMES = 2                       # 갈림길 방향 차로가 연속 이만큼 비어 보여야 미리 옮긴다
CROSS_FRAMES = 12                           # 두 칸 건너기: 가운데·도착 차로에 아무것도 안 보이는 상태가 연속 이만큼 (라이다 4바퀴)
                                            # 시뮬레이션: 섬 앞끝까지 1.9 m 부터 성공, 1.5~1.8 m 는 섬 앞에 갇힘 -> 2.45 m 안에 섬·기둥이 안 보일 때만
GOAL_TIE = 0.40                             # 두 차로의 빈 거리 차이가 이 안이면 갈림길 방향 쪽을 고른다
STUCK_FRAMES = 25                           # 정지 상태가 이만큼(약 1.2초) 이어지면 후진해서 다시 길을 찾는다
REVERSE_MIN = 0.25                          # 최소 후진 거리 [m]
REVERSE_MAX = 1.20                          # 최대 후진 거리 [m]
REVERSE_KEEP = 0.35                         # 지나온 길 중 이만큼은 남기고 후진 (라이다는 뒤를 못 본다)
REVERSE_GO_AHEAD = 0.90                     # 후진 중 옮길 길이 보여도, 앞 장애물과 이만큼 벌어질 때까지는 더 물러난다 (넉넉히 꺾으려고)
V_REVERSE = 0.12
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
        self._glare_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (GLARE_MARGIN_PX, GLARE_MARGIN_PX))
        self._rng = np.random.default_rng(0)
        self.yellow = self.white = None

    def _thin(self, mask, margin_kernel=None):
        # 15 cm 보다 굵은 덩어리(밝은 바닥, 상자)는 선이 아니다
        fat = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel), margin_kernel if margin_kernel is not None else self._kernel)
        return cv2.bitwise_and(mask, cv2.bitwise_not(fat))

    def masks(self, image, tune=None):
        tune = tune or {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (tune.get('h_min', 15), max(60, tune.get('s_min', 40)), max(70, tune.get('v_min', 60))),
                             (tune.get('h_max', 40), 255, 255))
        white = cv2.inRange(hsv, (0, 0, 150), (179, 60, 255))
        warp = lambda m: cv2.warpPerspective(m, self.i2b, (self.w, self.h), flags=cv2.INTER_NEAREST)  # noqa: E731
        # 흰색은 햇빛 받은 바닥(매트 밖)·반사광도 잡힌다. 큰 덩어리는 가장자리가 지저분하게 남아 가짜 선이 되므로
        # 덩어리 둘레 GLARE_MARGIN 까지 같이 지운다 (11/15 12:38 낮 주행: 매트 밖 밝은 바닥을 차로 선으로 착각)
        self.yellow, self.white = self._thin(warp(yellow)), self._thin(warp(white), self._glare_kernel)
        return self.yellow, self.white

    def fit(self, image, th_prior=None, tune=None):
        """반환: None 또는 dict(th, lines=[(u, n, yellow_ratio)], score)."""
        yellow, white = self.masks(image, tune)
        pts = []
        white_full = yellow_full = None
        for mask, is_yellow in ((yellow, 1.0), (white, 0.0)):
            py, px = np.nonzero(mask)
            if len(px) == 0:
                continue
            x = -XW + (px + 0.5) * RES
            f = F1 - (py + 0.5) * RES
            keep = f < FIT_FMAX
            pts.append(np.stack([x[keep], f[keep], np.full(int(keep.sum()), is_yellow)], 1))
            if not is_yellow:
                white_full = (x[keep], f[keep])
            else:
                yellow_full = (x[keep], f[keep])
        if not pts:
            return None
        P = np.concatenate(pts)
        if len(P) < 120:
            return None
        if len(P) > 3000:
            P = P[self._rng.choice(len(P), 3000, replace=False)]
        if th_prior is None:
            angles = np.arange(-math.radians(80), math.radians(80) + 1e-6, math.radians(2.0))
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
        # 끊김 없이 이어진 흰 선은 차로 점선이 아니다 (트랙 매트 가장자리의 흰 띠 등). 점선으로 착각하면
        # 위치가 한 차로씩 밀린다 (11/15 04:21 실차: 왼쪽 바깥의 흰 띠를 점선으로 보고 오른쪽 차로라고 착각)
        edges = []
        if white_full is not None and len(white_full[0]):
            uw = -white_full[0] * math.cos(th) - white_full[1] * math.sin(th)
            tw = -white_full[0] * math.sin(th) + white_full[1] * math.cos(th)
            kept = []
            for line in lines:
                if line[2] <= 0.3:
                    t = np.sort(tw[np.abs(uw - line[0]) < 0.04])
                    if t.size >= 10 and t[-1] - t[0] >= SOLID_MIN_SPAN and float(np.diff(t).max()) < SOLID_MAX_GAP:
                        edges.append((line[0], line[1]))
                        continue
                kept.append(line)
            lines = kept
        if not lines:
            return None
        lines = sorted(lines)
        short = set()                                  # 짧은 노란 조각 (갈림길 섬 꼭짓점 등): 바깥 선이 아니다
        ystart = {}                                    # 노란 선마다: 그 선이 실제로 시작하는 곳 (도로 방향 거리, 카메라 기준)
        if yellow_full is not None and len(yellow_full[0]):
            uy = -yellow_full[0] * math.cos(th) - yellow_full[1] * math.sin(th)
            ty = -yellow_full[0] * math.sin(th) + yellow_full[1] * math.cos(th)
            for line in lines:
                if line[2] >= 0.7:
                    t = ty[np.abs(uy - line[0]) < 0.04]
                    if t.size < 10 or float(t.max() - t.min()) < YELLOW_MIN_SPAN:
                        short.add(line[0])
                    if t.size >= 10:
                        ystart[line[0]] = float(np.percentile(t, 5))
        return dict(th=float(th), lines=lines, score=score / float(len(P)) ** 2, n=len(P),
                    yellow_xy=P[P[:, 2] > 0.5, :2], edges=sorted(edges), short=short, ystart=ystart)


def _shrink(u):
    if abs(u) <= FAR_U:
        return u
    return math.copysign(FAR_U + (abs(u) - FAR_U) / FAR_SHRINK, u)


def _match_score(c, lines, c_prior=None, skip=()):
    """c_prior 가 있으면, 그 위치에서 봤을 때 도로 안쪽(|위치| < ISLAND_INSIDE)에 나란히 있는 노란 선 두 줄은
    갈림길 섬 테두리로 본다: 바깥 노란 선 자리가 아니라 가운데 차로 양쪽 선 자리(+-0.27)에 맞춘다.
    (섬 테두리를 바깥 노란 선으로 착각해 위치가 반대편으로 넘어가는 것을 막는다 - 11/15 3번 주행)"""
    island = set()
    inside = []
    if c_prior is not None:
        inside = [u for u, n, ratio in lines if ratio >= 0.7 and abs(c_prior + _shrink(u)) < ISLAND_INSIDE]
        for a in inside:                 # 섬은 노란 선 두 줄이 차로 폭만큼 떨어져 있다. 한 줄뿐이면 바깥 선일 수 있어 그대로 쓴다
            if any(0.38 <= abs(a - b) <= 0.68 for b in inside):
                island.add(a)
    total = 0.0
    for u, n, yellow_ratio in lines:
        if u in skip:
            continue
        pos = c + _shrink(u)
        if u in island:
            expected = WHITE_POS             # 섬 테두리는 가운데 차로 양쪽 선 자리(+-0.27)에 있다
        elif u in inside:
            expected = YELLOW_POS + WHITE_POS    # 한 줄만 보이면 섬 테두리일 수도, 바깥 선일 수도 있다
        elif yellow_ratio >= 0.7:
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
        skip = obs.get('short', ())
        # 바깥쪽에 길게 이어진 흰 실선(멀리서 희게 보이는 바깥 노란 선, 또는 매트 가장자리 흰 띠)은
        # 도로 가장자리 근처에 있다 -> 그와 맞지 않는 위치는 후보에서 뺀다 (차로 하나씩 밀리는 착각 방지)
        for u, n in obs.get('edges', ()):
            if n < EDGE_MIN_POINTS or not 0.5 <= abs(u) <= 1.4:
                continue
            c_edge = math.copysign(EDGE_POS, u) - _shrink(u)
            ok = np.abs(candidates - c_edge) <= EDGE_TOL
            if ok.any():                            # 후보를 줄이기만 한다. 안 맞는다고 위치를 통째로 옮기지 않는다
                candidates = candidates[ok]         # (11/15 06:01 실차: 섬 옆에서 한 프레임에 한 차로씩 튀어 좌우로 흔들림)
        scores = np.array([_match_score(float(c), lines, prior, skip) for c in candidates])
        k = int(np.argmax(scores))
        if scores[k] < FIX_MIN_SCORE:
            return
        # 가장 잘 맞는 c 주변을 가중 평균
        near = np.abs(candidates - candidates[k]) <= 0.041
        c = float((candidates[near] * scores[near]).sum() / scores[near].sum())
        if prior is None:                           # 처음이거나 오래 못 봤다 -> 그대로 받는다
            self.c, self.th = c, obs['th']
        else:                                       # 따라가는 중 -> 조금씩만 고친다 (잘못 본 몇 프레임에 끌려가지 않게)
            step = TRACK_C_STEP + TRACK_C_STEP_BLIND * min(self.blind_m, 1.0)
            self.c += float(np.clip(0.6 * (c - self.c), -step, step))
            self.th += float(np.clip(0.7 * (obs['th'] - self.th), -TRACK_TH_STEP, TRACK_TH_STEP))
        self.blind_m = 0.0
        self.fixed = True


def acquire(obs):
    """선 검출 한 장으로 위치를 처음 잡는다 (이전 위치를 쓰지 않음). 반환: None 또는 (c, th).
    다른 차로 위치도 비슷하게 잘 맞으면(= 어느 차로인지 가릴 수 없으면) None."""
    if obs is None:
        return None
    candidates = np.arange(-0.80, 0.8001, 0.02)
    skip = obs.get('short', ())
    scores = np.array([_match_score(float(c), obs['lines'], None, skip) for c in candidates])
    k = int(np.argmax(scores))
    if scores[k] < FIX_MIN_SCORE:
        return None
    other = np.abs(candidates - candidates[k]) > 0.30
    if scores[k] < ACQUIRE_MARGIN * float(scores[other].max()):
        return None
    near = np.abs(candidates - candidates[k]) <= 0.041
    return float((candidates[near] * scores[near]).sum() / scores[near].sum()), float(obs['th'])


def front_gap(ranges, angle_min, angle_increment, range_min):
    """차 정면 통로(차 폭)에서 가장 가까운 물체까지 거리 [m] (라이다 기준). 위치 추정을 쓰지 않는다."""
    r = np.asarray(ranges, dtype=np.float32)
    a = angle_min + np.arange(r.size) * angle_increment
    ok = np.isfinite(r) & (r >= max(range_min, 0.05)) & (r <= LIDAR_MAX)
    x, y = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
    d = np.sort(x[(np.abs(y) <= GUARD_HALF) & (x > 0.0)])
    return float(d[1]) if d.size >= 2 else LIDAR_MAX          # 점 2개 이상일 때만 (잡음 1개는 무시)


def scan_points(ranges, angle_min, angle_increment, range_min, c, th):
    """라이다 한 바퀴 -> 도로 기준 점들 (라이다에서 도로 방향 거리, 도로 가운데 기준 옆 위치[왼쪽 +])."""
    r = np.asarray(ranges, dtype=np.float32)
    a = angle_min + np.arange(r.size) * angle_increment
    ok = np.isfinite(r) & (r >= max(range_min, 0.05)) & (r <= LIDAR_MAX * 1.2)
    x, y = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
    return x * math.cos(th) + y * math.sin(th), c + (-x * math.sin(th) + y * math.cos(th))


def path_clearance(along, lateral, c, th, target, lookahead, hmax, steer0=0.0):
    """지금 자세(c, th)에서 차로 target 으로, 실제와 같은 조향 규칙으로 미리 굴려 본 경로에서
    차체와 점들(along, lateral) 사이의 최소 거리 [m]. 차체가 바깥 노란 선 밖으로 나가면 -1.
    along 은 카메라(라이다) 위치 기준 도로 방향 거리."""
    psi = -th                                              # 차 방향 (도로 기준, 왼쪽 +)
    x = -CAMERA_FORWARD_OFFSET * math.cos(psi)             # 뒤 차축
    y = c - CAMERA_FORWARD_OFFSET * math.sin(psi)
    steer = float(np.clip(steer0, -PATH_STEER, PATH_STEER))
    poses = []
    settled = 0
    for _ in range(int(PATH_MAX_M / PATH_STEP)):
        cam_y = y + CAMERA_FORWARD_OFFSET * math.sin(psi)
        cmd = limit_heading(steer_to_lane(cam_y, -psi, target, lookahead), -psi, hmax)
        cmd = max(-PATH_STEER, min(PATH_STEER, cmd))
        steer += (cmd - steer) * min(1.0, PATH_STEP / PATH_LAG_M)
        if abs(steer) > 1e-4:
            radius = WHEELBASE / math.tan(abs(steer)) + TRACK / 2.0
            psi += math.copysign(PATH_STEP / radius, steer)
        x += PATH_STEP * math.cos(psi)
        y += PATH_STEP * math.sin(psi)
        poses.append((x, y, psi))
        if abs(cam_y - LANE_CENTER[target]) < 0.06 and abs(psi) < math.radians(10.0):
            settled += 1
            if settled >= 8:                               # 도착한 뒤 0.25 m 더 본다
                break
    P = np.asarray(poses)
    cs, sn = np.cos(P[:, 2]), np.sin(P[:, 2])
    # 차체 네 모서리가 도로 안에 있는지
    for fx in (CAR_BODY_CENTER - CAR_BODY_HALF_LEN, CAR_BODY_CENTER + CAR_BODY_HALF_LEN):
        for fy in (-CAR_HALF_WIDTH, CAR_HALF_WIDTH):
            if float(np.abs(P[:, 1] + fx * sn + fy * cs).max()) > PATH_ROAD_HALF:
                return -1.0
    if along is None or len(along) == 0:
        return LIDAR_MAX
    dx = along[None, :] - P[:, 0:1]
    dy = lateral[None, :] - P[:, 1:2]
    fx = dx * cs[:, None] + dy * sn[:, None] - CAR_BODY_CENTER
    fy = -dx * sn[:, None] + dy * cs[:, None]
    d = np.hypot(np.maximum(np.abs(fx) - CAR_BODY_HALF_LEN, 0.0), np.maximum(np.abs(fy) - CAR_HALF_WIDTH, 0.0))
    return float(d.min())


def reverse_room(along, lateral, c, th, max_d):
    """지금 방향 그대로 곧게 후진할 수 있는 거리 [m]: 차체가 점들(지나오며 기억한 것 포함)에 REVERSE_MARGIN 안으로
    다가가거나 바깥 노란 선을 넘기 직전까지. 차체 가운데보다 앞에 있는 점은 멀어지기만 하므로 보지 않는다."""
    psi = -th
    cs, sn = math.cos(psi), math.sin(psi)
    x = -CAMERA_FORWARD_OFFSET * cs
    y = c - CAMERA_FORWARD_OFFSET * sn
    if along is not None and len(along):
        fx0 = (along - x) * cs + (lateral - y) * sn - CAR_BODY_CENTER
        back = fx0 < 0.0
        pa, pl = along[back], lateral[back]
    else:
        pa = pl = np.zeros(0)
    d = 0.0
    while d + PATH_STEP <= max_d:
        nx, ny = x - (d + PATH_STEP) * cs, y - (d + PATH_STEP) * sn
        for fx in (CAR_BODY_CENTER - CAR_BODY_HALF_LEN, CAR_BODY_CENTER + CAR_BODY_HALF_LEN):
            for fy in (-CAR_HALF_WIDTH, CAR_HALF_WIDTH):
                if abs(ny + fx * sn + fy * cs) > PATH_ROAD_HALF:
                    return d
        if pa.size:
            fx = (pa - nx) * cs + (pl - ny) * sn - CAR_BODY_CENTER
            fy = -(pa - nx) * sn + (pl - ny) * cs
            gap = np.hypot(np.maximum(np.abs(fx) - CAR_BODY_HALF_LEN, 0.0), np.maximum(np.abs(fy) - CAR_HALF_WIDTH, 0.0))
            if float(gap.min()) < REVERSE_MARGIN:
                return d
        d += PATH_STEP
    return d


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
        # 옆~앞: 거의 다 지나친 장애물(끝이 라이다 바로 앞까지 왔고 차 옆면과 떨어져 있음)은 세지 않는다.
        # 예전에는 끝이 라이다 뒤 10 cm 로 빠질 때까지 기다려, 그 사이 앞 장애물이 가까워져 못 옮겼다 (11/15 실차)
        passed = (along < SIDE_PASSED) & (np.abs(lateral - c) - CAR_HALF_WIDTH >= SIDE_GAP)
        d = np.sort(along[in_lane & (along > SIDE_BEHIND) & (along < LIDAR_MAX) & ~passed])
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
        self.tight = False           # 급하게 옮기는 중 (가까운 목표점 + 큰 각도 허용)
        self._goal_how = None
        self._hold = 0               # 방금 고른 차로를 유지할 남은 프레임 수
        self._left_island = False    # 섬 앞 가운데에서 한쪽으로 이미 빠졌다 (다시 가운데로 돌아가지 않는다)
        self._goal_clear = 0         # 갈림길 방향 옆 차로가 연속으로 비어 보인 프레임 수

    @staticmethod
    def _can(reach, j, urgent):
        r = reach(j)
        return r is not None and (r[1] >= PATH_GOOD or urgent)

    def _at_island(self, lane, goal, island, reach):
        """가운데 차로(또는 가운데로 가는 중)에서 섬을 앞에 두고: 계속 갈지, 어느 쪽으로 빠질지. None 이면 기존 판단에 맡긴다."""
        r = {j: reach(j) for j in (0, 2)}
        clr = {j: (r[j][1] if r[j] is not None else -1.0) for j in (0, 2)}
        state = 'CHANGE' if self.changing else 'KEEP'
        speed = V_SLOW if island < WAIT_SLOW_BELOW else V_CHANGE
        pick = None
        if goal in (0, 2):
            if r[goal] is not None:
                pick, why = goal, '표지판 쪽'
            elif island > STOP_DIST + 0.05:
                # 표지판 쪽으로 가는 경로가 안 나온다 (너무 가까이 와서 읽음). 반대쪽으로 가지 않고 섬 앞에서 선다 ->
                # 후진해서 거리를 벌린 뒤 표지판 쪽으로 간다 (_recover)
                return Plan(1, V_SLOW, state, f'섬 {island:.1f} m: 표지판 쪽 경로가 안 나옴 -> 섬 앞에서 서서 물러남')
            else:
                return Plan(1, 0.0, 'STOP', '섬 앞: 표지판 쪽 경로가 안 나옴')
        elif min(clr.values()) >= WAIT_MARGIN:
            return Plan(1, speed, state, f'섬 {island:.1f} m: 가운데에서 표지판 읽는 중')
        else:
            # 한쪽이 곧 못 가게 된다 -> 표지판 없이 정한다: 경로 여유가 더 큰 쪽, 비슷하면 기본 쪽
            can = [j for j in (0, 2) if r[j] is not None]
            if can:
                pick = max(can, key=lambda j: clr[j] + (0.05 if j == DEFAULT_SIDE else 0.0))
                why = '표지판을 못 읽음'
        if pick is None:
            return None
        self.target, self.changing = pick, True
        self.tight = r[pick][0] == 'tight'
        self._hold = HOLD_FRAMES
        self._left_island = True
        return Plan(pick, V_CHANGE, 'CHANGE', f'섬 {island:.1f} m 앞 -> 차로 {pick} ({why})')

    def _escape(self, cur, c, ahead, side, goal, reach=None):
        """막힌 차로 cur 대신 갈 차로. 없으면 None.
        reach(j) 가 있으면 (라이다로 경로를 미리 굴려 본 결과: None = 못 감, 'normal' / 'tight' = 그 방식으로 감)
        거리 문턱값 대신 그것으로 갈 수 있는지 정한다."""
        if reach is not None:
            need = ahead[cur] + (CLOSE_GAIN if ahead[cur] < CLOSE_BELOW else GAIN_MIN)
            urgent = ahead[cur] < URGENT_BELOW
            near = [j for j in range(3) if j != cur and ahead[j] >= need and self._can(reach, j, urgent)]
            if not near:
                return None
            clear = [j for j in near if ahead[j] >= NO_OBSTACLE]
            if clear:
                near = clear
            else:
                best = max(ahead[j] for j in near)
                near = [j for j in near if ahead[j] >= best - GOAL_TIE]
            if goal is not None:         # 표지판 쪽 -> 더 빈 쪽 -> 가까운 차로
                near.sort(key=lambda j: (abs(j - goal), -ahead[j], abs(j - cur)))
            else:
                near.sort(key=lambda j: (-ahead[j], abs(j - cur), abs(j - DEFAULT_SIDE)))
            return near[0]
        in_cur = abs(c - LANE_CENTER[cur]) < LANE_HALF
        if in_cur and ahead[cur] < MIN_START:
            return None                      # 너무 가까워서 지금 꺾으면 앞 장애물을 스친다
        need = ahead[cur] + GAIN_MIN
        near = [j for j in (cur - 1, cur + 1)
                if 0 <= j <= 2 and side[j] >= SIDE_CLEAR and ahead[j] >= need]
        if near:
            # 더 멀리까지 빈 차로가 먼저. 갈림길 방향은 빈 거리가 비슷할 때(GOAL_TIE 이내)만 우선한다.
            # (11/15 실차: 갈림길 방향이라고 곧 막히는 차로로 들어가 멈춰 섬)
            clear = [j for j in near if ahead[j] >= NO_OBSTACLE]
            if clear:                    # 장애물이 아예 안 보이는 차로가 있으면 그중에서만 고른다
                near = clear             # (라이다를 LIDAR_MAX 까지만 봐서 '없음'과 '2.4 m 앞'이 비슷해 보인다)
            else:
                best = max(ahead[j] for j in near)
                near = [j for j in near if ahead[j] >= best - GOAL_TIE]
            if goal is not None:
                near.sort(key=lambda j: (abs(j - goal), -ahead[j]))
            else:                        # 표지판을 모름: 더 빈 쪽, 같으면 기본 쪽
                near.sort(key=lambda j: (-ahead[j], abs(j - DEFAULT_SIDE)))
            return near[0]
        far = 2 - cur
        if cur != 1 and side[1] >= TRANSIT_CLEAR and side[far] >= FAR_CLEAR \
                and ahead[far] >= max(ahead[cur], ahead[1]) + GAIN_MIN:
            return far                       # 가운데 차로를 가로질러 두 칸 이동
        return None

    def update(self, lane: int, c: float, ahead: List[float], side: List[float],
               goal: Optional[int] = None, have_scan: bool = True, may_center: bool = False, reach=None,
               island: Optional[float] = None, mid_free: bool = False) -> Plan:
        """have_scan=False (라이다가 아직 없음) 이면 차로를 바꾸는 결정을 하지 않는다.
        goal: 확정된 표지판 방향 (0 = A, 2 = B, None = 아직 모름). may_center: 모를 때 가운데 차로로 돌아가도 되는지.
        장애물 회피가 먼저다: 아래 '가고 싶은 차로' 판단은 지금 차로가 막히지 않았을 때만 한다."""
        if self.target is None:
            self.target = lane
        if not have_scan:
            self._goal_clear = 0
            return Plan(self.target, V_SLOW, 'CHANGE' if self.changing else 'KEEP', '라이다 없음 -> 서행')
        if self.changing and abs(c - LANE_CENTER[self.target]) <= ARRIVE_TOL:
            self.changing = self.tight = False
        cur = self.target
        if self._hold > 0:
            self._hold -= 1
        # 섬 앞에서 표지판 읽기: 가운데 차로를 막는 것이 섬뿐이면 (mid_free), 가운데에서 섬을 보며 가다가
        # 읽히면 그쪽으로, 끝까지 안 읽히면 양쪽 다 갈 수 있는 마지막 순간에 빠진다
        if reach is not None and island is not None and mid_free and not self._left_island:
            if cur == 1:
                plan = self._at_island(lane, goal, island, reach)
                if plan is not None:
                    return plan
            elif (goal is None and not self.changing and ahead[cur] >= BLOCK_DIST
                    and island >= WAIT_START_MIN):
                ok = reach(1) is not None
                self._goal_clear = self._goal_clear + 1 if ok else 0
                if self._goal_clear >= GOAL_CLEAR_FRAMES:
                    self._goal_clear = 0
                    self.target, self.changing = 1, True
                    self.tight = reach(1)[0] == 'tight'
                    return Plan(1, V_CHANGE, 'CHANGE', f'섬 {island:.1f} m -> 가운데에서 표지판 읽기')
                return Plan(cur, V_KEEP, 'KEEP', '')
        if ahead[cur] >= BLOCK_DIST:
            if self.changing:
                if reach is not None:    # 옮기는 중에도 경로를 다시 본다: 좁아졌으면 더 급하게, 그래도 안 되면 천천히
                    how = reach(cur)
                    if how is None:
                        return Plan(cur, V_SLOW, 'CHANGE', f'차로 {lane} -> {cur} (경로가 좁아 서행)')
                    if how[0] == 'tight':
                        self.tight = True
                return Plan(cur, V_CHANGE, 'CHANGE', f'차로 {lane} -> {cur}')
            # 가고 싶은 차로: 표지판이 확정됐으면 그쪽, 아직 모르면 가운데 (어느 쪽이 나와도 한 칸이면 되게)
            want = goal if goal is not None else (1 if may_center else cur)
            if want != cur:
                if goal is None:         # 가운데로 돌아가 기다리기: 세 차로 모두 앞에 아무것도 안 보일 때만
                    ok = min(min(side), min(ahead)) >= NO_OBSTACLE       # (돌아간 뒤 어느 쪽으로든 나갈 수 있어야 한다)
                elif reach is not None:  # 표지판 쪽 차로: 그 차로 앞이 비어 있고, 거기까지 가는 경로가 나오면 간다
                    how = reach(want)    # (섬 앞을 가로지르는 것도 경로만 나오면 된다. 섬은 점으로 넣어 둔다)
                    urgent = min(ahead[1], side[1]) < ISLAND_URGENT
                    ok = ahead[want] >= 1.0 and how is not None and (how[1] >= PATH_GOOD or urgent)
                    if ok:
                        self._goal_how = how[0]
                elif abs(want - cur) == 1:
                    ok = side[want] >= NO_OBSTACLE and ahead[want] >= NO_OBSTACLE
                else:                    # 반대편 바깥 차로 -> 가운데 차로를 가로질러 한 번에 건넌다 (섬이 충분히 멀 때만)
                    ok = (min(side[1], ahead[1]) >= NO_OBSTACLE
                          and side[want] >= NO_OBSTACLE and ahead[want] >= NO_OBSTACLE)
                self._goal_clear = self._goal_clear + 1 if ok else 0
                frames = GOAL_CLEAR_FRAMES if (abs(want - cur) == 1 or reach is not None) else CROSS_FRAMES
                if self._goal_clear >= frames:
                    self._goal_clear = 0
                    self.target, self.changing = want, True
                    self.tight = reach is not None and goal is not None and self._goal_how == 'tight'
                    why = '갈림길 방향' if goal is not None else '표지판 대기'
                    return Plan(want, V_CHANGE, 'CHANGE', f'{why} -> 차로 {want}')
            else:
                self._goal_clear = 0
            return Plan(cur, V_KEEP, 'KEEP', '')
        # cur 가 막힘 (가는 중이든 옮기는 중이든)
        if self.changing and reach is not None and self._hold > 0:
            how = reach(cur)             # 방금 고른 차로: 경로가 나오는 한 그대로 간다 (프레임마다 왔다 갔다 하면 어느 쪽으로도 못 간다)
            if how is not None:
                self.tight = self.tight or how[0] == 'tight'
                return Plan(cur, V_CHANGE, 'CHANGE', f'차로 {lane} -> {cur}')
        best = self._escape(cur, c, ahead, side, goal, reach)
        if best is not None:
            if reach is not None:
                self.tight = reach(best)[0] == 'tight'
                self._hold = HOLD_FRAMES
            else:
                self.tight = ahead[cur] < TIGHT_BELOW    # 앞이 가까운 채로 빠져나감 -> 급하게 꺾는다
            self.target, self.changing = best, True
            return Plan(best, V_CHANGE, 'CHANGE', f'차로 {cur} 앞 {ahead[cur]:.2f} m 막힘 -> 차로 {best}')
        if ahead[cur] < STOP_DIST and abs(c - LANE_CENTER[cur]) < LANE_HALF:
            return Plan(cur, 0.0, 'STOP', f'앞 {ahead[cur]:.2f} m, 옮길 차로 없음')
        if self.changing and reach is not None:
            # 옮기는 중: 도착할 차로 앞이 가까워 보여도 하던 방식(급하게 꺾기 포함)을 그대로 끝낸다.
            # 중간에 조향 방식을 바꾸면 반쯤 건넌 채로 곧게 펴져 버린다 (시뮬레이션: 섬 앞에 멈춤). 정면 정지는 따로 있다
            return Plan(cur, V_SLOW, 'CHANGE', f'차로 {lane} -> {cur} (앞 {ahead[cur]:.2f} m, 서행)')
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


def limit_heading(steer, th, hmax=None):
    """도로 방향에서 HEADING_MAX 넘게 틀어지지 않게 한다. 넘었으면 넘은 만큼 되돌리는 조향을 최소로 건다.
    (위치를 잘못 알고 계속 한쪽으로 꺾어 도로와 60도까지 틀어진 적이 있다 - 11/15 04:21)"""
    heading = -th                                       # 차가 도로보다 왼쪽을 보면 +
    over = abs(heading) - (HEADING_MAX if hmax is None else hmax)
    if over <= 0.0:
        return steer
    back = min(MAX_STEER, math.radians(4.0) + 2.0 * over)
    return min(steer, -back) if heading > 0.0 else max(steer, back)


def yellow_wall(lines, th_obs, ystart=None, only_side=0.0, skip=()):
    """노란 선은 벽: 3차선에서 넘어도 되는 것은 흰 점선뿐이다 (노란 선 = 바깥 선 또는 섬 테두리).
    노란 선에 닿기 전에 나란해질 수 없을 만큼 가까워지면, 위치 추정과 상관없이 그 선 반대쪽으로 꺾는다.
    ystart: 선마다 실제로 시작하는 곳 [m]. 내가 그 선 자리를 넘는 지점보다 한참 앞에서 시작하는 선(섬 테두리)은 벽이 아니다.
    반환: None 또는 (조향각[왼쪽 +], 설명)."""
    worst = None
    for u, n, ratio in lines:
        if ratio < 0.75 or n < 50 or abs(u) > 0.60 or u in skip:
            continue
        side = 1.0 if u > 0 else -1.0                 # 선이 차의 왼쪽(+) / 오른쪽(-)
        if only_side and side != only_side:
            continue
        approach = -side * th_obs                     # 그 선 쪽으로 향한 각 (+ 면 다가가는 중)
        start = ystart.get(u) if ystart else None
        if start is not None:
            if approach > math.radians(3.0):
                if start > abs(u) / math.tan(approach) + WALL_AHEAD_MARGIN:
                    continue                          # 선이 시작하기 전에 그 자리를 지나간다
            elif start > WALL_BESIDE_MAX:
                continue                              # 아직 옆에 없는 선
        need = WALL_MARGIN
        if approach > 0.0:
            need += WALL_RADIUS * (1.0 - math.cos(approach)) + WALL_LAG_M * math.sin(approach)
        gap = abs(u) - need
        if gap <= 0.0 and (worst is None or gap < worst[0]):
            worst = (gap, side, approach, u)
    if worst is None:
        return None
    gap, side, approach, u = worst
    angle = MAX_STEER if approach > math.radians(3.0) else math.radians(10.0)
    name = '왼쪽' if side > 0 else '오른쪽'
    return -side * angle, f'{name} 노란 선 {abs(u) * 100:.0f} cm (다가가는 각 {math.degrees(approach):+.0f}도) -> 반대로 꺾음'


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
    wall: Optional[float] = None     # 노란 선을 피하는 반사 조향 (걸렸을 때)
    wall_text: str = ''
    front: float = LIDAR_MAX         # 정면 통로에서 가장 가까운 물체 (라이다)


class ThreeLane:
    """한 프레임씩 호출. scan 은 가장 최근 라이다 (ranges, angle_min, angle_increment, range_min) 또는 None."""

    def __init__(self):
        self.fitter = LineFitter()
        self.road = RoadState()
        self.planner = Planner()
        self._pose = None
        self.travel = 0.0            # 이 모드에서 간 거리
        self._reset_recovery()

    def _reset_recovery(self):
        self.reversing = False       # 막혀서 후진하는 중
        self.reverse_done = 0.0      # 이번 후진에서 물러난 거리
        self.reverse_room = 0.0      # 이번 후진에서 물러나도 되는 거리
        self.reverse_count = 0
        self.gave_up = False         # 후진해도 길이 없었음 -> 그 자리에 선다
        self.dead_lane = None        # 막혔던 차로
        self.reverse_saw_way = False # 이번 후진 중에 옮길 길이 보인 적이 있다
        self.stuck = 0               # 정지 상태가 이어진 프레임 수
        self.lane_run = 1.0          # 지금 차로에 들어선 뒤(옮기기를 마친 뒤) 곧게 간 거리 (이만큼은 뒤가 비어 있다).
                                     # 처음에는 지나온 길 1.0 m 가 비어 있다고 본다
        self._last_target = None
        self._wall_side = 0.0        # 노란 선 벽에 걸려 있는 쪽 (+1 왼쪽 선, -1 오른쪽 선, 0 없음)
        self.s_road = 0.0            # 도로 방향으로 간 거리 (기억한 라이다 점의 위치 계산용)
        self._memory = []            # 지나친 라이다 점 [(도로 방향 절대 위치, 옆 위치)]
        self._steer_cmd = 0.0        # 직전 조향 명령
        self._mem_s = None           # 마지막으로 기억한 위치
        self._isl_s = None           # 기억한 섬 앞끝 위치 (도로 방향 절대 위치)
        self._isl_at = 0.0           # 섬을 마지막으로 본 위치
        self._points = None          # 이번 프레임의 점들 (along, lateral, c, th): 후진 여유 계산용
        self._retry = False          # 이번 후진은 옮기다가 정면이 막혀서 하는 것 (같은 차로로 다시 들어간다)
        self._rev_travel = 0.0       # 마지막 후진을 끝낸 위치

    def reset(self):
        self.road = RoadState()
        self.planner = Planner()
        self._pose = None
        self.travel = 0.0
        self._reset_recovery()

    def step(self, image, pose, scan, goal=None, tune=None) -> ThreeLaneOutput:
        self._advance(pose)
        # 선으로 맞춘 지 얼마 안 됐으면 그 방향 근처만, 아니면 넓게 찾는다 (odom 방향이 틀어졌을 수 있다)
        prior = self.road.th if (self.road.c is not None and self.road.blind_m < 0.3) else None
        obs = self.fitter.fit(image, prior, tune)
        return self._decide(obs, scan, goal)

    def look(self, image, tune=None):
        """이전 위치를 쓰지 않고 이 한 장으로만 본 결과: (c, th) 또는 None, 그리고 가까운 흰 점선이 보이는지."""
        obs = self.fitter.fit(image, None, tune)
        white = bool(obs) and any(ratio <= 0.3 and n >= 50 and abs(u) < 0.50 for u, n, ratio in obs['lines'])
        return acquire(obs), white

    def start(self, c, th, pose) -> None:
        """look() 으로 확인한 위치에서 주행을 시작한다."""
        self.reset()
        self.road.c, self.road.th, self.road.blind_m = float(c), float(th), 0.0
        self._pose = pose

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
        if self.reversing:                       # odom 위치 변화에는 부호가 없다 -> 후진 중이면 빼 준다
            self.reverse_done += ds
            ds = -ds
        self.travel += ds
        self.lane_run = max(0.0, self.lane_run + ds)
        self.s_road += ds * math.cos(self.road.th)
        self.road.predict(ds, dyaw)

    def _recover(self, plan, out, ahead=None) -> bool:
        """막혀서 선 상태가 이어지면 지나온 길로 조금 후진해 다시 길을 찾는다. True 면 후진 명령을 냈다."""
        if plan.target != self._last_target:
            if self._last_target is not None:
                self.lane_run = 0.0
            self._last_target = plan.target
        if self.gave_up:
            if plan.state == 'CHANGE' and plan.target != self.dead_lane:
                self.gave_up = False                 # 옮길 차로가 생겼다
                return False
            # 막힌 길: 후진해도 옮길 차로가 없었다. 같은 길로 다시 들어가지 않고 선다 (11/15 05:02 실차: 전진·후진 반복)
            out.speed, out.steer, out.state = 0.0, 0.0, 'STOP'
            out.reason = '막힌 길 (후진해도 옮길 차로 없음) -> 정지'
            return True
        if self.reversing:
            # 뒤로 물러나 앞이 비어 보이는 것은 '갈 수 있다'가 아니다. 다른 차로로 옮길 수 있어야 한다
            can_go = plan.state == 'CHANGE' and plan.target != self.dead_lane
            gap = ahead[self.dead_lane] if (ahead is not None and self.dead_lane is not None) else LIDAR_MAX
            if self._retry:              # 옮기다 선 경우: 정해 둔 만큼 물러난 뒤 같은 차로로 다시 꺾는다
                if self.reverse_done >= min(self.reverse_room, REVERSE_RETRY):
                    self.reversing, self.stuck, self._rev_travel = False, 0, self.travel
                    return False
                out.speed, out.steer, out.state = -V_REVERSE, 0.0, 'REVERSE'
                out.reason = f'옮기다 막힘 -> 후진 {self.reverse_done:.2f}/{min(self.reverse_room, REVERSE_RETRY):.2f} m'
                return True
            if can_go and self.reverse_done >= REVERSE_MIN and (
                    gap >= REVERSE_GO_AHEAD or self.reverse_done >= self.reverse_room):
                self.reversing, self.stuck, self._rev_travel = False, 0, self.travel
                return False
            if can_go:
                self.reverse_saw_way = True
            elif self.reverse_saw_way:
                # 길이 보였다가 다시 닫혔다 (방금 피한 옆 장애물 옆으로 되돌아옴). 더 물러나면 손해 ->
                # 여기서 후진을 끝내고, 다시 천천히 다가가다 길이 열리는 순간 나간다
                self.reversing, self.stuck, self._rev_travel = False, 0, self.travel
                return False
            if self.reverse_done >= self.reverse_room:
                self.reversing, self.stuck, self.gave_up = False, 0, True
                out.speed, out.steer, out.state = 0.0, 0.0, 'STOP'
                out.reason = '막힌 길 (후진해도 옮길 차로 없음) -> 정지'
                return True
            out.speed, out.steer, out.state = -V_REVERSE, 0.0, 'REVERSE'
            out.reason = f'막힘 -> 후진 {self.reverse_done:.2f}/{self.reverse_room:.2f} m'
            return True
        if plan.state != 'STOP':
            self.stuck = 0
            if self.reverse_count and self.travel - self._rev_travel > REVERSE_FORGET_M:
                self.reverse_count = 0               # 후진한 뒤 충분히 나아갔다 -> 다음에 막히면 다시 후진할 수 있다
            return False
        self.stuck += 1
        if self.stuck < STUCK_FRAMES:
            return False
        room = min(REVERSE_MAX, self.lane_run - REVERSE_KEEP)
        if self._points is not None:                 # 지나오며 기억한 점으로 확인한 만큼은 곧게 물러날 수 있다
            room = max(room, reverse_room(*self._points, REVERSE_SEEN_MAX))
        if self.reverse_count >= 2 or room < REVERSE_MIN:
            self.gave_up, self.dead_lane = True, plan.target
            return False
        self.reversing, self.reverse_done, self.reverse_room = True, 0.0, room
        self.reverse_saw_way = False
        self._retry = self.planner.changing          # 옮기는 중에 선 것이면 물러났다가 같은 차로로 다시 간다
        self.dead_lane = None if self._retry else plan.target
        self.reverse_count += 1
        out.speed, out.steer, out.state = -V_REVERSE, 0.0, 'REVERSE'
        out.reason = f'막힘 -> 후진 시작 (최대 {room:.2f} m)'
        return True

    def _decide(self, obs, scan, goal) -> ThreeLaneOutput:
        self.road.update(obs)
        out = ThreeLaneOutput(lines=obs['lines'] if obs else [])
        out.white_seen = bool(obs) and any(
            ratio <= 0.3 and n >= 50 and abs(u) < 0.50 for u, n, ratio in obs['lines'])
        # 노란 선 벽: 앞에서 시작하는 선(섬 테두리)은 그 자리를 먼저 지나가면 벽이 아니다. 다만 한 번 걸린 쪽은
        # 풀릴 때까지 시작 지점을 따지지 않는다 (코앞의 선은 카메라 사각 때문에 '앞에서 시작'으로 보인다)
        wall = None
        if obs:
            # 섬 앞에서 옆 차로로 빠져나가는 중이면, 나가는 쪽의 섬 테두리(도로 안쪽 노란 선)는 넘어야 하는 선이다.
            # 벽으로 보고 되돌리면 섬 한가운데로 들어간다 (시뮬레이션). 바깥 노란 선은 그대로 벽.
            cross = ()
            if self.planner.changing and self.road.c is not None and self.planner.target is not None:
                way = LANE_CENTER[self.planner.target] - self.road.c
                cross = tuple(u for u, n, ratio in obs['lines']
                              if ratio >= 0.75 and u * way > 0.0 and abs(self.road.c + _shrink(u)) < ISLAND_INSIDE)
            wall = yellow_wall(obs['lines'], obs['th'], obs.get('ystart'), skip=cross)
            if wall is None and self._wall_side:
                wall = yellow_wall(obs['lines'], obs['th'], None, self._wall_side, skip=cross)
            self._wall_side = (1.0 if wall[0] < 0 else -1.0) if wall is not None else 0.0
        if wall is not None:
            out.wall, out.wall_text = wall
        if scan is not None:
            out.front = front_gap(scan[0], scan[1], scan[2], scan[3])
        if self.road.c is None or self.road.blind_m > BLIND_MAX:
            out.state, out.reason = 'LOST', '차로 선으로 위치를 못 잡음'
            return out
        lane = self.road.lane
        c, th = self.road.c, self.road.th
        if scan is not None:
            ahead, side = lane_free(scan[0], scan[1], scan[2], scan[3], c, th)
        else:
            ahead, side = [LIDAR_MAX] * 3, [LIDAR_MAX] * 3
        island = island_distance(obs.get('yellow_xy') if obs else None, c, th)
        # 섬 앞끝이 카메라 밑으로 들어가면 옆 테두리의 '보이는 가장 가까운 곳'이 섬까지 거리로 나온다 (늘 0.4~0.5 m).
        # 앞끝 위치를 기억해 두고, 가까이 온 뒤에는 그 위치를 쓴다. 잠깐 안 보일 때도 마찬가지
        if island is not None:
            pos = self.s_road + island
            if self._isl_s is None or pos < self._isl_s or self._isl_s - self.s_road > ISLAND_NEAR_VIEW:
                self._isl_s = pos
            self._isl_at = self.s_road
            island = self._isl_s - self.s_road
        elif self._isl_s is not None and self.s_road - self._isl_at < ISLAND_KEEP_M:
            island = self._isl_s - self.s_road
        else:
            self._isl_s = None
        out.island = island
        mid_free = island is not None and min(ahead[1], side[1]) >= island - ISLAND_POST_TOL    # 가운데 차로에 섬 말고는 없다
        if island is not None:                   # 가운데 차로는 섬에서 끝난다 -> 장애물로 취급
            ahead[1], side[1] = min(ahead[1], island), min(side[1], island)
        reach = None
        self._points = None
        if scan is not None:
            pa, pl = scan_points(scan[0], scan[1], scan[2], scan[3], c, th)
            # 라이다는 뒤를 못 본다: 방금 지나친 점들을 기억해 두었다가 차 뒤쪽이 닿지 않는지 볼 때 쓴다
            self._memory = [(a, b) for a, b in self._memory if self.s_road - float(a.max()) < MEMORY_BEHIND_M][-40:]
            behind = [(a - self.s_road, b) for a, b in self._memory]
            keep = (pa > -0.4) & (pa < 0.6) & (np.abs(pl) <= ROAD_HALF + 0.1)
            if keep.any() and (self._mem_s is None or abs(self.s_road - self._mem_s) >= MEMORY_STEP_M):
                self._mem_s = self.s_road            # 거리 기준으로 기억 (프레임 기준이면 느리게 갈 때 금방 잊는다)
                self._memory.append((pa[keep] + self.s_road, pl[keep]))
            parts_a, parts_l = [pa], [pl]
            for a, b in behind:
                old = a < 0.05                   # 지금 라이다가 볼 수 없는 뒤쪽 것만
                if old.any():
                    parts_a.append(a[old]); parts_l.append(b[old])
            if island is not None:               # 섬: 앞끝과 양옆 테두리를 점으로
                edge = np.linspace(-ISLAND_HALF + 0.09, ISLAND_HALF - 0.09, 9)
                parts_a.append(np.full(9, island)); parts_l.append(edge)
                for k in np.arange(0.15, 1.51, 0.15):
                    parts_a.append(np.full(2, island + k)); parts_l.append(np.array([edge[0], edge[-1]]))
            all_a, all_l = np.concatenate(parts_a), np.concatenate(parts_l)
            near = (all_a > -1.0) & (all_a < PATH_MAX_M + 0.4) & (np.abs(all_l) < ROAD_HALF + 0.2)
            all_a, all_l = all_a[near], all_l[near]         # 도로 밖(벽, 사람)과 먼 점은 경로 계산에서 뺀다 (계산 시간)
            self._points = (all_a, all_l, c, th)
            cache = {}

            def reach(j, _a=all_a, _l=all_l, _c=c, _th=th, _cache=cache):
                """차로 j 로 가는 경로: None(못 감) 또는 (방식, 최소 여유). 여유가 넉넉하면 보통 방식, 아니면 더 넓은 쪽."""
                if j not in _cache:
                    best = None
                    for name, lookahead, hmax in (('normal', LOOKAHEAD_CHANGE, HEADING_MAX),
                                                  ('tight', LOOKAHEAD_TIGHT, HEADING_TIGHT)):
                        clr = path_clearance(_a, _l, _c, _th, j, lookahead, hmax, self._steer_cmd)
                        if clr >= PATH_MARGIN and (best is None or clr > best[1]):
                            best = (name, clr)
                        if clr >= PATH_GOOD:
                            break
                    _cache[j] = best
                return _cache[j]
        plan = self.planner.update(lane, c, ahead, side, goal, have_scan=scan is not None,
                                   may_center=goal is None, reach=reach, island=island, mid_free=mid_free)
        if self.planner.changing and not self.reversing:
            self.lane_run = 0.0                  # 옮기는 중에 지나온 길은 곧지 않다 -> 똑바로 후진하면 방금 피한 장애물에 닿는다
        if out.front < GUARD_DIST and plan.speed > 0.0:
            # 위치를 잘못 알아 '빈 차로'라고 믿고 장애물로 가는 것을 막는다 (11/15 06:01 실차: 장애물을 밀고 지나감)
            plan = Plan(plan.target, 0.0, 'STOP', f'정면 {out.front:.2f} m 에 물체 -> 정지')
        out.ok = True
        out.speed, out.state, out.reason = plan.speed, plan.state, plan.reason
        tight = plan.state == 'CHANGE' and self.planner.tight
        if plan.state != 'CHANGE':
            lookahead = LOOKAHEAD
        else:
            lookahead = LOOKAHEAD_TIGHT if tight else LOOKAHEAD_CHANGE
        out.steer = steer_to_lane(self.road.c, self.road.th, plan.target, lookahead)
        out.steer = limit_heading(out.steer, self.road.th, HEADING_TIGHT if tight else None)
        self._steer_cmd = out.steer
        if self._recover(plan, out, ahead):
            out.wall = None                      # 후진 중에는 조향을 풀고 곧게 물러난다
        elif wall is not None:
            out.steer = wall[0]
            out.reason = (out.reason + ' | ' if out.reason else '') + out.wall_text
        out.lane, out.target, out.c, out.th, out.ahead = lane, plan.target, self.road.c, self.road.th, ahead
        if scan is None:
            out.reason = (out.reason + ' 라이다 없음').strip()
        return out


# ---------------------------------------------------------------------------
# simple_lane.py 와 잇는 부분: 언제 3차선 주행을 켜고 끌지
# ---------------------------------------------------------------------------
SIGN_MIN_H = 33.0          # 표지판 상자 높이 [px]. 이보다 작게(= 약 1.7 m 보다 멀리) 보일 때 읽은 값은 쓰지 않는다
                           # (27 로 두었더니 실차에서 B 표지판을 29 px 에서 A 로 먼저 확정했다가 바꿈 - 11/15)
SIGN_CENTER_HALF = 0.30    # 3차선 주행 중에는 도로 가운데에서 이 안에 있을 때 읽은 값만 쓴다 (옆 차로에서 비스듬히 보면 틀린다). 끄려면 9.9
SIGN_WINDOW = 5            # 최근 이만큼 중
SIGN_NEED = 4              # 이만큼이 같은 쪽이면 확정
ARM_AFTER_M = 1.4          # M7 시작(횡단보도 출발) 후 이만큼 간 뒤부터 3차선 선을 찾는다 (그 전은 코너)
ENTRY_FRAMES = 3           # 가까운 흰 점선이 연속 이만큼 보이면 3차선 주행 시작
EXIT_FRAMES = 4            # 바깥 차로에서 양쪽 노란 선(갈림길 차로)이 연속 이만큼 보이면 기본 주행으로 넘김
DEFAULT_SIDE = 2           # 표지판을 끝까지 못 읽었고 양쪽이 똑같이 비어 있을 때 갈 쪽 (2 = 오른쪽 = B)
MAX_ON_M = 6.5             # 3차선 주행을 이 거리 넘게 하고 있으면 끝낸다 (도면상 점선 구간 약 3.7 m)
LOST_FRAMES = 30           # 위치를 못 잡은 채 이만큼 지나면 끝낸다
SIDE_TOL = 0.13


class SignVote:
    """갈림길 표지판(A/B) 판정. 멀리서는 모델이 표지판과 상관없이 A 라고 답한다 (11/15 측정: 실제 B 인데
    3.2 m 밖에서 정답 5 %, 2.1~1.7 m 에서 86 %, 1.4 m 안에서 100 %). 그래서 작게 보일 때 읽은 값은 버리고,
    최근 SIGN_WINDOW 개 중 SIGN_NEED 개가 같은 쪽일 때만 확정한다. 확정 뒤에도 같은 규칙으로만 바뀐다."""

    def __init__(self):
        self.recent = deque(maxlen=SIGN_WINDOW)
        self.goal: Optional[int] = None

    def reset(self) -> None:
        self.recent.clear()
        self.goal = None

    def add(self, label: str, height: float) -> Optional[int]:
        """label: 'a' 또는 'b', height: 상자 높이 [px]. 이번에 새로 확정되면 차로 번호(0 = A, 2 = B), 아니면 None."""
        if label not in ('a', 'b') or height < SIGN_MIN_H:
            return None
        self.recent.append(label)
        for name, lane in (('a', 0), ('b', 2)):
            if self.recent.count(name) >= SIGN_NEED and self.goal != lane:
                self.goal = lane
                return lane
        return None


class Lane3Supervisor:
    """mode: off(안 씀) -> armed(3차선 선을 기다림) -> on(3차선 주행) -> done(기본 주행으로 넘김)."""

    def __init__(self, forced=False, goal=None):
        self.tl = ThreeLane()
        self.forced = forced
        self.mode = 'armed' if forced else 'off'
        self.sign_goal = goal             # 확정된 갈림길 방향: 0 = A(왼쪽), 2 = B(오른쪽), None = 아직 모름
        self.fixed_goal = goal is not None    # 실행할 때 goal_a / goal_b 로 정해 둠 -> 표지판을 보지 않는다
        self.vote = SignVote()
        self.arm_travel = 0.0
        self._pose = None
        self._white = self._both = self._lost = 0
        self._seen = None                 # 시작 전 직전 프레임에서 본 (c, th, pose)
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
                self._seen = None
                self.vote.reset()             # 3차선에 들어오기 전에 읽은 것은 버린다
                if not self.fixed_goal:
                    self.sign_goal = None
        elif name != 'M2':
            self.mode = 'off'

    def set_sign(self, event):
        """갈림길 방향을 직접 정한다 (시험용). 주행 중에는 see_signs 로 판정한다."""
        if event == 'sign_a':
            self.sign_goal = 0
        elif event == 'sign_b':
            self.sign_goal = 2

    def see_signs(self, detections):
        """YOLO 한 프레임의 결과 [{'label', 'confidence', 'bbox': [x1, y1, x2, y2]}, ...] 로 표지판을 판정한다."""
        if self.fixed_goal or self.mode not in ('armed', 'on'):
            return
        signs = [d for d in detections if d.get('label') in ('a', 'b') and len(d.get('bbox', ())) == 4]
        if not signs:
            return
        if self.mode == 'on' and not (self.last is not None and self.last.ok and abs(self.last.c) <= SIGN_CENTER_HALF):
            return                                # 옆 차로에 있는 동안 읽은 값은 버린다
        best = max(signs, key=lambda d: d.get('confidence', 0.0))
        height = float(best['bbox'][3]) - float(best['bbox'][1])
        decided = self.vote.add(best['label'], height)
        if decided is not None:
            self.sign_goal = decided
            self.message = f"[3차선] 표지판 {'A(왼쪽)' if decided == 0 else 'B(오른쪽)'} 확정 (상자 높이 {height:.0f} px)"

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
            # 시작 조건: 한 장씩 따로 본 위치가 (1) 어느 차로인지 분명하고 (2) 연속 ENTRY_FRAMES 장이 서로 맞아야 한다.
            # (11/15 12:38 낮 주행: 반사광을 선으로 본 첫 장에서 시작해 도로 방향을 45도 틀리게 알고 왼쪽으로 나감)
            seen, white = self.tl.look(image, tune)
            if seen is None or not white or pose is None:
                self._white, self._seen = 0, None
                return None
            if self._seen is not None:
                c0, th0, pose0 = self._seen
                ds = math.hypot(pose[0] - pose0[0], pose[1] - pose0[1])
                th_exp = th0 - (pose[2] - pose0[2])
                same = (abs(seen[1] - th_exp) <= ACQUIRE_TH_TOL
                        and abs(seen[0] - (c0 - ds * math.sin(th0))) <= ACQUIRE_C_TOL)
                self._white = self._white + 1 if same else 1
            else:
                self._white = 1
            self._seen = (seen[0], seen[1], pose)
            if self._white < ENTRY_FRAMES:
                return None
            self.mode = 'on'
            self.tl.start(seen[0], seen[1], pose)
            self._both = self._lost = 0
            self.steer = 0.0
            self.message = (f'[3차선] 시작: 차로 {self.tl.road.lane} (도로 가운데에서 {seen[0] * 100:+.0f} cm, '
                            f'도로 방향 {math.degrees(seen[1]):+.0f}도)')
        goal = self.sign_goal                 # 확정 전에는 None -> 가운데 차로에서 기다린다 (Planner.update)
        out = self.tl.step(image, pose, scan, goal, tune)
        self.last = out
        if not out.ok:
            self._lost += 1
            if self._lost >= LOST_FRAMES:
                self.mode, self.message = 'done', '[3차선] 끝: 차로 선을 못 찾음 -> 기본 주행'
                return None
            if out.front < GUARD_DIST:
                return dict(v=0.0, steer=self.steer, state='LANE3',
                            reason=f'3차선: 위치 모름, 정면 {out.front:.2f} m 에 물체 -> 정지', note='L3 lost guard')
            if out.wall is not None:
                self.steer = out.wall
                return dict(v=V_SLOW, steer=self.steer, state='LANE3', reason='3차선: 위치 모름 | ' + out.wall_text,
                            note='L3 lost wall')
            return dict(v=V_SLOW, steer=self.steer, state='LANE3', reason='3차선: 선을 못 찾아 직전 조향 유지',
                        note='L3 lost')
        self._lost = 0
        on_side = (out.target in (0, 2) and out.state == 'KEEP'
                   and abs(out.c - LANE_CENTER[out.target]) < SIDE_TOL)
        # 갈림길 섬: 바깥 차로에서 도로 안쪽 이웃 선이 흰 점선이 아니라 노란 선(섬 테두리)으로 보인다
        inward = 1.0 if out.target == 2 else -1.0
        island = on_side and any(ratio >= 0.7 and n >= 40 and 0.12 < inward * u < 0.50
                                 for u, n, ratio in out.lines)
        # 표지판 쪽이 아닌 차로에 있거나 아직 표지판을 모르면, 섬 앞에서 건널 기회가 남아 있는 동안은 끝내지 않는다
        # (11/15 실차: 섬 1.5 m 앞에서 끝내 버려 그 뒤에 읽은 표지판을 못 씀)
        hold = ((self.sign_goal is None or out.target != self.sign_goal)
                and out.island is not None and out.island > EXIT_HOLD)
        self._both = self._both + 1 if (on_side and (both_yellow or island) and not hold) else 0
        if self._both >= EXIT_FRAMES:
            self.mode = 'done'
            self.message = f"[3차선] 끝: 갈림길 {'A(왼쪽)' if out.target == 0 else 'B(오른쪽)'} 차로 진입 -> 기본 주행"
            return None
        if self.tl.travel >= MAX_ON_M:
            self.mode, self.message = 'done', '[3차선] 끝: 최대 거리 -> 기본 주행'
            return None
        if out.wall is not None:
            self.steer = out.steer                # 벽 반사는 늦추지 않는다
        else:
            self.steer += 0.6 * (out.steer - self.steer)
        goal_text = '-' if goal is None else ('A' if goal == 0 else ('B' if goal == 2 else str(goal)))
        free = '/'.join(f'{a:.1f}' for a in out.ahead)
        reason = (f"3차선 {out.state} 차로 {out.lane}->{out.target} 빈거리 {free} 목표 {goal_text}"
                  + (f' | {out.reason}' if out.reason else ''))
        note = (f"L3 {out.state} lane{out.lane} tgt{out.target} c{out.c:+.2f} th{math.degrees(out.th):+.0f} "
                f"free{free} goal{goal_text} s{self.tl.travel:.2f}"
                + ('' if out.island is None else f' isl{out.island:.1f}') + ('' if self.tl.road.fixed else ' nofix')
                + ('' if out.wall is None else ' WALL'))
        return dict(v=out.speed, steer=self.steer, state='LANE3', reason=reason, note=note)
