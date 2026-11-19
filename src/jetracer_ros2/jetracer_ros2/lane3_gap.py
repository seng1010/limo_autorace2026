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

버전 claude-v7 (11/18): 실차에서 확인된 11/17 버전(LIMO 의 ~/lane3_gap_ok_1117.py) + 송도·대회장 bag 에서 나온 문제 수정.
  다른 팀원(다른 도구)의 lane3 수정과 섞지 않은 독립 버전. 터미널 "[3차선G] claude-v7 시작" 으로 구분한다.
  바뀐 것: 표지판 오인 차단(SIGN_EST_MAX_H, SIGN_GATE_*) / 흰 점선으로 방향(THW_*) / 입구 사선에서 안 섬(ENTRY_SOFT_M) /
          후진하면서 차 펴기(REV_STEER_*) / 뒤에 둔 차 기억(MEM_KEEP) / 최대 거리는 앞으로 간 거리로 /
          끝까지 막히면 선은 무시하고 장애물만 피해 빠져나감(DEAD_*) / 방지턱 구간 코너 조향 끔(M5_LATCH_OFF) /
          기본 주행 코너: 바깥 선이 정면에서 사라질 때까지 고정 조향 유지(CORNER_HOLD, HOLD_*) /
          기본 주행 직선: 한쪽 선에 붙는 것 고침(CENTER_FIX, CENTER_*)

터미널 줄:  3차선G <상태> 목표 +0.32 m, 갈 수 있는 거리 1.8 (좌 2.2 중 0.9 우 0.4) 표지판 A 섬 1.6 m | 사유
"""

import math
import sys
import time
import types
from collections import deque

import cv2
import numpy as np

VERSION = 'claude-v7'

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
# 켜고 끄기
ARM_AFTER_M = 1.4        # 매니저: M7 시작 후 이만큼 간 뒤부터 (그 전은 코너)
ENTRY_FRAMES = 3         # 흰 점선이 연속 이만큼 보이면 시작
ENTRY_WAIT_M = 1.2       # 그래도 못 보면 ARM_AFTER_M + 이 거리에서 그냥 시작
# --- 11/18: 송도 bag(1117_053209)과 대회장 bag(1117_194426)으로 넣은 것 ---
SIGN_EST_MAX_H = 60.0    # 섬 위치는 상자 높이가 이 값 이하(약 0.9 m 보다 멀리)일 때 본 것으로만 처음 잡는다.
                         #   대회장: 장애물 LIMO 를 높이 128 px 짜리 'a' 로 3번 연속 잘못 봐서 0.3 m 앞에 섬을 세우고 A 로 확정했다.
                         #   진짜 표지판은 멀리서 작게 보이기 시작해 커진다. 갑자기 크게 나타나는 것은 표지판이 아니다.
SIGN_GATE_DEG = 15.0     # 섬이 잡힌 뒤에는 그 방향에서 이 각도 안에 보이는 상자만 표지판으로 쓴다 (판정 투표 포함).
SIGN_GATE_BIG_M = 1.5    #   큰 상자(SIGN_EST_MAX_H 초과)는 섬이 이 거리 안일 때만. (거리로는 안 거른다: 섬 거리는 앞차에 가려 1 m 씩 틀릴 수 있다)
ENTRY_SOFT_M = 1.0       # 시작 후 이 거리까지는: 노란 선 때문에만 막혔고 앞에 장애물이 없으면 서지 않고 기본 주행 명령을 그대로 쓴다
                         #   (입구의 45도 사선을 벽으로 보고 후진을 반복했다 - 송도·대회장 둘 다)
REV_STEER_GAIN = 1.2     # 후진할 때 차를 도로와 나란히 펴는 조향 (도로 기준 방향 x 이 값). 곧게만 후진하면 같은 자리로 되돌아왔다
REV_STEER_MAX = math.radians(20.0)
DEAD_MODE = False        # 11/20 끔: 아래 '선 무시' 탈출을 쓰지 않는다. 3차선의 노란 선은 전부 바깥 경계라 무시하면 그대로 차로 밖으로 나갔다
                         #   (11/19 19:37: 왼쪽 앞에서 막힌 뒤 오른쪽 노란 선을 넘어 나감). 끄면 후진을 다 쓴 뒤 그 자리에 서 있는다. True 면 전과 같다
DEAD_SEC = 3.0           # 후진을 다 쓰고도(또는 뒤가 막혀 후진을 못 하고) 이만큼 서 있으면: 노란 선·가상 벽은 무시하고
DEAD_M_OBS = 0.02        #   장애물만 이 여유로 피해 빠져나간다 (선 -1점이 그 자리에 계속 서 있는 것보다 낫다)
DEAD_RUN_M = 0.60        #   그렇게 이 거리만큼 나아가면 다시 원래 규칙
# --- 코너 (기본 주행의 코너 고정 조향을 더 오래 잡아 준다. simple_lane.py 는 고치지 않고 여기서 덧씌운다) ---
# 대회장 bag(자이로로 잼): 코너 고정 조향은 odom 으로 turn_deg(60도) 돌면 풀리는데 실제로는 50도쯤이다 (odom 이 1.2배 더 돈다).
#   풀린 순간에도 바깥 선이 정면 0.6~1.0 m 에서 38~48도 기울어 가로지르고 있었고, 그 뒤 차선 추종은 10도만 꺾어서
#   남은 30~40도를 도는 데 7초가 걸렸다 = 코너 바깥으로 밀림. 송도 bag 도 같다.
# -> turn_deg 만큼 돈 뒤에도 '바깥 선이 아직 정면을 기울어 가로지르는 동안'은 고정 조향을 유지한다.
CORNER_HOLD = True       # False 면 끔 (기본 주행 원래대로)
HOLD_TILT = math.radians(17.0)    # 선 기울기가 이 이상 남아 있고
HOLD_CROSS = 1.20                 #   바깥 선이 정면 이 거리 [m] 안에서 가로지르면 계속 꺾는다. 코너에서 아직 밀리면 HOLD_TILT ↓, 안쪽으로 말리면 ↑
HOLD_EXTRA = math.radians(35.0)   # 그래도 turn_deg 에 odom 으로 이만큼 더 돌면 푼다 (실제 약 29도)
# --- 직선에서 한쪽 선에 붙는 것 (기본 주행에 덧씌움) ---
# 대회장 bag 출발 직선: 왼쪽 선만 보고 가는데, 기본 주행은 차로 폭을 0.43 m(연습 트랙 값)로 알고 있어 선에서 21.5 cm 를 가운데로 본다
#   (대회장 차로는 선 가운데끼리 0.52 m). 게다가 자기가 4~5 cm 치우친 것을 알면서도 조향을 1도만 내고 그대로 간다
#   (조향 1도는 차가 실제로 안 움직이는 크기) -> 실제로는 가운데에서 8~9 cm 왼쪽으로 달렸다.
CENTER_FIX = True        # False 면 끔 (기본 주행 원래대로)
CENTER_SHIFT = 0.045     # 한쪽 선만 보고 갈 때 그 선에서 이만큼 더 떨어져서 간다 [m] ((0.52 - 0.43) / 2). 그래도 보이는 선 쪽으로 붙으면 ↑ (0.07), 반대쪽으로 가면 ↓ (0.02)
CENTER_KI = 0.12         # 직선에서 치우침이 남아 있으면 조향을 조금씩 더한다 [rad/(m·s)]. 좌우로 흔들리면 ↓ (0.06), 가운데로 너무 늦게 오면 ↑
CENTER_I_MAX = math.radians(2.5)     # 그렇게 더하는 조향의 한도
M5_LATCH_OFF = (0.5, 5.6)    # M5 시작 후 이 거리 [m] 구간에서는 기본 주행의 코너 고정 조향을 끈다 (방지턱)
                             #   대회장: 방지턱의 노란 줄무늬에 올라선 순간 최대 조향이 4초 걸려 차로를 벗어났다 (출발 후 3.8 m).
                             #   첫 코너는 출발 후 약 7.4 m 라 이 구간에는 코너가 없다. 방지턱을 옮기면 이 값을 고칠 것
MAX_ON_M = 9.0           # 시작한 곳에서 앞으로 이만큼 넘게 나아가면 끝낸다 (후진으로 오간 거리는 세지 않는다)
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
MEM_STEP, MEM_KEEP = 0.05, 40          # 지나친 점 기억: 앞으로 5 cm 갈 때마다, 2 m 치 (후진으로 오간 거리는 세지 않는다 - 세면 뒤에 둔 차를 잊고 후진한다)
TH_GAIN, TH_ACCEPT = 0.30, math.radians(12.0)     # 선 맞추기(노란 선 포함)로 본 방향: 가끔 크게 튄다 -> 좁게만 받는다
THW_GAIN, THW_ACCEPT = 0.40, math.radians(25.0)   # 흰 점선 조각으로 본 방향: 사선·섬에 안 흔들린다 -> 우선
HINT_COST = 0.10                       # 차로 가운데(보이는 선으로 짐작)가 아닌 줄에 붙는 벌점 [m]
KEEP_BONUS = 0.20                      # 가던 줄을 유지하는 쪽으로 주는 덤 [m]
SIDE_BIAS = 0.04                       # 똑같으면 표지판 쪽(모르면 DEFAULT_SIDE)
LANE_HALF = 0.2675


# --- 흰 점선 (11/20). three_lane.py 는 고치지 않고 여기서 덧씌운다 ---
# three_lane.LineFitter 는 흰색을 V>=150 으로만 본다. 11/19 저녁 경기장(노출 350~400)의 3차선 입구 점선은 V 100~150 이라
#   거의 안 잡혔다 (V>=150 / >=100 인 픽셀: 341/2127, 728/3034, 856/2834, 154/2125. 바닥은 V 15~19).
#   -> 점선을 못 봐 3차선 로직이 늦게 켜지고(그 사이 기본 주행은 '선 없음' 정지), 도로 방향을 0 으로 잡고 시작해 같은 자리에서 막혔다.
# 100 으로 낮추면 흰 LIMO 차체·사람 다리(흰 바지)도 잡힌다 (19:03 bag: 방향이 12도 넘게 튄 프레임 18 %)
#   -> 라이다에 잡힌 물체보다 앞쪽 바닥의 흰색만 쓴다 (bag 7구간에서 튄 프레임 0 %).
WHITE_V_MIN = 100            # 3차선 흰 점선 밝기 기준 (HSV V). 150 이면 전과 같다
WHITE_LIDAR_MARGIN = 0.15    # 라이다 물체보다 이만큼 [m] 앞쪽 바닥의 흰색만 쓴다. 음수면 거르지 않는다


def _floor_before_lidar(fit, scan):
    """BEV 칸마다: 그 방향으로 라이다에 잡힌 가장 가까운 물체보다 WHITE_LIDAR_MARGIN 넘게 앞(=바닥)이면 True."""
    r = np.asarray(scan[0], dtype=np.float32).copy()
    r[~np.isfinite(r) | (r < max(scan[3], 0.12))] = 99.0
    idx = np.round((fit._bev_a - scan[1]) / scan[2]).astype(int)
    inside = (idx >= 0) & (idx < r.size)                     # 라이다가 안 보는 방향은 거르지 않는다
    near = np.minimum.reduce([r[np.clip(idx + k, 0, r.size - 1)] for k in (-2, -1, 0, 1, 2)])    # 좌우 2줄(약 2도)까지
    return ~inside | (fit._bev_r < near - WHITE_LIDAR_MARGIN)


def _masks(fit, image, tune=None):
    """three_lane.LineFitter.masks 와 같다. 흰색 기준만 WHITE_V_MIN 이고, 라이다 물체 뒤(위)의 흰색은 버린다."""
    tune = tune or {}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (tune.get('h_min', 15), max(60, tune.get('s_min', 40)), max(70, tune.get('v_min', 60))),
                         (tune.get('h_max', 40), 255, 255))
    white = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (179, 60, 255))
    warp = lambda m: cv2.warpPerspective(m, fit.i2b, (fit.w, fit.h), flags=cv2.INTER_NEAREST)  # noqa: E731
    fit.yellow, fit.white = fit._thin(warp(yellow)), fit._thin(warp(white), fit._glare_kernel)
    if WHITE_LIDAR_MARGIN >= 0.0 and fit.scan is not None:
        try:
            fit.white = np.where(_floor_before_lidar(fit, fit.scan), fit.white, 0).astype(np.uint8)
        except Exception:                                    # 라이다 값이 이상하면 거르지 않은 흰색을 그대로 쓴다
            pass
    return fit.yellow, fit.white


def make_fitter():
    """three_lane.LineFitter 를 만들되, 카메라 바닥 보정을 고친 값으로 바꾸고 흰 점선 기준을 덧씌운다."""
    fit = T.LineFitter()
    if USE_BEV_FIX:
        src = np.float32(T.BEV_SRC).reshape(4, 2)
        src[:, 0] *= T.IMG_W - 1
        src[:, 1] *= T.IMG_H - 1
        i2g = cv2.getPerspectiveTransform(src, np.float32(BEV_GROUND_FIX).reshape(4, 2))
        g2b = np.array([[1 / T.RES, 0, T.XW / T.RES], [0, -1 / T.RES, T.F1 / T.RES], [0, 0, 1.0]])
        fit.i2b = g2b @ i2g
    fit.scan = None                                          # GapSupervisor.step 이 프레임마다 넣는다 (ranges, angle_min, angle_increment, range_min)
    py, px = np.mgrid[0:fit.h, 0:fit.w]
    bx, bf = -T.XW + (px + 0.5) * T.RES, T.F1 - (py + 0.5) * T.RES       # BEV 칸마다 오른쪽 / 앞 [m]
    fit._bev_r, fit._bev_a = np.hypot(bf, bx), np.arctan2(-bx, bf)      # 라이다에서 본 거리 / 각(왼쪽 +). 라이다 위치는 LIDAR_AHEAD(0) 가정 그대로
    fit.masks = lambda image, tune=None: _masks(fit, image, tune)
    return fit


def white_heading(mask):
    """BEV 흰 마스크에서 점선 조각들의 방향(도로 방향, 차 정면 기준 왼쪽 +)의 중앙값. 조각이 2개 미만이면 None.
    흰 점선은 3차선에서 늘 도로와 나란하다. 노란 선은 입구 사선·섬·합류에서 꺾여 있어 선 맞추기를 속인다."""
    if mask is None:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    angs = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 20 or h < 10:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == k)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        long_, short_ = max(rw, rh), min(rw, rh)
        if long_ < 10 or short_ > 10 or long_ < 2.2 * max(short_, 1.0):
            continue
        vx, vy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()[:2]
        if vy > 0:
            vx, vy = -vx, -vy
        th = math.atan2(-vx, -vy)
        if abs(th) <= math.radians(50):
            angs.append(th)
    if len(angs) < 2:
        return None
    a = np.array(angs)
    ok = a[np.abs(a - np.median(a)) <= math.radians(8)]
    return float(np.median(ok)) if len(ok) >= 2 else None


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
        self.fwd = 0.0
        self.tgt = None              # 가고 있는 줄의 옆 위치 (self.lat 과 같은 기준)
        self.steer = 0.0
        self.v = 0.0
        self.t_prev = None
        self.mem = deque(maxlen=MEM_KEEP)   # [(장애물 점 world, 노란 점 world)]
        self.mem_at = -1.0
        # 갈림길
        self.isl = None              # (섬 앞끝 s, 섬 가운데 lat)
        self._isl_est = deque(maxlen=7)
        self.sign_q = deque(maxlen=8)   # YOLO 표지판 상자: (이름, 방향[왼쪽 +], 높이, 받은 시각)
        self.sign_ok = []            # 이번 프레임에 표지판으로 인정된 것: (이름, 높이)
        self._th_bad = 0
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
        self.rev_count = 0
        self.back_room = 0.4
        self.s_stuck = -9.0
        self.message = ''
        self.give_up = False
        self.dead_at = None          # 마지막 수단(선 무시)을 켠 자리
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
        self.fwd += max(ds, 0.0)
        if ds < 0:
            self.rev_left -= abs(ds)
            self.back_room = max(0.0, self.back_room + ds)
        else:
            self.back_room = min(1.0, self.back_room + ds)

    def _camera_heading(self, obs):
        if not obs:
            return
        thw = obs.get('thw')
        if thw is not None:
            d = -thw - self.psi
            if abs(d) <= THW_ACCEPT:
                self.psi += THW_GAIN * d
                self._th_bad = 0
            else:
                self._th_bad += 1
                if self._th_bad >= 5:                 # 흰 점선이 계속 다른 방향을 가리킨다 -> 내 방향이 틀린 것
                    self.psi = -thw
                    self._th_bad = 0                  # 기억해 둔 점은 그대로 둔다: 이제 차 뒤쪽 확인(후진)에만 쓰여서, 지우면 뒤 차를 잊는다
            return
        if obs.get('n', 0) < 300:
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
        """YOLO 표지판 상자 -> 섬 앞끝 위치 + 판정에 쓸 상자 고르기.
        거리는 그 방향 라이다 값이 상자 높이로 본 거리와 비슷하면 라이다 값을 쓴다."""
        self.sign_ok = []
        while self.sign_q:
            label, bearing, h, t = self.sign_q.popleft()
            if now - t > 0.4:
                continue
            self.sign_txt = f' sg{math.degrees(bearing):+.0f}/{h:.0f}'
            if h < SIGN_SEE_H:
                continue
            dist = SIGN_K / h
            if lidar is not None:
                r, a = lidar
                near = r[np.abs(a - bearing) < math.radians(5.0)]
                near = near[(near > 0.65 * dist) & (near < 1.35 * dist)]
                if near.size >= 2:
                    dist = float(np.sort(near)[1]) + LIDAR_AHEAD
            if dist > 3.2:
                continue
            s, lat = self._car_to_road(dist * math.cos(bearing), dist * math.sin(bearing))
            s, lat = float(s), float(lat)
            if self.isl is not None:
                # 섬이 잡힌 뒤: 그 자리 근처에 보이는 상자만 표지판이다 (장애물 LIMO 등을 잘못 본 것은 버린다)
                ix, iy = self._road_to_car(np.array([self.isl[0]]), np.array([self.isl[1]]))
                ix, iy = float(ix[0]), float(iy[0])
                off = abs(bearing - math.atan2(iy, max(ix, 0.05)))
                if off > math.radians(SIGN_GATE_DEG) or (h > SIGN_EST_MAX_H and math.hypot(ix, iy) > SIGN_GATE_BIG_M):
                    self.sign_txt += 'x'
                    continue
                self.sign_ok.append((label, h))
                if h > SIGN_EST_MAX_H:
                    continue                                  # 가까이서는 상자가 잘려 위치가 틀리므로 자리는 안 고친다
            elif h > SIGN_EST_MAX_H:
                self.sign_txt += 'x'
                continue                                      # 처음부터 크게 보이는 것은 표지판이 아니다
            self._isl_est.append((s, lat, now))
            est = np.array([e for e in self._isl_est if now - e[2] <= SIGN_NEED_SEC])
            if len(est) < SIGN_NEED_N or np.ptp(est[:, 1]) > 0.30 or np.ptp(est[:, 0]) > 0.60:
                continue                                      # 한두 번 보인 것, 자리가 흩어진 것은 표지판이 아니다
            new = (float(np.median(est[:, 0])), float(np.median(est[:, 1])))
            if self.isl is None:
                self.message = f'[3차선G] 표지판(섬) 보임: 앞 {new[0] - self.sP:.2f} m, 옆 {new[1] - self.latP:+.2f} m'
                self.sign_ok.append((label, h))
            if self.isl is None or new[0] - self.sP > 0.5:    # 아주 가까우면 상자가 잘려 틀리므로 그 전 값을 쓴다
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
        bad = g < min(0.0, g0) - 0.02                # 여유 경계에 딱 걸린 채 지나갈 때 2 cm 흔들림으로 서지 않게
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
        if self.fwd - self.mem_at >= MEM_STEP:
            self.mem_at = self.fwd
            near_o, near_y = lx < 0.9, yrx < 0.9
            self.mem.append((np.stack(self._car_to_road(lx[near_o], ly[near_o]), 1),
                             np.stack(self._car_to_road(yrx[near_y], yry[near_y]), 1)))
        if self.mem:
            for kind in (0, 1):
                w = np.concatenate([m[kind] for m in self.mem])
                if not len(w):
                    continue
                mx_, my_ = self._road_to_car(w[:, 0], w[:, 1])
                if kind == 0:
                    # 라이다가 지금 직접 보는 곳(좌우 105도 안)은 지금 값만 믿는다. 기억은 그 바깥(차 뒤쪽)에만 쓴다.
                    # 기억은 방향 추정 오차만큼 몇 cm 밀려서, 옆을 스치듯 지날 때 없는 벽을 만들어 세웠다 (대회장 163~169초)
                    keep = (mx_ < 0.15) & (np.abs(np.arctan2(my_, mx_ - LIDAR_AHEAD)) > math.radians(105.0))
                else:
                    keep = mx_ < 0.45
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

        # 후진 중
        if self.rev_left > 0.0:
            bx = np.concatenate([np.array(BODY_OFF) - CAM - b for b in (0.06, 0.12, 0.18)])
            behind = float(self._look(g_obs, bx, np.zeros(bx.size)).min()) - BODY_R
            if behind < 0.03 or self.back_room <= 0.02:
                self.rev_left = 0.0
            else:
                # 물러나면서 차를 도로와 나란히 편다 (도로보다 오른쪽을 보고 있으면 오른쪽으로 꺾고 후진 -> 차 앞이 왼쪽으로 돈다).
                # 그 호를 따라 차체가 장애물·노란 선에 더 가까워지면 곧게 물러난다
                rs = float(np.clip(REV_STEER_GAIN * self.psi, -REV_STEER_MAX, REV_STEER_MAX))
                if abs(rs) > math.radians(2.0):
                    rx, ry = self._arc_rev(rs)
                    if ((self._look(g_obs, rx, ry) - BODY_R).min() < min(0.03, o0 - 0.01)
                            or (self._look(g_line, rx, ry) - BODY_R - M_LINE).min() < min(0.0, l0c - 0.02)):
                        rs = 0.0
                else:
                    rs = 0.0
                self.v, self.steer = -V_REVERSE, rs
                return self._out('REVERSE', f'후진 {self.rev_left:.2f} m 남음', 0.0, 0.0, to_isl)

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
                if self.dead_at is not None and free.max() < MIN_GO:
                    no_line = np.full_like(g_line, 9.0)
                    free3 = self._free(g_obs, no_line, cx2, cy2, DEAD_M_OBS, o0 - DEAD_M_OBS)
                    if free3.max() > free.max():
                        free, tight, level = free3, True, '선 무시'
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
        # 정지 조건
        stop = ''
        if fk < STOP_FREE:
            stop = f'갈 길 없음 ({level}, 차체 옆 여유 장애물 {min(o0, 9):.2f} / 노란 선 {min(l0c + M_LINE, 9):.2f} m)'
        else:
            gx, gy = self._arc(self.steer)
            g = (self._look(g_obs, gx, gy) - BODY_R).min(1)
            if (g < min(0.02, o0 - 0.02)).any():     # 이미 옆에 붙어 있는 것(지나치는 중)은 더 가까워질 때만
                stop = f'정면 {GUARD_DIST:.2f} m 안에 장애물'
        if (stop.startswith('갈 길 없음') and self.sP < ENTRY_SOFT_M and self.rev_count == 0 and o0 >= 0.15
                and self.isl is None):
            fx, fy = self._arc(0.0, 0.80)
            if (self._look(g_obs, fx, fy) - BODY_R).min() >= 0.10:
                # 입구: 장애물은 없고 노란 선(사선)만 걸린다 -> 서지 않고 기본 주행(노란 선 사이 주행)에 이 프레임을 맡긴다
                self.v = V_SLOW
                self.stop_since = None
                out = self._out('ENTRY', '입구: 노란 선만 걸림 -> 기본 주행 명령', dk, fk, to_isl)
                out['pass'] = True
                return out
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
            if (DEAD_MODE and self.dead_at is None and now - self.stop_since >= DEAD_SEC
                    and (self.rev_count >= REVERSE_MAX or self.back_room <= 0.15)):
                self.dead_at = self.sP
                self.message = '[3차선G] 후진으로도 안 풀림 -> 노란 선은 무시하고 장애물만 피해서 빠져나감'
            if now - self.stop_since >= STUCK_SEC and self.rev_count < REVERSE_MAX and self.back_room > 0.15:
                if self.sP > self.s_stuck + 0.8:
                    self.rev_count = 0
                self.s_stuck = max(self.s_stuck, self.sP)
                self.rev_count += 1
                self.rev_left = min(REVERSE_M, self.back_room)
                self.stop_since = None
                self.tgt = None
                self.message = f'[3차선G] 막힘 -> 후진 {self.rev_left:.2f} m ({self.rev_count}번째)'
            return self._out('BLOCKED', stop, dk, fk, to_isl)
        self.stop_since = self.block_since = None
        if self.dead_at is not None and self.sP > self.dead_at + DEAD_RUN_M:
            self.dead_at = None
        self.v = v
        if self.isl is not None and self.side_taken is None and to_isl < -0.5:
            self.side_taken = 0 if self.latP > self.isl[1] else 2
        return self._out(state, f'{reason} {level}'.strip(), dk, fk, to_isl)

    def _arc_rev(self, steer, dist=0.20):
        """지금 조향으로 dist 만큼 후진할 때 차체 원 가운데들: (걸음, 원 개수)."""
        n = int(dist / STEP)
        ax, ay, h = -CAM, 0.0, 0.0
        dh = 0.0 if abs(steer) < 1e-3 else -math.copysign(STEP / float(_radius(steer)), steer)
        gx, gy = np.zeros((n, NB)), np.zeros((n, NB))
        for i in range(n):
            ax, ay, h = ax - STEP * math.cos(h + dh / 2), ay - STEP * math.sin(h + dh / 2), h + dh
            for j, off in enumerate(BODY_OFF):
                gx[i, j], gy[i, j] = ax + off * math.cos(h), ay + off * math.sin(h)
        return gx, gy

    def _arc(self, steer, dist=GUARD_DIST):
        """지금 조향대로 GUARD_DIST 까지 갈 때 차체 원 가운데들: (걸음, 원 개수)."""
        n = int(dist / STEP)
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
        self._mission = ''
        self._odo = 0.0              # odom 누적 거리 (미션과 무관하게 계속 센다)
        self._m5_at = -99.0
        self.route_complete = False  # 갈림길 차로에 실제로 들어가 합류까지 끝냄 (매니저가 M2 -> M8 넘길 때 읽는다)
        self._entered = False

    def _reset(self):
        self.route_complete = self._entered = False
        self.core = GapCore()
        self.vote.reset()
        self.arm_travel = 0.0
        self._white = self._both = 0

    def set_mission(self, name):
        if name != self._mission:
            self._mission = name
            if name == 'M5':
                self._m5_at = self._odo
        if self.forced:
            return
        if name == 'M7':
            if self.mode == 'off':
                self._reset()
                self.mode = 'armed'
        elif name != 'M2':
            self.mode = 'off'

    def see_signs(self, detections, now=None):
        if self.mode != 'on':
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
        # 여기서는 받아 두기만 한다. 표지판으로 인정할지는 위치를 아는 _sign_update 가 정하고, 판정은 step_obs 에서 한다
        self.core.sign_q.append((label, -math.atan((frac - 0.5) * BEAR_K), height, time.monotonic() if now is None else now))

    def _vote_signs(self):
        if self.fixed_goal is not None:
            return
        for label, height in self.core.sign_ok:
            if height >= GUESS_MIN_H:
                self.core.guess[0 if label == 'a' else 2] += 1
            decided = self.vote.add(label, height)
            if decided is not None and decided != self.core.goal:
                self.core.goal = decided
                self.message = f"[3차선G] 표지판 {'A(왼쪽)' if decided == 0 else 'B(오른쪽)'} 확정 (상자 높이 {height:.0f} px)"
        self.core.sign_ok = []

    @property
    def latch_off(self):
        """True 면 기본 주행의 코너 고정 조향을 끈다 (3차선 입구 사선, 섬 뒤 합류의 꺾인 선을 코너로 보지 않게)."""
        if self.mode == 'fork' or (self.mode in ('armed', 'on') and (self.forced or self.arm_travel >= ARM_AFTER_M)):
            return True
        return self._mission == 'M5' and M5_LATCH_OFF[0] <= self._odo - self._m5_at <= M5_LATCH_OFF[1]      # 방지턱

    def step(self, image, pose, scan, both_yellow, tune=None):
        if self.mode in ('armed', 'on') and (self.forced or self.arm_travel >= ARM_AFTER_M - 0.05):
            prior = -self.core.psi if self.mode == 'on' else None
            self.fitter.scan = scan
            obs = self.fitter.fit(image, prior, tune)
            if obs:
                obs['thw'] = white_heading(getattr(self.fitter, 'white', None))
        else:
            obs = None
        return self.step_obs(obs, pose, scan, both_yellow, time.monotonic())

    def step_obs(self, obs, pose, scan, both_yellow=False, now=0.0):
        """반환: None (기본 주행 명령을 그대로 씀) 또는 dict(v, steer, state, reason, note)."""
        if pose is not None and self._pose is not None:
            step = math.hypot(pose[0] - self._pose[0], pose[1] - self._pose[1])
            if step < 0.5:
                self.arm_travel += step
                self._odo += step
        if pose is not None:
            self._pose = pose
        if self.mode in ('off', 'done'):
            return None
        if self.mode == 'fork':                 # 갈림길 차로 ~ 합류: 기본 주행이 몰고, latch 만 꺼 둔다
            self.core._advance(pose, now)
            if self.core.sP >= self._fork_end:
                self.route_complete = self._entered
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
            # 시작 방향: 흰 점선으로 본 값이 있으면 그것. 없으면 선 맞추기 값은 작을 때만 믿는다 (입구 사선에 속아 -20도로 시작한 적 있음)
            if obs and obs.get('thw') is not None:
                psi0 = -obs['thw']
            elif obs and abs(obs['th']) <= TH_ACCEPT:
                psi0 = -obs['th']
            else:
                psi0 = 0.0
            self.core.start(psi0, pose)
            self.message = f"[3차선G] {VERSION} 시작 (도로 방향 {math.degrees(-self.core.psi):+.0f}도)"
        core = self.core
        if scan is None and core.v > 0.0:
            core.v = 0.0
            return dict(v=0.0, steer=core.steer, state='LANE3', reason='3차선G: 라이다가 끊김 -> 정지', note='L3G no scan')
        out = core.step_obs(obs, pose, scan, self.fixed_goal, now)
        self._vote_signs()
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
        if core.sP >= MAX_ON_M:
            # 앞으로 나아간 거리로만 센다. 후진으로 오간 거리까지 세면 섬 앞에서 '최대 거리'에 걸려
            # 틀어진 자세로 기본 주행에 넘어가고 코너 조향이 바로 켜져 섬을 밟았다 (송도 169초, 대회장 225초)
            done, self._fork_end = '최대 거리', core.sP
        if done:
            self._entered = core.isl is not None and core.sP - core.isl[0] >= HANDOVER_IN and not core.give_up
            self.mode = 'fork'
            side = core.side_taken if core.isl is None else (0 if core.latP > core.isl[1] else 2)
            self.side_taken = side
            self.message = (f"[3차선G] {done}: {({0: 'A(왼쪽)', 2: 'B(오른쪽)', None: '?'})[side]} -> 기본 주행에 넘김 "
                            f'(합류 끝까지 코너 조향 끔)')
            return None
        if out.get('pass'):
            return None                 # 이 프레임은 기본 주행 명령을 그대로 쓴다 (입구)
        return out


def _patch_corner(SL):
    """simple_lane.LaneEngine 의 코너 고정 조향을 '바깥 선이 정면에서 사라질 때까지' 이어 준다 (원본 파일은 그대로)."""
    Eng = SL.LaneEngine
    orig_command, orig_done = Eng.command, Eng._lock_done
    max_sec = getattr(SL, 'LOCK_MAX_SEC', 5.0)

    def command(self, result, tune, now, camera_ok=True):
        self._g_res = result
        v, angle, state, why = orig_command(self, result, tune, now, camera_ok)
        if not CENTER_FIX:
            return v, angle, state, why
        acc = getattr(self, '_g_i', 0.0)
        dt = min(0.2, max(0.0, now - getattr(self, '_g_it', now)))
        self._g_it = now
        lat = result.get('lateral') if result else None
        straight = (state == 'TRACK' and not self.locked and getattr(self, 'mode', '') != 'CURVE'
                    and lat is not None and abs(lat) < 0.20 and abs(angle) < math.radians(6.0))
        if straight:
            src = result.get('source')
            err = lat + (CENTER_SHIFT if src == 'left_only' else (-CENTER_SHIFT if src == 'right_only' else 0.0))
            acc = max(-CENTER_I_MAX, min(CENTER_I_MAX, acc + CENTER_KI * err * dt))
        else:
            acc *= 0.97                                   # 코너·정지·선 놓침: 서서히 뺀다
        self._g_i = acc
        return v, angle - acc, state, why

    def _lock_done(self, now, yaw_limit):
        if not orig_done(self, now, yaw_limit):
            return False
        hold = False
        r = getattr(self, '_g_res', None)
        if (CORNER_HOLD and r and r.get('valid') and yaw_limit < math.radians(95.0)      # 95도 이상 = 선이 안 보일 때 쓰는 한도: 그대로 둔다
                and now - self.lock_t <= max_sec and self.yaw is not None and self.lock_yaw is not None
                and abs(self.yaw - self.lock_yaw) < yaw_limit + HOLD_EXTRA):
            tilt, cross = r.get('tilt'), r.get('cross')
            hold = (tilt is not None and abs(tilt) >= HOLD_TILT and (tilt > 0) == (self.lock_dir < 0)
                    and cross is not None and cross[1] < HOLD_CROSS and (cross[0] == 'left') == (self.lock_dir < 0))
        if hold and not getattr(self, '_g_hold', False):
            print(f"\n[코너G] 고정 조향 연장 (선 기울기 {math.degrees(abs(tilt)):.0f}도, 바깥 선 정면 {cross[1]:.2f} m)")
        if not hold and getattr(self, '_g_hold', False) and self.yaw is not None and self.lock_yaw is not None:
            print(f"\n[코너G] 연장 끝 (odom 으로 {math.degrees(abs(self.yaw - self.lock_yaw)):.0f}도 돎)")
        self._g_hold = hold
        return not hold

    Eng.command, Eng._lock_done = command, _lock_done


def main():
    try:
        import simple_lane as SL                      # 같은 폴더의 simple_lane.py (고치지 않는다)
    except ImportError:
        from jetracer_ros2 import simple_lane as SL
    SL._three_lane = types.SimpleNamespace(Lane3Supervisor=GapSupervisor)
    _patch_corner(SL)
    if 'l3simple' in sys.argv:
        sys.argv.remove('l3simple')
    print(f'[3차선G] lane3_gap.py {VERSION}: 3차선 + 갈림길 + 방지턱 + 코너 고정 조향 연장 + 직선 가운데 맞추기 (simple_lane.py 파일은 그대로)')
    SL.main()


if __name__ == '__main__':
    main()
