#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""백업 주행 코드 1단계: 노란 차선 따라가기 (팀 코드와 별개인 단독 파일)

처리 순서
  카메라 -> 왜곡 보정 -> 640x480 -> [원본에서] 노란색 마스크 -> 마스크만 BEV(미터 격자)
  -> 노란 덩어리 중 왼쪽/오른쪽 선 고르기 (지난 프레임 선과 가까운 것)
  -> 각 선을 차에서 가까운 순서의 점열로 만들고 법선으로 0.26m 띄워 중심선 추정 (급커브에서도 동작)
  -> 중심선 위 앞쪽 목표점(pure pursuit) -> 조향각 -> 속도

실행 (LIMO, 먼저 source ~/wego_ws/install/setup.bash)
  python3 ~/simple_lane.py                         보기 전용 (명령 발행 안 함)
  python3 ~/simple_lane.py drive                   실제 주행 (/cmd_vel 로 발행)
  python3 ~/simple_lane.py drive --topic inference/cmd_vel   매니저 연결용 (2단계)
  python3 ~/simple_lane.py calib                   BEV 보정 확인 (직선 정중앙에 놓고)
  python3 ~/simple_lane.py image 사진.jpg          사진 한 장으로 확인 (ROS 불필요)
  python3 ~/simple_lane.py rec                     보기 전용 + 기록
  python3 ~/simple_lane.py circle -20              조향 고정 원 주행 테스트 (음수=오른쪽, 차가 실제로 명령만큼 도는지 확인)
  python3 ~/simple_lane.py reset                   조향 초기화 (리모컨으로 코드 모드로 바꾼 뒤 바퀴가 꺾여 있을 때)
  python3 ~/simple_lane.py sweep                   바퀴 띄우고 angular.z 를 단계별로 보내 조향 정도 확인
  python3 ~/simple_lane.py turncal                 실제 회전반경 측정 (넓은 바닥에서. 배율별로 오른쪽으로 돌며 카메라로 회전량을 잼)
  python3 ~/simple_lane.py bridge                  /cmd_vel -> /limo/ack_cmd 변환 (limo_base_ackerman.launch.py 와 함께. 조향 제한 해제용)
  python3 ~/simple_lane.py drive rec               주행 + 기록 (프레임 저장, 문제 분석용)
  python3 ~/simple_lane.py drive rec p1            프리셋 p1/p2/p3 로 주행 + 기록 (곡선 파라미터 비교용)
  python3 ~/simple_lane.py drive mem               경로 기억 방식으로 주행 (/odom 필요, limo_base 켜져 있어야 함)
  python3 ~/simple_lane.py drive mem rec           경로 기억 방식 + 기록 (odom 도 같이 저장)
  python3 ~/simple_lane.py mem                     보기 전용으로 기억 경로 확인 (리모컨으로 몰면서 노란 경로가 차 밑까지 이어지는지)

경로 기억 방식 (mem)
  카메라는 차 앞 약 55cm 부터만 보이고 그 폭도 좁아서, 커브에서는 차 바로 앞의 차선을 볼 수 없다.
  그래서 조금 전에 본 차로 중심선을 /odom(바퀴 이동 기록)으로 옮겨 가며 기억해 두고,
  [기억한 경로(차 옆~55cm)] + [지금 보이는 경로(55cm~1.5m)] 를 이은 경로 위의 가까운 목표점(mem_ld)을 따라간다.
  미리 꺾기(ff)·강제 최소 조향(latch)은 이 방식에서는 쓰지 않는다. mem 을 안 붙이면 기존 방식 그대로.
  python3 ~/simple_lane.py replay ~/simple_lane_log/폴더   기록 다시 보기

디버거 창 키 (NoMachine 터미널에서 실행했을 때)
  스페이스: 화면 멈춤 / s: 캡처 저장 / q: 종료
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np

# =====================================================================
# 파라미터 (트랙에서 바꿀 값은 대부분 디버거 슬라이더로 조절 가능)
# =====================================================================
IMG_W, IMG_H = 640, 480

# --- 카메라 보정: 성윤 님 실측값 그대로 (params.yaml) ---
# 이미지 4점 (640x480 비율, TL, TR, BR, BL) - 왜곡 보정 후 이미지 기준
BEV_SRC = [0.3605, 0.46, 0.6395, 0.46, 1.00, 0.82, 0.00, 0.82]
# 위 4점의 바닥 좌표 [오른쪽 m, 카메라 투영점에서 전방 m]
BEV_GROUND = [-0.24, 1.326, 0.24, 1.326, 0.22, 0.421, -0.22, 0.421]
CAMERA_FORWARD_OFFSET = 0.184   # 카메라 투영점이 뒤 차축보다 앞에 있는 거리 (m)

# --- 우리 BEV 영역 (바닥 미터 격자) ---
X_MIN, X_MAX = -0.45, 0.45      # 좌우 (m, 오른쪽 +)
F_MIN, F_MAX = 0.35, 1.50       # 카메라 앞 거리 (m). 0.35 = 화면 맨 아래 근처 (곡선에서 선이 차에 가까워져도 놓치지 않게)
RES = 0.005                     # m / 픽셀

# --- 차로 ---
LANE_WIDTH = 0.43               # 노란선 중심 간 거리 (m). 연습 트랙 calib 실측값(BEV 기준 약 42~43cm)
HALF_LANE = LANE_WIDTH / 2.0
EXPECT_TOL = 0.12               # 예상 위치에서 이만큼(m) 안의 선만 그쪽 선 후보로 인정 (옆 차로 선은 약 19cm 더 바깥)
SIDE_MARGIN = 0.05              # 왼쪽 선의 가까운 쪽이 차 중심보다 이만큼(m) 넘게 오른쪽이면 왼쪽 선 아님
WIDTH_TOL = 0.08                # 양쪽 선 간격이 기억한 차로 폭과 이만큼(m) 넘게 다르면 다시 고름
TRACK_TOL = 0.10                # 지난 프레임의 같은 쪽 선과 이만큼(m) 안이면 같은 선으로 이어 감
TRACKED_SIDE_LIMIT = 0.20       # 이어 가는 선이라도 반대편으로 이만큼(m) 넘어가면 잘못 잡은 것
WIDTH_MIN, WIDTH_MAX = 0.33, 0.72   # 차로 폭으로 인정하는 범위 (m). 연습 트랙 오른쪽 구간은 약 0.55~0.6m로 넓음
F_REF = 0.70                    # 차로 중심을 기억하는 기준 거리 (카메라 앞 m)
ARC_MIN = 0.25                  # 목표점은 보이는 경로 시작점에서 경로를 따라 최소 이만큼(m) 앞

# --- 선 덩어리 ---
LINE_MIN_PIX = 100              # 선으로 인정하는 최소 픽셀 수 (BEV, 5mm 격자)
LINE_MIN_SPAN = 0.15            # 선으로 인정하는 최소 길이 (차에서 거리 기준, m)
FRAG_MIN_PIX = 15               # 선 끝에 이어 붙일 조각의 최소 픽셀 수
FRAG_JOIN = 0.10                # 선 끝에서 이 거리(m) 안의 조각은 같은 선으로 이어 붙임 (먼 곳 끊긴 곡선)

# --- 차량 / 조향 (성윤 님 코드 값) ---
WHEELBASE = 0.20
TRACK = 0.172
# 실차 측정(turncal, 다리 사용): 최대 조향에서 실제 회전반경 약 0.51m -> 안쪽 바퀴 기준 약 25도.
# (다리 없이 limo_base.launch.py 의 /cmd_vel 로 보내면 드라이버가 조향을 11.3도로 제한해 회전반경이 약 1.05m 밖에 안 나옴)
MAX_STEER = math.radians(25.0)
BRIDGE_GAIN = 1.3               # 다리(bridge)에서 조향각에 곱하는 값. 실측: 1.0 이면 계산 0.48m 에 실제 0.62m, 1.3 이면 계산과 거의 같음
STEER_ALPHA = 0.5               # (지금은 슬라이더 smooth 로 조절)
# 조향 명령 배율. 1.0 = 성윤 님 코드와 같은 계산식 그대로 (실차 확인: angular.z 약 0.45 에서 최대 조향).
# 필요하면 실행할 때 --gain 숫자 로 바꿔 볼 수 있음 (예: python3 simple_lane.py drive --gain 1.2)
STEER_GAIN = 1.0

# --- 속도 / 안전 ---
V_MIN = 0.20
LOST_HOLD_SEC = 0.5             # 선을 놓쳐도 마지막 조향으로 버티는 시간
LOST_HOLD_SPEED = 0.15
RESUME_FRAMES = 5               # 선을 놓쳐 정지한 뒤, 연속 이만큼 정상이어야 다시 출발 (덜컹거림 방지)

# --- 급커브 미리 꺾기 (곡률 피드포워드) ---
# 보이는 선을 원으로 맞춰 커브 반경을 구하고, 커브가 차에 가까워질수록(원 중심의 전방 거리 b가 작을수록)
# 그 반경을 도는 데 필요한 조향을 섞는다. 실차 기록에서 순수 추종만으로는 꺾기 시작이 늦어 바깥으로 밀렸음.
# 섞기 시작 거리는 슬라이더 ff_cm (원 중심이 뒤 차축 앞 이 거리보다 가까워지면 시작, 0이면 끔)
FF_RAMP = 0.35                  # 시작 후 이 거리(m)만큼 더 다가오면 100% 반영
FF_MAX_R = 1.2                  # 이보다 큰 반경은 직선으로 봄 (m)
FF_MAX_RMS = 0.015              # 원 맞춤 오차 상한 (m)
CURVE_SPEED = 0.22              # (지금은 슬라이더 curve_v_cm 으로 조절)
LATCH_MIN = math.radians(10.0)  # 미리 꺾기가 이 이상 걸리면 커브 끝날 때까지 유지
LATCH_CROSS = 1.30              # 바깥 선이 차 앞 이 거리(m) 안에서 진행 방향을 가로지르면 아직 커브 중
LATCH_FLOOR = math.radians(22.0)  # (지금은 슬라이더 latch_deg 로 조절)
CURVE_STAY = math.radians(12.0)   # 곡선 모드 유지: 선 기울기가 이 이상이면 아직 곡선
PP_DISAGREE = math.radians(5.0)   # 경로 추종이 커브 반대 방향으로 이만큼 이상 꺾으라 하면 최소 조향 강제 안 함
# --- 경로 기억 (실행 인자 mem). 카메라는 차 앞 약 55cm부터만 보여서, 지나온 중심선을 /odom 으로 옮겨 가며 기억 ---
MEM = False                     # 실행할 때 mem 을 붙이면 켜짐
ODOM_TOPIC = '/odom'
MEM_JUMP = 0.15                 # 이번에 본 선의 시작점이 기억한 같은 쪽 선에서 이보다(m) 멀면 일단 무시
MEM_BAD_FRAMES = 5              # 위 상황이 이만큼 연속되면 기억을 버리고 새 선으로 교체
MEM_TAIL_TOL = 0.08             # 이번에 본 선의 먼 쪽 끝이 기억한 선 위(이 거리 안)에 있으면, 그보다 먼 기억은 그대로 둠
MEM_BEHIND = 0.35               # 차 뒤로 이만큼(m) 지난 선은 지움
MEM_AHEAD = 2.00                # 차 앞으로 이보다(m) 먼 기억은 지움
MEM_FAR = 1.50                  # 기억한 선이 차에서 이보다(m) 멀리 있으면 버림
MEM_HUG_EXTRA = 0.03            # 넓어진 구간에서 기준 선과 띄우는 거리 상한 = 직선 반폭 + 이 값 (m)
MEM_TURN_MIN = math.radians(20) # 기억한 선의 방향이 이만큼 이상 변하면 커브로 보고 안쪽 선 기준으로 감
MEM_LINE_KEEP = 1.20            # 화면에서 사라진 선을 차가 이만큼(m) 갈 때까지 기억 (커브 안쪽 선)
MEM_BLIND_SEC = 4.0             # 선이 하나도 안 보일 때 기억만으로 달리는 최대 시간
ODOM_STALE = 0.3                # odom 이 이 시간(초) 넘게 안 오면 기억 방식을 쉬고 기존 방식으로
PARALLEL_TILT = math.radians(4.0)    # 두 선의 기울기 차이가 이보다 작아야 나란한 것으로 봄
STRAIGHT_TILT = math.radians(12.0)   # 선 기울기가 이보다 작을 때만 차로 폭을 기억 (곡선 입구에서 벌어진 폭을 기억하지 않게)
# --- 코너 고정 조향 (슬라이더 latch=1 일 때) ---
# 곡선 모드에서 바깥 선이 차 정면 turn_in_cm 안으로 들어오면 커브 방향으로 latch_deg 를 고정해서 돌고,
# odom 기준으로 turn_deg 만큼 돌면 풀어서 다시 차선을 따라감. 도는 중에 선이 안 보여도 멈추지 않음.
# (실차 기록: 경로만 따라가면 최대 조향에 도달하는 시점이 정면 0.54m 로 늦어 바깥 선까지 밀렸고, 선을 놓치자 63도 돈 곳에서 멈췄음)
LOCK_BLIND_YAW = math.radians(100.0) # 선이 안 보이는 채로는 이만큼까지만 돌고 정지
LOCK_MAX_SEC = 5.0                   # 고정 조향 최대 시간 (odom 이 멈추거나 없어도 이 시간이 지나면 끝냄)
CAMERA_TIMEOUT = 0.3            # 영상이 이 시간 넘게 안 오면 정지
RESET_SEC = 1.5                 # drive 시작 후 이 시간 동안은 정지·직진 명령만 보내 조향 초기화

# --- 슬라이더로 조절하는 값 (시작값). 종료 시 ~/simple_lane_tune.json 에 저장되고 다음 실행 때 불러옴 ---
DEFAULT_TUNE = {
    'h_min': 15, 'h_max': 40,   # HSV hue (0~179)
    's_min': 40,                # 채도 최소 (먼 곳 노란선은 채도 30~60으로 옅음. 흰 바닥은 30 미만)
    'v_min': 60,                # 밝기 최소
    'ld_cm': 70,                # 목표점 거리 (뒤 차축 기준, cm). 크면 곡선에서 일찍 꺾고(안쪽 파고듦), 작으면 늦게 꺾음(바깥 밀림)
    'vmax_cm': 30,              # 직선 최고 속도 (cm/s)
    'ff_cm': 90,                # 앞의 커브가 이 거리(cm) 안에 들어오면 커브로 보고 미리 감속 (0이면 끔)
    'pre_steer': 0,             # 1: 커브가 보이면 미리 꺾기 / 0: 끔. 조향 제한을 풀기 전(회전반경 1m)에 쓰던 보정이라 기본은 끔
    'latch': 1,                 # 1: 코너 고정 조향 사용 / 0: 끔 (경로만 따라감)
    'latch_deg': 25,            # 코너 고정 조향 각도 (도). 25 = 최대 조향 (회전반경 약 0.51m)
    'turn_in_cm': 75,           # 바깥 선이 차 정면 이 거리(cm) 안에 들어오면 고정 조향 시작. 바깥으로 밀리면 ↑, 안쪽 선에 붙으면 ↓
    'turn_deg': 60,             # 고정 조향으로 이만큼(도) 돌면 풀고 차선 추종으로 넘김. 코너 끝에서 안쪽으로 말리면 ↓, 덜 돌면 ↑
    'curve_v_cm': 22,           # 커브에서 속도 상한 (cm/s). 작으면 곡선에서 더 느리게
    'smooth': 50,               # 조향 반응 속도 (%). 크면 빠르게 반응, 작으면 부드럽지만 늦음
    'curve_in_deg': 25,         # 선 기울기가 이 각도(도) 이상이면 곡선 모드로 (작으면 곡선을 쉽게 인정)
    'curve_out': 6,             # 곡선 근거가 이만큼 연속(프레임) 없어야 직선 모드로 복귀 (크면 곡선 모드를 오래 유지)
    'mem_ld': 50,               # [mem] 기억 경로 위 목표점 거리 (뒤 차축 기준 cm). 커브에서 바깥으로 밀리면 ↑, 안쪽 선에 붙으면 ↓
}

# 곡선 파라미터 비교용 프리셋 (색 검출 값은 저장된 값 그대로 쓰고, 아래 값만 덮어씀)
#   p1 안전:  느리게 + 최대 조향. 이걸로도 못 돌면 파라미터가 아니라 인식/타이밍 문제
#   p2 기본:  추천 중간값
#   p3 일찍:  p2 와 속도·조향 같고, 꺾기 시작만 더 일찍 (p2 와 비교하면 타이밍 효과가 보임)
PRESETS = {
    'p1': dict(vmax_cm=25, curve_v_cm=16, latch=1, latch_deg=27, smooth=60, ld_cm=60, ff_cm=100,
               curve_in_deg=20, curve_out=8),
    'p2': dict(vmax_cm=30, curve_v_cm=20, latch=1, latch_deg=24, smooth=60, ld_cm=70, ff_cm=90,
               curve_in_deg=25, curve_out=6),
    'p3': dict(vmax_cm=30, curve_v_cm=20, latch=1, latch_deg=24, smooth=85, ld_cm=55, ff_cm=100,
               curve_in_deg=18, curve_out=8),
}
PRESET = None     # 실행 인자로 p1/p2/p3 를 주면 설정됨
TUNE_FILE = os.path.expanduser('~/simple_lane_tune_v9.json')   # v8: 조향 제한 해제 후 기본값 (미리 꺾기·최소 조향 끔)
SNAP_DIR = os.path.expanduser('~/simple_lane_snaps')
LOG_DIR = os.path.expanduser('~/simple_lane_log')   # 주행 기록 (프레임 + log.csv)

IMAGE_TOPIC = '/camera/color/image_raw/compressed'
CAMERA_INFO_TOPIC = '/camera/color/camera_info'


# =====================================================================
# 조향 변환: 성윤 님 코드 (lane_ackermann.py, ackermann_utils.py) 그대로
# =====================================================================
def target_to_inner_angle(lateral_right_m, forward_from_rear_m,
                          wheelbase_m=WHEELBASE, track_m=TRACK,
                          maximum_inner_angle_rad=MAX_STEER):
    """목표점(오른쪽 +, 뒤 차축 전방) -> 안쪽 바퀴 조향각 (왼쪽 +)."""
    lateral = float(lateral_right_m)
    if abs(lateral) < 1e-9 or forward_from_rear_m <= 0:
        return 0.0
    curvature = 2.0 * abs(lateral) / (forward_from_rear_m ** 2 + lateral ** 2)
    center_radius = 1.0 / curvature
    inside_radius = max(1e-6, center_radius - track_m / 2.0)
    angle = math.atan(wheelbase_m / inside_radius)
    return math.copysign(min(angle, maximum_inner_angle_rad), -lateral)


def inner_angle_to_omega(inner_angle, linear_speed, wheelbase=WHEELBASE, track=TRACK):
    """조향각 -> limo_base 에 보낼 angular.z (속도 0이면 0).
    STEER_GAIN: 조향 명령 배율 (기본 1.0)."""
    if abs(inner_angle) < 1e-4 or linear_speed == 0.0:
        return 0.0
    r = wheelbase / math.tan(abs(inner_angle)) + track / 2.0
    central_angle = math.atan(wheelbase / r)
    if inner_angle < 0:
        central_angle = -central_angle
    return STEER_GAIN * linear_speed * math.tan(central_angle) / wheelbase


# =====================================================================
# 기하: 이미지 <-> 바닥(m) <-> BEV 픽셀
# =====================================================================
class Geometry:
    def __init__(self):
        src = np.float32(BEV_SRC).reshape(4, 2)
        src[:, 0] *= IMG_W - 1
        src[:, 1] *= IMG_H - 1
        ground = np.float32(BEV_GROUND).reshape(4, 2)
        self.img2ground = cv2.getPerspectiveTransform(src, ground)
        self.ground2img = np.linalg.inv(self.img2ground)
        self.bev_w = int(round((X_MAX - X_MIN) / RES))
        self.bev_h = int(round((F_MAX - F_MIN) / RES))
        g2b = np.array([[1.0 / RES, 0, -X_MIN / RES],
                        [0, -1.0 / RES, F_MAX / RES],
                        [0, 0, 1]], dtype=np.float64)
        self.img2bev = g2b @ self.img2ground
        # 카메라가 실제로 볼 수 있는 BEV 영역 (이미지 밖은 0)
        full = np.full((IMG_H, IMG_W), 255, np.uint8)
        self.visible = self.warp(full) > 127

    def warp(self, mask):
        return cv2.warpPerspective(mask, self.img2bev, (self.bev_w, self.bev_h),
                                   flags=cv2.INTER_LINEAR, borderValue=0)

    def warp_color(self, image):
        return cv2.warpPerspective(image, self.img2bev, (self.bev_w, self.bev_h),
                                   flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    def px_to_m(self, px, py):
        return X_MIN + (np.asarray(px) + 0.5) * RES, F_MAX - (np.asarray(py) + 0.5) * RES

    def m_to_px(self, x, f):
        return (np.asarray(x) - X_MIN) / RES - 0.5, (F_MAX - np.asarray(f)) / RES - 0.5

    def ground_to_image(self, x, f):
        pts = np.stack([np.asarray(x, np.float64), np.asarray(f, np.float64)], axis=-1)
        pts = pts.reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.ground2img).reshape(-1, 2)
        return out


# =====================================================================
# 차선 검출 + 조향 계산 (ROS 없음)
# =====================================================================
class LaneLine:
    """한 줄의 노란선. 차에서 가까운 쪽부터 먼 쪽 순서의 점열(미터)."""
    def __init__(self, side, xs, fs, px, py, near_x):
        self.side, self.xs, self.fs = side, xs, fs
        self.px, self.py = px, py          # BEV 픽셀 (그리기용)
        self.n = len(px)
        self.near_x = near_x
        self.f_lo, self.f_hi = float(fs.min()), float(fs.max())

    def x_at(self, f):
        """전방 거리 f 에서의 좌우 위치 (직선에 가까운 선에서만 의미 있음)."""
        order = np.argsort(self.fs)
        return np.interp(f, self.fs[order], self.xs[order])


def _smooth(values, k=3):
    if len(values) < k:
        return values
    pad = np.pad(values, (k // 2, k // 2), mode='edge')
    return np.convolve(pad, np.ones(k) / k, mode='valid')


def _polyline_by_distance(xm, fm, step=0.02, min_pts=3):
    """점들을 차(뒤 차축)에서의 거리로 묶어 순서 있는 점열로 만든다. 급커브에서도 동작."""
    d = np.hypot(xm, fm + CAMERA_FORWARD_OFFSET)
    bins = np.floor((d - d.min()) / step).astype(int)
    xs, fs = [], []
    for b in np.unique(bins):
        sel = bins == b
        if np.count_nonzero(sel) >= min_pts:
            xs.append(float(xm[sel].mean()))
            fs.append(float(fm[sel].mean()))
    if len(xs) < 2:
        return None, None
    return _smooth(np.array(xs)), _smooth(np.array(fs))


def _offset(xs, fs, sign, half=HALF_LANE):
    """점열을 진행방향 법선으로 half 만큼 띄운다. sign=+1 오른쪽, -1 왼쪽."""
    dx, df = np.gradient(xs), np.gradient(fs)
    norm = np.hypot(dx, df) + 1e-9
    # 진행 방향 (dx, df) 의 오른쪽 법선 = (df, -dx)
    return xs + sign * half * df / norm, fs - sign * half * dx / norm


def _fit_circle(x, y):
    """점들에 원을 맞춘다. 반환 (R, 중심x, 중심y, rms)."""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    p, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    R = math.sqrt(max(1e-9, p[2] + p[0] ** 2 + p[1] ** 2))
    rms = float(np.sqrt(np.mean((np.hypot(x - p[0], y - p[1]) - R) ** 2)))
    return R, float(p[0]), float(p[1]), rms


def _inner_angle_for_radius(rc):
    """뒤 차축 중심 회전반경 rc 를 돌기 위한 안쪽 바퀴 조향각 (크기, 27도 제한)."""
    return min(math.atan(WHEELBASE / max(1e-3, rc - TRACK / 2.0)), MAX_STEER)


def _line_tilt(line):
    """선의 가까운 쪽 방향이 차 정면에서 몇 rad 틀어졌는지 (오른쪽 +).
    시작점에서 직선거리 25cm 떨어진 점까지의 방향이라, 지그재그 선은 꺾임이 평균돼서 작게 나옴."""
    seg = np.hypot(line.xs - line.xs[0], line.fs - line.fs[0])
    k = int(np.argmax(seg >= 0.25)) if np.any(seg >= 0.25) else len(seg) - 1
    if k == 0:
        return 0.0
    return math.atan2(line.xs[k] - line.xs[0], line.fs[k] - line.fs[0])


def _perp_width(left, right):
    """두 선 사이의 수직 거리 (m). 곡선에서 가로로 재면 실제보다 넓게 나와서 선에 수직으로 잰다. 못 재면 None"""
    d = np.sqrt((left.xs[:, None] - right.xs[None, :]) ** 2 + (left.fs[:, None] - right.fs[None, :]) ** 2)
    j = d.argmin(axis=1)
    ok = (j > 0) & (j < len(right.xs) - 1)          # 상대 선의 끝점에 걸린 것은 수직 거리가 아님
    if np.count_nonzero(ok) < 3:
        return None
    return float(np.median(d[np.arange(len(left.xs)), j][ok]))


def _pursuit_on_path(X, Y, i0, look):
    """차 좌표(뒤 차축 원점, 오른쪽 X, 앞 Y) 경로에서, 차에 가장 가까운 점(i0)부터 앞으로 가며
    차에서 look(m) 떨어진 점을 목표로 한 조향각. 반환 (각도, (tx, ty)) 또는 None"""
    d = np.hypot(X[i0:], Y[i0:])
    far = np.nonzero(d >= look)[0]
    if len(far) == 0:
        if d[-1] < 0.25:                    # 남은 경로가 너무 짧음
            return None
        tx, ty = float(X[-1]), float(Y[-1])
    else:
        k = i0 + int(far[0])
        if k > i0 and d[k - i0] > d[k - i0 - 1]:         # 정확히 look 거리가 되도록 두 점 사이를 보간
            w = (look - d[k - i0 - 1]) / (d[k - i0] - d[k - i0 - 1])
            w = min(1.0, max(0.0, w))
            tx = float(X[k - 1] + w * (X[k] - X[k - 1]))
            ty = float(Y[k - 1] + w * (Y[k] - Y[k - 1]))
        else:
            tx, ty = float(X[k]), float(Y[k])
    if ty <= 0.05:
        return None
    return target_to_inner_angle(tx, ty), (tx, ty)


def _heading_change(P):
    """점열의 처음과 끝 진행 방향 차이 (rad, 오른쪽으로 휘면 +). 각 끝은 약 15cm 구간으로 방향을 잼."""
    if P is None or len(P) < 8:
        return 0.0
    seg = np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1])
    a = int(np.argmax(seg >= 0.15)) if np.any(seg >= 0.15) else len(P) - 1
    seg2 = np.hypot(P[:, 0] - P[-1, 0], P[:, 1] - P[-1, 1])
    b = len(P) - 1 - int(np.argmax(seg2[::-1] >= 0.15)) if np.any(seg2 >= 0.15) else 0
    if a == 0 or b >= len(P) - 1:
        return 0.0
    h0 = math.atan2(P[a, 0] - P[0, 0], P[a, 1] - P[0, 1])
    h1 = math.atan2(P[-1, 0] - P[b, 0], P[-1, 1] - P[b, 1])
    return math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))


def _side_path(P, Q, sign, half):
    """선 P 를 기준으로 한 중심 후보: P 에서 차로 안쪽(sign=+1 오른쪽, -1 왼쪽)으로 띄운 선.
    띄우는 거리 = 맞은편 선 Q 까지 거리의 절반. 단 half + MEM_HUG_EXTRA 를 넘지 않음
    (코너에서 차로가 넓어져도 이 선에서 반 차로 폭 남짓만 떨어져서 감). Q 가 없거나 너무 멀면 half."""
    off = np.full(len(P), half)
    if Q is not None and len(Q) >= 3:
        d = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(axis=2))
        j = d.argmin(axis=1)
        dm = d[np.arange(len(P)), j]
        pair = (j > 0) & (j < len(Q) - 1) & (dm >= WIDTH_MIN - 0.05) & (dm <= WIDTH_MAX + 0.05)
        off[pair] = np.minimum(dm[pair] / 2.0, half + MEM_HUG_EXTRA)
        off = _smooth(off, 7)
    dx, dy = np.gradient(P[:, 0]), np.gradient(P[:, 1])
    n = np.hypot(dx, dy) + 1e-9
    return np.stack([P[:, 0] + sign * off * dy / n, P[:, 1] - sign * off * dx / n], axis=1)


def _center_from_lines(L, R, half):
    """기억한 왼쪽 선 L, 오른쪽 선 R (차 좌표 (N,2), 진행 순서; 없으면 None) 로 따라갈 경로를 만든다.
    - 직선: 두 선의 한가운데
    - 커브: 안쪽 선(휘는 쪽 선)을 기준으로, 그 선에서 반 차로 폭 남짓 떨어진 선.
      (이 트랙의 코너는 안쪽 선이 먼저 급하게 휘고 바깥 선은 한참 직진하다 크게 돈다. 바깥 선 기준으로 가면
       꺾기 시작이 늦어 LIMO 회전반경으로는 못 돌았음: 실차 기록. 안쪽 선 기준이면 일찍 꺾기 시작하고 바깥 여유가 남음)
      안쪽 선 기억이 끝나는 곳부터는 바깥 선 기준 경로로 이어 감
    반환 ((M,2) 경로, 커브 방향 [-1 왼쪽, 0 직선, +1 오른쪽]) 또는 (None, 0)"""
    ok_l = L is not None and len(L) >= 3
    ok_r = R is not None and len(R) >= 3
    if not ok_l and not ok_r:
        return None, 0
    if not ok_r:
        return _side_path(L, None, +1, half), 0
    if not ok_l:
        return _side_path(R, None, -1, half), 0
    c_l = _side_path(L, R, +1, half)
    c_r = _side_path(R, L, -1, half)
    hl, hr = _heading_change(L), _heading_change(R)
    turn = hl if abs(hl) > abs(hr) else hr
    if abs(turn) < MEM_TURN_MIN:
        prim, sec, direction = (c_l, c_r, 0) if len(c_l) >= len(c_r) else (c_r, c_l, 0)
    elif turn > 0:
        prim, sec, direction = c_r, c_l, 1          # 오른쪽 커브 -> 오른쪽(안쪽) 선 기준
    else:
        prim, sec, direction = c_l, c_r, -1
    # 기준 경로가 닿지 않는 앞/뒤 구간은 다른 쪽 경로로 채움
    d0 = np.hypot(sec[:, 0] - prim[0, 0], sec[:, 1] - prim[0, 1])
    d1 = np.hypot(sec[:, 0] - prim[-1, 0], sec[:, 1] - prim[-1, 1])
    i0, i1 = int(np.argmin(d0)), int(np.argmin(d1))
    t0 = prim[min(2, len(prim) - 1)] - prim[0]
    t1 = prim[-1] - prim[max(0, len(prim) - 3)]
    before = sec[:i0][((sec[:i0] - prim[0]) @ t0) < 0]
    after = sec[i1 + 1:][((sec[i1 + 1:] - prim[-1]) @ t1) > 0]
    return np.vstack([before, prim, after]), direction


class LineMemory:
    """왼쪽 선과 오른쪽 선을 각각 바닥(odom) 좌표로 기억한다.
    카메라는 차 앞 약 55cm 부터, 그것도 좁은 폭만 보여서 커브에서는 안쪽 선이 금방 화면 밖으로 나간다.
    선 자체를 기억해 두면, 화면에서 사라진 뒤에도 "두 선의 한가운데"를 계속 알 수 있다.
    (처음엔 중심선만 기억했는데, 안쪽 선이 사라지자 바깥 선만으로 중심을 다시 계산해서 커브에서 직진했음: 실차 기록)
    매 프레임, 선마다: [기억 중 이번에 보인 구간보다 차 쪽] + [이번에 보인 선] + [기억 중 이번에 보인 구간보다 먼 쪽]
    pose = (x, y, yaw): /odom 값 (x 앞, y 왼쪽, yaw 반시계 +). 뒤 차축 중심의 위치로 가정."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.lines = {'left': None, 'right': None}     # (N, 2) odom 좌표, 진행 순서
        self.bad = {'left': 0, 'right': 0}
        self.seen = {'left': None, 'right': None}      # 마지막으로 그 선을 봤을 때의 차 위치
        self.err = None                                # 이번에 본 선과 기억한 선의 어긋남 (m)

    @staticmethod
    def to_world(X, Y, pose):
        ox, oy, th = pose
        c, s_ = math.cos(th), math.sin(th)
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        return np.stack([ox + Y * c + X * s_, oy + Y * s_ - X * c], axis=1)

    @staticmethod
    def to_car(W, pose):
        ox, oy, th = pose
        c, s_ = math.cos(th), math.sin(th)
        dx, dy = W[:, 0] - ox, W[:, 1] - oy
        return dx * s_ - dy * c, dx * c + dy * s_          # (오른쪽 X, 앞 Y)

    def _merge(self, side, new):
        old = self.lines[side]
        if old is None or len(old) < 2:
            self.lines[side], self.bad[side] = new, 0
            return None
        d_a = np.hypot(old[:, 0] - new[0, 0], old[:, 1] - new[0, 1])
        i = int(np.argmin(d_a))
        m = min(len(new), 15)
        err = float(np.median(np.sqrt(((new[:m, None, :] - old[None, :, :]) ** 2).sum(axis=2)).min(axis=1)))
        if d_a[i] > MEM_JUMP:                       # 기억한 선과 동떨어진 선 -> 일단 무시, 계속되면 교체
            self.bad[side] += 1
            if self.bad[side] >= MEM_BAD_FRAMES:
                self.lines[side], self.bad[side] = new, 0
            return err
        self.bad[side] = 0
        d_b = np.hypot(old[:, 0] - new[-1, 0], old[:, 1] - new[-1, 1])
        k = int(np.argmin(d_b))
        tail = old[k + 1:] if (k > i and d_b[k] <= MEM_TAIL_TOL) else old[:0]
        self.lines[side] = np.vstack([old[:i], new, tail])
        return err

    def update(self, pose, left=None, right=None):
        """left/right: 이번 프레임에 본 선 (X, Y) (차 좌표, 가까운 쪽부터) 또는 None.
        반환 {'left': (N,2) 또는 None, 'right': ...} - 차 좌표로 본 기억한 선"""
        errs = []
        for side, cur in (('left', left), ('right', right)):
            if cur is not None and len(cur[0]) >= 5:
                e = self._merge(side, self.to_world(cur[0], cur[1], pose))
                self.seen[side] = (pose[0], pose[1])
                if e is not None:
                    errs.append(e)
            elif self.lines[side] is not None and self.seen[side] is not None and math.hypot(
                    pose[0] - self.seen[side][0], pose[1] - self.seen[side][1]) > MEM_LINE_KEEP:
                self.lines[side] = None             # 너무 오래(멀리) 못 본 선은 잊음
        self.err = max(errs) if errs else None
        out = {'left': None, 'right': None}
        for side in ('left', 'right'):
            W = self.lines[side]
            if W is None or len(W) < 3:
                continue
            Xc, Yc = self.to_car(W, pose)
            d0 = np.hypot(Xc, Yc)
            i0 = int(np.argmin(d0))
            if d0[i0] > MEM_FAR:
                self.lines[side] = None
                continue
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(Xc), np.diff(Yc)))])
            k0 = int(np.searchsorted(arc, arc[i0] - MEM_BEHIND))
            k1 = int(np.searchsorted(arc, arc[i0] + MEM_AHEAD))
            if k0 > 0 or k1 < len(W):
                self.lines[side] = W = W[k0:k1]
                Xc, Yc = Xc[k0:k1], Yc[k0:k1]
            if len(W) >= 3:
                out[side] = np.stack([Xc, Yc], axis=1)
        return out


class LaneEngine:
    def __init__(self):
        self.geo = Geometry()
        self.mem = LineMemory()
        self.blind_t = None             # 선이 안 보여 기억만으로 달리기 시작한 시각
        self.yaw = None                 # /odom 방향 (rad, 연속값). 없으면 None
        self.locked = False             # 코너 고정 조향 중인지
        self.lock_dir = 0
        self.lock_yaw = None
        self.lock_t = 0.0
        self.lock_block = False
        self.unlock_t = -1e9
        self.prev_center = 0.0          # F_REF 근처 차로 중심 (오른쪽 +, m)
        self.prev_lines = {'left': None, 'right': None}   # 지난 프레임 선 점열 (x, f)
        self.prev_path = None           # 지난 프레임 중심 점열 (x, f)
        self.lane_width = LANE_WIDTH    # 양쪽 선이 다 보일 때 실제로 잰 폭으로 계속 갱신 (선 고르기에 사용)
        self.single_width = LANE_WIDTH  # 직선 구간에서만 기억하는 폭 (한쪽 선만 보일 때 중심 잡기에 사용)
        self.last_valid_t = None
        self.angle = 0.0                # 평활된 조향각 (왼쪽 +)
        self.frags = []
        self.tracked = {'left': False, 'right': False}
        self.stopped = False
        self.resume_count = 0
        self.prev_ff = 0.0
        self.ff_used = 0.0
        self.latch = 0.0
        self.mode = 'STRAIGHT'
        self.curve_dir = 0
        self.last_v = 0.0
        self.in_count = 0
        self.out_count = 0
        self.latch_clear = 0

    # ---------- 1. 노란색 마스크 (원본에서, 밝기 보정 없음) ----------
    @staticmethod
    def yellow_mask(image, tune):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lo = np.array([tune['h_min'], tune['s_min'], tune['v_min']], np.uint8)
        hi = np.array([max(tune['h_min'], tune['h_max']), 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        # (3x3 열기 연산은 먼 곳의 가는 노란선을 점선처럼 끊어서 뺐음. 작은 잡티는 BEV 덩어리 크기로 거름)
        return mask

    # ---------- 2. BEV 에서 선 덩어리 찾기 ----------
    def _components(self, bev):
        """연결된 노란 덩어리마다 (가까운 쪽 x, 미터 점들, 픽셀) 반환."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bev, connectivity=8)
        comps = []
        self.frags = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < FRAG_MIN_PIX:
                continue
            py, px = np.nonzero(labels == i)
            xm, fm = self.geo.px_to_m(px, py)
            d = np.hypot(xm, fm + CAMERA_FORWARD_OFFSET)
            if stats[i, cv2.CC_STAT_AREA] < LINE_MIN_PIX or d.max() - d.min() < LINE_MIN_SPAN:
                self.frags.append(dict(xm=xm, fm=fm, px=px, py=py, d=d))
                continue
            near = d <= d.min() + 0.08
            comps.append(dict(near_x=float(xm[near].mean()), near_f=float(fm[near].mean()),
                              xm=xm, fm=fm, px=px, py=py))
        return comps

    def _expected(self, c, sign):
        """지난 중심선(없으면 차 중심)에서 반폭 떨어진 예상 위치와의 차이."""
        if self.prev_path is not None:
            cx, cf = self.prev_path
            order = np.argsort(cf)
            center = float(np.interp(c['near_f'], cf[order], cx[order]))
        else:
            center = 0.0
        return abs(c['near_x'] - (center + sign * self.lane_width / 2.0))

    def _track_error(self, c, side):
        """지난 프레임 같은 쪽 선과의 거리 (중앙값, m). 기록 없으면 None."""
        prev = self.prev_lines[side]
        if prev is None:
            return None
        sel = np.linspace(0, len(c['xm']) - 1, min(150, len(c['xm']))).astype(int)
        dx = c['xm'][sel, None] - prev[0][None, :]
        df = c['fm'][sel, None] - prev[1][None, :]
        return float(np.median(np.sqrt(dx * dx + df * df).min(axis=1)))

    def _pick_lines(self, comps):
        """왼쪽/오른쪽 선 고르기.
        1) 지난 프레임 같은 쪽 선과 TRACK_TOL 안에 있는 덩어리는 그대로 이어 감 (곡선에서 선이 차 중심을 넘어가도 유지)
           단, 왼쪽 선이 차보다 TRACKED_SIDE_LIMIT 넘게 오른쪽이면(반대도) 잘못 잡은 것으로 보고 버림
        2) 이어 갈 선이 없는 쪽은: 차 중심 기준 그쪽에 있는 덩어리 중 가장 안쪽 = 내 차로 선
           (옆 차로 선은 항상 더 바깥이라, 잘못 잡았던 상태에서도 바로 돌아옴)
        3) 양쪽 간격이 기억한 폭과 다르면, 새로 고른 쪽을 다음 후보로 바꿔 봄
        4) 새로 고른 선이 혼자이고 지난 기록이 있는데 예상 위치와 너무 다르면 버림 (옆 차로 선 오인 방지)"""
        picked = {'left': None, 'right': None}
        tracked = {'left': False, 'right': False}
        used = set()
        for side, sign in (('left', -1), ('right', +1)):
            best = None
            for i, c in enumerate(comps):
                if (side == 'left' and c['near_x'] > TRACKED_SIDE_LIMIT) or \
                        (side == 'right' and c['near_x'] < -TRACKED_SIDE_LIMIT):
                    continue
                err = self._track_error(c, side)
                if err is not None and err <= TRACK_TOL and (best is None or err < best[0]):
                    best = (err, i)
            if best is not None and best[1] not in used:
                picked[side] = (self._expected(comps[best[1]], sign), comps[best[1]])
                tracked[side] = True
                used.add(best[1])

        def width_ok(lc, rc):
            if abs(lc['near_f'] - rc['near_f']) > 0.10:      # 곡선 등으로 비교 위치가 다르면 판단 보류
                return True
            # 새로 고르는 선은 기억한 폭과 비슷해야 함 (옆 차로 선 오인 방지). 이어 가는 선끼리는 넓어져도 허용
            return abs((rc['near_x'] - lc['near_x']) - self.lane_width) <= WIDTH_TOL

        for side, sign in (('left', -1), ('right', +1)):
            if picked[side] is not None:
                continue
            lst = [(i, c) for i, c in enumerate(comps) if i not in used and
                   (c['near_x'] <= SIDE_MARGIN if side == 'left' else c['near_x'] >= -SIDE_MARGIN)]
            lst.sort(key=lambda ic: sign * ic[1]['near_x'])          # 안쪽부터
            other = picked['right' if side == 'left' else 'left']
            for i, c in lst:
                if other is not None:
                    lc, rc = (c, other[1]) if side == 'left' else (other[1], c)
                    if not width_ok(lc, rc):
                        continue
                err = self._expected(c, sign)
                if other is None and self.prev_path is not None and err > EXPECT_TOL:
                    continue
                picked[side] = (err, c)
                used.add(i)
                break
        # 둘 다 새로 골랐는데 폭이 안 맞으면 예상 위치에서 먼 쪽 버림
        if picked['left'] and picked['right'] and not (tracked['left'] or tracked['right']):
            if not width_ok(picked['left'][1], picked['right'][1]):
                if picked['left'][0] > picked['right'][0]:
                    picked['left'] = None
                else:
                    picked['right'] = None
        self.tracked = tracked
        return picked

    def _extend(self, comp):
        """선의 먼 쪽 끝에 이어지는 작은 조각들을 붙인다 (먼 곳 곡선이 점선처럼 끊겨 보이는 경우)."""
        xm, fm, px, py = comp['xm'], comp['fm'], comp['px'], comp['py']
        used = set()
        for _ in range(6):
            d = np.hypot(xm, fm + CAMERA_FORWARD_OFFSET)
            far = d >= d.max() - 0.05
            ex, ef, dmax = xm[far], fm[far], d.max()
            best = None
            for k, fr in enumerate(self.frags):
                if k in used or fr['d'].max() <= dmax:
                    continue
                gap = np.sqrt((fr['xm'][:, None] - ex[None, :]) ** 2 + (fr['fm'][:, None] - ef[None, :]) ** 2).min()
                if gap <= FRAG_JOIN and (best is None or gap < best[0]):
                    best = (gap, k)
            if best is None:
                break
            fr = self.frags[best[1]]
            used.add(best[1])
            xm, fm = np.concatenate([xm, fr['xm']]), np.concatenate([fm, fr['fm']])
            px, py = np.concatenate([px, fr['px']]), np.concatenate([py, fr['py']])
        return dict(comp, xm=xm, fm=fm, px=px, py=py)

    def _make_line(self, comp, side):
        comp = self._extend(comp)
        xs, fs = _polyline_by_distance(comp['xm'], comp['fm'])
        if xs is None or len(xs) < 5:
            return None
        return LaneLine(side, xs, fs, comp['px'], comp['py'], comp['near_x'])

    # ---------- 3. 차로 중심선 ----------
    @staticmethod
    def _merge(cands):
        """중심 후보 점열(선마다 하나)을 합친다. 점열의 원래 순서(가까운 쪽 -> 먼 쪽)를 유지한다.
        (차와의 거리로 다시 묶으면 급커브에서 안쪽 원호가 뭉개져 경로가 짧아지는 문제가 있었음)
        반환 (x, f, 두 후보 간 불일치 m)"""
        if len(cands) == 1:
            return _smooth(cands[0][0]), _smooth(cands[0][1]), 0.0
        (ax, af), (bx, bf) = cands
        # 더 긴 쪽을 기준으로, 다른 쪽의 가장 가까운 점과 평균
        if len(bx) > len(ax):
            (ax, af), (bx, bf) = (bx, bf), (ax, af)
        dx = ax[:, None] - bx[None, :]
        df = af[:, None] - bf[None, :]
        dist = np.sqrt(dx * dx + df * df)
        j = dist.argmin(axis=1)
        dmin = dist[np.arange(len(ax)), j]
        close = dmin < 0.15
        xs = np.where(close, (ax + bx[j]) / 2.0, ax)
        fs = np.where(close, (af + bf[j]) / 2.0, af)
        # 두 선이 겹치는 구간의 불일치 (겹치는 곳이 없으면 큰 값)
        both = dmin[dmin < 0.30]
        gap = float(np.median(dmin)) if len(both) < 3 else float(np.median(both))
        return _smooth(xs), _smooth(fs), gap

    # ---------- 4. 전체 처리 ----------
    def process(self, image, tune, now, pose=None):
        """image: 640x480 BGR (왜곡 보정 후). pose: /odom 의 (x, y, yaw) - mem 방식에서만 사용. 결과 dict 반환."""
        t0 = time.perf_counter()
        self.yaw = None if pose is None else pose[2]
        mask = self.yellow_mask(image, tune)
        bev = self.geo.warp(mask)
        bev = np.where(bev > 127, 255, 0).astype(np.uint8)
        bev = cv2.morphologyEx(bev, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # 오래 놓쳤으면 기준 중심을 가운데로 되돌림
        if self.last_valid_t is not None and now - self.last_valid_t > 2.0:
            self.prev_center = 0.0

        if self.last_valid_t is not None and now - self.last_valid_t > LOST_HOLD_SEC:
            self.prev_lines = {'left': None, 'right': None}
            self.prev_path = None
            self.prev_center = 0.0
        comps = self._components(bev)
        picked = self._pick_lines(comps)
        left = self._make_line(picked['left'][1], 'left') if picked['left'] else None
        right = self._make_line(picked['right'][1], 'right') if picked['right'] else None

        result = dict(mask=mask, bev=bev, left=left, right=right, valid=False,
                      source='none', center=None, target=None, lateral=0.0,
                      raw_angle=0.0, note='', comps=len(comps), lane_width=self.lane_width,
                      ff_angle=0.0, ff_w=0.0, cross=None, tilt=None, wall=None,
                      mem_path=None, mem_angle=None, mem_target=None, mem_err=None, mem_lines=None, mem_turn=0)
        # 차로 폭 (선에 수직으로 잼. 가로로 재면 곡선에서 실제보다 넓게 나옴: 실차 기록에서 60cm 가 76cm 로 나왔음)
        #  - 양쪽 선이 다 보이면: 지금 잰 폭의 절반씩 띄워 두 선의 한가운데를 중심으로
        #  - 한쪽 선만 보이면: 직선 구간에서 기억한 폭(single_width)의 절반을 띄움
        #    (곡선 입구는 폭이 벌어지는데, 그 폭을 기억한 채 바깥 선만 보고 돌면 중심이 안쪽으로 치우쳐 안쪽 선을 밟음)
        half = self.single_width / 2.0
        if left is not None and right is not None:
            w = _perp_width(left, right)
            both_tracked = self.tracked['left'] and self.tracked['right']
            if w is not None and WIDTH_MIN <= w <= WIDTH_MAX and (
                    both_tracked or abs(w - self.lane_width) <= WIDTH_TOL):
                self.lane_width = 0.5 * self.lane_width + 0.5 * w
                half = self.lane_width / 2.0
                tl_, tr_ = _line_tilt(left), _line_tilt(right)
                # 두 선이 곧고 서로 나란할 때만 (커브 입구는 기울기가 작아도 두 선이 벌어지고 있어서 제외)
                if max(abs(tl_), abs(tr_)) < STRAIGHT_TILT and abs(tl_ - tr_) < PARALLEL_TILT:
                    self.single_width = 0.8 * self.single_width + 0.2 * w
            else:
                half = self.lane_width / 2.0
        cands = []
        if left is not None:
            cands.append(_offset(left.xs, left.fs, +1, half))
        if right is not None:
            cands.append(_offset(right.xs, right.fs, -1, half))
        if cands:
            cx, cf, gap = self._merge(cands)
            if len(cands) == 2 and gap > 0.15 and not (self.tracked['left'] and self.tracked['right']):
                # 두 선이 같은 차로라고 보기 어려움 -> 예상 위치에서 더 먼 쪽 버림
                if picked['left'][0] <= picked['right'][0]:
                    right, cands = None, cands[:1]
                else:
                    left, cands = None, cands[1:]
                cx, cf, _ = self._merge(cands)
                result['note'] = f'좌우 불일치 {gap * 100:.0f}cm -> 한쪽 버림'
                result.update(left=left, right=right)
            source = 'both' if left is not None and right is not None else (
                'left_only' if left is not None else 'right_only')
            dist = np.hypot(cx, cf + CAMERA_FORWARD_OFFSET)
            ld = tune['ld_cm'] / 100.0
            # 목표점: 차에서 ld 이상 떨어져 있고, 보이는 경로 시작점에서 경로를 따라 ARC_MIN 이상 간 점.
            # (경로가 0.7m 앞부터 보이는데 ld 가 70cm면 목표가 경로 맨 앞 점이 되어,
            #  곡선 입구에서 앞쪽 곡선을 무시하고 반대로 꺾는 일이 실차 기록에서 있었음)
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cx), np.diff(cf)))])
            ok = (dist >= ld) & (arc >= ARC_MIN)
            k = int(np.argmax(ok)) if np.any(ok) else len(cx) - 1
            tx, tf = float(cx[k]), float(cf[k])
            raw = target_to_inner_angle(tx, tf + CAMERA_FORWARD_OFFSET)
            j = int(np.argmin(np.abs(cf - F_REF)))
            ref = float(cx[j]) if abs(cf[j] - F_REF) < 0.15 else float(cx[0])
            self.prev_center = float(np.clip(0.6 * self.prev_center + 0.4 * ref, -0.35, 0.35))
            self.prev_path = (cx, cf)
            self.prev_lines = {'left': None if left is None else (left.xs, left.fs),
                               'right': None if right is None else (right.xs, right.fs)}
            result['lane_width'] = 2.0 * half
            # 급커브 미리 꺾기: 가장 잘 맞는 선의 원으로 커브 반경과 위치 추정
            ff_angle, ff_w = 0.0, 0.0
            best = None
            for line, side in ((left, 'left'), (right, 'right')):
                if line is None or len(line.xs) < 12:
                    continue
                # 먼 쪽 절반만 맞춤: 직선+곡선이 섞인 선 전체를 원으로 맞추면 오차가 커서 곡선 입구에서 판단을 못 했음
                h = len(line.xs) // 2
                R, a_, b_, rms = _fit_circle(line.xs[h:], line.fs[h:] + CAMERA_FORWARD_OFFSET)
                if R < FF_MAX_R and rms < FF_MAX_RMS and (best is None or rms < best[4]):
                    inside = (a_ > line.xs[0]) if side == 'left' else (a_ < line.xs[0])
                    path_r = R - half if inside else R + half
                    best = (path_r, a_, b_, side, rms)
            if best is not None:
                path_r, a_, b_, _, _ = best
                start_b = tune.get('ff_cm', 45) / 100.0
                ff_w = 0.0 if start_b <= 0 else float(np.clip((start_b - b_) / FF_RAMP, 0.0, 1.0))
                ff_angle = _inner_angle_for_radius(path_r) * (-1.0 if a_ > 0 else 1.0)
            result.update(ff_angle=ff_angle, ff_w=ff_w)
            # 바깥 선이 차 진행 방향(차 중심선 x=0)을 가로지르는 거리 (커브가 아직 계속되는지 판단용)
            cross = None
            for line, side in ((left, 'left'), (right, 'right')):
                if line is None:
                    continue
                sel = line.xs >= 0.0 if side == 'left' else line.xs <= 0.0
                starts_outside = line.xs[0] < 0.0 if side == 'left' else line.xs[0] > 0.0
                if starts_outside and np.any(sel):
                    k2 = int(np.argmax(sel))
                    d2 = math.hypot(line.xs[k2], line.fs[k2] + CAMERA_FORWARD_OFFSET)
                    if cross is None or d2 < cross[1]:
                        cross = (side, d2)
            result['cross'] = cross
            # 정면 막힘: 차 정면 폭 10cm 통로 안에 노란 점(끊긴 조각 포함)이 있으면 그 거리.
            # 커브 중 바깥 선이 끊겨서 먼 쪽이 따로 떨어진 조각으로 보여도 "아직 앞이 막혀 있음"을 알 수 있음
            wall = None
            for c in list(comps) + list(self.frags):
                inside = np.abs(c['xm']) < 0.05
                if np.any(inside):
                    d3 = float(np.min(np.hypot(c['xm'][inside], c['fm'][inside] + CAMERA_FORWARD_OFFSET)))
                    wall = d3 if wall is None else min(wall, d3)
            result['wall'] = wall
            # 선의 기울기: 가까운 쪽 25cm 구간의 진행 방향이 차 정면에서 몇 도 틀어졌는지 (오른쪽으로 기울면 +)
            # 커브 한가운데서는 보이는 선이 짧고 거의 곧아서 원 맞추기로는 직선처럼 보이지만, 기울기는 크게 나옴
            tilt = None
            for line in (left, right):
                if line is None or len(line.xs) < 3:
                    continue
                ang = _line_tilt(line)
                if tilt is None or abs(ang) > abs(tilt):
                    tilt = ang
            result['tilt'] = tilt
            result.update(valid=True, source=source, center=(cx, cf), target=(tx, tf),
                          lateral=float(cx[0]), raw_angle=raw)
            self.last_valid_t = now
        # 선 기억 (mem): 이번에 본 양쪽 선을 기억에 반영 -> 기억한 두 선으로 차 옆부터 이어진 중심선 -> 가까운 목표점으로 조향
        if MEM and pose is not None:
            cur = {}
            for side in ('left', 'right'):
                line = result[side]
                cur[side] = None if line is None else (line.xs, line.fs + CAMERA_FORWARD_OFFSET)
            mem_lines = self.mem.update(pose, cur['left'], cur['right'])
            result['mem_err'] = self.mem.err
            result['mem_lines'] = mem_lines
            C, result['mem_turn'] = _center_from_lines(mem_lines['left'], mem_lines['right'], self.single_width / 2.0)
            if C is not None and len(C) >= 5:
                Xc, Yc = _smooth(C[:, 0], 7), _smooth(C[:, 1], 7)
                i0 = int(np.argmin(np.hypot(Xc, Yc)))
                result['mem_path'] = (Xc, Yc - CAMERA_FORWARD_OFFSET)       # 그리기용 (카메라 앞 거리 기준)
                out = _pursuit_on_path(Xc, Yc, i0, tune.get('mem_ld', 50) / 100.0)
                if out is not None:
                    result.update(mem_angle=out[0], mem_target=(out[1][0], out[1][1] - CAMERA_FORWARD_OFFSET))
        result['ms'] = (time.perf_counter() - t0) * 1000.0
        return result

    def _lock_done(self, now, yaw_limit):
        """코너 고정 조향을 시작한 뒤 yaw_limit 만큼 돌았는지 (odom 이 없으면 속도로 추정).
        어떤 경우든 LOCK_MAX_SEC 를 넘기면 끝낸다 (실차 기록: 주행 코드를 켠 채 차를 들어 옮기자 odom 이 멈춰서
        고정 조향이 12초 동안 안 풀렸음)."""
        if now - self.lock_t > LOCK_MAX_SEC:
            return True
        if self.yaw is not None and self.lock_yaw is not None:
            return abs(self.yaw - self.lock_yaw) >= yaw_limit
        radius = WHEELBASE / math.tan(MAX_STEER) + TRACK / 2.0
        return now - self.lock_t > yaw_limit * radius / max(0.10, self.last_v)

    # ---------- 5. 명령 (조향, 속도, 정지 판단) ----------
    def command(self, result, tune, now, camera_ok=True):
        """반환: (speed, inner_angle, state, reason). self.mode = STRAIGHT / APPROACH / CURVE"""
        vmax = tune['vmax_cm'] / 100.0
        if not camera_ok:
            return 0.0, 0.0, 'STOP', '카메라 영상 끊김'
        use_mem = MEM and result.get('mem_angle') is not None
        if result['valid']:
            self.blind_t = None
        if result['valid'] and self.stopped:
            self.resume_count += 1
            if self.resume_count < RESUME_FRAMES:
                return 0.0, 0.0, 'WAIT', f'재출발 확인 중 {self.resume_count}/{RESUME_FRAMES}'
            self.stopped = False
        if result['valid']:
            self.resume_count = 0
            raw = result['raw_angle']
            ff = result['ff_angle'] * result['ff_w']
            # 커브 판단이 한두 프레임 끊겨도 미리 꺾기가 풀리지 않게, 이전 값을 프레임당 15%씩만 줄임
            if ff * self.prev_ff >= 0 and abs(ff) < abs(self.prev_ff) * 0.85:
                ff = self.prev_ff * 0.85
            self.prev_ff = ff

            # ---- 모드 판단: 곡률(원 맞추기) + 선 기울기 + 바깥 선이 정면을 가로지르는지 ----
            tilt, cross = result.get('tilt'), result.get('cross')
            ff_strong = result['ff_w'] >= 0.5 and abs(result['ff_angle']) >= LATCH_MIN
            tilt_dir = 0 if tilt is None else (-1 if tilt > 0 else 1)       # 오른쪽으로 기울면 오른쪽(-) 조향
            ff_dir = 0 if not ff_strong else (1 if result['ff_angle'] > 0 else -1)
            in_deg = math.radians(tune.get('curve_in_deg', 25))
            # 경로 추종(PP)이 커브 방향과 반대로 PP_DISAGREE 이상 꺾으라고 하면 곡선 모드를 믿지 않음
            # (실차 영상: 지그재그 구간에서 차가 오른쪽 선 위에 있는데 곡선 모드가 오른쪽 27도를 강제한 적 있음)
            pp = result['raw_angle']
            if self.mode != 'CURVE':
                cand = ff_dir if ff_dir else tilt_dir
                evidence = (ff_strong or (tilt is not None and abs(tilt) >= in_deg)) and not (
                    pp * cand < 0 and abs(pp) >= PP_DISAGREE)
                self.in_count = self.in_count + 1 if evidence else 0
                if self.in_count >= 2:
                    self.mode = 'CURVE'
                    self.curve_dir = cand
                    self.out_count = 0
                else:
                    self.mode = 'APPROACH' if result['ff_w'] > 0 else 'STRAIGHT'
            else:
                cross_ok = cross is not None and cross[1] < LATCH_CROSS and (
                    (cross[0] == 'left') == (self.curve_dir < 0))
                wall = result.get('wall')
                wall_ok = wall is not None and wall < LATCH_CROSS
                stay = (ff_strong and ff_dir == self.curve_dir) or cross_ok or wall_ok or (
                    tilt is not None and abs(tilt) >= CURVE_STAY and tilt_dir == self.curve_dir)
                disagree = pp * self.curve_dir < 0 and abs(pp) >= PP_DISAGREE
                self.out_count = 0 if (stay and not disagree) else self.out_count + (2 if disagree else 1)
                if self.out_count >= int(tune.get('curve_out', 6)):
                    self.mode, self.in_count = 'STRAIGHT', 0

            # ---- 조향 ----
            reason = result['source']
            self.latch = 0.0
            if use_mem:
                raw = result['mem_angle']           # 기억한 선으로 만든 경로 위의 가까운 목표점을 따라감
                reason += ' 기억경로'
            else:
                if MEM:
                    reason += ' (기억 없음->기본)'
                if tune.get('pre_steer', 0) and ff * raw >= 0 and abs(ff) > abs(raw):
                    raw = ff
                    reason += f' +미리꺾기 {result["ff_w"] * 100:.0f}%'
            # 코너 고정 조향: 바깥 선이 정면 turn_in 안으로 들어오면 시작, 충분히 돌면 끝
            front = result.get('wall')
            if cross is not None and (front is None or cross[1] < front):
                front = cross[1]
            if self.mode != 'CURVE' and now - self.unlock_t > 1.5:
                self.lock_block = False             # 같은 코너에서 다시 걸리지 않게, 곡선 모드를 한 번 벗어나야 다음 고정 허용
            if not self.locked:
                if (tune.get('latch', 1) and not self.lock_block and self.mode == 'CURVE' and self.curve_dir != 0
                        and front is not None and front <= tune.get('turn_in_cm', 75) / 100.0
                        and not (raw * self.curve_dir < 0 and abs(raw) >= PP_DISAGREE)):
                    self.locked, self.lock_dir, self.lock_yaw, self.lock_t = True, self.curve_dir, self.yaw, now
            else:
                # 충분히 돌면 고정을 풂. (도는 중에는 경로 추종의 의견을 듣지 않음: 경로는 바깥 선을 따라 늦게 휘어서,
                #  일찍 꺾은 차를 "안쪽으로 들어왔다"고 보고 반대로 꺾게 만들었음 - 시뮬레이션에서 확인)
                if self._lock_done(now, math.radians(tune.get('turn_deg', 60))) or not tune.get('latch', 1):
                    self.locked, self.lock_block, self.unlock_t = False, True, now
            if self.locked:
                floor = min(MAX_STEER, math.radians(tune.get('latch_deg', 25)))
                raw = self.lock_dir * max(floor, abs(raw) if raw * self.lock_dir > 0 else 0.0)
                self.latch = raw
                reason += ' 코너고정'
            alpha = max(5, tune.get('smooth', 50)) / 100.0
            self.angle = alpha * raw + (1 - alpha) * self.angle

            # ---- 속도 (모드별로 따로) ----
            cv_ = tune.get('curve_v_cm', 22) / 100.0
            if self.mode == 'CURVE' or self.locked:
                v = cv_
            else:
                ratio = min(1.0, abs(self.angle) / MAX_STEER)
                v = vmax - (vmax - min(V_MIN, vmax)) * ratio
                if self.mode == 'APPROACH':                      # 곡선이 보이면 미리 감속
                    v = min(v, cv_ + (vmax - cv_) * (1 - result['ff_w']))
            self.last_v = v
            return v, self.angle, 'TRACK', f'{self.mode} {reason}'
        if self.locked and not self.stopped:
            # 코너를 도는 중에 선이 안 보임 (카메라 시야가 좁아 흔한 일) -> 멈추지 않고 고정 조향으로 계속 돎
            if not self._lock_done(now, LOCK_BLIND_YAW):
                floor = min(MAX_STEER, math.radians(tune.get('latch_deg', 25)))
                alpha = max(5, tune.get('smooth', 50)) / 100.0
                self.angle = alpha * self.lock_dir * floor + (1 - alpha) * self.angle
                return tune.get('curve_v_cm', 22) / 100.0, self.angle, 'TURN', '코너 고정 조향 (선 안 보임)'
            self.locked, self.lock_block, self.unlock_t = False, True, now
        if use_mem and not self.stopped:
            # 선이 안 보여도 기억한 경로가 앞에 남아 있으면 그 경로를 따라감 (최대 MEM_BLIND_SEC)
            if self.blind_t is None:
                self.blind_t = now
            if now - self.blind_t <= MEM_BLIND_SEC:
                alpha = max(5, tune.get('smooth', 50)) / 100.0
                self.angle = alpha * result['mem_angle'] + (1 - alpha) * self.angle
                v = min(LOST_HOLD_SPEED, vmax, tune.get('curve_v_cm', 22) / 100.0)
                return v, self.angle, 'MEM', f'{self.mode} 선 안 보임 -> 기억한 경로로 주행'
        if self.last_valid_t is not None and now - self.last_valid_t <= LOST_HOLD_SEC:
            v = min(LOST_HOLD_SPEED, vmax)
            if self.mode == 'CURVE':
                v = min(v, tune.get('curve_v_cm', 22) / 100.0)
            return v, self.angle, 'HOLD', f'{self.mode} 선 놓침 -> 마지막 조향 유지'
        self.angle = 0.0
        self.prev_ff = 0.0
        self.latch = 0.0
        self.mode, self.in_count = 'STRAIGHT', 0
        self.stopped = True
        self.resume_count = 0
        self.blind_t = None
        self.locked = False
        self.mem.reset()
        return 0.0, 0.0, 'STOP', '선 없음'


# =====================================================================
# 디버그 화면
# =====================================================================
def _poly(img, pts, color, thick=2):
    pts = np.asarray(pts)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) >= 2:
        cv2.polylines(img, [np.int32(np.round(pts))], False, color, thick, cv2.LINE_AA)


def _put(img, text, org, color=(255, 255, 255), scale=0.5):
    """한글은 OpenCV 기본 글꼴로 안 나오므로 영문/숫자만 그린다."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_debug(engine, image, result, cmd, extra):
    geo = engine.geo
    pw, ph = 480, 360
    # 1) 원본 + 선/목표점
    view = image.copy()
    ys = np.linspace(F_MIN, F_MAX, 30)
    for xg in (X_MIN, X_MAX):
        _poly(view, geo.ground_to_image(np.full_like(ys, xg), ys), (0, 255, 255), 1)
    _poly(view, geo.ground_to_image(np.linspace(X_MIN, X_MAX, 30), np.full(30, F_MIN)), (0, 255, 255), 1)
    _poly(view, geo.ground_to_image(np.linspace(X_MIN, X_MAX, 30), np.full(30, F_MAX)), (0, 255, 255), 1)
    for line, color in ((result['left'], (255, 128, 0)), (result['right'], (0, 0, 255))):
        if line is not None:
            _poly(view, geo.ground_to_image(line.xs, line.fs), color, 3)
    if result['center'] is not None:
        _poly(view, geo.ground_to_image(*result['center']), (0, 255, 0), 2)
        tp = geo.ground_to_image([result['target'][0]], [result['target'][1]])[0]
        if np.all(np.isfinite(tp)):
            cv2.circle(view, (int(tp[0]), int(tp[1])), 7, (255, 0, 255), -1)
    p1 = cv2.resize(view, (pw, ph))
    _put(p1, 'camera + path', (8, 20))

    # 2) 노란색 마스크
    p2 = cv2.resize(cv2.cvtColor(result['mask'], cv2.COLOR_GRAY2BGR), (pw, ph))
    _put(p2, 'yellow mask (original)', (8, 20))

    # 3) BEV
    bev = cv2.cvtColor(result['bev'], cv2.COLOR_GRAY2BGR)
    bev[~geo.visible] = (40, 40, 40)
    for xg in np.arange(-0.4, 0.41, 0.1):
        px, _ = geo.m_to_px(xg, 0)
        cv2.line(bev, (int(px), 0), (int(px), geo.bev_h), (70, 70, 70) if abs(xg) > 1e-6 else (110, 110, 110), 1)
    for fg in np.arange(0.5, F_MAX, 0.1):
        _, py = geo.m_to_px(0, fg)
        cv2.line(bev, (0, int(py)), (geo.bev_w, int(py)), (70, 70, 70), 1)
    for line, color in ((result['left'], (255, 128, 0)), (result['right'], (0, 0, 255))):
        if line is not None:
            bev[line.py, line.px] = np.array(color) // 2 + 60
            _poly(bev, np.stack(geo.m_to_px(line.xs, line.fs), axis=1), color, 2)
    if result['center'] is not None:
        _poly(bev, np.stack(geo.m_to_px(*result['center']), axis=1), (0, 255, 0), 2)
        tx, ty = geo.m_to_px(*result['target'])
        cv2.circle(bev, (int(tx), int(ty)), 5, (255, 0, 255), -1)
    p3 = np.zeros((ph, pw, 3), np.uint8)
    if MEM:
        # 차 주변 지도: 좌우 ±0.6m, 카메라 앞 -0.40 ~ 1.50m (BEV 는 0.35m 부터라 그 아래는 기억으로만 채워짐)
        gx0, gf0, gf1 = -0.6, -0.40, F_MAX
        sc = ph / (gf1 - gf0)
        mw = int(round(1.2 * sc))
        cmap = np.zeros((ph, mw, 3), np.uint8)
        bw, bh = int(round((X_MAX - X_MIN) * sc)), int(round((F_MAX - F_MIN) * sc))
        bx = int(round((X_MIN - gx0) * sc))
        cmap[0:bh, bx:bx + bw] = cv2.resize(bev, (bw, bh), interpolation=cv2.INTER_NEAREST)

        def g2p(x, f):
            return np.stack([(np.asarray(x, np.float64) - gx0) * sc, (gf1 - np.asarray(f, np.float64)) * sc], axis=-1)
        r0, r1 = g2p(-0.11, -CAMERA_FORWARD_OFFSET + 0.26), g2p(0.11, -CAMERA_FORWARD_OFFSET - 0.06)
        cv2.rectangle(cmap, (int(r0[0]), int(r0[1])), (int(r1[0]), int(r1[1])), (160, 160, 160), 1)   # 차체
        ax = g2p(0.0, -CAMERA_FORWARD_OFFSET)
        cv2.circle(cmap, (int(ax[0]), int(ax[1])), 3, (255, 255, 255), -1)                             # 뒤 차축 중심
        for side, color in (('left', (255, 190, 120)), ('right', (130, 130, 255))):                   # 기억한 선 (연한 색)
            ml = (result.get('mem_lines') or {}).get(side)
            if ml is not None:
                _poly(cmap, g2p(ml[:, 0], ml[:, 1] - CAMERA_FORWARD_OFFSET), color, 1)
        if result.get('mem_path') is not None:
            _poly(cmap, g2p(*result['mem_path']), (0, 255, 255), 2)                                    # 중심 경로 (노랑)
        if result.get('mem_target') is not None:
            tp = g2p(*result['mem_target'])
            cv2.circle(cmap, (int(tp[0]), int(tp[1])), 5, (255, 0, 255), -1)
        off = (pw - mw) // 2
        p3[:, off:off + mw] = cmap
        _put(p3, 'BEV + remembered lines / path (yellow)', (8, 20))
    else:
        scale = ph / bev.shape[0]
        bev_s = cv2.resize(bev, (int(bev.shape[1] * scale), ph), interpolation=cv2.INTER_NEAREST)
        off = (pw - bev_s.shape[1]) // 2
        p3[:, off:off + bev_s.shape[1]] = bev_s
        _put(p3, 'BEV (0.1m grid)', (8, 20))

    # 4) 상태 글자
    p4 = np.zeros((ph, pw, 3), np.uint8)
    v, ang, state, reason = cmd
    color = {'TRACK': (0, 220, 0), 'HOLD': (0, 200, 255), 'MEM': (0, 255, 255), 'TURN': (0, 140, 255), 'STOP': (0, 0, 255),
             'WAIT': (0, 200, 255)}.get(state, (255, 255, 255))
    if not MEM:
        mem_text, mem_color = 'MEM: off (old steering)', (150, 150, 150)
    elif not extra.get('pose_ok', True):
        mem_text, mem_color = 'MEM: NO ODOM -> old steering', (0, 0, 255)
    elif result.get('mem_angle') is None:
        mem_text, mem_color = 'MEM: no path yet -> old steering', (0, 200, 255)
    else:
        n_mem = len(result['mem_path'][0]) if result.get('mem_path') is not None else 0
        mem_text = (f'MEM: ON  path {n_mem}pts  err '
                    + ('-' if result.get('mem_err') is None else f'{result["mem_err"] * 100:.1f}cm')
                    + f'  ld {extra["tune"].get("mem_ld", 0)}')
        mem_color = (0, 255, 255)
    lines = [
        (f'STATE: {state}', color, 0.8),
        (f'output: {extra.get("output", "OFF (view only)")}', (255, 255, 255), 0.5),
        (f'left line : {"OK" if result["left"] is not None else "-"}   '
         f'right line: {"OK" if result["right"] is not None else "-"}', (255, 255, 255), 0.55),
        (f'source: {result["source"]}   blobs: {result.get("comps", 0)}   width: {result.get("lane_width", 0) * 100:.0f}cm', (255, 255, 255), 0.5),
        (f'lateral (near): {result["lateral"] * 100:+.1f} cm (right +)', (255, 255, 255), 0.5),
        (f'steer: {math.degrees(ang):+.1f} deg (left +)', (255, 255, 255), 0.55),
        (f'speed: {v:.2f} m/s', (255, 255, 255), 0.55),
        (f'MODE: {getattr(engine, "mode", "-")}' + ('  [CORNER LOCK]' if getattr(engine, 'locked', False) else ''),
         {'CURVE': (0, 140, 255), 'APPROACH': (0, 220, 255)}.get(getattr(engine, 'mode', ''), (200, 200, 200)), 0.75),
        (f'  ff {math.degrees(result.get("ff_angle", 0)):+.0f}deg x {result.get("ff_w", 0) * 100:.0f}%   tilt '
         + ('-' if result.get('tilt') is None else f'{math.degrees(result["tilt"]):+.0f}deg')
         + ('   front -' if result.get('wall') is None else f'   front {result["wall"]:.2f}m'), (200, 200, 200), 0.5),
        (mem_text, mem_color, 0.5),
        (f'proc: {result["ms"]:.1f} ms   fps: {extra.get("fps", 0):.1f}   H {extra["tune"]["h_min"]}-{extra["tune"]["h_max"]}  '
         f'S>={extra["tune"]["s_min"]}  V>={extra["tune"]["v_min"]}', (200, 200, 200), 0.45),
        (f'Ld {extra["tune"]["ld_cm"]}  vmax {extra["tune"]["vmax_cm"]}  curve_v {extra["tune"].get("curve_v_cm", 0)}  '
         f'ff {extra["tune"].get("ff_cm", 0)}/{extra["tune"].get("pre_steer", 0)}  latch {extra["tune"].get("latch", 0)}/{extra["tune"].get("latch_deg", 0)}deg  '
         f'turn_in {extra["tune"].get("turn_in_cm", 0)}/{extra["tune"].get("turn_deg", 0)}deg  smooth {extra["tune"].get("smooth", 0)}  in {extra["tune"].get("curve_in_deg", 0)}  out {extra["tune"].get("curve_out", 0)}',
         (200, 200, 200), 0.42),
        ('space: freeze  s: save  q: quit', (150, 150, 150), 0.45),
    ]
    y = 34
    for text, c, s in lines:
        _put(p4, text, (12, y), c, s)
        y += 24
    return np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])


# =====================================================================
# 슬라이더 / 설정 저장
# =====================================================================
def load_tune():
    tune = dict(DEFAULT_TUNE)
    try:
        with open(TUNE_FILE) as f:
            saved = json.load(f)
        for k in tune:
            if k in saved:
                tune[k] = int(saved[k])
    except (OSError, ValueError):
        pass
    if PRESET:
        tune.update(PRESETS[PRESET])
    return tune


def save_tune(tune):
    if PRESET:          # 프리셋 주행은 평소 설정 파일을 덮어쓰지 않음
        return
    try:
        with open(TUNE_FILE, 'w') as f:
            json.dump(tune, f, indent=1)
    except OSError:
        pass


class Gui:
    WIN = 'simple_lane'
    LIMITS = {'h_min': 179, 'h_max': 179, 's_min': 255, 'v_min': 255, 'ld_cm': 150, 'vmax_cm': 50,
              'ff_cm': 100, 'pre_steer': 1, 'latch': 1, 'latch_deg': 27, 'turn_in_cm': 130, 'turn_deg': 90, 'curve_v_cm': 50, 'smooth': 100,
              'curve_in_deg': 60, 'curve_out': 30, 'mem_ld': 90}

    def __init__(self, tune):
        self.ok = bool(os.environ.get('DISPLAY'))
        self.frozen = False
        if not self.ok:
            print('[안내] 화면(DISPLAY)이 없어 창 대신 ~/simple_lane_snaps/live.jpg 를 갱신합니다. '
                  '창을 보려면 NoMachine 터미널에서 실행하세요.')
            return
        try:
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            for k, lim in self.LIMITS.items():
                cv2.createTrackbar(k, self.WIN, int(tune[k]), lim, lambda _v: None)
        except cv2.error as e:
            print(f'[안내] OpenCV 창을 열 수 없음 ({e}). live.jpg 로 대신합니다.')
            self.ok = False

    def read(self, tune):
        if not self.ok:
            return tune
        try:
            for k in self.LIMITS:
                tune[k] = cv2.getTrackbarPos(k, self.WIN)
        except cv2.error:
            pass
        tune['ld_cm'] = max(40, tune['ld_cm'])
        return tune

    def show(self, canvas):
        """반환: 누른 키 문자 또는 ''"""
        if not self.ok:
            os.makedirs(SNAP_DIR, exist_ok=True)
            tmp = os.path.join(SNAP_DIR, 'live.tmp.jpg')
            cv2.imwrite(tmp, canvas)
            os.replace(tmp, os.path.join(SNAP_DIR, 'live.jpg'))
            return ''
        try:
            cv2.imshow(self.WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            self.ok = False
            return ''
        if key == ord(' '):
            self.frozen = not self.frozen
            print('\n[화면 멈춤]' if self.frozen else '\n[화면 재개]')
        return chr(key) if key not in (255, ord(' ')) else ''


def save_snapshot(canvas, image, tag=''):
    os.makedirs(SNAP_DIR, exist_ok=True)
    stamp = time.strftime('%m%d_%H%M%S')
    cv2.imwrite(os.path.join(SNAP_DIR, f'{stamp}{tag}_debug.jpg'), canvas)
    cv2.imwrite(os.path.join(SNAP_DIR, f'{stamp}{tag}_raw.png'), image)
    return os.path.join(SNAP_DIR, f'{stamp}{tag}_debug.jpg')


# =====================================================================
# 보정 확인 (calib)
# =====================================================================
def calib_view(engine, image, result):
    geo = engine.geo
    color_bev = geo.warp_color(image)
    for xg in np.arange(-0.4, 0.41, 0.1):
        px, _ = geo.m_to_px(xg, 0)
        cv2.line(color_bev, (int(px), 0), (int(px), geo.bev_h), (90, 90, 90), 1)
    for xg in (-HALF_LANE, HALF_LANE):
        px, _ = geo.m_to_px(xg, 0)
        cv2.line(color_bev, (int(px), 0), (int(px), geo.bev_h), (0, 255, 0), 1)
    big = cv2.resize(color_bev, (color_bev.shape[1] * 3, color_bev.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    msg = []
    l, r = result['left'], result['right']
    for name, line in (('왼쪽', l), ('오른쪽', r)):
        if line is None:
            msg.append(f'{name} 선: 못 찾음')
        else:
            f0, f1 = max(line.f_lo, 0.6), min(line.f_hi, 1.1)
            ang = math.degrees(math.atan2(float(line.x_at(f1) - line.x_at(f0)), max(1e-6, f1 - f0)))
            msg.append(f'{name} 선: {line.x_at(F_REF) * 100:+.1f}cm (기대 {"-" if line is l else "+"}{HALF_LANE * 100:.0f}cm), 기울기 {ang:+.1f}°')
    if l is not None and r is not None:
        msg.append(f'폭: {(r.x_at(F_REF) - l.x_at(F_REF)) * 100:.1f}cm (설정 {LANE_WIDTH * 100:.0f}cm)')
    return big, msg


# =====================================================================
# 사진 한 장 모드 (ROS 불필요)
# =====================================================================
def run_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f'사진을 못 읽음: {path}')
        return
    image = cv2.resize(image, (IMG_W, IMG_H))
    engine = LaneEngine()
    tune = load_tune()
    gui = Gui(tune)
    print('사진 모드: 왜곡 보정 없이 처리합니다 (보정값이 없으므로 위치가 조금 다를 수 있음). q 로 종료.')
    last = None
    while True:
        tune = gui.read(tune)
        engine.prev_center = 0.0
        res = engine.process(image, tune, time.monotonic())
        cmd = engine.command(res, tune, time.monotonic())
        canvas = draw_debug(engine, image, res, cmd, {'tune': tune, 'fps': 0})
        status = (f"{cmd[2]} | 왼쪽 {'O' if res['left'] is not None else 'X'} "
                  f"오른쪽 {'O' if res['right'] is not None else 'X'} | 중심 {res['lateral'] * 100:+.1f}cm | "
                  f"조향 {math.degrees(cmd[1]):+.1f}° | {res['ms']:.1f}ms {res['note']}")
        if status != last:
            print(status)
            last = status
        key = gui.show(canvas)
        if not gui.ok:
            out = save_snapshot(canvas, image, '_image')
            print(f'결과 저장: {out}')
            break
        if key == 'q':
            break
        if key == 's':
            print('저장:', save_snapshot(canvas, image, '_image'))
    save_tune(tune)


# =====================================================================
# ROS 노드
# =====================================================================
def run_replay(folder):
    """기록 폴더의 프레임을 순서대로 다시 처리 (ROS 불필요). 스페이스: 멈춤, q: 종료"""
    import glob
    files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    if not files:
        print(f'프레임이 없음: {folder}')
        return
    engine = LaneEngine()
    tune = load_tune()
    gui = Gui(tune)
    t = 0.0
    poses = {}
    try:                                    # 기록에 odom 이 있으면 mem 방식도 그대로 다시 볼 수 있음
        import csv
        with open(os.path.join(folder, 'log.csv')) as f:
            for row in csv.DictReader(f):
                if row.get('ox'):
                    poses[int(row['frame'])] = (float(row['ox']), float(row['oy']), math.radians(float(row['oyaw_deg'])))
    except (OSError, ValueError, KeyError):
        pass
    if MEM and not poses:
        print('[안내] 이 기록에는 odom 이 없어 mem 방식은 볼 수 없음 (기존 방식으로 표시)')
    for i, fn in enumerate(files):
        image = cv2.imread(fn)
        tune = gui.read(tune)
        pose = poses.get(i + 1)
        res = engine.process(image, tune, t, pose=pose)
        cmd = engine.command(res, tune, t)
        t += 1.0 / 20.0
        print(f"{i + 1:5d} {cmd[2]:5s} 왼쪽 {'O' if res['left'] is not None else 'X'} "
              f"오른쪽 {'O' if res['right'] is not None else 'X'} {res['source']:10s} "
              f"중심 {res['lateral'] * 100:+6.1f}cm 조향 {math.degrees(cmd[1]):+6.1f}° {res['note']}")
        canvas = draw_debug(engine, image, res, cmd, {'tune': tune, 'fps': 0, 'pose_ok': pose is not None})
        key = gui.show(canvas)
        while gui.ok and gui.frozen:
            key = gui.show(canvas)
            time.sleep(0.03)
            if key == 'q':
                break
        if key == 'q':
            break
        if gui.ok:
            time.sleep(0.03)


def run_reset(seconds=2.0, topic='/cmd_vel'):
    """조향 초기화: 정지·직진 명령(Twist 0)을 일정 시간 계속 보낸다.
    리모컨으로 코드 모드로 바꿨을 때 이전 명령이 남아 바퀴가 꺾여 있으면 이걸 먼저 실행."""
    import rclpy
    from geometry_msgs.msg import Twist
    rclpy.init()
    node = rclpy.create_node('simple_lane_reset')
    pub = node.create_publisher(Twist, topic, 10)
    print(f'조향 초기화: {seconds:.0f}초 동안 정지·직진 명령 전송')
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            pub.publish(Twist())
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    print('완료. 앞바퀴가 똑바로 섰는지 확인하세요.')
    node.destroy_node()
    rclpy.shutdown()


def run_sweep(topic='/cmd_vel', speed=0.20):
    """바퀴를 띄운 상태에서 angular.z 를 단계별로 보내, 몇에서 바퀴가 끝까지 꺾이는지 눈으로 확인."""
    import rclpy
    from geometry_msgs.msg import Twist
    values = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5]
    rclpy.init()
    node = rclpy.create_node('simple_lane_sweep')
    pub = node.create_publisher(Twist, topic, 10)

    def hold(z, sec):
        msg = Twist()
        if z is not None:
            msg.linear.x = speed
            msg.angular.z = z
        t_end = time.monotonic() + sec
        while time.monotonic() < t_end:
            pub.publish(msg)
            time.sleep(0.05)

    print('바퀴를 띄웠는지 확인하세요. 왼쪽으로 꺾는 값을 작은 것부터 차례로 보냅니다.')
    print('각 단계마다 앞바퀴가 [조금 / 절반 / 끝까지] 중 어느 정도인지 메모하세요.\n')
    try:
        hold(None, 2.0)
        for i, z in enumerate(values, 1):
            print(f'  {i}번: angular.z = {z}  (3초)')
            hold(z, 3.0)
            print('      -> 초기화 (1.5초)')
            hold(None, 1.5)
    except KeyboardInterrupt:
        pass
    finally:
        hold(None, 0.5)
        print('\n끝. 몇 번부터 "끝까지" 꺾였는지 알려주세요.')
        node.destroy_node()
        rclpy.shutdown()


def run_circle(angle_deg, speed=0.20, seconds=8.0, topic='/cmd_vel'):
    """조향 고정 원 주행: 명령한 조향만큼 실제로 도는지 확인 (카메라 불필요).
    기대 지름(뒤 차축 중심 궤적) = 2 x (휠베이스/tan(조향) + 트랙/2)"""
    import rclpy
    from geometry_msgs.msg import Twist
    ang = math.radians(abs(angle_deg)) * (-1 if angle_deg < 0 else 1)
    ang = max(-MAX_STEER, min(MAX_STEER, ang))
    expect = 2 * (WHEELBASE / math.tan(abs(ang)) + TRACK / 2.0) if abs(ang) > 1e-3 else float('inf')
    rclpy.init()
    node = rclpy.create_node('simple_lane_circle')
    pub = node.create_publisher(Twist, topic, 10)
    side = '왼쪽' if ang > 0 else '오른쪽'
    print(f'원 주행 테스트: {side} {abs(math.degrees(ang)):.0f}°, 속도 {speed} m/s, {seconds:.0f}초')
    print(f'기대 지름(뒤 차축 중심): 약 {expect * 100:.0f} cm  (바퀴 바깥 자국은 이보다 약 20cm 큼)')
    print('주변을 비우세요. 3초 뒤 출발, 멈추려면 Ctrl+C')
    msg = Twist()
    try:
        for k in (3, 2, 1):
            print(f'  {k}...')
            for _ in range(20):            # 기다리는 동안 조향 초기화 (정지·직진 명령)
                pub.publish(Twist())
                time.sleep(0.05)
        msg.linear.x = speed
        msg.angular.z = inner_angle_to_omega(ang, speed)
        print(f'  출발: angular.z = {msg.angular.z:.2f} (조향 배율 {STEER_GAIN})')
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            pub.publish(msg)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10):
            pub.publish(Twist())
            time.sleep(0.03)
        print('정지. 바닥에 그린 원의 지름을 대충 재서 기대값과 비교해 주세요.')
        node.destroy_node()
        rclpy.shutdown()


def run_bridge(in_topic='/cmd_vel', out_topic='/limo/ack_cmd', gain=1.0):
    """/cmd_vel (Twist) 을 받아 limo_ackerman_base 드라이버용 명령(AckermannDrive)으로 바꿔 보낸다.

    왜 필요한가 (wego_ws limo_base 소스 + 실차 측정으로 확인):
      limo_base.launch.py 의 /cmd_vel 경로는 조향각을 28도로 제한한 뒤 2.47 로 나눠 차에 보낸다 -> 차가 받는 최대값 0.198rad(11.3도)
      -> 실제 최소 회전반경 약 1.05m (turncal 측정: 배율을 2.6배로 올려도 1.0~1.2m).
      limo_base_ackerman.launch.py 의 드라이버는 0.42rad(24도)까지 보낸다 -> 회전반경 약 0.4m.
    이 다리를 켜 두면 /cmd_vel 을 쓰는 기존 코드(이 파일, 팀 코드)를 고치지 않고 그대로 쓸 수 있다.

    변환: 회전반경 r = v / w,  조향(중심 기준) = atan(축간거리 / r)  -> AckermannDrive.steering_angle (왼쪽 +)
    드라이버가 받은 값을 1.215 로 나눈 뒤 +-0.42 로 자르므로 여기서는 +-0.51rad 로만 제한.
    0.5초 넘게 /cmd_vel 이 안 오면 정지 명령을 보낸다."""
    import rclpy
    from ackermann_msgs.msg import AckermannDrive
    from geometry_msgs.msg import Twist
    rclpy.init()
    node = rclpy.create_node('cmdvel_to_ackermann')
    pub = node.create_publisher(AckermannDrive, out_topic, 10)
    limit = 0.42 * 1.215
    st = {'v': 0.0, 'steer': 0.0, 't': 0.0, 'n': 0, 'print': 0.0}

    def send(v, steer):
        msg = AckermannDrive()
        msg.speed = float(v)
        msg.steering_angle = float(steer)
        pub.publish(msg)

    def on_cmd(msg):
        v, w = msg.linear.x, msg.angular.z
        steer = 0.0 if (abs(v) < 1e-3 or abs(w) < 1e-4) else math.atan(WHEELBASE / (v / w)) * gain
        steer = max(-limit, min(limit, steer))
        st.update(v=v, steer=steer, t=time.monotonic())
        st['n'] += 1
        send(v, steer)

    def on_timer():
        now = time.monotonic()
        alive = now - st['t'] <= 0.5
        if not alive:
            send(0.0, 0.0)
        if now - st['print'] >= 1.0:
            st['print'] = now
            side = '왼쪽' if st['steer'] > 0 else '오른쪽'
            print(f"\r[다리] 받은 명령 {st['n']}개 | " + (
                f"속도 {st['v']:.2f} 조향 {side} {abs(math.degrees(st['steer'])):4.1f}도" if alive else '명령 없음 -> 정지 유지')
                + '        ', end='', flush=True)

    node.create_subscription(Twist, in_topic, on_cmd, 10)
    node.create_timer(0.05, on_timer)
    print(f'[다리] {in_topic} (Twist) -> {out_topic} (AckermannDrive), 조향 배율 {gain}')
    print('       먼저 limo_base.launch.py 는 끄고  ros2 launch limo_base limo_base_ackerman.launch.py  를 켜 두세요.')
    print('       이 창은 켜 둔 채로 다른 터미널에서 평소처럼 주행 코드를 실행합니다. 끝내려면 Ctrl+C')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            send(0.0, 0.0)
            time.sleep(0.03)
        print('\n[다리] 종료: 정지 명령 전송')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run_turncal(topic='/cmd_vel', speed=0.20, gains=(1.0, 1.4, 1.8, 2.2, 2.6), seconds=3.0, settle=0.8):
    """실제 회전반경 측정. 조향 명령 배율(gain)을 바꿔 가며 오른쪽으로 돌고, 카메라에 보이는 먼 배경이
    옆으로 얼마나 흘렀는지로 차가 실제로 몇 도 돌았는지 잰다 (줄자 필요 없음).
    실차 기록 분석: 최대 조향 명령(27도)에서 계산상 회전반경은 0.48m 인데 실제로는 약 1.1m 였음."""
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage
    rclpy.init()
    node = rclpy.create_node('simple_lane_turncal')
    pub = node.create_publisher(Twist, topic, 10)
    state = {'img': None, 'img_t': 0.0, 'fx': None, 'odom': None, 'n': 0}

    def on_image(msg):
        raw = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            state['img'] = cv2.resize(raw, (IMG_W, IMG_H))[0:int(IMG_H * 0.45)]     # 위쪽(먼 배경)만 사용
            state['img_t'] = time.monotonic()
            state['n'] += 1

    def on_info(msg):
        if state['fx'] is None and len(msg.k) == 9 and msg.k[0] > 0 and msg.width > 0:
            state['fx'] = msg.k[0] * IMG_W / msg.width

    def on_odom(msg):
        state['odom'] = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(CompressedImage, IMAGE_TOPIC, on_image, qos)
    node.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, on_info, qos_profile_sensor_data)
    node.create_subscription(Odometry, ODOM_TOPIC, on_odom, qos_profile_sensor_data)
    node.create_subscription(Odometry, '/limo' + ODOM_TOPIC, on_odom, qos_profile_sensor_data)   # ackerman 드라이버 쪽

    def spin(sec, twist=None):
        t_end = time.monotonic() + sec
        while time.monotonic() < t_end and rclpy.ok():
            if twist is not None:
                pub.publish(twist)
            rclpy.spin_once(node, timeout_sec=0.02)

    orb = cv2.ORB_create(1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def shift(a, b):
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None:
            return None
        m = matcher.match(da, db)
        if len(m) < 12:
            return None
        dx = np.array([kb[x.trainIdx].pt[0] - ka[x.queryIdx].pt[0] for x in m])
        dy = np.array([kb[x.trainIdx].pt[1] - ka[x.queryIdx].pt[1] for x in m])
        ok = np.abs(dy) < 4
        return float(np.median(dx[ok])) if np.count_nonzero(ok) >= 12 else None

    w0 = speed * math.tan(math.atan(WHEELBASE / (WHEELBASE / math.tan(MAX_STEER) + TRACK / 2.0))) / WHEELBASE
    nominal = WHEELBASE / math.tan(MAX_STEER) + TRACK / 2.0
    print('실제 회전반경 측정 (오른쪽으로 돕니다)')
    print('  - 차를 넓은 바닥에 놓고 주변 1.5m 를 비우세요. 카메라 앞쪽 멀리 벽·기둥 같은 배경이 보여야 합니다.')
    print(f'  - 배율 {len(gains)}가지를 각 {seconds:.0f}초씩 돌고 자동으로 멈춥니다. 중간에 멈추려면 Ctrl+C')
    results = []
    try:
        spin(2.0, Twist())
        if state['img'] is None:
            print('!! 카메라 영상이 안 들어옴 -> 카메라를 켜고 다시 실행')
            return
        fx = state['fx'] or 480.0
        print(f'  - 카메라 초점거리 fx = {fx:.0f}px' + ('' if state['fx'] else ' (camera_info 없음, 추정값)')
              + f', odom {"있음" if state["odom"] else "없음 (거리는 속도x시간으로 계산)"}')
        for k in (3, 2, 1):
            print(f'  {k}...')
            spin(1.0, Twist())
        for g in gains:
            msg = Twist()
            msg.linear.x = speed
            msg.angular.z = -g * w0
            spin(settle, msg)                                  # 조향이 자리 잡을 때까지
            frames, dist, last_n = [], 0.0, state['n']
            prev_o, t0 = state['odom'], time.monotonic()
            while time.monotonic() - t0 < seconds - settle and rclpy.ok():
                pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.02)
                if state['n'] != last_n:
                    last_n = state['n']
                    frames.append(state['img'])
                    if state['odom'] is not None and prev_o is not None:
                        dist += math.hypot(state['odom'][0] - prev_o[0], state['odom'][1] - prev_o[1])
                    prev_o = state['odom']
            dur = time.monotonic() - t0
            spin(0.6, Twist())
            if dist < 0.05:
                dist = speed * dur
            yaw, bad = 0.0, 0
            for i in range(0, len(frames) - 2, 2):
                sdx = shift(frames[i], frames[i + 2])
                if sdx is None:
                    bad += 1
                else:
                    yaw += math.atan(-sdx / fx)
            radius = dist / abs(yaw) if abs(yaw) > 0.03 else float('inf')
            results.append((g, -g * w0, math.degrees(yaw), dist, radius, len(frames), bad))
            print(f'  배율 {g:.1f} (angular.z {-g * w0:+.2f}): {dur:.1f}초 동안 {math.degrees(abs(yaw)):5.1f}도 회전, '
                  f'{dist:.2f}m 이동 -> 실제 회전반경 {radius:.2f}m' + (f'  (매칭 실패 {bad}회)' if bad else ''))
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10):
            pub.publish(Twist())
            time.sleep(0.03)
        node.destroy_node()
        rclpy.shutdown()
    if results:
        lines = [f'계산상 회전반경(배율 1.0, 조향 27도): {nominal:.2f}m', '배율, angular.z, 회전(도), 이동(m), 실제 반경(m), 프레임, 실패']
        lines += [f'{g:.1f}, {w:+.2f}, {y:.1f}, {d:.2f}, {r:.2f}, {n}, {b}' for g, w, y, d, r, n, b in results]
        out = os.path.expanduser('~/simple_lane_turncal.txt')
        with open(out, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('\n결과 (이 표를 그대로 보내 주세요)')
        print('\n'.join('  ' + x for x in lines))
        print(f'저장: {out}')


def run_ros(mode, topic, record=False):
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage
    from collections import deque
    from nav_msgs.msg import Odometry

    class SimpleLane(Node):
        def __init__(self):
            super().__init__('simple_lane')
            self.engine = LaneEngine()
            self.tune = load_tune()
            self.gui = Gui(self.tune)
            self.drive = mode == 'drive'
            self.calib = mode == 'calib'
            self.pub = self.create_publisher(Twist, topic, 10) if self.drive else None
            self.K = self.D = None
            self.maps = None
            self.maps_size = None
            self.last_img_t = None
            self.frames = 0
            self.fps = 0.0
            self.fps_t = time.monotonic()
            self.fps_n = 0
            self.last_print = 0.0
            self.quit = False
            self.last_canvas = None
            self.last_image = None
            self.t_start = time.monotonic()
            self.warned_info = False
            self.rec_dir = None
            self.rec_csv = None
            self.rec_n = 0
            self.odom = deque(maxlen=200)       # (메시지 시각, 받은 시각, x, y, yaw[연속])
            self.odom_mode = None
            self.warned_odom = 0.0
            if record:
                self.rec_dir = os.path.join(LOG_DIR, time.strftime('%m%d_%H%M%S') + ('_drive' if self.drive else '_view')
                                            + (f'_{PRESET}' if PRESET else '') + ('_mem' if MEM else ''))
                os.makedirs(self.rec_dir, exist_ok=True)
                with open(os.path.join(self.rec_dir, 'tune_start.json'), 'w') as f:
                    json.dump(dict(self.tune, preset=PRESET or '', gain=STEER_GAIN, mem=int(MEM)), f, indent=1)
                self.rec_csv = open(os.path.join(self.rec_dir, 'log.csv'), 'w')
                self.rec_csv.write('frame,t,state,left,right,source,lateral_cm,steer_deg,speed,lane_width_cm,ms,'
                                   'mode,tilt_deg,ff_deg,ff_w,front_m,'
                                   'ld,vmax,ff_cm,latch,latch_deg,curve_v,smooth,in_deg,out,'
                                   'ox,oy,oyaw_deg,mem_used,mem_deg,mem_err_cm,mem_ld,single_w_cm,note\n')
                print(f'[기록] 프레임과 상태를 저장합니다: {self.rec_dir}')
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(CompressedImage, IMAGE_TOPIC, self.on_image, qos)
            self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.on_info, qos_profile_sensor_data)
            self.create_subscription(Odometry, ODOM_TOPIC, self.on_odom, qos_profile_sensor_data)
            self.create_subscription(Odometry, '/limo' + ODOM_TOPIC, self.on_odom, qos_profile_sensor_data)
            if MEM:
                print(f'[mem] 선 기억 방식 켜짐: {ODOM_TOPIC} 로 지나온 차선을 기억합니다.')
            self.create_timer(0.05, self.on_timer)
            if self.drive:
                print('\n' + '!' * 60)
                print(f'  주행 모드: 실제로 바퀴가 움직입니다  (출력 토픽: {topic})')
                print('  멈추려면 이 창에서 q 또는 Ctrl+C')
                print('!' * 60 + '\n')
            else:
                print(f'\n{"보정 확인 모드" if self.calib else "보기 전용 모드"}: 명령을 보내지 않습니다.\n')

        def on_info(self, msg):
            if self.K is None and len(msg.k) == 9 and msg.k[0] > 0:
                self.K = np.array(msg.k, np.float64).reshape(3, 3)
                self.D = np.array(msg.d, np.float64)
                print(f'[카메라 보정값 수신] {msg.width}x{msg.height}, 왜곡계수 {len(self.D)}개')

        def on_odom(self, msg):
            q = msg.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            if self.odom:                                   # yaw 를 끊김 없이 이어 붙임 (+-180도 넘어갈 때)
                prev = self.odom[-1][4]
                yaw = prev + math.atan2(math.sin(yaw - prev), math.cos(yaw - prev))
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.odom.append((stamp, time.monotonic(), msg.pose.pose.position.x, msg.pose.pose.position.y, yaw))

        def pose_for(self, img_stamp, now):
            """영상에 맞는 odom 자세 (x, y, yaw). 영상 시각이 odom 기록 범위 안이면 그 시각으로 보간,
            아니면(카메라 시계가 다르면) 가장 최근 값. odom 이 끊겼으면 None."""
            if not self.odom or now - self.odom[-1][1] > ODOM_STALE:
                return None
            if self.odom[0][0] <= img_stamp <= self.odom[-1][0] + 0.2:
                mode = 'stamp'
                if img_stamp >= self.odom[-1][0]:
                    pose = self.odom[-1][2:]
                else:
                    pose = self.odom[0][2:]
                    for k in range(len(self.odom) - 1, 0, -1):
                        a, b = self.odom[k - 1], self.odom[k]
                        if a[0] <= img_stamp:
                            w = 0.0 if b[0] <= a[0] else (img_stamp - a[0]) / (b[0] - a[0])
                            pose = tuple(a[n] + w * (b[n] - a[n]) for n in (2, 3, 4))
                            break
            else:
                mode, pose = 'latest', self.odom[-1][2:]
            if mode != self.odom_mode:
                self.odom_mode = mode
                print('\n[odom] 연결됨: ' + ('영상 시각에 맞춰 사용' if mode == 'stamp' else
                                              '영상 시각을 쓸 수 없어 가장 최근 값 사용'))
            return tuple(pose)

        def undistort(self, image):
            if self.K is None:
                return image
            h, w = image.shape[:2]
            if self.maps is None or self.maps_size != (w, h):
                self.maps = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (w, h), cv2.CV_16SC2)
                self.maps_size = (w, h)
            return cv2.remap(image, self.maps[0], self.maps[1], cv2.INTER_LINEAR)

        def send(self, v, angle):
            if self.pub is None:
                return
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(inner_angle_to_omega(angle, v))
            self.pub.publish(msg)

        def stop(self, repeat=5):
            if self.pub is None:
                return
            for _ in range(repeat):
                self.pub.publish(Twist())
                time.sleep(0.03)

        def on_timer(self):
            now = time.monotonic()
            if not self.warned_info and self.K is None and now - self.t_start > 3.0:
                self.warned_info = True
                print(f'[주의] {CAMERA_INFO_TOPIC} 를 못 받음 -> 왜곡 보정 없이 진행 (위치가 조금 어긋날 수 있음)')
            if self.drive and now - self.t_start < RESET_SEC:
                self.send(0.0, 0.0)                 # 시작 직후: 이전 명령이 남아 있지 않게 초기화
            if self.last_img_t is None or now - self.last_img_t > CAMERA_TIMEOUT:
                if self.drive:
                    self.send(0.0, 0.0)
                if self.last_img_t is None and now - self.t_start > 3.0 and int(now) != int(now - 0.05):
                    print(f'[대기] 카메라 영상이 안 들어옴 ({IMAGE_TOPIC}). 카메라 켜졌는지 확인')

        def on_image(self, msg):
            now = time.monotonic()
            raw = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                return
            self.last_img_t = now
            image = cv2.resize(self.undistort(raw), (IMG_W, IMG_H))
            self.tune = self.gui.read(self.tune)
            pose = self.pose_for(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, now)
            if pose is None and now - self.warned_odom > 3.0 and now - self.t_start > 2.0:
                self.warned_odom = now
                print(f'\n[주의] {ODOM_TOPIC} 가 안 들어옴 -> 코너에서 돈 각도를 시간으로 추정합니다'
                      + (' / 선 기억은 못 씀' if MEM else '') + ' (드라이버 확인)')
            try:
                res = self.engine.process(image, self.tune, now, pose=pose)
                cmd = self.engine.command(res, self.tune, now)
            except (cv2.error, ValueError, np.linalg.LinAlgError) as e:
                print(f'\n[처리 오류] {e} -> 정지')
                self.send(0.0, 0.0)
                return
            if self.drive and now - self.t_start < RESET_SEC:
                cmd = (0.0, 0.0, 'WAIT', '시작 전 조향 초기화')
            self.send(cmd[0], cmd[1])
            if self.rec_dir is not None:
                self.rec_n += 1
                cv2.imwrite(os.path.join(self.rec_dir, f'{self.rec_n:05d}.jpg'), image,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.rec_csv.write(
                    f"{self.rec_n},{now - self.t_start:.3f},{cmd[2]},{int(res['left'] is not None)},"
                    f"{int(res['right'] is not None)},{res['source']},{res['lateral'] * 100:.1f},"
                    f"{math.degrees(cmd[1]):.1f},{cmd[0]:.2f},{res['lane_width'] * 100:.1f},{res['ms']:.1f},"
                    f"{self.engine.mode},"
                    f"{'' if res.get('tilt') is None else format(math.degrees(res['tilt']), '.1f')},"
                    f"{math.degrees(res.get('ff_angle', 0.0)):.1f},{res.get('ff_w', 0.0):.2f},"
                    f"{'' if res.get('wall') is None else format(res['wall'], '.2f')},"
                    + ','.join(str(self.tune.get(k, '')) for k in
                               ('ld_cm', 'vmax_cm', 'ff_cm', 'latch', 'latch_deg', 'curve_v_cm', 'smooth',
                                'curve_in_deg', 'curve_out'))
                    + (',,,' if pose is None else f",{pose[0]:.4f},{pose[1]:.4f},{math.degrees(pose[2]):.2f}")
                    + f",{int(MEM and res.get('mem_angle') is not None)},"
                    + ('' if res.get('mem_angle') is None else format(math.degrees(res['mem_angle']), '.1f')) + ','
                    + ('' if res.get('mem_err') is None else format(res['mem_err'] * 100, '.1f'))
                    + f",{self.tune.get('mem_ld', '')},{self.engine.single_width * 100:.1f}"
                    + f",{res['note']}\n")
            # fps
            self.fps_n += 1
            if now - self.fps_t >= 1.0:
                self.fps = self.fps_n / (now - self.fps_t)
                self.fps_t, self.fps_n = now, 0
            if now - self.last_print >= 0.5:
                self.last_print = now
                v, ang, state, reason = cmd
                side = '오른쪽' if ang < 0 else '왼쪽'
                print(f"\r[{state:5s}] 왼쪽 {'O' if res['left'] is not None else 'X'} "
                      f"오른쪽 {'O' if res['right'] is not None else 'X'} | 중심 {res['lateral'] * 100:+5.1f}cm | "
                      f"조향 {side} {abs(math.degrees(ang)):4.1f}° | 속도 {v:.2f} | "
                      f"폭 {res['lane_width'] * 100:.0f}cm | 왜곡보정 {'O' if self.K is not None else 'X'} | "
                      f"{self.fps:4.1f}fps {res['ms']:4.1f}ms | {reason} {res['note']}      ",
                      end='', flush=True)
            if self.gui.frozen:
                key = self.gui.show(self.last_canvas) if self.last_canvas is not None else ''
            else:
                extra = {'tune': self.tune, 'fps': self.fps, 'pose_ok': pose is not None,
                         'output': f'ON -> {topic}' if self.drive else 'OFF (view only)'}
                if self.calib:
                    big, info = calib_view(self.engine, image, res)
                    if now - self.last_print < 0.05:
                        print('\n  ' + ' | '.join(info))
                    canvas = big
                else:
                    canvas = draw_debug(self.engine, image, res, cmd, extra)
                self.last_canvas, self.last_image = canvas, image
                key = self.gui.show(canvas)
            if key == 'q':
                self.quit = True
            elif key == 's' and self.last_canvas is not None:
                print('\n[저장] ' + save_snapshot(self.last_canvas, self.last_image))
                save_tune(self.tune)

    rclpy.init()
    node = SimpleLane()
    try:
        while rclpy.ok() and not node.quit:
            rclpy.spin_once(node, timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        print('\n종료: 정지 명령 전송' if node.drive else '\n종료')
        node.stop()
        save_tune(node.tune)
        if node.rec_csv is not None:
            node.rec_csv.close()
            print(f'[기록 저장됨] {node.rec_dir}  ({node.rec_n}프레임)')
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    global STEER_GAIN, PRESET, MEM
    args = sys.argv[1:]
    if 'mem' in args:
        MEM = True
    if '--gain' in args:
        i = args.index('--gain')
        if i + 1 < len(args):
            STEER_GAIN = float(args[i + 1])
            print(f'[조향 배율] {STEER_GAIN}')
    if args and args[0] == 'image':
        if len(args) < 2:
            print('사용법: python3 simple_lane.py image 사진.jpg')
            return
        run_image(args[1])
        return
    if args and args[0] == 'reset':
        run_reset()
        return
    if args and args[0] == 'sweep':
        run_sweep()
        return
    if args and args[0] == 'turncal':
        run_turncal()
        return
    if args and args[0] == 'bridge':
        run_bridge(gain=BRIDGE_GAIN * STEER_GAIN)
        return
    if args and args[0] == 'circle':
        deg = float(args[1]) if len(args) > 1 else -20.0
        run_circle(deg)
        return
    if args and args[0] == 'replay':
        if len(args) < 2:
            print('사용법: python3 simple_lane.py replay 기록폴더')
            return
        run_replay(args[1])
        return
    mode = args[0] if args and args[0] in ('drive', 'calib') else 'view'
    record = 'rec' in args      # 기록은 rec 를 붙였을 때만 (예: python3 simple_lane.py drive rec)
    for p in PRESETS:
        if p in args:
            PRESET = p
            print(f'[프리셋 {p}] ' + ', '.join(f'{k} {v}' for k, v in PRESETS[p].items()))
    topic = '/cmd_vel'
    if '--topic' in args:
        i = args.index('--topic')
        if i + 1 < len(args):
            topic = args[i + 1]
    run_ros(mode, topic, record)


if __name__ == '__main__':
    main()
