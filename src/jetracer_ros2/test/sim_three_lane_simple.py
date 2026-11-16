"""three_lane_simple 닫힌 고리 시뮬레이션 (ROS 없음).

도로 좌표: x = 도로 방향 [m] (진입 지점 = 흰 점선 시작 0), y = 왼쪽 + [m]. 차로 가운데 +0.535 / 0 / -0.535, 바깥 노란 선 +-0.80.
가운데 차로는 섬 앞끝(isl)에서 끝나고, 섬은 y +-0.27, 길이 1.7 m. 섬 앞끝에 표지판 기둥(라이다에 보임).
정차 LIMO 0.32 x 0.22 m. 라이다 250점 -110~110도, 카메라 투영점 위치.
odom: 거리 배율 오차, 방향 배율 오차 + 1 m 당 흐름. 진입 위치/방향 오차. 조향은 1차 지연.

python3 -I test/sim_three_lane_simple.py [횟수]   (src/jetracer_ros2 에서)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetracer_ros2 import three_lane as T  # noqa: E402
from jetracer_ros2 import three_lane_simple as S  # noqa: E402

CAM = T.CAMERA_FORWARD_OFFSET
CAR_L, CAR_W = 0.32, 0.22
BODY_C = 0.10                     # 차체 가운데: 뒤 차축 앞
ISL_LEN, ISL_HALF = 1.74, 0.27
JOG0, JOG1 = 1.79, 2.29          # 섬 앞끝 기준: 바깥 선이 45도로 꺾여 합류하는 구간 (도면)
DT = 0.05
STEER_LAG = float(os.environ.get('SIM_STEER_LAG', '0.15'))   # 조향 1차 지연 [s]
YOLO_EVERY = int(os.environ.get('SIM_YOLO_EVERY', '2'))   # 2 = 10 Hz, 5 = 4 Hz (차에서 실제 속도를 재서 맞출 것)


def rect(cx, cy, yaw, L, W):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx, dy in ((L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)):
        pts.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return np.array(pts)


def overlap(p, q):
    for poly in (p, q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def raycast(ox, oy, yaw, polys, n=250, a0=-1.92, inc=3.84 / 249, rng=None):
    ang = yaw + a0 + np.arange(n) * inc
    dx, dy = np.cos(ang), np.sin(ang)
    r = np.full(n, np.inf)
    for poly in polys:
        for i in range(len(poly)):
            p1, p2 = poly[i], poly[(i + 1) % len(poly)]
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            den = dx * ey - dy * ex
            ok = np.abs(den) > 1e-9
            wx, wy = p1[0] - ox, p1[1] - oy
            t = np.where(ok, (wx * ey - wy * ex) / np.where(ok, den, 1), np.inf)
            u = np.where(ok, (wx * dy - wy * dx) / np.where(ok, den, 1), -1)
            hit = ok & (t > 0) & (u >= 0) & (u <= 1)
            r = np.where(hit & (t < r), t, r)
    if rng is not None:
        r = r + rng.normal(0, 0.01, n)
    r[r > 8.0] = np.inf
    return r.astype(np.float32), a0, inc, 0.15


GAP_LO, GAP_HI = 1.0, 1.5


def feasible(rows, isl, goal=None):
    """줄마다 빈 차로 집합. 줄 간격 1.0 m 이상이면 한 칸, 1.9 m 이상이면 두 칸 옮길 수 있다고 본다.
    goal: 주면 마지막 줄 뒤 섬까지 남은 거리로 표지판 쪽 차로까지 갈 수 있는 배치만 (fair)."""
    reach, prev_s = {1}, 0.0
    for s, blocked in rows:
        g = s - prev_s
        step = 2 if g >= 1.9 else (1 if g >= 1.0 else 0)
        reach = {j for j in range(3) if j not in blocked and any(abs(j - r) <= step for r in reach)}
        if not reach:
            return False
        prev_s = s
    if goal is not None:
        g = isl - prev_s
        step = 2 if g >= 1.9 else (1 if g >= 1.0 else 0)
        return any(abs(goal - r) <= step for r in reach)
    return bool(reach & {0, 2}) or (isl - prev_s >= 1.2)


def layout(rng, isl, goal=None):
    """정차 LIMO 를 줄 단위로 놓는다. 한 줄은 1~2개 (최소 한 차로는 빈다). 최대 7대. 지나갈 수 있는 배치만."""
    for _ in range(200):
        rows, s, n = [], rng.uniform(0.9, 1.3), 0
        while n < 7 and s < isl - 0.5:
            k = min(7 - n, int(rng.integers(1, 3)))
            rows.append((s, set(int(j) for j in rng.choice(3, size=k, replace=False))))
            n += k
            s += rng.uniform(GAP_LO, GAP_HI)
        if feasible(rows, isl, goal):
            break
    cars = []
    for s, blocked in rows:
        for j in blocked:
            cars.append((s + rng.uniform(-0.05, 0.05), T.LANE_CENTER[j] + rng.uniform(-0.04, 0.04)))
    return cars


def camera_obs(cx, cy, psi, isl, rng, spurious=0.0):
    """카메라 결과 흉내: 도로 방향(잡음), 노란 선(바깥 +-0.80, 섬 테두리), 흰 점선(+-0.27, 섬 앞까지), 섬 노란 점.
    spurious: 프레임마다 이 확률로 가짜 흰 선 하나 (매트 가장자리·반사광 흉내)."""
    th = -psi + rng.normal(0, math.radians(1.0))
    lines, ystart = [], {}
    def rel(px, py):
        sx, ly = px - cx, py - cy
        f = sx * math.cos(psi) + ly * math.sin(psi)
        u = -sx * math.sin(psi) + ly * math.cos(psi)
        return f, u
    def bev(u):                    # 실제 BEV 는 먼 선을 더 멀리 보여 준다 (three_lane._shrink 의 반대)
        return u if abs(u) <= T.FAR_U else math.copysign(T.FAR_U + (abs(u) - T.FAR_U) * T.FAR_SHRINK, u)
    for y0 in (0.80, -0.80):
        u = bev((y0 - cy) * math.cos(psi) + rng.normal(0, 0.02))
        if abs(u) < 1.0:
            lines.append((u, 150, 1.0))
            ystart[u] = 0.42
    xs = np.arange(isl, isl + ISL_LEN, 0.02)
    pts = []
    for y0 in (ISL_HALF, -ISL_HALF):
        for x in xs:
            pts.append(rel(x, y0))
    for y in np.arange(-ISL_HALF, ISL_HALF, 0.02):
        pts.append(rel(isl, y))
    pts = np.array(pts)
    vis = (pts[:, 0] > 0.42) & (pts[:, 0] < 1.9) & (np.abs(pts[:, 1]) < 1.0)
    yellow_xy = None
    if vis.sum() > 10:
        yellow_xy = np.stack([-pts[vis, 1], pts[vis, 0]], 1)      # (오른쪽, 앞)
        for y0 in (ISL_HALF, -ISL_HALF):
            u0 = (y0 - cy) * math.cos(psi)
            sel = vis & (np.abs(pts[:, 1] - u0) < 0.05)
            u = bev(u0)
            if sel.sum() > 10 and abs(u) < 1.0:
                lines.append((u, 120, 1.0))
                ystart[u] = float(np.percentile(pts[sel, 0], 5))
    if cx < isl - 0.3:                     # 흰 점선 (섬 앞까지)
        for y0 in (0.27, -0.27):
            u = bev((y0 - cy) * math.cos(psi) + rng.normal(0, 0.02))
            if abs(u) < 1.0:
                lines.append((u, 100, 0.0))
    if rng.random() < spurious:
        lines.append((float(rng.uniform(-1.0, 1.0)), int(rng.integers(60, 200)), 0.0))
    lines = sorted(lines)
    merged = []                            # three_lane.LineFitter 처럼 0.20 m 안의 선은 하나로
    for ln in lines:
        if merged and abs(ln[0] - merged[-1][0]) < 0.20:
            if ln[1] > merged[-1][1]:
                merged[-1] = ln
            continue
        merged.append(ln)
    return dict(th=th, lines=merged, n=1000, ystart=ystart, yellow_xy=yellow_xy, short=set(), edges=[])


def p_correct(d, label, side):
    """표지판을 맞게 읽을 확률. 11/15 측정(실제 B): 3.2 m 밖 5 %, 2.1~1.7 m 86 %, 1.4 m 안 100 %.
    모델은 멀리서 A 라고 하는 쪽으로 틀린다 -> 실제 A 는 거의 맞는다. 옆 차로(비스듬히)는 B 를 더 틀린다."""
    if label == 'a':
        return 0.97
    if d > 3.2:
        p = 0.05
    elif d > 2.1:
        p = 0.05 + (3.2 - d) / 1.1 * 0.81
    elif d > 1.7:
        p = 0.86
    elif d > 1.4:
        p = 0.95
    else:
        p = 1.0
    return p - (0.25 if side else 0.0)


def yolo_sign(cx, cy, psi, isl, goal, rng):
    """섬 앞끝 표지판 YOLO 결과 흉내 (못 보면 None). 상자 높이 33 px ≈ 1.7 m."""
    dx, dy = isl - cx, 0.0 - cy
    d = math.hypot(dx, dy)
    ang = math.atan2(dy, dx) - psi
    if dx < 0.3 or d > 4.5 or abs(ang) > math.radians(35) or rng.random() < 0.2:
        return None
    true = 'a' if goal == 0 else 'b'
    lab = true if rng.random() < p_correct(d, true, abs(cy) > 0.3) else ('b' if true == 'a' else 'a')
    h = 33.0 * 1.7 / d
    return {'label': lab, 'confidence': 0.8, 'bbox': [300.0, 100.0, 300.0 + h * 0.8, 100.0 + h]}


def run(seed, goal=None, verbose=False, max_t=60.0, spurious=0.0, fair=False, cars=None, x0=None):
    """cars: 장애물 (x, y) 목록을 직접 줄 때 (진입=흰 점선 시작 기준). x0: 처음 카메라 투영점 위치 (기본: 점선 시작)"""
    rng = np.random.default_rng(seed)
    isl = S.START_TO_ISLAND - S.START_TO_DASH + rng.uniform(-0.10, 0.10)   # 진입(점선 시작) 기준. 트랙 오차 ±10 cm
    goal = int(rng.choice([0, 2])) if goal is None else goal
    cars = layout(rng, isl, goal if fair else None) if cars is None else cars
    car_polys = [rect(x, y, 0.0, CAR_L, CAR_W) for x, y in cars]
    post = rect(isl - 0.03, 0.0, 0.0, 0.06, 0.06)
    # 섬: 양 끝이 뾰족한 육각형 (도면: 끝 0.21 m 동안 좁아짐)
    island_poly = np.array([(isl, 0.0), (isl + 0.21, ISL_HALF), (isl + ISL_LEN - 0.21, ISL_HALF), (isl + ISL_LEN, 0.0),
                            (isl + ISL_LEN - 0.21, -ISL_HALF), (isl + 0.21, -ISL_HALF)])

    def road_half(xx):
        if xx < isl + JOG0:
            return 0.80
        if xx < isl + JOG1:
            return 0.80 - (xx - isl - JOG0)
        return 0.27
    yolo_rng = np.random.default_rng(seed + 10_000)
    # 참 상태 (뒤 차축)
    x, y, psi = (-CAM if x0 is None else x0 - CAM), rng.uniform(-0.04, 0.04), rng.normal(0, math.radians(2.0))
    steer_act = 0.0
    k_ds, k_yaw = 1 + rng.normal(0, 0.03), 1 + rng.normal(0, 0.03)
    drift = rng.normal(0, math.radians(1.5))   # 1 m 당 방향 흐름
    ox, oy, oyaw = 5.0, -2.0, 1.3               # odom 은 임의 좌표계
    c_true = y + CAM * math.sin(psi)
    c0, th0 = c_true + rng.normal(0, 0.04), -psi + rng.normal(0, math.radians(2.0))
    sup = S.Lane3SimpleSupervisor(forced=True)
    sup.mode = 'on'
    sup.core.start(c0, th0, (ox, oy, oyaw), S.START_TO_DASH + (0.0 if x0 is None else x0) + rng.normal(0, 0.10))   # 시작선 기준 위치 추정 오차
    t, contact, out_frames, island_frames, frames = 0.0, 0, 0, 0, 0
    touched = set()
    exit_lane = None
    max_psi = max_edge = 0.0
    while t < max_t:
        cx, cy = x + CAM * math.cos(psi), y + CAM * math.sin(psi)
        if frames % YOLO_EVERY == 0:              # YOLO 속도 (DT 0.05 s 의 배수)
            det = yolo_sign(cx, cy, psi, isl, goal, yolo_rng)
            if det is not None:
                sup.see_signs([det])
        polys = car_polys + [post]
        scan = raycast(cx, cy, psi, polys, rng=rng)
        obs = camera_obs(cx, cy, psi, isl, rng, spurious) if rng.random() > 0.1 else None
        out = sup.step_obs(obs, (ox, oy, oyaw), scan, both_yellow=False, now=t)
        if out is None:
            if sup.mode == 'done':
                exit_lane = sup.side_taken
            break
        v, cmd = out['v'], max(-T.MAX_STEER, min(T.MAX_STEER, out['steer']))
        steer_act += (cmd - steer_act) * min(1.0, DT / STEER_LAG)
        ds = v * DT
        if abs(steer_act) > 1e-4:
            radius = T.WHEELBASE / math.tan(abs(steer_act)) + T.TRACK / 2.0
            dpsi = math.copysign(abs(ds) / radius, steer_act) * (1.0 if ds >= 0 else -1.0)   # 후진이면 반대로 돈다
        else:
            dpsi = 0.0
        x += ds * math.cos(psi + dpsi / 2)
        y += ds * math.sin(psi + dpsi / 2)
        psi += dpsi
        # odom (흐름 포함)
        dyaw_o = dpsi * k_yaw + drift * abs(ds)
        oyaw += dyaw_o
        dso = abs(ds) * k_ds
        sgn = 1.0 if ds >= 0 else -1.0
        ox += sgn * dso * math.cos(oyaw)
        oy += sgn * dso * math.sin(oyaw)
        body = rect(x + BODY_C * math.cos(psi), y + BODY_C * math.sin(psi), psi, CAR_L, CAR_W)
        for i, p in enumerate(car_polys):
            if overlap(body, p) and i not in touched:
                touched.add(i)
                contact += 1
        if any(abs(py) > road_half(px) for px, py in body):
            out_frames += 1
        max_psi = max(max_psi, abs(psi))
        max_edge = max(max_edge, max(abs(py) for px, py in body) if x < isl else 0.0)
        if overlap(body, island_poly):
            island_frames += 1
        frames += 1
        t += DT
        if verbose and frames % 10 == 0:
            print(f't{t:5.1f} x{x:5.2f} y{y:+.2f} psi{math.degrees(psi):+4.0f} v{v:+.2f} | {out["note"]} | {out["reason"]}')
    passed = max(0.0, x)
    ok = exit_lane in (0, 2) and contact == 0 and out_frames == 0 and island_frames == 0
    return dict(seed=seed, ok=ok, exit_lane=exit_lane, goal=goal, contact=contact, out=out_frames,
                island=island_frames, t=t, x=passed, cars=len(cars), reverse=sup.core.reverse_count,
                state=sup.core.state, reason=sup.core.reason, max_psi=math.degrees(max_psi), max_edge=max_edge)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    spurious = 0.15 if 'noise' in sys.argv else 0.0
    fair = 'fair' in sys.argv            # 표지판 쪽으로 갈 수 있는 배치만
    nums = [a for a in sys.argv[2:] if a.isdigit()]
    if nums:
        print(run(int(nums[0]), verbose=True, spurious=spurious, fair=fair))
        sys.exit()
    res = [run(i, spurious=spurious, fair=fair) for i in range(n)]
    ok = sum(r['ok'] for r in res)
    right = sum(r['exit_lane'] == r['goal'] for r in res)
    print(f'통과(접촉·이탈 없이 갈림길 차로까지) {ok}/{n}, 표지판 쪽 {right}/{n}, '
          f'접촉 {sum(r["contact"] for r in res)}, 이탈 프레임 {sum(r["out"] for r in res)}, '
          f'섬 침범 {sum(r["island"] > 0 for r in res)}, 후진 {sum(r["reverse"] for r in res)}')
    for r in res:
        if not r['ok'] or r['exit_lane'] != r['goal']:
            print(r)
