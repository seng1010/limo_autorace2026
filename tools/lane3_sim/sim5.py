"""sim4 확장: 장애물 크기 자유, 라이다 위치/차체 위치 지정, 후진 반영. 장애물 = (앞끝 x0, 뒤끝 x1, 가운데 y, 반폭)."""
import sys, math, importlib.util, numpy as np
def load(path, **over):
    spec = importlib.util.spec_from_file_location('tl_' + str(abs(hash((path, tuple(sorted(over.items())))))), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for k, v in over.items(): setattr(m, k, v)
    return m
CAR_HL, CAR_HW, CAR_CENTER = 0.16, 0.11, 0.10        # 차체: 뒤 차축 +0.10 이 가운데, 길이 0.32, 폭 0.22
ISL_HALF, ISL_LEN = 0.27, 3.0
def ray_rect(ox, oy, dx, dy, cx, cy, hl, hw):
    t0, t1 = 0.0, 1e9
    for o, d, c, h in ((ox, dx, cx, hl), (oy, dy, cy, hw)):
        if abs(d) < 1e-9:
            if abs(o - c) > h: return None
        else:
            a, b = (c - h - o) / d, (c + h - o) / d
            if a > b: a, b = b, a
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1: return None
    return t0 if t0 > 0 else None
def box(T, x0, lane, length=0.32, width=0.22, dy=0.0):
    return (x0, x0 + length, T.LANE_CENTER[lane] + dy, width / 2)
def run(T, obstacles, XI=30.0, x0=0.0, c0=0.0, psi0=0.0, sign=None, d_sign=2.0, noise=0.0, lag=0.15, yaw_err=1.0, vis_drop=0.0,
        seed=0, verbose=0, cam_dt=1 / 22.0, scan_dt=0.1, lidar=0.20, xend=None, tmax=120):
    rng = np.random.default_rng(seed)
    X, Y, psi = x0, c0, psi0
    steer = 0.0; tl = T.ThreeLane(); t = 0.0; scan = None; next_scan = 0.0
    oyaw = psi; ox = oy = 0.0
    worst = None; minclr = 9.0; out_max = -9.0; isl_in = -9.0; stopped = 0.0; rev = 0.0; side_at = None; log = []; last = None
    rects = [((a + b) / 2, y, (b - a) / 2, hw) for a, b, y, hw in obstacles] + [(XI + 0.12, 0.0, 0.03, 0.03)]
    isl_pts = np.array([(XI, y) for y in np.arange(-ISL_HALF, ISL_HALF + 1e-6, 0.02)]
                       + [(x, s * ISL_HALF) for s in (-1, 1) for x in np.arange(XI, XI + ISL_LEN, 0.02)])
    xend = XI + 0.8 if xend is None else xend
    while X < xend and t < tmax:
        camx = X + T.CAMERA_FORWARD_OFFSET * math.cos(psi); camy = Y + T.CAMERA_FORWARD_OFFSET * math.sin(psi)
        if t >= next_scan:
            next_scan += scan_dt
            n = 250; a0 = -1.92; inc = 3.84 / (n - 1); r = np.full(n, np.inf, np.float32)
            lx, ly = X + lidar * math.cos(psi), Y + lidar * math.sin(psi)
            for i in range(n):
                ang = psi + a0 + i * inc; dx, dy = math.cos(ang), math.sin(ang); best = None
                for (bx, by, hl, hw) in rects:
                    h = ray_rect(lx, ly, dx, dy, bx, by, hl, hw)
                    if h is not None and (best is None or h < best): best = h
                if best is not None and best < 8: r[i] = best + rng.normal(0, 0.01)
            scan = (r, a0, inc, 0.15)
        obs = None
        if rng.random() >= vis_drop:
            lines = []; raw = []; gap = XI - camx
            mid = 0.0 if gap > 1.9 else (0.5 if gap > 1.0 else 1.0)
            for pos, yel in ((0.80, 1.0), (0.27, mid), (-0.27, mid), (-0.80, 1.0)):
                u = pos - camy                     # 선까지 옆 거리 (도로 기준)
                if abs(u) > 0.35: u = math.copysign(0.35 + (abs(u) - 0.35) * 1.25, u)
                if abs(u) < 1.05 and rng.random() > 0.1 * (abs(u) > 0.5) and abs(psi) < math.radians(45):   # 45도 넘으면 선을 못 본다 (실차 기록)
                    un = u + rng.normal(0, noise * (2.0 if abs(psi) > math.radians(30) else 1.0)); lines.append((un, 150, yel)); raw.append((un, pos, yel))
            dxw, dyw = isl_pts[:, 0] - camx, isl_pts[:, 1] - camy
            f = dxw * math.cos(psi) + dyw * math.sin(psi); left = -dxw * math.sin(psi) + dyw * math.cos(psi)
            keep = (f > 0.4) & (f < 1.9) & (np.abs(left) < 1.0)
            yxy = np.stack([-left[keep], f[keep]], 1) if keep.any() else np.zeros((0, 2))
            if lines:
                ystart = {u: (max(0.4, gap) if abs(abs(u0) - 0.27) < 1e-6 else 0.45) for u, u0, yel in raw if yel >= 0.7}
                obs = dict(th=-psi + rng.normal(0, noise * 2), lines=sorted(lines), score=1.0, n=1000, yellow_xy=yxy, ystart=ystart)
        goal = sign if (sign is not None and XI - X <= d_sign) else None
        o = tl.step_obs(obs, (ox, oy, oyaw), scan, goal)
        v, cmd = (o.speed, o.steer) if o.ok else (0.0, 0.0)
        key = (o.state, o.target, o.reason[:10])
        if key != last:
            last = key; log.append((round(X, 2), round(Y, 2), round(math.degrees(psi)), o.state, o.target, [round(a, 1) for a in o.ahead], o.reason))
            if verbose: print(f'  X={X:4.2f} Y={Y:+.2f} psi={math.degrees(psi):+4.0f} {o.state:8s} tgt={o.target} free={[round(a,1) for a in o.ahead]} v={v:+.2f} st={math.degrees(cmd):+4.0f} | {o.reason}')
        if v == 0.0: stopped += cam_dt
        if v < 0.0: rev += -v * cam_dt
        steps = 5; dt = cam_dt / steps
        for _ in range(steps):
            steer += (cmd - steer) * min(1.0, dt / lag)
            if abs(steer) > 1e-4:
                R = T.WHEELBASE / math.tan(abs(steer)) + T.TRACK / 2
                w = math.copysign(v / R, steer)
            else: w = 0.0
            X += v * math.cos(psi) * dt; Y += v * math.sin(psi) * dt; psi += w * dt
            ds = abs(v) * dt                      # odom 은 위치만 (부호 없는 이동) + yaw
            ox += v * math.cos(oyaw) * dt; oy += v * math.sin(oyaw) * dt; oyaw += w * dt * yaw_err
            t += dt
        cx, cy = X + CAR_CENTER * math.cos(psi), Y + CAR_CENTER * math.sin(psi)
        for sx in np.linspace(-CAR_HL, CAR_HL, 5):
            for sy in (-CAR_HW, CAR_HW):
                px = cx + sx * math.cos(psi) - sy * math.sin(psi); py = cy + sx * math.sin(psi) + sy * math.cos(psi)
                for (bx, by, hl, hw) in rects[:-1]:
                    dd = math.hypot(max(abs(px - bx) - hl, 0), max(abs(py - by) - hw, 0))
                    if dd < minclr:
                        minclr = dd; worst = dict(X=round(X, 2), Y=round(Y, 2), psi=round(math.degrees(psi)), v=round(v, 2), state=o.state, tgt=o.target, pt=(round(px, 2), round(py, 2)), ob=(round(bx - hl, 2), round(bx + hl, 2), round(by, 2), round(hw, 2)), corner=(round(float(sx), 2), sy))
                out_max = max(out_max, abs(py) - 0.80)
                if px >= XI: isl_in = max(isl_in, ISL_HALF - abs(py))
        if side_at is None and X >= XI + 0.5: side_at = 'A' if Y > 0 else 'B'
    return dict(done=X >= xend, t=t, minclr=minclr, out=out_max, island=isl_in, stopped=stopped, reversed=rev, side=side_at, log=log, endX=X, worst=worst)
