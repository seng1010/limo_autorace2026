import sys, math
sys.path.insert(0, '.'); import sim6
XI = 6.0
NOISE = (dict(), dict(noise=0.02, lag=0.25, yaw_err=1.15, vis_drop=0.3, seed=1, c0=0.10, psi0=math.radians(6)),
         dict(noise=0.02, lag=0.35, yaw_err=0.85, vis_drop=0.3, seed=2, c0=0.05, psi0=math.radians(-4)),
         dict(noise=0.02, lag=0.25, yaw_err=1.0, vis_drop=0.3, seed=3, lidar=0.10))
def layout(T, kind, rest):
    last = XI - rest
    if kind == 'R':   # 왼쪽·가운데 막힘 -> 오른쪽 차로로 옴
        return [sim6.box(T, 1.3, 0), sim6.box(T, 1.6, 1), sim6.box(T, last - 1.2, 0), sim6.box(T, last - 0.32, 1), sim6.box(T, last - 0.32, 0)]
    if kind == 'L':   # 오른쪽·가운데 막힘 -> 왼쪽 차로로 옴
        return [sim6.box(T, 1.3, 2), sim6.box(T, 1.6, 1), sim6.box(T, last - 1.2, 2), sim6.box(T, last - 0.32, 1), sim6.box(T, last - 0.32, 2)]
    return []         # 장애물 없음 -> 가운데로 옴
CASES = [('R', 2.2, 0, 1.6), ('R', 2.2, 2, 1.6), ('R', 1.7, 0, 1.6), ('R', 1.7, 2, 1.6), ('R', 2.2, 0, 1.2), ('R', 2.2, None, 0),
         ('L', 2.2, 2, 1.6), ('L', 1.7, 0, 1.6), ('C', 0, 0, 1.6), ('C', 0, 2, 1.6), ('C', 0, None, 0)]
for tag in sys.argv[1:]:
    T = sim6.load(tag + '.py')
    for kind, rest, sign, d in CASES:
        res = []
        for kw in NOISE:
            r = sim6.run(T, layout(T, kind, rest), XI=XI, sign=sign, d_sign=d, tmax=70, read_half=0.30, read_m=0.10, **kw)
            res.append((r['side'] or '멈춤') + ('!섬' if r['island'] > 0 else '') + ('!선' if r['out'] > 0 else '') + ('!접촉' if r['minclr'] <= 0 else '')
                       + (f"(정지{r['stopped']:.0f}s,후진{r['reversed']:.1f})" if r['stopped'] > 1 else ''))
        name = '못읽음' if sign is None else ('A' if sign == 0 else 'B')
        where = {'R': f'오른쪽으로 옴(장애물 끝 섬 {rest} m 앞)', 'L': f'왼쪽으로 옴(장애물 끝 섬 {rest} m 앞)', 'C': '가운데로 옴(장애물 없음)'}[kind]
        print(f'[{tag}] {where}, 표지판 {name}, 가운데에서 섬 {d} m 안에서만 읽힘: ' + ' '.join(res)); sys.stdout.flush()
