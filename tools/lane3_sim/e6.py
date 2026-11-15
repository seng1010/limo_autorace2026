import sys, math, numpy as np
sys.path.insert(0, '.'); import sim5
def trials(n, seed, row_gap=(1.35, 0.5), tight=False):
    rng = np.random.default_rng(seed); out = []
    for k in range(n):
        XI = 6.2; rest = 0.8 + rng.random() * 2.6
        open_lane = int(rng.integers(0, 3)); x = 1.3 + rng.random() * 0.4; obs = []
        while x < XI - rest:
            nxt = int(np.clip(open_lane + rng.choice([-1, 1]), 0, 2))
            if nxt == open_lane: nxt = 1
            blocked = [l for l in range(3) if l != nxt]
            if rng.random() < 0.35: blocked = [l for l in blocked if l == open_lane] or blocked[:1]
            if open_lane not in blocked: blocked = [open_lane]
            for l in blocked:
                obs.append((x + rng.normal(0, 0.05), l, 0.32 + rng.random() * 0.2, 0.22 + rng.random() * 0.18, rng.normal(0, 0.03)))
            open_lane = nxt; x += row_gap[0] + rng.random() * row_gap[1]
        out.append(dict(XI=XI, obs=obs, sign=int(rng.choice([0, 2])),
                        kw=dict(noise=0.02, lag=float(rng.choice([0.2, 0.3])), yaw_err=float(rng.choice([0.8, 1.0, 1.25])), vis_drop=0.3, seed=k,
                                c0=float(rng.normal(0, 0.05)), psi0=float(rng.normal(0, math.radians(4))), lidar=float(rng.choice([0.10, 0.20])))))
    return out
def evaluate(tag, path, tr_list):
    T = sim5.load(path); st = dict(n=0, done=0, hit=0, near=0, island=0, out=0, right=0, stop=0.0, rev=0); worst = 9
    for tr in tr_list:
        obs = [sim5.box(T, x, l, ln, wd, dy) for x, l, ln, wd, dy in tr['obs']]
        r = sim5.run(T, obs, XI=tr['XI'], sign=tr['sign'], d_sign=1.7, **tr['kw'])
        want = 'A' if tr['sign'] == 0 else 'B'
        st['n'] += 1; st['done'] += r['done']; st['hit'] += r['minclr'] <= 0.0; st['near'] += 0.0 < r['minclr'] < 0.04
        st['island'] += r['island'] > 0.0; st['out'] += r['out'] > 0.0; st['right'] += r['side'] == want; st['stop'] += r['stopped']; st['rev'] += r['reversed'] > 0.05
        worst = min(worst, r['minclr'])
    print(f"  [{tag}] {st['n']}개: 통과 {st['done']}, 접촉 {st['hit']}, 4cm 미만 스침 {st['near']}, 섬 밟음 {st['island']}, 바깥선 {st['out']}, 표지판대로 {st['right']}, 후진한 주행 {st['rev']}, 최소 여유 {worst*100:.0f} cm")
if __name__ == '__main__':
    for title, tl_ in (('보통 간격 (줄 사이 1.35~1.85 m) 100개', trials(100, 21)), ('촘촘한 간격 (줄 사이 1.0~1.4 m) 100개', trials(100, 22, (1.0, 0.4)))):
        print(title)
        for a in sys.argv[1:]:
            tag, path = a.split('='); evaluate(tag, path, tl_)
