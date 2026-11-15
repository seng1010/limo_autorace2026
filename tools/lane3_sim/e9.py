import sys, math, numpy as np
sys.path.insert(0, '.'); import sim5
def seven(n, seed, XI=5.0, x_lo=0.9, x_hi=4.0):
    """정차 LIMO 7대를 3차선 구간에 무작위로. 어느 0.6 m 구간에서도 세 차로가 다 막히지는 않게."""
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        obs = []
        for _ in range(400):
            if len(obs) == 7: break
            x, l = x_lo + rng.random() * (x_hi - x_lo), int(rng.integers(0, 3))
            if any(l2 == l and abs(x2 - x) < 0.75 for x2, l2 in obs): continue
            lanes_near = {l2 for x2, l2 in obs if abs(x2 - x) < 0.60} | {l}
            if len(lanes_near) == 3: continue
            obs.append((x, l))
        if len(obs) < 7: continue
        out.append(dict(XI=XI, obs=sorted(obs), sign=int(rng.choice([0, 2])),
                        kw=dict(noise=0.02, lag=float(rng.choice([0.2, 0.3])), yaw_err=float(rng.choice([0.85, 1.0, 1.15])), vis_drop=0.3,
                                seed=len(out), c0=float(rng.normal(0.1, 0.08)), psi0=float(rng.normal(0, math.radians(5))), lidar=float(rng.choice([0.10, 0.20])))))
    return out
def evaluate(tag, path, tl_, d_sign=1.7, show=0):
    T = sim5.load(path); st = dict(done=0, hit=0, island=0, out=0, right=0, rev=0, stop=0.0); worst = 9; shown = 0
    for i, tr in enumerate(tl_):
        obs = [sim5.box(T, x, l) for x, l in tr['obs']]
        r = sim5.run(T, obs, XI=tr["XI"], sign=tr["sign"], d_sign=d_sign, tmax=70, **tr["kw"])
        want = 'A' if tr['sign'] == 0 else 'B'
        st['done'] += r['done']; st['hit'] += r['minclr'] <= 0.0; st['island'] += r['island'] > 0.0; st['out'] += r['out'] > 0.0
        st['right'] += r['side'] == want; st['rev'] += r['reversed'] > 0.05; st['stop'] += r['stopped']; worst = min(worst, r['minclr'])
        if show and shown < show and (not r['done'] or r['minclr'] <= 0 or r['island'] > 0 or r['out'] > 0):
            shown += 1
            print(f"    #{i} 표지판 {want} 끝X {r['endX']:.2f} 간쪽 {r['side']} 여유 {r['minclr']*100:.0f}cm 섬 {r['island']:.2f} 선 {r['out']:.2f} | {[(round(x,2), l) for x, l in tr['obs']]}")
            for e in r['log'][-5:]: print('        X=%.2f Y=%+.2f psi=%+d %s tgt=%s free=%s | %s' % (e[0], e[1], e[2], e[3], e[4], e[5], e[6][:60]))
    n = len(tl_)
    print(f"  [{tag}] {n}개: 통과 {st['done']}, 접촉 {st['hit']}, 섬 밟음 {st['island']}, 바깥선 {st['out']}, 표지판대로 {st['right']}, 후진 {st['rev']}, 평균 정지 {st['stop']/n:.1f}s, 최소 여유 {worst*100:.0f} cm")
if __name__ == '__main__':
    tl_ = seven(80, 7)
    print('정차 LIMO 7대 무작위 배치 80개 (섬은 3차선 시작 5.0 m 뒤)')
    for a in sys.argv[1:]:
        tag, path = a.split('='); evaluate(tag, path, tl_, show=3 if tag == '새' else 0)
