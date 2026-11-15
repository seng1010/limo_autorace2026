import sys, math, numpy as np
sys.path.insert(0, '.'); import sim5, e9
def rows7(n, seed, smin, smax, XI_after=1.6):
    """정차 LIMO 7대: 줄 4개 (2대, 2대, 2대, 1대). 줄마다 한 차로만 열려 있고, 열린 차로는 옆 차로로 한 칸씩 바뀐다."""
    rng = np.random.default_rng(seed); out = []
    for k in range(n):
        x = 1.0 + rng.random() * 0.3; open_lane = int(rng.integers(0, 3)); obs = []
        for row in range(4):
            blocked = [l for l in range(3) if l != open_lane]
            if row == 3: blocked = [blocked[int(rng.integers(0, 2))]]
            obs += [(x + rng.normal(0, 0.04), l) for l in blocked]
            nxt = open_lane + int(rng.choice([-1, 1]))
            open_lane = nxt if 0 <= nxt <= 2 else 1
            last_x = x; x += smin + rng.random() * (smax - smin)
        out.append(dict(XI=last_x + 0.32 + XI_after + rng.random() * 0.8, obs=sorted(obs), sign=int(rng.choice([0, 2])),
                        kw=dict(noise=0.02, lag=float(rng.choice([0.2, 0.3])), yaw_err=float(rng.choice([0.85, 1.0, 1.15])), vis_drop=0.3,
                                seed=k, c0=float(rng.normal(0.1, 0.08)), psi0=float(rng.normal(0, math.radians(5))), lidar=float(rng.choice([0.10, 0.20])))))
    return out
if __name__ == '__main__':
    tag, path = sys.argv[1].split('=')
    for title, smin, smax in (('줄 간격 1.2~1.4 m', 1.2, 1.4), ('줄 간격 1.0~1.2 m', 1.0, 1.2), ('줄 간격 0.85~1.0 m', 0.85, 1.0)):
        print('LIMO 7대 (줄 4개), ' + title + ', 40개'); sys.stdout.flush()
        e9.evaluate(tag, path, rows7(40, 31, smin, smax), show=int(sys.argv[2]) if len(sys.argv) > 2 else 0); sys.stdout.flush()
