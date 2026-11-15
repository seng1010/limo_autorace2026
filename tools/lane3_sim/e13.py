import sys, math
sys.path.insert(0, '.'); import sim5, repro
tag, path = sys.argv[1].split('='); T = sim5.load(path); XI = 6.0
NOISE = (dict(), dict(noise=0.02, lag=0.25, yaw_err=1.15, vis_drop=0.3, seed=1, c0=0.10, psi0=math.radians(6)),
         dict(noise=0.02, lag=0.35, yaw_err=0.85, vis_drop=0.3, seed=2, c0=0.05, psi0=math.radians(-4)),
         dict(noise=0.02, lag=0.25, yaw_err=1.0, vis_drop=0.3, seed=3, lidar=0.10))
print(f'[{tag}] 왼쪽·가운데 차로에 장애물이 섬 앞 rest m 까지 이어짐 -> 오른쪽 차로로 감. 표지판 A 를 섬 d m 앞에서 확정.')
print('   칸 = A 로 간 횟수/4, 못 지나감, 섬 밟음, 바깥선, 최소 여유 cm')
for rest in (2.2, 1.9, 1.7, 1.5, 1.3, 1.1):
    row = []
    for d_sign in (1.7, 1.5, 1.3):
        a = stuck = isl = out = 0; mc = 9
        for kw in NOISE:
            last = XI - rest                                   # 마지막 장애물의 뒤끝
            obs = [sim5.box(T, 1.3, 0, 0.32, 0.22), sim5.box(T, 1.6, 1, 0.32, 0.22),
                   sim5.box(T, last - 1.2, 0, 0.32, 0.22), sim5.box(T, last - 0.32, 1, 0.32, 0.22), sim5.box(T, last - 0.32, 0, 0.32, 0.22)]
            r = sim5.run(T, obs, XI=XI, sign=0, d_sign=min(d_sign, rest), tmax=70, **kw)
            a += r['side'] == 'A'; stuck += not r['done']; isl += r['island'] > 0; out += r['out'] > 0; mc = min(mc, r['minclr'])
        row.append(f'd{d_sign}: A {a}/4 멈춤{stuck} 섬{isl} 선{out} {mc*100:.0f}cm')
    print(f'   rest {rest:.1f} | ' + ' | '.join(row)); sys.stdout.flush()
