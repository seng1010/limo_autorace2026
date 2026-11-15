import sys, math
sys.path.insert(0, '.'); import sim5
def layout(T, gap=0.82, wide=0.30):
    # 11/15 실차 로그 재현: 가운데 차로 긴 장애물(1.8~2.55), 오른쪽 차로 2개, 왼쪽 차로 장애물은 가운데 장애물 끝에서 gap 뒤
    return [sim5.box(T, 1.80 + 0.20, 1, 0.75, wide), sim5.box(T, 1.45 + 0.20, 2, 0.32, wide), sim5.box(T, 2.20 + 0.20, 2, 0.32, wide),
            sim5.box(T, 2.55 + 0.20 + gap, 0, 0.32, wide)]
NOISE = (dict(), dict(noise=0.02, lag=0.25, yaw_err=1.25, vis_drop=0.3, seed=1, c0=0.10, psi0=math.radians(8)),
         dict(noise=0.02, lag=0.35, yaw_err=0.8, vis_drop=0.3, seed=2, c0=0.05, psi0=math.radians(-4)))
if __name__ == '__main__':
    T = sim5.load(sys.argv[1])
    r = sim5.run(T, layout(T), xend=5.5, verbose=1, c0=0.10)
    print('결과: 통과', r['done'], '최소여유 %.0f cm' % (r['minclr'] * 100), '정지 %.1f s' % r['stopped'], '후진 %.2f m' % r['reversed'], '끝 X %.2f' % r['endX'])
