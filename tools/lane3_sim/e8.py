import sys, math, numpy as np
sys.path.insert(0, '.'); import sim5, repro, e6
def gapsweep(path):
    T = sim5.load(path); row = []
    for gap in (0.4, 0.5, 0.6, 0.7, 0.82, 1.0):
        n = 0; mc = 9; st = 0
        for lidar in (0.20, 0.10):
            for wide in (0.22, 0.30, 0.40):
                for kw in repro.NOISE:
                    r = sim5.run(T, repro.layout(T, gap, wide), xend=5.5, lidar=lidar, **kw)
                    n += r['done']; mc = min(mc, r['minclr']); st += r['stopped']
        row.append(f'{gap:.2f}: {n:2d}/18 {mc*100:3.0f}cm')
    return ' | '.join(row)
if __name__ == '__main__':
    for a in sys.argv[1:]:
        tag, path = a.split('=')
        print(f'[{tag}] 실차 배치 간격별 (통과/18, 최소 여유): ' + gapsweep(path))
    for title, tl_ in (('보통 간격 100개', e6.trials(100, 21)), ('촘촘한 간격 100개', e6.trials(100, 22, (1.0, 0.4)))):
        print(title)
        for a in sys.argv[1:]:
            tag, path = a.split('='); e6.evaluate(tag, path, tl_)
