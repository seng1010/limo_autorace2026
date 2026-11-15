"""성적표의 배치 몇 개만 다시 돌린다:  python3 cases.py <three_lane.py 경로>[,상수=값] 100B 108A ..."""
import sys
from multiprocessing import Pool
sys.path.insert(0, '.')
import score
if __name__ == '__main__':
    parts = sys.argv[1].split(',')
    over = {k: score._value(v) for k, v in (p.split('=') for p in parts[1:])}
    L = score.layouts()
    work = [(int(a[:-1]), L[int(a[:-1])][0], L[int(a[:-1])][1], 0 if a[-1] == 'A' else 2) for a in sys.argv[2:]]
    with Pool(2, score._init, (parts[0], over)) as pool:
        res = pool.map(score._one, work, chunksize=1)
    ok = 0
    for r in res:
        good = r['done'] and not r['hit'] and not r['island'] and not r['out'] and r['right']
        ok += good
        print(f"#{r['i']}{r['sign']} {'OK ' if good else '-- '} 완주 {r['done']} 간쪽 {r['side']} 접촉 {r['hit']} 섬 {r['island']} 바깥선 {r['out']} 여유 {r['minclr']*100:.0f}cm 끝X {r['endX']:.2f} 시간 {r['t']:.0f}s")
    print(f'무결점 {ok}/{len(res)}')
