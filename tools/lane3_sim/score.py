"""3차선 판단 성적표: 무작위 배치 200개 x 표지판 A/B = 400회 주행을 돌려 표로 찍는다.

  python3 score.py 옛=../../src/jetracer_ros2/jetracer_ros2/three_lane.py,PLANNER=old 새=../../src/jetracer_ros2/jetracer_ros2/three_lane.py
  (이름=경로[,상수=값,...] 을 여러 개. 옵션: -n 배치 수(기본 200), -j 프로세스 수(기본 2), --show 실패 로그 몇 개, --json 결과 저장 파일)

배치 200개 = 완전 무작위 7대 100개 (e9.seven) + 4줄 배치 100개 (e10.rows7, 줄 간격 1.2~1.4 / 1.0~1.2 / 0.85~1.0 m).
4개 중 1개는 왼쪽으로 치우쳐 들어온다 (실차: 3차선을 +28~36 cm 에서 시작).
표지판은 sim6 방식: 섬 1.6 m 안, 도로 가운데 +-0.30 m 안에서 0.10 m 간 뒤에야 읽힌다.
"""
import sys, math, json, time
from multiprocessing import Pool
sys.path.insert(0, '.')
import sim6, e9, e10

D_SIGN, READ_HALF, READ_M, TMAX = 1.6, 0.30, 0.10, 70


def layouts(n=200):
    L = [('무작위 7대', tr) for tr in e9.seven(n // 2, 7)]
    rest = n - len(L)
    sets = (('4줄 1.2~1.4', 1.2, 1.4, 31), ('4줄 1.0~1.2', 1.0, 1.2, 32), ('4줄 0.85~1.0', 0.85, 1.0, 33))
    for k, (name, smin, smax, seed) in enumerate(sets):
        m = rest // 3 + (1 if k < rest % 3 else 0)
        L += [(name, tr) for tr in e10.rows7(m, seed, smin, smax)]
    for i, (name, tr) in enumerate(L):
        if i % 4 == 3:                                  # 왼쪽으로 치우쳐 진입
            tr['kw'] = dict(tr['kw'], c0=0.32 + 0.04 * ((i // 4) % 3 - 1), psi0=math.radians(-4.0))
    return L


_T = None


def _init(path, over):
    global _T
    _T = sim6.load(path, **over)


def _one(job):
    i, name, tr, sign = job
    obs = [sim6.box(_T, x, l) for x, l in tr['obs']]
    r = sim6.run(_T, obs, XI=tr['XI'], sign=sign, d_sign=D_SIGN, tmax=TMAX, read_half=READ_HALF, read_m=READ_M, **tr['kw'])
    want = 'A' if sign == 0 else 'B'
    return dict(i=i, set=name, sign=want, done=bool(r['done']), hit=bool(r['minclr'] <= 0.0), island=bool(r['island'] > 0.0),
                out=bool(r['out'] > 0.0), right=r['side'] == want, side=r['side'], rev=bool(r['reversed'] > 0.05),
                minclr=float(r['minclr']), t=float(r['t']), stopped=float(r['stopped']), endX=float(r['endX']), XI=tr['XI'],
                obs=[(round(x, 2), l) for x, l in tr['obs']], log=[list(e) for e in r['log'][-8:]])


def run(path, over, L, jobs=2):
    work = [(i, name, tr, sign) for i, (name, tr) in enumerate(L) for sign in (0, 2)]
    with Pool(jobs, _init, (path, over)) as pool:
        return pool.map(_one, work, chunksize=4)


def table(tag, res):
    names = []
    for r in res:
        if r['set'] not in names:
            names.append(r['set'])
    print(f'[{tag}]')
    print('  배치          주행  완주  멈춤  접촉  섬밟음 바깥선 갈림길정답 후진  무결점  평균시간')
    for name in names + ['합계']:
        rs = [r for r in res if name == '합계' or r['set'] == name]
        n = len(rs)
        clean = sum(r['done'] and not r['hit'] and not r['island'] and not r['out'] and r['right'] for r in rs)
        done = sum(r['done'] for r in rs)
        tm = sum(r['t'] for r in rs if r['done']) / max(1, done)
        print(f"  {name:12s} {n:5d} {done:5d} {n - done:5d} {sum(r['hit'] for r in rs):5d} {sum(r['island'] for r in rs):6d} "
              f"{sum(r['out'] for r in rs):6d} {sum(r['right'] for r in rs):9d} {sum(r['rev'] for r in rs):5d} {clean:6d} {tm:8.1f}s")
    sys.stdout.flush()


def show(res, k):
    bad = [r for r in res if not r['done'] or r['hit'] or r['island'] or r['out'] or not r['right']]
    for r in bad[:k]:
        what = ('멈춤 ' if not r['done'] else '') + ('접촉 ' if r['hit'] else '') + ('섬 ' if r['island'] else '') \
            + ('바깥선 ' if r['out'] else '') + ('' if r['right'] else f"갈림길 {r['side']} ")
        print(f"  #{r['i']} {r['set']} 표지판 {r['sign']} -> {what}| 끝 X {r['endX']:.2f} 섬 {r['XI']:.2f} 여유 {r['minclr'] * 100:.0f} cm | {r['obs']}")
        for e in r['log'][-5:]:
            print('      X=%.2f Y=%+.2f psi=%+d %s tgt=%s free=%s | %s' % (e[0], e[1], e[2], e[3], e[4], e[5], str(e[6])[:70]))


def _value(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


if __name__ == '__main__':
    args = sys.argv[1:]
    n, jobs, nshow, out = 200, 2, 0, None
    specs = []
    while args:
        a = args.pop(0)
        if a == '-n':
            n = int(args.pop(0))
        elif a == '-j':
            jobs = int(args.pop(0))
        elif a == '--show':
            nshow = int(args.pop(0))
        elif a == '--json':
            out = args.pop(0)
        else:
            specs.append(a)
    L = layouts(n)
    saved = {}
    for spec in specs:
        tag, rest = spec.split('=', 1)
        parts = rest.split(',')
        over = {k: _value(v) for k, v in (p.split('=') for p in parts[1:])}
        t0 = time.time()
        res = run(parts[0], over, L, jobs)
        table(f'{tag}  ({time.time() - t0:.0f}초)', res)
        if nshow:
            show(res, nshow)
        saved[tag] = res
    if out:
        with open(out, 'w') as f:
            json.dump(saved, f, ensure_ascii=False)
