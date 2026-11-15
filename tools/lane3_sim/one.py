"""성적표의 배치 하나를 자세히 본다:  python3 one.py <three_lane.py 경로> <배치 번호> <A|B> [상수=값 ...]"""
import sys
sys.path.insert(0, '.')
import sim6, score
path, i, sign = sys.argv[1], int(sys.argv[2]), 0 if sys.argv[3] == 'A' else 2
over = {k: score._value(v) for k, v in (a.split('=') for a in sys.argv[4:])}
T = sim6.load(path, **over)
name, tr = score.layouts()[i]
print(name, '섬', round(tr['XI'], 2), tr['obs'], tr['kw'])
r = sim6.run(T, [sim6.box(T, x, l) for x, l in tr['obs']], XI=tr['XI'], sign=sign, d_sign=score.D_SIGN, tmax=score.TMAX,
             read_half=score.READ_HALF, read_m=score.READ_M, verbose=1, **tr['kw'])
print('결과: 통과', r['done'], '간 쪽', r['side'], '최소여유 %.0f cm' % (r['minclr'] * 100), '정지 %.1f s' % r['stopped'],
      '후진 %.2f m' % r['reversed'], '끝 X %.2f' % r['endX'], '섬', r['island'] > 0, '바깥선', r['out'] > 0, r['worst'])
