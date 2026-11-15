import sys
sys.path.insert(0, '.')
import sim6, score, numpy as np
path, i, sign, x0, x1 = sys.argv[1], int(sys.argv[2]), 0 if sys.argv[3] == 'A' else 2, float(sys.argv[4]), float(sys.argv[5])
T = sim6.load(path)
name, tr = score.layouts()[i]
orig = T.PathPlanner.update
def upd(self, c, th, along, lateral, island=None, goal=None, steer0=0.0, have_scan=True, desperate=False, block_y=None):
    p = orig(self, c, th, along, lateral, island, goal, steer0, have_scan, desperate, block_y)
    upd.n += 1
    print(f'   #{upd.n} c={c:+.2f} th={np.degrees(th):+.0f} isl={island} goal={goal} side={self.side} desp={desperate} -> y={p.y:+.2f} {p.state} v={p.speed} F={p.free:.2f} | {p.reason[:90]}')
    return p
upd.n = 0
class Gate:
    pass
T.PathPlanner.update = upd
import builtins
r = sim6.run(T, [sim6.box(T, x, l) for x, l in tr['obs']], XI=tr['XI'], sign=sign, d_sign=score.D_SIGN, tmax=score.TMAX,
             read_half=score.READ_HALF, read_m=score.READ_M, verbose=1, **tr['kw'])
