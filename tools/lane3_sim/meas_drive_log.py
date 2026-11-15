import csv, glob, math, numpy as np
L = 0.20
files = sorted(set(glob.glob('lanelog*/*/log.csv') + glob.glob('lanelog9b/log.csv')))
seen = set(); allR = []; lags = []; widths = []
for f in files:
    key = f.split('/')[-2] if f.count('/') == 2 else f
    if key in seen: continue
    rd = csv.DictReader(open(f))
    if 'ox' not in (rd.fieldnames or []): continue
    seen.add(key)
    rows = [r for r in rd if r.get('ox') not in ('', None) and r.get('oyaw_deg') not in ('', None)]
    if len(rows) < 100: continue
    t = np.array([float(r['t']) for r in rows]); x = np.array([float(r['ox']) for r in rows]); y = np.array([float(r['oy']) for r in rows])
    yaw = np.unwrap(np.radians([float(r['oyaw_deg']) for r in rows])); st = np.radians([float(r['steer_deg']) for r in rows]); v = np.array([float(r['speed']) for r in rows])
    widths += [float(r['lane_width_cm']) for r in rows if r['source'] == 'both' and r['lane_width_cm']]
    tu = np.arange(t[0], t[-1], 0.02); xi, yi, yawi, sti = (np.interp(tu, t, a) for a in (x, y, yaw, st))
    k = 10
    ds = np.hypot(xi[k:] - xi[:-k], yi[k:] - yi[:-k]); dyaw = yawi[k:] - yawi[:-k]
    kappa = np.where(ds > 0.02, dyaw / np.maximum(ds, 1e-6), 0.0)
    tc = tu[:-k] + 0.1
    sc = np.interp(tc, tu, sti); cmdk = np.tan(sc) / L
    moving = ds > 0.03
    big = (np.abs(sc) >= math.radians(22)) & moving & (np.abs(kappa) > 0.5)
    if big.sum() > 10: allR += list(1 / np.abs(kappa[big]))
    if moving.sum() > 200 and np.std(cmdk[moving]) > 0.3:
        best = None
        for d in np.arange(0.0, 0.8, 0.02):
            c = np.interp(tc - d, tc, cmdk); m = moving & (tc - d > tc[0])
            cc = np.corrcoef(c[m], kappa[m])[0, 1]
            if best is None or cc > best[1]: best = (d, cc)
        c = np.interp(tc - best[0], tc, cmdk); m = moving & (np.abs(c) > 0.5)
        gain = np.median(kappa[m] / c[m]) if m.sum() > 20 else float('nan')
        lags.append((key, best[0], best[1], gain, float(np.median(v[v > 0.05])) if (v > 0.05).any() else 0, len(rows)))
allR = np.array(allR)
if len(allR): print('조향 22도 이상 구간의 회전 반경: n=%d  5%% %.2f  25%% %.2f  중앙 %.2f  75%% %.2f m' % (len(allR), *np.percentile(allR, [5, 25, 50, 75])))
print('지연(명령 -> 실제 곡률):')
for k, d, cc, g, vv, n in lags: print('  %-28s 지연 %.2f s  상관 %.2f  실제/명령 곡률 %.2f  속도 %.2f  n=%d' % (k, d, cc, g, vv, n))
widths = np.array(widths)
if len(widths): print('양쪽 선이 다 보일 때 차로 폭: n=%d 중앙 %.1f cm, 10~90%% %.1f~%.1f' % (len(widths), np.median(widths), *np.percentile(widths, [10, 90])))
