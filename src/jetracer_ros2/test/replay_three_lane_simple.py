"""기록(프레임 + odom)으로 위치 추정만 다시 돌려 본다. python3 replay.py 기록폴더"""
import csv, math, sys, os
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jetracer_ros2 import three_lane_simple as S
d = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(d, 'log.csv'))))
sup = S.Lane3SimpleSupervisor(forced=True)
show = set(int(x) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else set()
for r in rows:
    k = int(r['frame'])
    if not r['ox']:
        continue
    pose = (float(r['ox']), float(r['oy']), math.radians(float(r['oyaw_deg'])))
    img = cv2.imread(os.path.join(d, f'{k:05d}.jpg'))
    if sup.mode != 'on':
        sup._try_enter(img, pose, None)
        if sup.mode == 'on':
            print('enter', k, sup.message)
        continue
    core = sup.core
    obs = sup.fitter.fit(img, core.th, None)
    core._advance(pose)
    core._camera_heading(obs)
    core._camera_lines(obs)
    core._camera_island(obs)
    if k in show or (not show and k % 20 == 0):
        old = r['note'][4:60]
        print(k, f'new c{core.c:+.2f} th{math.degrees(core.th):+.0f} s{core.s:.2f}', '| old', old)
