"""Exactly-seven closed-loop M7 scenarios, with first-return lidar occlusion.

Run from repo root: PYTHONPATH=src/jetracer_ros2 python .../sim_m7_occlusion.py
The simulator owns obstacle truth. The controller receives only scans, camera
line observations and odometry. Collision checks use independent polygon SAT.
These are synthetic sensor-level tests, not camera/ROS/hardware certification.
"""
import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import replace

import numpy as np

from jetracer_ros2 import three_lane as T
from jetracer_ros2 import three_lane_simple as S
from jetracer_ros2.m7_navigation import MappedSupervisor, NavigationConfig
from sim_three_lane_simple import rect, overlap, raycast


def polygon_overlap(p,q):
    # Independent SAT supporting the six-vertex island (legacy helper assumed 4).
    for poly in (p,q):
        for edge in np.roll(poly,-1,axis=0)-poly:
            axis = np.array([-edge[1],edge[0]])
            a,b = p@axis,q@axis
            if a.max()<b.min() or b.max()<a.min():
                return False
    return True


def layout(kind, mirror=False, seed=0):
    rng = np.random.default_rng(seed)
    if kind == 'weave':
        entries = [(1.40,1), (1.65,2), (2.05,2), (2.65,0),
                   (3.05,0), (3.60,1), (3.95,0)]
        goal = 2
    elif kind == 'blocked':
        entries = [(1.4,j) for j in range(3)] + [(2.3,0),(2.3,1),(3.2,1),(3.2,2)]
        goal = 0
    else:
        entries = [(1.35,1),(1.35,2),(2.10,1),(2.10,2),(2.85,1),(2.85,2),(3.60,1)]
        goal = 0
    if mirror:
        entries = [(s, 2-j) for s,j in entries]
        goal = 2-goal
    # Each obstacle covers essentially the entire 0.535 m lane. Translation
    # noise changes longitudinal clearance, never opens a gap inside its lane.
    polys = [rect(s+rng.uniform(-.015,.015), T.LANE_CENTER[j], 0., .32, .55)
             for s,j in entries]
    assert len(polys) == 7
    return polys, goal


def observation(c, psi, rng, yellow=True, s=0.):
    # LineFitter reports distance normal to the road, after heading rotation.
    lines = []
    for p, ratio in ((-.27,0.),(.27,0.),(-.80,1.),(.80,1.)):
        if s >= S.START_TO_ISLAND and ratio==0:
            continue
        if ratio:
            width = max(.27,.80-1.06*max(0.,s-S.START_TO_ISLAND-1.79))
            p = math.copysign(width,p)
        if ratio and not yellow:
            continue
        u = p-c+rng.normal(0,.003)
        if abs(u) > T.FAR_U:
            u = math.copysign(T.FAR_U+(abs(u)-T.FAR_U)*T.FAR_SHRINK,u)
        if abs(u) <= 1.0:
            lines.append((u,150,ratio))
    return dict(th=-psi+rng.normal(0,math.radians(.3)), n=1000, lines=lines,
                short=set(), ystart={}, yellow_xy=None)


def run_case(kind='corridor', mirror=False, seed=0, faults=False, verbose=False, signs=False,
             profile='baseline', planner='hybrid', duration=120.):
    rng = np.random.default_rng(seed)
    polys, goal = layout(kind, mirror, seed)
    i = S.START_TO_ISLAND
    island = np.array([(i,0.),(i+.21,.27),(i+S.ISLAND_LEN-.21,.27),
                       (i+S.ISLAND_LEN,0.),(i+S.ISLAND_LEN-.21,-.27),(i+.21,-.27)])
    x, y, psi = .30-S.CAM_FROM_ODOM, rng.uniform(-.01,.01), 0.
    sup = MappedSupervisor(forced=True, goal=None if signs else goal,
                           config=replace(NavigationConfig(),planner=planner))
    sup.mode = 'on'
    ox, oy, oyaw = 5., -2., 1.3
    sup.core.start(y,0.,(ox,oy,oyaw),.30)
    distance_scale = rng.uniform(.985,1.015) if faults else 1.
    yaw_scale = rng.uniform(.99,1.01) if faults else 1.
    steer = 0.
    dt = .10 if profile=='baseline' else .05
    velocity = 0.
    last_scan,last_camera = -math.inf,-math.inf
    scan_queue = []
    latest_scan = None
    scan_ident = 0
    touched, out_count, island_count = set(), 0, 0
    max_yaw, max_stop, stopped = 0., 0., 0.
    compute, states, reverses = [], set(), 0
    trace = []
    for frame in range(int(duration/dt)):
        t = frame*dt
        cx, cy = x+S.CAM_FROM_ODOM*math.cos(psi), y+S.CAM_FROM_ODOM*math.sin(psi)
        pose = (ox,oy,oyaw)
        if t-last_scan>=.10-1e-8:
            raw = list(raycast(cx,cy,psi,polys))  # Painted island is a road constraint.
            if profile in ('partial','delayed'):
                raw[0][rng.random(len(raw[0]))<.40] = 0.
                valid = np.isfinite(raw[0]) & (raw[0]>0)
                raw[0][valid] += .008+rng.uniform(-.008,.008,int(valid.sum()))
            elif profile=='blind':
                angles = raw[1]+np.arange(len(raw[0]))*raw[2]
                raw[0][np.abs(angles)<math.radians(85)] = 0.
            scan_ident += 1
            scan_queue.append((t+(.10 if profile=='delayed' else 0.),tuple(raw)+(pose,scan_ident,8.)))
            last_scan = t
        while scan_queue and scan_queue[0][0]<=t+1e-8:
            _,latest_scan = scan_queue.pop(0)
        scan = latest_scan
        obs = None
        if t-last_camera>=(.10 if profile!='baseline' else dt)-1e-8:
            obs = observation(cy,psi,rng,yellow=abs(psi)<math.radians(15),s=cx)
            obs['_observed_at'] = t
            last_camera = t
        if signs and abs(cy)<.3 and 1.4<S.START_TO_ISLAND-cx<1.75:
            sup.see_signs([dict(label='sign_a' if goal==0 else 'sign_b',confidence=.9,bbox=[100,100,140,145])],
                          stamp=t,capture_pose=pose,now=t,monotonic_now=t)
        if faults:
            if 2.0 <= t < 2.5:
                pose = None
            if 6.0 <= t < 6.5:
                scan = None
            if 10.0 <= t < 11.0 or (abs(psi)>math.radians(20) and frame%4==0):
                obs = None
        before = time.perf_counter()
        result = sup.step_obs(obs,pose,scan,False,now=t)
        compute.append(time.perf_counter()-before)
        if result is None:
            break
        states.add(sup.core.state)
        v, cmd = result['v'], result['steer']
        stopped = stopped+dt if abs(v)<1e-5 else 0.
        max_stop = max(max_stop,stopped)
        steer += (cmd-steer)*min(1.,dt/.15)
        if profile=='baseline':
            velocity = v
        else:
            # Independent plant: commands do not stop the chassis instantly.
            limit = (.35 if abs(v)<abs(velocity) or v*velocity<0 else .25)*dt
            next_velocity = velocity+float(np.clip(v-velocity,-limit,limit))
            velocity = next_velocity
        ds = velocity*dt
        curvature = (math.copysign(1/(T.WHEELBASE/math.tan(abs(steer))+T.TRACK/2),steer)
                     if abs(steer)>1e-5 else 0.)
        dpsi = ds*curvature
        x += ds*math.cos(psi+dpsi/2)
        y += ds*math.sin(psi+dpsi/2)
        psi += dpsi
        ox += ds*distance_scale*math.cos(oyaw+dpsi*yaw_scale/2)
        oy += ds*distance_scale*math.sin(oyaw+dpsi*yaw_scale/2)
        oyaw += dpsi*yaw_scale
        max_yaw = max(max_yaw,abs(psi))
        body = rect(x+.10*math.cos(psi),y+.10*math.sin(psi),psi,.32,.22)
        touched.update(i for i,p in enumerate(polys) if overlap(body,p))
        road_half = np.minimum(.80,.275+np.maximum(0.,body[:,0]-.05))
        road_half = np.minimum(road_half,np.maximum(.27,.80-1.06*np.maximum(0.,body[:,0]-i-1.79)))
        out_count += int(np.any(np.abs(body[:,1])>road_half))
        island_count += int(polygon_overlap(body,island))
        if frame%10==0:
            trace.append(dict(t=round(t,1),s=round(cx,3),c=round(cy,3),v=v,
                              state=sup.core.state,reason=sup.core.reason))
            if verbose:
                print(trace[-1],flush=True)
        if touched or out_count or island_count or stopped>15.1:
            break
    lane = int(np.argmin(np.abs(np.asarray(T.LANE_CENTER)-y)))
    safe = not touched and not out_count and not island_count
    complete = (sup.mode=='done' and lane==1 and sup.side_taken==goal
                and (not signs or sup.sign_goal==goal))
    expected_stop = kind=='blocked' or profile=='blind'
    result = dict(kind=kind,mirror=mirror,seed=seed,faults=faults,signs=signs,sign_goal=sup.sign_goal,obstacles=len(polys),
                  profile=profile,planner=planner,expected='stop' if expected_stop else 'complete',
                  width=.55,ok=safe and (not complete and max_stop>=15 if expected_stop else complete),
                  complete=complete,contacts=len(touched),out_frames=out_count,island_frames=island_count,
                  t=round(t,1),s=round(x,3),c=round(y,3),max_yaw_deg=round(math.degrees(max_yaw),1),
                  max_stop_sec=round(max_stop,1),reverse_count=sup.core.reverse_count,
                  state=sup.core.state,reason=sup.core.reason,
                  median_ms=round(float(np.median(compute))*1000,1),
                  p95_ms=round(float(np.percentile(compute,95))*1000,1),
                  p99_ms=round(float(np.percentile(compute,99))*1000,1),plan_count=sup.core.planner.plan_count,
                  control_ticks=frame+1,scan_updates=sup.core.visibility.sequence)
    result['trace'] = trace
    return result


def worker(args):
    return run_case(*args)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--kind',choices=['all','corridor','weave','blocked'],default='all')
    parser.add_argument('--output')
    parser.add_argument('--signs',action='store_true',help='Use synthetic detections from the centre lane (weave layout).')
    parser.add_argument('--profile',choices=['baseline','partial','delayed','blind'],default='baseline')
    parser.add_argument('--planner',choices=['hybrid','lattice'],default='hybrid')
    parser.add_argument('--duration',type=float,default=120.)
    args = parser.parse_args()
    kinds = ['corridor','weave','blocked'] if args.kind=='all' else [args.kind]
    cases = [(k,m,s,f,False,args.signs,args.profile,args.planner,args.duration)
             for k in kinds for m in (False,True) for s in range(args.seeds) for f in (False,True)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker,cases))
    report = dict(cases=len(results),passed=sum(r['ok'] for r in results),results=results)
    if args.output:
        trace_path = Path(args.output).with_suffix('.traces.jsonl')
        with trace_path.open('w') as stream:
            for result in results:
                trace = result.pop('trace')
                case = {k:result[k] for k in ('kind','mirror','seed','faults','signs','profile','planner')}
                stream.write(json.dumps(dict(case,trace=trace))+'\n')
        report['trace_file'] = trace_path.name
        with open(args.output,'w') as stream:
            json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed']==report['cases'] else 1)
