"""Reproducible desktop comparison; not a LIMO CPU or success-rate benchmark."""
import argparse
from dataclasses import replace
import json
import time

import numpy as np

from jetracer_ros2.m7_navigation import MappedThreeLane, NavigationConfig
from test_m7_navigation import identified_scan, white_observation
from sim_three_lane_simple import rect, overlap


def benchmark(repeats=5):
    results=[]
    # The free corridor is offset from all three lane centres, forcing actual
    # search instead of comparing two identical successful warm starts.
    objects=[rect(1.7,.65,0.,1.5,.55),rect(1.7,-.38,0.,1.5,.55)]
    for name in ('hybrid','lattice'):
        for repeat in range(repeats):
            before=time.perf_counter()
            k=MappedThreeLane(replace(NavigationConfig(),planner=name))
            init_ms=(time.perf_counter()-before)*1000
            k.start(0.,0.,(0,0,0),.4)
            before=time.perf_counter()
            out=k.step_obs(white_observation(),k.pose,identified_scan(k.pose,objects,1),now=0.)
            elapsed=(time.perf_counter()-before)*1000
            path=k.planner.selected['path']
            world,angles=k.planner.world_path(path,k)
            clear=all(not any(overlap(rect(p[0]+.1*np.cos(a),p[1]+.1*np.sin(a),a,.32,.22),o)
                              for o in objects) for p,a in zip(world,angles))
            results.append(dict(planner=name,repeat=repeat,source=k.planner.source,
                init_ms=init_ms,tick_ms=elapsed,plan_ms=k.planner.search_ms,
                expanded=k.planner.expanded,progress_m=float(path[-1,0]-k.planner.start_pose(k)[0]) if len(path) else 0.,
                certified_m=float(k.planner.selected['certified']),moving=bool(out['v']>0),
                independent_collision_free=bool(len(path) and clear)))
    return dict(method='desktop, default 80ms search budget; one off-centre corridor; independent polygon check',
                repeats=repeats,hardware_cpu_percent=None,results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    report=benchmark(args.repeats)
    with open(args.output,'w') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if all(r['independent_collision_free'] and r['moving'] for r in report['results']) else 1)
