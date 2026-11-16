"""Reverse only to obtain a checked forward departure, with bounded candidates."""
from copy import copy
from dataclasses import replace
import math
import numpy as np

from . import three_lane_simple as S
from .m7_planning import integrate


def forward_room(core, world_pose, now):
    """Cheap lane escape probe. Does not change the live planner or map."""
    from .m7_navigation import FootprintPlanner
    probe=copy(core)
    probe.pose=tuple(world_pose)
    road=core.planner.road_path(np.asarray([world_pose]),core)[0]
    probe.s=road[0]+S.CAM_FROM_ODOM*math.cos(road[2])
    probe.c=road[1]+S.CAM_FROM_ODOM*math.sin(road[2])
    probe.th=-road[2]
    cfg=replace(core.config,horizon=1.2)
    checker=FootprintPlanner(cfg)
    # Forward departure follows the stop/re-observation phase, which commands
    # centred steering. Do not carry the final reverse lock into that rollout.
    checker.previous_steer=0.
    checker.merge_distance=core.island_at+S.MERGE_TURN-probe.s
    best=0.
    targets=[(j,0.) for j in range(3)]
    if core.goal is None:targets += [(1,-.18),(1,.18)]
    for lane,offset in targets:
        path,_=checker.rollout(probe.s,probe.c,probe.th,lane,read_offset=offset)
        certified,_,_,blocked,_=checker.assess(path,probe,now)
        if not blocked.any() and certified>=cfg.review_departure_m:
            best=max(best,certified)
    return best


def reverse_for_departure(core, now):
    """Choose the shortest useful observed reverse; preserve unknown rear space."""
    cfg=core.config
    history=core.history.reverse_path(core.pose,now,cfg.reverse_max_distance,cfg.step)
    options=[]
    if len(history)>1:
        options.append(('executed_history',history,cfg.history_age))
    start=core.planner.start_pose(core)
    for steer in (0.,-cfg.max_steer,cfg.max_steer):
        road,_=integrate(start,core.planner.previous_steer,steer,
                         -cfg.reverse_max_distance,min(cfg.step,.004),cfg.reverse_speed,
                         steering_rate=1.4)
        xy,angles=core.planner.world_path(road,core)
        options.append((f'observed_reverse_{math.degrees(steer):+.0f}',
                        np.column_stack((xy,angles)),cfg.history_age))
    usable=[]
    for source,world,age in options:
        road=core.planner.road_path(world,core)
        room=core.planner.assess(road,core,now,reverse=True,max_age=age)[0]-cfg.stop_buffer
        # The route must contain room to brake, not just a reachable endpoint.
        usable.append((source,world,age,max(0.,room)))
    for distance in (.05,.10,.15,.25,.40,.60):
        if distance>cfg.reverse_max_distance:
            break
        candidates=[]
        for source,world,age,room in usable:
            if room<distance or len(world)<2:
                continue
            arcs=np.cumsum(np.linalg.norm(np.diff(np.vstack((core.pose[:2],world[:,:2])),axis=0),axis=1))
            at=int(np.searchsorted(arcs,distance-1e-9))
            if at>=len(world):
                continue
            forward=forward_room(core,world[at],now)
            if forward>0:
                candidates.append((forward,source,world,age,float(arcs[at])))
        if candidates:
            best=max(candidates,key=lambda c:c[0])
            return dict(source=best[1],path=best[2],age=best[3],distance=best[4],forward_m=best[0])
    return None
