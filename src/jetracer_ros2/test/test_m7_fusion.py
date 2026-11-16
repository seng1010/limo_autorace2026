"""Timing, geometry, execution and optional-depth regressions for M7."""
from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from jetracer_ros2.m7_sensors import PoseHistory, RoadFilter, DepthProjector, fresh, move_observation, road_hypotheses
from jetracer_ros2.m7_navigation import NavigationConfig, VisibilityMap, MappedThreeLane
from jetracer_ros2.m7_planning import TravelHistory, LatticePrimitives, curvature
from jetracer_ros2.m7_grid import FreeGrid
from jetracer_ros2.m7_runtime import SafetyGate, LatestInputs, M7Worker, load_config

sys.path.insert(0,str(Path(__file__).parent))
from test_m7_navigation import white_observation, identified_scan, core
from sim_three_lane_simple import raycast, rect, overlap


def test_joint_colour_spacing_overrides_wrong_central_lane_prior():
    obs = dict(th=math.radians(-10.65),n=802,
               lines=[(-.3096921168,65,0.),(.2792347168,195,1.)])
    h = road_hypotheses(obs,prior=.055)[0]
    assert h.support==2 and h.confidence>.6 and .50<h.c<.59


def test_road_filter_uses_elapsed_time_not_image_count():
    results = []
    for hz in (10,30):
        filt = RoadFilter(); c=.05; heading=.02
        for t in np.arange(0.,1.01,1./hz):
            c,heading,accepted,_ = filt.update(white_observation(),c,heading,float(t))
            assert accepted
        results.append((c,heading))
    assert np.allclose(results[0],results[1],atol=1e-4)


def test_single_line_cannot_reanchor_to_adjacent_lane():
    f=RoadFilter()
    obs=dict(th=0.,n=500,lines=[(.25,150,1.)])
    c,heading,accepted,large=f.update(obs,0.,0.,0.)
    assert c==0. and not accepted


def test_missing_image_ticks_do_not_consume_camera_timestamp():
    f=RoadFilter()
    f.update(white_observation(),0.,0.,1.)
    f.update(None,0.,0.,1.10)
    assert f.update(white_observation(),0.,0.,1.08)[2]
    assert not f.update(white_observation(),0.,0.,1.08)[2]


def test_large_joint_correction_stops_before_changing_pose():
    k=core(); pose=k.pose
    obs=dict(th=0.,n=500,lines=[(-.265,100,0.),(.265,180,1.)])
    first=k.step_obs(obs,pose,identified_scan(pose,[],1),now=0.)
    assert first['v']==0 and k.c==0.
    for i in range(1,12):
        k.step_obs(obs,pose,identified_scan(pose,[],i+1),now=i*.1)
    assert .49<k.c<.59


def test_localise_accumulates_rest_before_large_camera_correction():
    k=core();pose=k.pose;k.blind_travel=3.
    obs=dict(th=0.,n=500,lines=[(-.265,100,0.),(.265,180,1.)])
    first=k.step_obs(obs,pose,identified_scan(pose,[],1),now=0.)
    assert first['state']=='LOCALISE' and first['v']==0 and k.c==0.
    for i in range(1,12):
        k.step_obs(obs,pose,identified_scan(pose,[],i+1),now=i*.1)
    assert .49<k.c<.59 and k.blind_travel==0.


def test_pose_history_does_not_use_latest_pose_for_old_image():
    h=PoseHistory();h.add(10.,(0,0,math.radians(179)));h.add(10.05,(.1,0,math.radians(-179)))
    assert h.at(9.9) is None and h.at(10.1) is None
    assert h.at(10.025)[0]==pytest.approx(.05)
    assert h.at(10.025)[2]==pytest.approx(math.pi)
    assert not h.add(10.01,(10,0,0))


def test_delayed_callback_does_not_refresh_acquisition_age():
    assert not fresh(9.,10.,100.,100.,.35)
    assert fresh(9.9,10.,99.9,100.,.35)
    assert not fresh(10.1,10.,99.9,100.,.35)


def test_per_ray_pose_requires_valid_time_span_and_history():
    h=PoseHistory()
    for t in np.arange(10.,10.201,.01):h.add(float(t),(float(t-10),0,0))
    p=h.rays(10.02,.01,8,.10)
    assert p.shape==(8,3) and p[-1,0]-p[0,0]==pytest.approx(.07)
    assert h.rays(10.02,.03,8,.10) is None
    assert h.rays(9.,.01,8,.10) is None


def test_observation_moves_with_vehicle_without_changing_world_road():
    obs=white_observation(c=0.,th=0.)
    moved=move_observation(obs,(0,0,0),(.1,.05,.2))
    h=road_hypotheses(moved)[0]
    assert h.c==pytest.approx(.05+.184*math.sin(.2))
    assert h.heading==pytest.approx(-.2)


def test_grid_recentering_keeps_evidence_without_wraparound():
    g=FreeGrid(.02,half_size=1.)
    g[(20,0)]=1.;g[(-40,0)]=1.
    g.recenter((45,0))
    assert g.get((20,0))==1.
    assert not math.isfinite(g.get((60,0)))
    assert not math.isfinite(g.get((-40,0)))


def test_raster_wedges_do_not_fill_a_missing_angular_sector():
    m=VisibilityMap();scan=list(raycast(.184,0,0,[]))
    angles=scan[1]+np.arange(len(scan[0]))*scan[2]
    scan[0][np.abs(angles)<.2]=0.
    assert m.observe(tuple(scan),(0,0,0),0.)
    assert not m.known(np.array([[1.,0.]]),0.)[0]
    assert m.known(np.array([[.6,.4]]),0.)[0]


def test_depth_requires_proper_extrinsic_and_rejects_floor_invalid_values():
    with pytest.raises(ValueError):DepthProjector(np.zeros((4,4)))
    # Optical z -> base forward, optical x -> base right, optical y -> down.
    transform=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,.3],[0,0,0,1.]])
    projector=DepthProjector(transform,stride=1)
    depth=np.array([[1.,0.,np.nan],[1.,1.,1.]])
    k=np.array([[10,0,1],[0,2,0],[0,0,1.]])
    points=projector.project(depth,k,(2.,3.,0.))
    assert points.shape==(1,2) and np.allclose(points[0],[3.,3.1])


def test_depth_occupancy_is_not_cleared_by_a_low_lidar_ray():
    m=VisibilityMap();point=np.array([[.8,.1]])
    m.observe_depth(point,1,0.)
    for i in range(3):m.observe(raycast(.184,0,0,[])+((0,0,0),i),(0,0,0),i*.1)
    assert m.depth_hits and not m.known(point,.2)[0]
    assert not m.observe_depth(point,1,.3)


def test_raw_occupied_bounds_are_not_replaced_by_storage_cell_centres():
    m=VisibilityMap();points=np.array([[.801,.101],[.806,.108]])
    m.observe_depth(points,1,0.)
    assert np.allclose(m.points[0],points.mean(axis=0))
    assert m.point_padding[0]==pytest.approx(np.linalg.norm(points[1]-points[0])/2)
    assert m.observe_depth(points,2,.1) and len(m.added)==0
    assert not m.known(points,.1).any()


def test_new_depth_obstacle_stops_before_replanning(monkeypatch):
    k=core();pose=k.pose
    assert k.step_obs(white_observation(),pose,identified_scan(pose,[],1),now=0.)['v']>0
    k.pending_depth=(np.array([[.85,.5]]),11)
    monkeypatch.setattr(k.planner,'plan',lambda *a:pytest.fail('must stop before expensive search'))
    out=k.step_obs(white_observation(),pose,identified_scan(pose,[],2),now=.1)
    assert out['v']==0 and k.visibility.depth_hits


def test_reuses_route_while_new_clear_scans_arrive():
    k=core();pose=k.pose
    for i in range(20):
        assert k.step_obs(white_observation(),pose,identified_scan(pose,[],i),now=i*.05)['v']>0
    assert k.planner.plan_count==1


@pytest.mark.parametrize('planner',['hybrid','lattice'])
def test_search_finds_off_centre_corridor_missing_from_lane_seeds(planner,monkeypatch):
    k=MappedThreeLane(replace(NavigationConfig(),planner=planner,search_budget=.5,search_nodes=3000))
    k.start(0.,0.,(0,0,0),.4)
    objects=[rect(1.7,.65,0.,1.5,.55),rect(1.7,-.38,0.,1.5,.55)]
    # Exercise search itself even though the new centre-offset seeds fit here.
    monkeypatch.setattr(k.planner.checker,'seed_paths',lambda *a:[])
    out=k.step_obs(white_observation(),k.pose,identified_scan(k.pose,objects,1),now=0.)
    assert k.planner.source==planner
    assert out['v']>0
    world,angles=k.planner.world_path(k.planner.selected['path'],k)
    for p,yaw in zip(world,angles):
        body=rect(p[0]+.1*math.cos(yaw),p[1]+.1*math.sin(yaw),yaw,.32,.22)
        assert not any(overlap(body,o) for o in objects)


def test_lattice_has_exact_endpoints_and_tangent_directions():
    library=LatticePrimitives(curvature(math.radians(24)))
    assert all(library.by_heading[h] for h in range(library.headings))
    assert any(nh!=0 for _,nh in library.by_heading[0])
    for h,items in library.by_heading.items():
        for path,nh in items:
            assert np.allclose(path[-1,:2]/library.resolution,np.round(path[-1,:2]/library.resolution))
            assert math.sin(path[-1,2]-nh*library.angle)==pytest.approx(0.,abs=1e-8)


def test_history_reverses_a_curve_and_expires_without_refreshing_at_rest():
    history=TravelHistory()
    for i in range(41):
        angle=i*.015
        history.observe((math.sin(angle),1-math.cos(angle),angle),i*.1)
    pose=(math.sin(.6),1-math.cos(.6),.6)
    path=history.reverse_path(pose,5.,.5,.025)
    assert len(path)>=19 and path[-1,2]<path[0,2]-.3
    assert np.ptp(path[:,1])>.08
    history.observe(pose,30.)
    assert not len(history.reverse_path(pose,30.,.5,.025))


def gate_inputs(now=100.):
    values=LatestInputs()
    for t in (9.85,9.9,9.95,10.):values.odom((0.,0.,0.),t,now-(10-t))
    scan=raycast(.184,0,0,[])
    values.put('image',b'x',9.99,now)
    values.put('scan',dict(ranges=scan[0],angle_min=scan[1],angle_increment=scan[2],range_min=scan[3],range_max=8.),9.9,now)
    return values.snapshot()


def test_independent_gate_stops_hung_worker_and_old_sensor_headers():
    gate=SafetyGate(NavigationConfig());data=gate_inputs()
    result=dict(v=.1,steer=0.,active=True,started=100.)
    assert not gate.check(result,data,10.,100.)
    assert 'deadline' in gate.check(dict(result,started=99.7),data,10.,100.)
    data['scan']=dict(data['scan'],stamp=9.)
    assert 'stale' in gate.check(result,data,10.,100.)


def test_independent_gate_uses_new_raw_hit_without_waiting_for_worker():
    gate=SafetyGate(NavigationConfig());data=gate_inputs()
    result=dict(v=.18,steer=0.,active=True,started=100.)
    scan=raycast(.184,0,0,[rect(.55,0.,0.,.32,.55)])
    data['scan']['data']['ranges']=scan[0]
    assert 'braking' in gate.check(result,data,10.,100.)


def test_optional_depth_and_deskew_fail_closed_without_calibration(tmp_path):
    path=tmp_path/'config.json'
    path.write_text('{"depth":{"enabled":true}}')
    with pytest.raises(ValueError):load_config(path)
    path.write_text('{"deskew":true}')
    with pytest.raises(ValueError):load_config(path)
    assert not load_config()[1].get('depth',{}).get('enabled',False)


def test_worker_runs_m7_without_duplicate_base_lane_processing(monkeypatch):
    from types import SimpleNamespace
    class Base:
        def process(self,*a,**k):pytest.fail('base lane engine must not run while M7 owns control')
    values=LatestInputs()
    worker=M7Worker(values,NavigationConfig(),load_config()[1],{},SimpleNamespace(LaneEngine=Base),True,None)
    worker.sup.mode='on';worker.sup.core.start(0.,0.,(0,0,0),1.)
    monkeypatch.setattr(worker,'_image',lambda *a:np.zeros((480,640,3),np.uint8))
    monkeypatch.setattr(worker.sup.fitter,'fit',lambda *a:white_observation())
    data=gate_inputs()
    first=worker.tick(data,10.,100.)
    assert first['v']>0 and first['frame']==1
    second=worker.tick(data,10.05,100.05)
    assert second['v']>0 and worker.sup.core.planner.plan_count==1
    # Stale headers stop, and the loss breaks stop/reverse confirmation.
    worker.sup.core.reversing=True
    stale=worker.tick(data,10.5,100.5)
    assert stale['v']==0 and not worker.sup.core.reversing
    assert worker.sup.core.review_active


def test_hold_and_resume_require_new_rest_confirmation(monkeypatch):
    from types import SimpleNamespace
    values=LatestInputs()
    worker=M7Worker(values,NavigationConfig(),load_config()[1],{},SimpleNamespace(LaneEngine=lambda:None),True,None)
    worker.sup.mode='on';worker.sup.core.start(0.,0.,(0,0,0),1.)
    monkeypatch.setattr(worker,'_image',lambda *a:np.zeros((480,640,3),np.uint8))
    monkeypatch.setattr(worker.sup.fitter,'fit',lambda *a:white_observation())
    data=gate_inputs();data['events']=((1,'hold',None),)
    assert worker.tick(data,10.,100.)['state']=='HOLD'
    data['events']=((1,'hold',None),(2,'resume',None))
    result=worker.tick(data,10.01,100.01)
    assert result['v']==0 and worker.sup.core.review_active


def test_clear_reference_with_unexecutable_command_does_not_restart_stop_timer(monkeypatch):
    k=core();pose=k.pose
    k.step_obs(white_observation(),pose,identified_scan(pose,[],1),now=0.)
    best=dict(k.planner.selected)
    k.begin_review('tracker braking arc blocked',.1)
    monkeypatch.setattr(k.planner,'plan',lambda *a:(0.,0.,best))
    for i in range(2,22):
        out=k.step_obs(white_observation(),pose,identified_scan(pose,[],i),now=i*.1)
        assert out['v']==0 and k.review_active
    assert k.stop_since==pytest.approx(.1)
