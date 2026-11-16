"""Fixed-map, passable-gap, sign-viewpoint and purposeful-reverse regressions."""
from dataclasses import replace
import math
import numpy as np
import pytest

from jetracer_ros2.m7_navigation import MappedThreeLane, MappedSupervisor, NavigationConfig
from jetracer_ros2.m7_signs import SignEvidence
from jetracer_ros2.m7_recovery import reverse_for_departure
from jetracer_ros2.m7_recovery import forward_room
from jetracer_ros2.m7_planning import integrate,curvature,curve_steering
from jetracer_ros2.m7_runtime import SafetyGate
from test_m7_navigation import core, white_observation, identified_scan
from test_m7_fusion import gate_inputs
from sim_three_lane_simple import rect, raycast


def full_scan(pose=(0.,0.,0.),ident=1):
    return (np.full(360,np.inf),-math.pi,2*math.pi/360,.02,pose,ident,8.)


def test_camera_correction_moves_vehicle_not_road_or_obstacles():
    k=core();k.visibility.observe_depth(np.array([[1.,.4]]),1,0.)
    points=k.visibility.points.copy();anchor=k.road_origin.copy();yaw=k.road_yaw
    path=np.array([[1.5,.1,0.],[1.6,.1,0.]])
    before=k.planner.world_path(path,k)[0]
    k.correct_road_pose(.10,.05)
    assert np.array_equal(k.visibility.points,points)
    assert np.array_equal(k.road_origin,anchor) and k.road_yaw==yaw
    assert np.allclose(k.planner.world_path(path,k)[0],before)
    assert not np.allclose(k.pose,k.odom_pose)
    world_pose=k.map_pose((.1,0.,0.))
    k._advance((.1,0.,0.))
    assert np.allclose(k.pose,world_pose)
    road=k.planner.road_path(np.array([k.pose]),k)[0]
    assert road[1]+.184*math.sin(road[2])==pytest.approx(k.c)


@pytest.mark.parametrize('gap,blocked',[(.31,False),(.27,True)])
def test_passable_gap_does_not_get_an_extra_whole_grid_cell_margin(gap,blocked):
    k=core()
    points=np.array([(x,y) for x in np.arange(.4,1.5,.02) for y in (-gap/2,gap/2)])
    k.visibility.observe_depth(points,1,0.)
    p=np.array([[1.616,0.,0.]])  # rear axle at map x=.8
    assert bool(k.planner.collision_mask(p,k)[0])==blocked


def test_half_metre_group_keeps_its_internal_gap_open():
    k=core();points=np.array([(x,y) for x in (.5,.53,.56) for y in (-.20,.20)])
    k.visibility.observe_depth(points,1,0.)
    tracks=k.visibility.tracks;tracks.update(k.visibility.points,k.visibility.revision,0.)
    groups=tracks.groups(.5)
    assert len(groups)==1 and len(groups[0]['surface_ids'])==2
    assert len(groups[0]['points'])==len(k.visibility.points)
    assert not k.planner.collision_mask(np.array([[1.30,0.,0.]]),k)[0]


def test_yellow_boundary_checks_vehicle_corner_not_just_centre():
    k=core()
    assert k.planner.collision_mask(np.array([[1.2,.64,math.radians(30)]]),k)[0]
    assert not k.planner.collision_mask(np.array([[1.2,.50,0.]]),k)[0]


def test_unread_sign_forbids_both_fork_exits():
    k=core();k.require_sign=True
    for y in (-.535,.535):
        assert k.planner.collision_mask(np.array([[k.island_at-.6,y,0.]]),k)[0]
    k.goal=0
    assert not k.planner.collision_mask(np.array([[k.island_at-.6,.535,0.]]),k)[0]


def test_distant_blockage_preserves_a_safe_forward_prefix(monkeypatch):
    k=core();pose=k.odom_pose
    wall=rect(1.7,0.,0.,.10,1.6)
    k.visibility.observe(identified_scan(pose,[wall],1),pose,0.)
    monkeypatch.setattr(k.planner,'_search',lambda *a:None)
    k.planner._make_route(k,0.)
    assert k.planner.source=='safe_prefix'
    assert k.planner.selected is None  # route build does not issue a command
    remaining,_=k.planner.remaining(k)
    assert len(remaining)>4 and not k.planner.collision_mask(remaining,k).any()


def test_repeated_review_does_not_restart_physical_rest_timer():
    k=core();k.begin_review('first',0.)
    k.review_stationary(k.pose,0.)
    anchor=k._rest_anchor
    for i in range(1,8):
        k.begin_review('another observation',i*.1)
        k.visibility.sequence+=1
        ready=k.review_stationary(k.pose,i*.1)
    assert ready and k._rest_anchor==anchor and k.stop_since==0.


def detections(label='b',height=45,confidence=.9):
    return [dict(label=label,confidence=confidence,bbox=[0,0,40,height])]


@pytest.mark.parametrize('c,heading,height,conf,age',[
    (.4,0.,45,.9,0.),(0.,.4,45,.9,0.),(0.,0.,20,.9,0.),
    (0.,0.,45,.3,0.),(0.,0.,45,.9,.5)])
def test_unreliable_sign_does_not_choose_a_fork(c,heading,height,conf,age):
    gate=SignEvidence();cfg=NavigationConfig()
    for i in range(8):gate.observe(detections(height=height,confidence=conf),10.+i*.1,10.+i*.1+age,c,heading,cfg)
    assert gate.goal is None


def test_sign_requires_distinct_recent_frames_and_latches_the_verified_goal():
    gate=SignEvidence();cfg=NavigationConfig()
    for _ in range(10):gate.observe(detections(),10.,10.,0.,0.,cfg)
    assert gate.goal is None
    for i in range(1,4):gate.observe(detections(),10.+i*.1,10.+i*.1,0.,0.,cfg)
    assert gate.goal==2
    for i in range(4,10):gate.observe(detections('a'),10.+i*.1,10.+i*.1,0.,0.,cfg)
    assert gate.goal==2


def test_reverse_selects_short_turn_only_when_it_opens_a_forward_route(monkeypatch):
    k=core();k.visibility.observe(full_scan(),k.pose,0.);k.visibility.mark_footprint(k.pose,0.)
    monkeypatch.setattr('jetracer_ros2.m7_recovery.forward_room',
                        lambda core,p,now:1. if abs(p[2])>.03 else 0.)
    choice=reverse_for_departure(k,0.)
    assert choice and choice['distance']<=.075
    assert choice['source']!='observed_reverse_+0'
    monkeypatch.setattr('jetracer_ros2.m7_recovery.forward_room',lambda *a:0.)
    assert reverse_for_departure(k,0.) is None


def test_reverse_never_uses_unseen_rear_space(monkeypatch):
    k=core();k.visibility.mark_footprint(k.pose,0.)
    monkeypatch.setattr('jetracer_ros2.m7_recovery.forward_room',lambda *a:1.)
    assert reverse_for_departure(k,0.) is None


def test_reverse_ends_when_forward_route_opens(monkeypatch):
    k=core();k.reverse_world=np.array([[-.1,0.,0.],[-.2,0.,0.]])
    k.reverse_left=.5;k.reverse_since=0.;k._reverse_probe_time=0.
    monkeypatch.setattr('jetracer_ros2.m7_recovery.forward_room',lambda *a:1.)
    assert k.reverse_command(.11) is None
    assert 'end reverse now' in k.recovery_status


@pytest.mark.parametrize('side',[-1.,1.])
def test_short_reverse_turn_opens_opposite_forward_direction_in_real_geometry(side):
    k=core();k.visibility.observe(full_scan(),k.pose,0.)
    points=np.array([(x,side*y) for x in np.arange(.56,.881,.02)
                     for y in np.arange(-.125,.426,.02)])
    k.visibility.observe_depth(points,1,0.);k.visibility.mark_footprint(k.pose,0.)
    assert forward_room(k,k.pose,0.)==0.
    assert k.prepare_reverse(0.)
    assert k.reverse_left<=.15
    x=y=psi=steer=distance=0.
    for i in range(100):
        now=i*.05
        k.visibility.observe(full_scan((x,y,psi),i+2),(x,y,psi),now)
        command=k.reverse_command(now)
        if command is None:
            break
        assert command['v']<0.
        steer+=(command['steer']-steer)*(.05/.15)
        ds=command['v']*.05;dpsi=ds*curvature(steer)
        x+=ds*math.cos(psi+dpsi/2);y+=ds*math.sin(psi+dpsi/2);psi+=dpsi
        distance+=abs(ds);k._advance((x,y,psi))
        assert not k.planner.collision_mask(np.asarray([k.planner.start_pose(k)]),k)[0]
    assert distance<.15 and forward_room(k,k.pose,now)>0
    # Occupancy is on +side: rear goes +side, nose faces -side (the free lane).
    assert y*side>0 and psi*side<0
    assert abs(steer)>.9*k.config.max_steer
    assert 'end reverse now' in k.recovery_status


def test_retained_mapped_rear_space_can_support_a_turn_but_new_hit_overrides_it():
    k=core();k.visibility.observe(full_scan(),k.pose,0.)
    points=np.array([(x,y) for x in np.arange(.56,.881,.02) for y in np.arange(-.125,.426,.02)])
    k.visibility.observe_depth(points,1,0.)
    assert reverse_for_departure(k,2.) is not None
    # A newly observed wall behind the car invalidates retained free evidence.
    k.visibility.observe_depth(np.array([(-.14,y) for y in np.arange(-.8,.81,.01)]),2,2.)
    assert reverse_for_departure(k,2.) is None


def test_generated_curve_retains_its_ackermann_steering():
    angle=math.radians(20.)
    path,_=integrate((0.,0.,0.),angle,angle,1.,.01)
    profile=curve_steering(path)
    assert np.allclose(profile,angle,atol=1e-4)
    radius=1./curvature(angle)
    assert np.allclose(path[:,0]**2+(path[:,1]-radius)**2,radius**2,atol=1e-4)


@pytest.mark.parametrize('metadata',[None,'480',float('nan'),True])
def test_invalid_sign_metadata_is_rejected_without_interrupting_control(metadata):
    gate=SignEvidence()
    assert gate.observe(detections(),1.,1.,0.,0.,NavigationConfig(),metadata) is None
    assert gate.goal is None and 'invalid' in gate.reason


def test_speed_ramps_and_never_exceeds_point_four():
    k=core();pose=k.odom_pose;speeds=[]
    for i in range(45):
        out=k.step_obs(white_observation(),pose,identified_scan(pose,[],i),now=i*.05)
        speeds.append(out['v'])
    assert 0<speeds[0]<=.01251 and max(speeds)<=.4
    assert speeds[-1]>.3
    assert max(np.diff(speeds))<=.01251


def test_gate_brakes_for_actual_speed_even_after_command_was_reduced():
    data=gate_inputs();data['poses']=tuple((t,.4*(t-10.),y,yaw) for t,x,y,yaw in data['poses'])
    data['scan']['data']['ranges']=raycast(.184,0.,0.,[rect(.7,0.,0.,.12,.55)])[0]
    reason=SafetyGate(NavigationConfig()).check(dict(v=.05,steer=0.,active=True,started=100.),data,10.,100.)
    assert 'braking' in reason
