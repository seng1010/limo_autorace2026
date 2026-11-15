"""M7 safety and visibility regressions; no ROS or vehicle required."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from jetracer_ros2 import three_lane as T
from jetracer_ros2 import three_lane_simple as S
from jetracer_ros2.m7_navigation import MappedSupervisor, MappedThreeLane, VisibilityMap, central_white_pair

sys.path.insert(0, str(Path(__file__).parent))
from sim_three_lane_simple import raycast, rect


def white_observation(c=0.0, th=0.0):
    def bev(u):
        return u if abs(u)<=T.FAR_U else math.copysign(T.FAR_U+(abs(u)-T.FAR_U)*T.FAR_SHRINK,u)
    return dict(th=th, n=1000, lines=[(bev(-0.27-c), 120, 0.0), (bev(0.27-c), 120, 0.0)],
                short=set(), ystart={})


def core():
    k = MappedThreeLane()
    k.start(0.0, 0.0, (0.0, 0.0, 0.0), 1.0)
    return k


def test_occluded_lane_is_unknown_not_free():
    m = VisibilityMap()
    pose = (0.0, 0.0, 0.0)
    front = rect(1.134, 0.0, 0.0, 0.32, 0.55)
    hidden = rect(2.584, 0.535, 0.0, 0.32, 0.55)
    a = raycast(S.CAM_FROM_ODOM, 0.0, 0.0, [front])
    b = raycast(S.CAM_FROM_ODOM, 0.0, 0.0, [front, hidden])
    assert np.array_equal(a[0], b[0])
    assert m.observe(b, pose, 0.0)
    assert not m.known(np.array([[2.5, 0.535]]), 0.0)[0]
    assert m.known(np.array([[0.6, 0.0]]), 0.0)[0]


def test_empty_and_invalid_scans_do_not_mean_free_space():
    k = core()
    for ranges in ([], [float('nan')] * 250, [0.0] * 250):
        out = k.step_obs(white_observation(), k.pose, (ranges, -1.92, 3.84/249, .15), now=0.0)
        assert out['v'] == 0.0 and out['state'] == 'NO_SCAN'


def test_stale_lidar_or_odometry_never_completes_m7():
    sup = MappedSupervisor(forced=True)
    sup.mode = 'on'
    for i in range(100):
        out = sup.step_obs(None, None, None, False, now=i*.1)
        assert out is not None and out['v'] == 0.0
    assert sup.mode == 'on'
    out = sup.step_obs(white_observation(), (0,0,0), None, False, now=10.)
    assert out['v'] == 0.0 and sup.mode == 'on'


def test_legacy_localisation_loss_keeps_m7_guard(monkeypatch):
    sup = T.Lane3Supervisor()
    sup.set_mission('M7')
    sup.mode = 'on'
    monkeypatch.setattr(sup.tl, 'step', lambda *a, **k: T.ThreeLaneOutput(ok=False, front=.2))
    for _ in range(60):
        out = sup.step(None, (0,0,0), None, False)
        assert out is not None and out['v'] == 0.0
    sup.set_mission('M7')
    assert sup.mode == 'on'


def test_rear_obstacle_caps_legacy_recovery_distance():
    k = T.ThreeLane()
    k.start(0,0,(0,0,0))
    k.lane_run = 1.0
    k.stuck = T.STUCK_FRAMES - 1
    k._last_target = 1
    k._points = (np.array([-.4]), np.array([0.]), 0., 0.)
    out = T.ThreeLaneOutput()
    k._recover(T.Plan(1,0.,'STOP','blocked'), out, [.3]*3)
    # The remembered obstacle leaves less than the minimum reverse budget.
    assert out.speed == 0.0 and not k.reversing


def test_simultaneous_turn_and_translation_uses_midpoint_heading():
    k = S.SimpleThreeLane()
    k.start(0,0,(0,0,0),1.)
    angle, distance = .2, .1
    k._advance((distance*math.cos(angle/2), distance*math.sin(angle/2), angle))
    expected = distance*math.sin(angle/2) + S.CAM_FROM_ODOM*math.sin(angle)
    assert k.c == pytest.approx(expected)


def test_repeated_scan_cannot_clear_an_obstacle_twice():
    m = VisibilityMap()
    pose = (0,0,0)
    blocked = raycast(.184,0,0,[rect(1.,0.,0.,.32,.55)])
    clear = raycast(.184,0,0,[])
    m.observe(blocked + (pose, 1), pose, 0.)
    hits = set(m.hits)
    assert hits
    m.observe(clear + (pose, 2), pose, .1)
    m.observe(clear + (pose, 2), pose, .2)
    assert hits & set(m.hits)
    m.observe(clear + (pose, 3), pose, .3)
    assert len(hits & set(m.hits)) < len(hits)


def test_odometry_map_does_not_move_when_camera_corrects_road_position():
    k = core()
    scan = raycast(.184,0,0,[rect(1.,0.,0.,.32,.55)])
    k.visibility.observe(scan, k.pose, 0.)
    before = k.visibility.points.copy()
    k.c = -.15
    for _ in range(20):
        k._camera_lines(white_observation())
    assert abs(k.c) < .01
    assert np.array_equal(before, k.visibility.points)


def test_no_reverse_into_unobserved_space():
    k = core()
    # All lanes blocked near the front; no prior trajectory exists behind us.
    scan = raycast(.184,0,0,[rect(.65,y,0,.32,.55) for y in T.LANE_CENTER])
    for t in (0., 2., 5., 10.):
        out = k.step_obs(white_observation(), k.pose, scan, now=t)
        assert out['v'] >= 0.0


def test_manager_missions_do_not_activate_mapped_controller():
    sup = MappedSupervisor()
    for mission in ('M5','M6','M3','M8','M1','M4'):
        sup.set_mission(mission)
        assert sup.mode == 'off'
        assert sup.step(None, None, None, False) is None
    sup._entry_seen,sup._entry_count = (0.,0.,(0,0,0)),5
    sup.set_mission('M7')
    assert sup.mode=='armed' and sup._entry_seen is None and sup._entry_count==0


def test_invalid_heading_does_not_refresh_lane_confidence():
    k = core()
    k.blind_travel = .9
    k.step_obs(white_observation(th=math.radians(60)),k.pose,raycast(.184,0,0,[]),now=0.)
    assert k.blind_travel == .9
    assert not k.diagnostics['line_accepted']


def test_coasting_backwards_is_not_counted_as_forward_after_stop():
    k = core()
    k.reversing = False
    before = k.s
    k._advance((-.05,0,0))
    assert k.s == pytest.approx(before-.05)


def test_sensor_failure_at_merge_end_does_not_complete_mission():
    sup = MappedSupervisor(forced=True)
    sup.mode = 'on'
    sup.core.start(0,0,(0,0,0),S.START_TO_ISLAND+S.MERGE_END+.3)
    for i in range(20):
        out = sup.step_obs(white_observation(),(0,0,0),None,True,now=i*.1)
        assert out is not None and out['v']==0
    assert sup.mode=='on'


def test_unknown_space_caps_forward_motion_without_fabricating_hits():
    k = core()
    # Side-looking returns are valid; a blind forward cone is not free space.
    scan = list(raycast(.184,0,0,[]))
    angles = scan[1]+np.arange(len(scan[0]))*scan[2]
    scan[0][np.abs(angles)<1.0] = np.nan
    out = k.step_obs(white_observation(),k.pose,tuple(scan),now=0.)
    assert out['v']==0 and not k.visibility.hits


def test_obstacle_diagnostics_are_sorted_and_allow_multiple_lanes():
    k = core()
    scan = raycast(.184,0,0,[rect(1.,.27,0,.32,.55),rect(1.8,-.535,0,.32,.55)])
    k.step_obs(white_observation(),k.pose,scan,now=0.)
    items = k.diagnostics['obstacles']
    assert len(items)>=2
    assert [c['near_m'] for c in items]==sorted(c['near_m'] for c in items)
    assert any(0 in c['lanes'] and 1 in c['lanes'] for c in items)


def test_sign_aliases_are_voted_only_from_central_lane():
    sup = MappedSupervisor(forced=True)
    sup.mode = 'on'
    sup.core.c = .535
    det = [dict(label='sign_a',confidence=.9,bbox=[0,0,40,45])]
    for _ in range(8):
        sup.see_signs(det)
    assert sup.sign_goal is None
    sup.core.c = 0.
    for _ in range(T.SIGN_NEED):
        sup.see_signs(det)
    assert sup.sign_goal==0


def test_open_road_does_not_deadlock_on_current_body_overlap():
    k = core()
    assert k.step_obs(white_observation(),k.pose,raycast(.184,0,0,[]),now=0.)['v']>0


def test_range_max_limits_free_evidence_and_invalid_scan_id_is_not_cached():
    m = VisibilityMap()
    scan = raycast(.184,0,0,[])
    assert m.observe(scan+((0,0,0),1,.7),(0,0,0),0.)
    assert m.known(np.array([[.5,0],[1.2,0]]),0.).tolist()==[True,False]
    for _ in range(2):
        assert not m.observe(scan+((0,0,0),2,0.),(0,0,0),0.)
    invalid = (np.full(250,9.),scan[1],scan[2],scan[3],(0,0,0),3,8.)
    assert not m.observe(invalid,(0,0,0),0.)
    blind = VisibilityMap()
    assert blind.observe((scan[0],scan[1],scan[2],.5),(0,0,0),0.)
    assert not blind.known(np.array([[.4,0]]),0.)[0]


def test_entry_constructs_mapped_core_after_confirmed_pair(monkeypatch):
    sup = MappedSupervisor(forced=True)
    monkeypatch.setattr(sup.entry_fitter,'fit',lambda *a: white_observation())
    monkeypatch.setattr(S,'dash_start_ahead',lambda *a: None)
    for _ in range(sup.entry_frames):
        entered = sup._try_enter(np.zeros((480,640,3),np.uint8),(0,0,0),None)
    assert entered and sup.mode=='on' and isinstance(sup.core,MappedThreeLane)
    assert sup.core.c==0 and sup.core.th==0


def test_handoff_only_after_merge_and_ready_for_normal_lane_control():
    sup = MappedSupervisor(forced=True,goal=0)
    sup.mode = 'on'
    sup.side_taken = 0
    sup.core.start(0,0,(0,0,0),S.START_TO_ISLAND+S.MERGE_END+.3)
    for i in range(S.EXIT_FRAMES):
        out = sup.step_obs(white_observation(),(0,0,0),raycast(.184,0,0,[]),True,now=i*.1)
    assert out is None and sup.mode=='done'
    assert sup.step(None,None,None,False) is None


def test_unread_sign_gets_time_for_multiple_central_lane_votes():
    k = core()
    k.s = S.START_TO_ISLAND-1.7
    out = k.step_obs(white_observation(),k.pose,raycast(.184,0,0,[]),goal=None,now=0.)
    assert 0 < out['v'] <= S.V_READ
    assert k.diagnostics['reading_sign']


def test_island_post_does_not_falsely_occupy_the_whole_exit_lane():
    k = core()
    k.c,k.s = -.535,S.START_TO_ISLAND-.6
    scan = raycast(.184,0,0,[rect(.784,.235,0,.06,.06)])
    out = k.step_obs(None,k.pose,scan,goal=2,now=0.)
    assert out['v']>0 and k.lane==2
    assert k.visibility.hits  # Keep the actual obstacle, check physical clearance.


def test_entry_rejects_floor_reflections_and_unaligned_pose():
    assert central_white_pair(white_observation(),.02) is None
    assert central_white_pair(white_observation(th=math.radians(31)),.95) is None
    assert central_white_pair(white_observation(c=.5),.95) is None
    assert central_white_pair(white_observation(),.95)==(0.,0.)


def test_forced_waiting_for_entry_never_falls_back_to_normal_drive(monkeypatch):
    sup = MappedSupervisor(forced=True)
    monkeypatch.setattr(sup.entry_fitter,'fit',lambda *a: None)
    image = np.zeros((480,640,3),np.uint8)
    for _ in range(10):
        out = sup.step(image,(0,0,0),None,False)
        assert out is not None and out['v']==0 and sup.mode=='armed'
    assert sup.core.diagnostics['state']=='WAIT_ENTRY'


def test_stationary_relocalisation_retains_longitudinal_and_obstacle_map(monkeypatch):
    sup = MappedSupervisor(forced=True)
    sup.mode = 'on'
    sup.core.start(.45,0.,(0,0,0),3.)
    sup.core.state = 'BLOCKED'
    sup.core.visibility.hits[(20,5)] = 0.
    monkeypatch.setattr(sup.entry_fitter,'fit',lambda *a: white_observation(c=.14))
    for t in np.arange(0.,.8,.1):
        sup._try_relocalise(np.zeros((480,640,3),np.uint8),(0,0,0),None,float(t))
    assert sup.core.c==pytest.approx(.14) and sup.core.s==3.
    assert sup.core.visibility.hits=={(20,5):0.}
    assert sup.core.relocalizations==1


def test_large_relocation_cannot_silently_reanchor_road(monkeypatch):
    sup = MappedSupervisor(forced=True)
    sup.core.start(-.96,math.radians(-178),(0,0,0),3.88)
    sup.core.state = 'LOCALISE'
    monkeypatch.setattr(sup.entry_fitter,'fit',lambda *a: white_observation())
    for i in range(12):
        sup._try_relocalise(np.zeros((480,640,3),np.uint8),(0,0,0),None,i*.1)
    assert sup.core.c==-.96 and sup.core.relocalizations==0
    assert 'restart at the entrance' in sup.core.recovery_status


def test_relocalisation_requires_stationary_vehicle(monkeypatch):
    sup = MappedSupervisor(forced=True)
    sup.core.start(.45,0.,(0,0,0),3.)
    sup.core.state = 'BLOCKED'
    monkeypatch.setattr(sup.entry_fitter,'fit',lambda *a: white_observation(c=.14))
    for i in range(12):
        sup._try_relocalise(np.zeros((480,640,3),np.uint8),(i*.03,0,0),None,i*.1)
    assert sup.core.c==.45 and sup.core.relocalizations==0


def test_localise_diagnostics_are_current_and_do_not_repeat_old_path():
    k = core()
    k.blind_travel = 1.3
    k.diagnostics = dict(s=-99,c=99,blind_m=0.,path=[[1,2,3]])
    out = k.step_obs(None,(.05,0,0),raycast(.234,0,0,[]),now=0.)
    assert out['v']==0 and k.diagnostics['state']=='LOCALISE'
    assert k.diagnostics['s']==k.s and k.diagnostics['c']==k.c
    assert k.diagnostics['blind_m']==k.blind_travel and k.blind_travel>1.3
    assert k.diagnostics['path']==[] and k.diagnostics['observed_m'] is None


def test_floor_yellow_outlier_cannot_move_island_to_two_metres(monkeypatch):
    k = core()
    monkeypatch.setattr(T,'island_distance',lambda *a: 1.0)
    for _ in range(6):
        k._camera_island(dict(yellow_xy=np.zeros((3,2))))
    assert k.island_s is None and k.island_at==S.START_TO_ISLAND


def test_stopped_recovery_can_use_two_weak_but_consistent_dashes(monkeypatch):
    sup = MappedSupervisor(forced=True)
    sup.core.start(.45,0.,(0,0,0),3.)
    sup.core.state = 'BLOCKED'
    obs = white_observation(c=.14)
    obs['lines'] = [(u,35,ratio) for u,_,ratio in obs['lines']]
    monkeypatch.setattr(sup.entry_fitter,'fit',lambda *a: obs)
    for n in range(7):
        sup._try_relocalise(np.zeros((480,640,3),np.uint8),(0,0,0),None,n*.1)
    assert sup.core.c==pytest.approx(.14) and sup.core.relocalizations==1
    assert central_white_pair(obs,.95) is None  # Entry still requires stronger evidence.
