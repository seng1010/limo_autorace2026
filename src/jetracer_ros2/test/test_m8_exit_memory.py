import numpy as np
import pytest

from jetracer_ros2.cone_corridor import Params, _rotation
from jetracer_ros2.m8_exit_memory import ConeExitMemory


P = Params()
A = (-1.919862, 1.919862, .0153623, 0., .1, .15, 64.)
SCAN = A, np.full(250, 4.)


def measurement(x, *, road=True, groups=True, predicted=False):
    cloud = [np.array([[.50-x, y-.02], [.52-x, y+.02]]) for y in (.32, -.32)]
    return dict(state='tracking' if road else 'no_gate', groups=cloud if groups else [],
                road=dict(ends=np.array([g.mean(axis=0) for g in cloud]),
                          normal=np.array([1., 0.]), predicted=predicted) if road else None)


def pose(x):
    return (x-P.rear_from_odom_x, 0., 0.)


def passed_memory(*, view=SCAN, one_hit=False, yaw=0.):
    mem = ConeExitMemory(P)
    for i,x in enumerate(np.linspace(0., .68, 15)):
        z=measurement(x, road=(i==0 if one_hit else i<3), groups=i<8)
        # Rigidly rotate/translate the entire experiment: no image-axis assumptions.
        od = np.asarray(pose(x))
        od[:2] = od[:2] @ _rotation(yaw).T + (2., -1.)
        od[2] = yaw
        status=mem.update(1.+i*.1, od, z, view)
    return mem,status


@pytest.mark.parametrize('yaw', [0., .6, -1.3])
def test_whole_footprint_must_pass_measured_gate(yaw):
    mem,status=passed_memory(yaw=yaw)
    assert status['cleared'] and status['gates']==1
    assert status['rear_clearance'] > 0


def test_missing_cones_while_stopped_is_not_exit():
    mem=ConeExitMemory(P)
    for i in range(3):
        mem.update(1+i*.1, pose(0.), measurement(0.), SCAN)
    for i in range(1,6):
        z=mem.update(1.2+i*.1, pose(0.), measurement(0.,road=False,groups=False), SCAN)
        assert not z['cleared'] and z['rear_clearance']<0


def test_front_axle_or_center_crossing_does_not_mean_rear_cleared():
    mem=ConeExitMemory(P)
    for i,x in enumerate(np.linspace(0., .60, 13)):
        z=mem.update(1+i*.1, pose(x), measurement(x,road=i<3,groups=i<8), SCAN)
    assert not z['cleared']  # rear=.51, observed cone extends to .52 + .035 margin


def test_one_observation_or_unobserved_forward_sector_is_not_exit():
    assert not passed_memory(one_hit=True)[1]['cleared']
    assert not passed_memory(view=(A,np.full(250,np.inf)))[1]['cleared']


def test_predicted_gate_never_creates_measured_history():
    mem=ConeExitMemory(P)
    for i in range(4):
        z=mem.update(1+i*.1, pose(0.), measurement(0.,predicted=True), SCAN)
    assert z['gates']==0 and not z['cleared']


@pytest.mark.parametrize('cause', ['reverse','jump','gap','missing'])
def test_discontinuities_invalidate_history(cause):
    mem,z=passed_memory()
    t=2.5
    od=mem.last_pose.copy()
    od[:2] -= _rotation(od[2]) @ np.array([P.rear_from_odom_x,0.])
    if cause=='reverse': od[0]-=.03
    elif cause=='jump': od[0]+=.4
    elif cause=='gap': t+=1.
    else: od=None
    z=mem.update(t,od,measurement(0.,road=False,groups=False),SCAN)
    assert not z['cleared'] and z['gates']==0


def test_unseen_future_cone_and_expired_history_cannot_be_erased_to_finish():
    mem,z=passed_memory()
    od=mem.last_pose.copy();od[0]-=P.rear_from_odom_x
    data=measurement(.0,road=False)
    z=mem.update(2.5,od,data,SCAN)
    assert z['reason']=='forward_cones_remain' and not z['cleared']
    mem=ConeExitMemory(P)
    for i in range(150):
        z=mem.update(1+i*.1,pose(0.),measurement(0.,road=i<3,groups=i<3),SCAN)
    assert z['reason']=='history_expired' and not z['cleared']


def test_clearance_configuration_is_not_modified():
    assert P.cone_padding==.015 and P.clearance==.020 and P.speed==.10


def test_rotated_rear_corner_must_also_clear_the_gate():
    mem=ConeExitMemory(P)
    empty=measurement(0.,road=False,groups=False)
    for i,x in enumerate(np.linspace(0.,.60,13)):
        mem.update(1+i*.1,pose(x),measurement(x,road=i<3,groups=i<8),SCAN)
    for t,x,yaw in ((2.3,.64,.3),(2.4,.68,.6)):
        od=np.array([x,0.,yaw]);od[:2]-=_rotation(yaw)@np.array([P.rear_from_odom_x,0.])
        z=mem.update(t,od,empty,SCAN)
    assert not z['cleared'] and z['rear_clearance']<0.
    od[:2]+=(.07,0.)
    z=mem.update(2.5,od,empty,SCAN)
    assert z['cleared']
