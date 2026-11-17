"""A reverse manoeuvre is travelled distance, not progress through a fork."""
import numpy as np
from jetracer_ros2 import lane3_gap as G


def laser():
    return (np.full(250,3.2),-1.92,3.84/249,.12)


def test_reverse_travel_does_not_finish_fork_early():
    s=G.GapSupervisor();s.mode='on';c=s.core
    c.travel=G.MAX_ON_M+2.;c.isl=(2.,0.);c.goal=2
    c.start(0.,(0.,0.,0.))
    s.step_obs(None,(0.,0.,0.),laser(),False,1.)
    assert s.mode=='on' and not s.route_complete


def test_reverse_travel_does_not_trigger_no_sign_exit():
    s=G.GapSupervisor();s.mode='on';s.core.travel=G.MAX_ON_M+2.
    s.core.start(0.,(0.,0.,0.))
    for i in range(10):s.step_obs(None,(0.,0.,0.),laser(),True,1.+i*.1)
    assert s.mode=='on'


def test_forward_watchdog_does_not_erase_known_island_exit():
    s=G.GapSupervisor();s.mode='on';c=s.core;c.isl=(G.MAX_ON_M,0.);c.goal=2
    c.s=G.MAX_ON_M;c.pose=(0.,0.,0.)
    s.step_obs(None,(0.,0.,0.),laser(),False,1.)
    assert s.mode=='fork' and s._fork_end==c.isl[0]+G.EXIT_AFTER
    assert not s.route_complete
