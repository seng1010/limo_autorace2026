"""Rearward motion probes the SAME forward planner, without new steering policy."""
import numpy as np
import pytest
from jetracer_ros2 import lane3_gap as G


def reversing():
    core = G.GapCore()
    core.rev_left = .5
    core.v = -.1
    core.back_room = .8
    core.pose = (0., 0., 0.)
    core.t_prev = 0.
    return core


def step(core, t, scan=True):
    # Fresh clear laser fan; frozen pose means this test doesn't invent motion.
    laser = (np.full(250, 3.2), -1.92, 3.84/249, .12) if scan else None
    return core.step_obs(None, (0.,0.,0.), laser, None, t)


def test_replan_stops_reverse_before_distance_budget_then_rechecks_forward():
    c = reversing()
    assert step(c,.1)['v'] < 0
    assert step(c,.2)['v'] < 0
    assert step(c,.3)['v'] == 0
    assert c.rev_left == 0
    assert c.travel == 0  # Did NOT need to spend the old 50 cm budget.
    assert step(c,.4)['v'] == 0
    assert step(c,.5)['v'] > 0


def test_forward_probe_runs_each_reverse_frame(monkeypatch):
    c = reversing()
    original = c._rollout
    calls = []
    def track(*a):
        calls.append(True)
        return original(*a)
    monkeypatch.setattr(c,'_rollout',track)
    step(c,.1); step(c,.2)
    assert len(calls) >= 2


@pytest.mark.parametrize('free',[.12,.29,.40])
def test_blocked_or_marginal_forward_route_does_not_end_reverse(monkeypatch,free):
    c = reversing()
    monkeypatch.setattr(c,'_free',lambda go,gl,x,y,*a: np.full(x.shape[0],free))
    for t in (.1,.2,.3,.4):
        assert step(c,t)['v'] == -.1
        assert c.steer == 0
    assert c.rev_left > 0


def test_missing_scan_stops_reverse_and_breaks_confirmation():
    c = reversing()
    step(c,.1);step(c,.2)
    assert step(c,.3,scan=False)['v'] == 0
    assert step(c,.4)['v'] < 0
    assert step(c,.5)['v'] < 0
    assert step(c,.6)['v'] == 0


def test_repeated_timestamp_cannot_confirm_route():
    c = reversing()
    for _ in range(8):
        assert step(c,.1)['v'] < 0
    assert c.rev_left > 0


def test_rear_budget_exhaustion_stops_before_forward():
    c = reversing();c.back_room = .01
    assert step(c,.1)['v'] == 0
    assert c.rev_left == 0


def test_new_block_after_reverse_end_is_not_ignored(monkeypatch):
    c = reversing()
    for t in (.1,.2,.3):step(c,t)
    monkeypatch.setattr(c,'_free',lambda go,gl,x,y,*a: np.zeros(x.shape[0]))
    assert step(c,.5)['v'] == 0


def test_normal_clear_forward_does_not_wait_for_reverse_confirmation():
    c = G.GapCore()
    assert step(c,.1)['v'] > 0
