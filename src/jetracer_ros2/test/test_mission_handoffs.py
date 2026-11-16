"""Controller-produced fork evidence through real manager callbacks, no hardware."""
import json
from types import SimpleNamespace as NS

import pytest

from jetracer_ros2.mission_handoffs import ForkProgressReporter, lane3_progress
from jetracer_ros2 import three_lane_simple as S
from jetracer_ros2 import three_lane as T
from jetracer_ros2.lane3_gap import GapSupervisor
from test_mission_manager_transitions import modules, owner, lane, step, proposal, event
from test_three_lane_simple import core_at, scan_with


def signed_simple(sign=0):
    sup = S.Lane3SimpleSupervisor()
    sup.set_mission('M7')
    for _ in range(T.SIGN_NEED):
        sup.see_signs([dict(label='a' if sign == 0 else 'b', confidence=.9,
                             bbox=[250, 0, 350, 200])])
    assert sup.vote.goal == sign
    sup.mode = 'on'
    sup.core = core_at()
    sup.core.s = sup.core.island_at-S.LEAVE_DIST+.05
    return sup


def report(n, payload):
    n._lane3_progress_callback(NS(data=json.dumps(payload)))


@pytest.mark.parametrize('side', [0, 2])
def test_real_sign_vote_plan_motion_report_and_manager_handoff(modules, side):
    n = owner(modules, 'M7')
    sup = signed_simple(side)
    r = ForkProgressReporter()
    r.set_mission('M7')
    for i in range(2):
        pose = (i*.02, 0., 0.)
        out = sup.step_obs(None, pose, scan_with([]), False, now=i*.1)
        assert sup.core.lane == side and out['v'] > 0
        step(n, dt=.1, speed=.2)
        payload = r.update(sup, pose, out['v'], n.now)
        report(n, payload)
        assert n._active == ('M7' if i == 0 else 'M2')
    sup.set_mission('M2')
    assert sup.mode == 'on'  # Keep the same controller/plan across the boundary.
    report(n, payload)
    assert n._active == 'M2'  # Repeated report cannot skip M2.


def test_wrong_selected_branch_is_logged_but_still_enters_m2(modules):
    n = owner(modules, 'M7')
    sup = signed_simple(0)
    sup.core.lane = 2  # A observed, but obstacle avoidance selected right.
    r = ForkProgressReporter()
    r.set_mission('M7')
    for i in range(2):
        step(n, dt=.1, speed=.2)
        payload = r.update(sup, (i*.02, 0., 0.), .15, n.now)
        report(n, payload)
    assert payload['sign_side'] == 0 and payload['fork_side'] == 2
    assert n._active == 'M2'


@pytest.mark.parametrize('speed,pose', [(0., (.02, 0., 0.)), (-.1, (-.02, 0., 0.)),
                                       (.1, (0., 0., 0.)), (.1, (2., 0., 0.)),
                                       (.1, (-.02, 0., 0.)), (.1, None)])
def test_command_without_forward_motion_does_not_report_start(speed, pose):
    sup = signed_simple()
    sup.core.lane = 0
    r = ForkProgressReporter()
    r.set_mission('M7')
    r.update(sup, (0., 0., 0.), .15, 10**9)
    out = r.update(sup, pose, speed, 1_100_000_000)
    assert out is None or out['fork_side'] is None


def test_obstacle_lane_change_far_from_fork_and_fixed_goals_are_not_start():
    sup = signed_simple()
    sup.core.lane = 0
    sup.core.s = sup.core.island_at-3.
    assert lane3_progress(sup)[0] is None
    sup.core.s = sup.core.island_at-.5
    sup.vote.reset()
    assert lane3_progress(sup)[0] is None  # configured/default goal is not a seen sign


def test_gap_and_full_report_actual_selected_branch_not_sign_correctness():
    gap = GapSupervisor()
    gap.mode, gap.vote.goal = 'on', 0
    gap.core.isl, gap.core.s, gap.core.tgt = (2., 0.), 1., -.5
    assert lane3_progress(gap)[0] == 2
    gap.vote.goal = None
    gap.core.goal = 2  # timed guessing does not count as a detected sign
    assert lane3_progress(gap)[0] is None
    full = T.Lane3Supervisor()
    full.mode, full.vote.goal = 'on', 0
    full.last = NS(ok=True, target=2, island=1.)
    assert lane3_progress(full)[0] == 2


@pytest.mark.parametrize('failure', ['stale', 'future', 'pre_entry', 'reverse', 'stopped', 'estop', 'invalid_sign'])
def test_manager_rejects_invalid_fork_start(modules, failure):
    n = owner(modules, 'M7')
    step(n, speed=.2)
    payload = dict(stamp_ns=n.now, fork_side=0, sign_side=0, route_complete=False)
    if failure == 'stale':
        payload['stamp_ns'] -= 1_000_000_000
    elif failure == 'future':
        payload['stamp_ns'] += 1
    elif failure == 'pre_entry':
        payload['stamp_ns'] = n._mission_enter_ns-1
    elif failure in ('reverse', 'stopped'):
        n._odom_speed = -.1 if failure == 'reverse' else 0.
    elif failure == 'estop':
        event(n, 'ESTOP')
    else:
        payload['sign_side'] = None
    report(n, payload)
    assert n._active == 'M7'


def m7_owner(modules):
    return owner(modules, 'M7', m8_entry_arm_after_m=-1.,
                 m8_entry_arm_near_line_m=.45, m8_entry_precheck_m=.55)


def allow_watch(n):
    # Route completion is observed success, not a timeout; may arrive stopped.
    report(n, dict(stamp_ns=n.now, fork_side=None, route_complete=True))
    assert n._active == 'M7' and n.m8._m8_phase_sent == 'watch_entry'


def full_entry(n):
    for _ in range(3):
        n.now += 50_000_000
        n._odom_ns = n.now
        n.m8._m8_visual_line_callback(NS(data=.54))
    n._odom.s += .12
    step(n, speed=.2)


def test_m7_obstacles_and_lines_do_not_arm_cone_before_fork_passage(modules):
    n = m7_owner(modules)
    full_entry(n)
    n.m8._store_cone_cmd(proposal(n))
    step(n, speed=.2)
    assert n._active == 'M7' and n.m8._m8_phase_sent == 'off'
    assert n.cmd_pub.messages[-1].angular.z == .10
    assert not n.started


def test_m7_direct_m8_retains_entry_anchor_and_advances_to_m1_not_m8_again(modules):
    n = m7_owner(modules)
    allow_watch(n)
    full_entry(n)
    anchor = n.m8._m8_entry.line_s
    # Checked cone absent: M7's original command must remain unchanged.
    assert n._active == 'M7'
    assert n.cmd_pub.messages[-1].linear.x == .25
    assert n.cmd_pub.messages[-1].angular.z == .10
    n.m8._store_cone_cmd(proposal(n))
    assert n._active == 'M8'
    assert n.m8._m8_entry.line_s == anchor and n.m8._m8_exit.entry_line == anchor
    assert n.cmd_pub.messages[-1].angular.z == -.10
    assert modules.manager.MISSION_ORDER[n._index] == 'M8'
    # Dispatch-only check; exit evidence tests exercise the actual exit gate.
    n._advance('test confirmed exit dispatch')
    assert n._active == 'M1'


@pytest.mark.parametrize('failure', ['no_line', 'stale_cone', 'reverse', 'estop'])
def test_m7_watch_is_not_unconditional_m8_handoff(modules, failure):
    n = m7_owner(modules)
    allow_watch(n)
    if failure != 'no_line':
        full_entry(n)
    if failure == 'reverse':
        n._odom_speed = -.1
    if failure == 'estop':
        event(n, 'ESTOP')
    p = proposal(n)
    if failure == 'stale_cone':
        p.header.stamp.sec -= 2
    n.m8._store_cone_cmd(p)
    step(n, speed=-.1 if failure == 'reverse' else .2)
    assert n._active == 'M7'


def test_m7_partial_entry_with_checked_cone_can_handover(modules):
    n = m7_owner(modules)
    allow_watch(n)
    for _ in range(3):
        n.now += 50_000_000
        n._odom_ns = n.now
        cue = proposal(n)
        cue.header.frame_id, cue.point.x = 'front_axle', .49
        n.m8._m8_partial_callback(cue)
        n.m8._store_cone_cmd(proposal(n))
        n._control_tick()
    assert n._active == 'M8'


def test_supervisor_failure_does_not_mark_route_complete():
    sup = signed_simple()
    for _ in range(S.LOST_FRAMES):
        sup.step_obs(None, None, None, False)
    assert sup.mode == 'done' and not sup.route_complete
