import unittest
from unittest.mock import patch

import numpy as np

from jetracer_ros2 import cone_corridor
from jetracer_ros2.m8_entry import M8EntryGate, M8ExitGate, m8_entry_policy, m8_phase, select_m8_command
from jetracer_ros2.cone_corridor import (
    Params, Planner, ResumeGate, approach_clearance_low, checked_exit_lane, curvature_limit, plan_m8,
    rollout, safe_lane_arc)
from jetracer_ros2.utils.ackermann_utils import TRACK, WHEELBASE, inner_angle_to_omega

A = (-1.919862, 1.919862, .0153623, 0., .1, .15, 64.)
P = Params()


def synthetic_scan(centers):
    theta = A[0] + np.arange(250)*A[2]
    rays = np.column_stack((np.cos(theta), np.sin(theta)))
    out = np.full(len(theta), np.inf)
    for x, y in centers:
        q = np.array([x-P.lidar_x, y-P.lidar_y])
        dot = rays @ q
        disc = .07**2 - q @ q + dot**2
        r = dot - np.sqrt(np.maximum(0, disc))
        out = np.minimum(out, np.where((disc >= 0) & (r >= A[5]), r, np.inf))
    return out


def parallel(half=.32, shift=.0):
    return [(x,.0+half) for x in (.35,.68,1.,1.33,1.66)] + [
        (x+shift,-half) for x in (.35,.68,1.,1.33,1.66)]


class TestConeCorridor(unittest.TestCase):



    def test_recovery_requires_consecutive_scans(self):
        gate = ResumeGate(2)
        self.assertEqual([gate.update(x) for x in (True, True, False, True, False, True, True)],
                         [False, True, False, False, False, False, True])

    def test_recovery_requires_same_mouth(self):
        gate = ResumeGate(2)
        self.assertFalse(gate.update(True, np.array([.5,.2])))
        self.assertFalse(gate.update(True, np.array([.5,-.2])))
        self.assertTrue(gate.update(True, np.array([.49,-.2])))

    def test_passage_gives_m8_exclusive_authority_independently_of_cone_availability(self):
        gate = M8EntryGate(.02, .5)
        self.assertEqual(gate.decide(2., True), 'off')
        gate.arm(2.)
        for i in range(3):
            gate.observe(2. + i*.01, .30-i*.01, i*.05)
        self.assertEqual(gate.decide(2.30, True), 'approach')
        self.assertEqual(gate.decide(2.33, False), 'wait_odom')
        self.assertEqual(gate.decide(2.33, True), 'passed')
        self.assertEqual(select_m8_command(None, 0.), ((0., 0.), 'waiting_for_cone'))
        self.assertEqual(gate.decide(2.25, True), 'passed')
        self.assertEqual(gate.decide(2.25, False), 'passed')
        gate.reset()
        self.assertEqual(gate.decide(2.25, True), 'off')

    def test_m8_passes_original_steering_without_rescaling(self):
        for omega in (-.135, 0., .135):
            self.assertEqual(select_m8_command((.1, omega), .20),
                             ((.1, omega), 'cone_tracking'))

    def test_early_entry_prefers_cone_but_never_unchecked_approach(self):
        self.assertEqual(m8_entry_policy('off', False, True, True, True), 'lane')
        self.assertEqual(m8_entry_policy('approach', True, True, True, True), 'cone')
        self.assertEqual(m8_entry_policy('approach', True, True, False, True), 'guarded_lane')
        self.assertEqual(m8_entry_policy('approach', True, True, False, False), 'stop')
        self.assertEqual(m8_entry_policy('approach', True, False, True, True), 'stop')
        self.assertEqual(m8_entry_policy('passed', True, True, False, True), 'stop')
        self.assertEqual(m8_entry_policy('line_not_found', True, True, True, False), 'cone')


    def test_m8_blocks_non_actuatable_stale_or_stopped_proposals(self):
        for command, age, estop, reason in (
                ((.072, -.13), .1, False, 'below_min_speed'),
                ((.0999, .08), .1, False, 'below_min_speed'),
                ((.10, -.13), .251, False, 'stale_cone'),
                ((.10, -.13), -.001, False, 'stale_cone'),
                ((.10, -.13), .10, True, 'operator_stop'),
                ((0., .2), .10, False, 'cone_stop'),
                ((.10, float('nan')), .10, False, 'invalid_cone'),
                ((.11, 0.), .10, False, 'invalid_cone'),
                ((-.10, 0.), .10, False, 'invalid_cone')):
            self.assertEqual(select_m8_command(command, age, estop), ((0., 0.), reason))

    def test_missing_or_glitch_line_does_not_authorize_handoff(self):
        gate = M8EntryGate(.02, .5)
        gate.arm(0.)
        for t, dist in enumerate((.30, .55, .28, None, 0.0)):
            gate.observe(0., dist, t*.05)
        self.assertIsNone(gate.line_s)
        self.assertEqual(gate.decide(.51, True), 'line_not_found')





    def test_far_preview_does_not_assume_missing_beams_are_free_space(self):
        from jetracer_ros2.cone_corridor import _prefix_observed
        all_seen = np.full(250, 4.)
        self.assertTrue(_prefix_observed(A, all_seen, P, 0.))
        phi = A[0]+np.arange(len(all_seen))*A[2]
        all_seen[abs(phi) < .25] = np.inf
        self.assertFalse(_prefix_observed(A, all_seen, P, 0.))

    def test_exit_does_not_confuse_entry_with_exit_or_trust_odometry_alone(self):
        gate = M8ExitGate(.60, .30, .15)
        self.assertEqual(gate.decide(3.0, True, True), 'no_entry_anchor')
        gate.reset(1.0)
        for t in (.1, .2, .3):
            gate.observe(.90, .10, t)
            gate.observe(1.30, .20, t)
        self.assertIsNone(gate.line_s)
        self.assertEqual(gate.decide(2.0, True, True), 'search_exit')

    def test_exit_owns_lane_only_after_distinct_line_pass_and_checked_lane(self):
        gate = M8ExitGate(.60, .30, .15)
        gate.reset(1.0)
        for t, s in enumerate((1.40, 1.41, 1.42)):
            gate.observe(s, 1.80-s, t*.05)
        self.assertAlmostEqual(gate.line_s, 1.80)
        self.assertEqual(gate.decide(1.90, True, True), 'approach_exit')
        self.assertEqual(gate.decide(2.11, False, True), 'wait_odom')
        self.assertEqual(gate.decide(2.11, True, False), 'wait_exit_lane')
        self.assertEqual(gate.decide(2.05, True, True), 'wait_exit_lane')
        self.assertEqual(gate.decide(2.11, True, True), 'exit_lane')
        self.assertEqual(gate.decide(2.27, True, False), 'wait_exit_lane')
        self.assertEqual(gate.decide(2.27, True, True), 'complete')

    def test_checked_exit_lane_keeps_curvature_and_checks_last_cone(self):
        points = np.array([[.45,.50], [.70,-.50], [.90,.60], [1.10,-.55]])
        self.assertEqual(checked_exit_lane((.20,-.10), points, P), (.10,-.05))
        blocked = np.vstack((points,[.54,0.]))
        self.assertEqual(checked_exit_lane((.20,-.10), blocked, P), (0.,0.))
        self.assertEqual(checked_exit_lane((.20,-.10), points[:2], P), (0.,0.))
        self.assertEqual(checked_exit_lane((.08,-.04), points, P), (0.,0.))
        self.assertEqual(checked_exit_lane((.20,5.), points, P), (0.,0.))
        unobserved = (A, np.full(250, np.inf))
        self.assertEqual(checked_exit_lane((.20,-.10), points, P, scan=unobserved), (0., 0.))

    def test_exit_detection_hands_over_before_passing_but_waits_without_a_safe_lane(self):
        gate = M8ExitGate(.60, .30, .15, handoff_on_detection=True)
        gate.reset(1.)
        self.assertFalse(gate.owns_lane)
        for t,s in enumerate((1.40, 1.41, 1.42)):
            gate.observe(s, 1.80-s, t*.05)
        self.assertTrue(gate.owns_lane)
        self.assertEqual(gate.decide(1.43, True, False), 'wait_exit_lane')
        self.assertEqual(gate.decide(1.43, True, True), 'exit_lane')
        self.assertEqual(gate.decide(1.65, True, False), 'wait_exit_lane')
        self.assertEqual(gate.decide(1.70, True, True), 'exit_lane')
        self.assertEqual(gate.decide(2.11, True, True), 'complete')

    def test_exit_longer_range_requires_three_consistent_distinct_frames_in_window(self):
        gate = M8ExitGate(.60, .30, .15, handoff_on_detection=True)
        gate.reset(1.)
        gate.observe(1.15, .82, 1.00)  # Too early relative to entry.
        gate.observe(1.30, .86, 1.01)  # Beyond exit search range.
        gate.observe(1.30, .82, 1.02)
        gate.observe(1.30, .82, 1.02)  # Duplicate acquisition time.
        gate.observe(1.31, None, 1.08)
        gate.observe(1.32, .80, 1.10)
        self.assertIsNone(gate.line_s)
        gate.observe(1.34, .78, 1.20)
        self.assertAlmostEqual(gate.line_s, 2.12)
        self.assertEqual(m8_phase(True, gate.owns_lane), 'exit_lane')
        self.assertEqual(m8_phase(True, False), 'cone')
        self.assertEqual(m8_phase(False, True), 'off')
        self.assertEqual(gate.decide(1.35, True, False), 'wait_exit_lane')
        gate.reset(1.)
        for t in (1.0, 1.13, 1.26):
            gate.observe(1.30, .82, t)
        self.assertIsNone(gate.line_s)

    def test_one_local_gate_starts_without_a_second_cone_row_pair(self):
        cones = [(.65, -.30), (.65, .30)]
        scan = synthetic_scan(cones)
        seen = np.where(np.isfinite(scan), scan, 4.)
        selected = plan_m8(A, seen, P)
        self.assertEqual(selected['mode'], 'local_gate')
        self.assertEqual(selected['speed'], .10)
        self.assertLessEqual(abs(selected['curvature']), curvature_limit(P))
        self.assertLess(abs(selected['steer']), .03)
        self.assertEqual(plan_m8(A, scan, P)['speed'], 0.)
        for unsafe in (cones[:1], cones+[(.48, 0.)], cones+[(.65, 0.)]):
            raw = synthetic_scan(unsafe)
            raw = np.where(np.isfinite(raw), raw, 4.)
            self.assertEqual(plan_m8(A, raw, P)['speed'], 0.)
        raw = synthetic_scan(cones+[(.38, 0.)])
        # With the measured smaller padding this is outside the CURRENT
        # footprint, but remains blocked on every forward 20 cm arc.
        self.assertEqual(plan_m8(A, raw, P)['state'], 'path_blocked')

    def test_guarded_lane_approach_checks_short_sweep_without_changing_exit(self):
        scan = A, np.full(250, 4.)
        points = np.array([[.58, 0.], [.50, .52], [.62, -.49], [1.15, .36]])
        self.assertEqual(checked_exit_lane((.10, 0.), points, P, scan=scan), (0., 0.))
        self.assertEqual(checked_exit_lane((.10, 0.), points, P, scan=scan, approach=True), (.10, 0.))
        points[0, 0] = .43
        self.assertEqual(checked_exit_lane((.10, 0.), points, P, scan=scan, approach=True), (0., 0.))

    def test_exit_connector_is_not_an_override_for_missing_rays_or_blocked_footprint(self):
        x = np.linspace(.35,.85,25)
        center = np.column_stack((x,.20*np.ones(25)))
        points = np.array([[.38,-.53], [.72,-.55]])
        scan = A, np.full(250, 4.)
        v, w = safe_lane_arc(scan, points, P, center, (.10, .2))
        self.assertEqual(v, .10)
        self.assertLessEqual(abs(w/v), curvature_limit(P))
        self.assertEqual(safe_lane_arc((A, np.full(250,np.inf)), points, P, center, (.10,.2)), (0.,0.))
        blocked = np.vstack((points, [.27, 0.]))
        self.assertEqual(safe_lane_arc(scan, blocked, P, center, (.10,.2)), (0.,0.))
        self.assertEqual(safe_lane_arc(scan, points, P, center*np.nan, (.10,.2)), (0.,0.))



def observed(cones):
    raw=synthetic_scan(cones)
    return np.where(np.isfinite(raw),raw,4.)

def test_single_pair_and_staggered_turn_are_drivable_without_full_row():
    for shift in (0.,.13):
        z=plan_m8(A,observed(parallel(shift=shift)),P)
        assert z['state']=='tracking'
        assert z['speed']==.10

def test_narrow_corridor_and_raw_obstacle_are_stops():
    for cones in (parallel(.18),parallel()+[(.40,0.)]):
        assert plan_m8(A,observed(cones),P)['speed']==0.
    assert plan_m8(A,np.full(250,np.inf),P)['speed']==0.


def test_far_obstacle_biases_turn_before_the_local_arc_needs_it():
    scan = np.full(250, 4.)
    gate = {'gate': np.array([.95, 0.]), 'cost': 0.}
    for side in (-1, 1):
        points = np.array([[.60, side*.15]])
        with patch.object(cone_corridor, 'clusters', return_value=(points, [])), \
                patch.object(cone_corridor, '_gates', return_value=[gate]):
            with patch.object(cone_corridor, 'PREVIEW_WEIGHT', 0.):
                old = plan_m8(A, scan, P)
            new = plan_m8(A, scan, P)
        assert old['speed'] == new['speed'] == .10
        assert old['curvature'] == 0.
        assert side*new['curvature'] < 0.
        assert new['min_clearance'] >= P.clearance
        assert np.isclose(np.linalg.norm(np.diff(new['arc'][:, :2], axis=0), axis=1).sum(),
                          .20, atol=1e-5)
        assert cone_corridor.swept_clearance(
            points, rollout(new['curvature'], .38), P) > P.clearance


def test_preview_never_scores_or_releases_candidates_with_missing_local_rays():
    points = np.array([[.60, -.15]])
    gate = {'gate': np.array([.95, 0.]), 'cost': 0.}
    with patch.object(cone_corridor, 'clusters', return_value=(points, [])), \
            patch.object(cone_corridor, '_gates', return_value=[gate]), \
            patch.object(cone_corridor, '_prefix_observed', return_value=False) as observed, \
            patch.object(cone_corridor, '_preview_penalty') as preview:
        assert plan_m8(A, np.full(250, 4.), P)['state'] == 'path_blocked'
    assert observed.call_count > 0
    assert all(call.kwargs['horizon'] == .20 for call in observed.call_args_list)
    preview.assert_not_called()


def test_approach_risk_observes_beyond_twenty_cm_without_altering_guarded_speed():
    rays = (A, np.full(250, 4.))
    points = np.array([[.62, 0.], [.50, .52], [.62, -.55], [1.15, .36]])
    assert checked_exit_lane((.30, 0.), points, P, scan=rays, approach=True) == (.10, 0.)
    assert approach_clearance_low((.30, 0.), points, P)
    assert not approach_clearance_low((.30, 0.), points[1:], P)

def test_inner_and_bridge_central_angles_are_not_the_same_value():
    for k in np.linspace(-curvature_limit(P),curvature_limit(P),11):
        inner=np.sign(k)*np.arctan(WHEELBASE/(1/max(abs(k),1e-9)-TRACK/2))
        w=inner_angle_to_omega(inner,.10)
        assert abs(w/.10-k)<1e-7
        bridge_angle=np.arctan(WHEELBASE*w/.10)
        assert abs(bridge_angle)<=abs(inner)+1e-8

def test_pose_shift_prediction_expires_and_never_bypasses_fresh_collision_scan():
    planner=Planner(P)
    z=planner.update(A,observed([(.65,-.30),(.65,.30)]),1.,(.1,0.,0.))
    assert z['state']=='tracking'
    # Empty gate observations but measured clear rays; target advances by odom.
    z=planner.update(A,np.full(250,4.),1.1,(.11,0.,0.))
    assert z['mode']=='tracked_gate'
    assert z['road']['gate'][0]<.65
    z=planner.update(A,observed([(.39,0.)]),1.2,(.12,0.,0.))
    assert z['speed']==0.
    for t in (1.3,1.4,1.5,1.6):
        z=planner.update(A,np.full(250,4.),t,(.12,0.,0.))
    assert z['speed']==0.  # Prediction does not renew its own age.

def test_history_without_odom_or_after_reposition_cannot_drive():
    for pose in (None,(1.,0.,0.)):
        planner=Planner(P)
        planner.update(A,observed([(.65,-.30),(.65,.30)]),1.,(.1,0.,0.))
        z=planner.update(A,np.full(250,4.),1.1,pose)
        assert z['speed']==0.

def test_lost_scan_stops_even_with_goal_history():
    planner=Planner(P)
    planner.update(A,observed([(.65,-.30),(.65,.30)]),1.,(.1,0.,0.))
    z=planner.update(A,np.full(250,np.nan),1.1,(.11,0.,0.))
    assert z['speed']==0.


def test_measured_padding_and_clearance_are_separate():
    assert P.cone_padding == .015
    assert P.clearance == .020
    body = np.zeros((1, 3))
    for raw_gap, expected in ((.034, False), (.036, True)):
        points = np.array([[P.front+raw_gap, 0.]])
        assert (cone_corridor.swept_clearance(points, body, P) >= P.clearance) == expected


def test_refined_forward_search_finds_narrow_band_and_reuses_checks():
    gate = {'gate': np.array([.8, 0.]), 'normal': np.array([1., 0.]), 'cost': 0.}
    points = np.array([[1.5, .8]])
    with patch.object(cone_corridor, 'clusters', return_value=(points, [])), \
            patch.object(cone_corridor, '_gates', return_value=[gate, gate]), \
            patch.object(cone_corridor, '_prefix_observed',
                         side_effect=lambda a, r, p, k, **kw: .97 < k < 1.04) as seen:
        z = plan_m8(A, np.full(250, 4.), P)
    assert z['state'] == 'tracking'
    assert z['search_stage'] == 'refined'
    assert .97 < z['curvature'] < 1.04
    ks = [round(call.args[3], 10) for call in seen.call_args_list]
    assert len(ks) == len(set(ks))
    assert z['candidate_count'] <= 83
    assert all(call.kwargs['horizon'] == .20 for call in seen.call_args_list)


def test_entry_far_search_is_time_and_odometry_bounded():
    window = cone_corridor.EntrySearchWindow()
    for phase in ('watch_entry', 'approach', 'exit_lane', 'off'):
        window.set_phase(phase)
        assert not window.allows(10., (0., 0., 0.))
    window.set_phase('cone')
    assert not window.allows(10., None)
    for i in range(8):
        assert window.allows(10.1+i*.1, (i*.05, 0., 0.))
    assert not window.allows(11., (.41, 0., 0.))
    assert not window.allows(11.1, (0., 0., 0.))
    window.set_phase('cone')
    assert window.allows(10., (0., 0., 0.))
    assert not window.allows(18., (0., 0., 0.))
    window.set_phase('cone')
    assert window.allows(10., (0., 0., 0.))
    assert not window.allows(10.1, None)
    assert not window.allows(10.2, (0., 0., 0.))


def test_far_entry_gate_does_not_leak_into_history_and_is_bounded():
    planner = Planner(P)
    far = observed([(1.33,-.30), (1.33,.30)])
    pose = (.1, 0., 0.)
    assert planner.update(A, far, 1., pose)['state'] == 'no_gate'
    z = planner.update(A, far, 1.1, pose, entry_far_search=True)
    assert z['state'] == 'tracking' and z['entry_far_gate']
    assert 1.15 < z['road']['gate'][0] <= 1.40
    assert planner.memory is None and planner.previous_curvature is None
    assert planner.update(A, far, 1.2, pose)['state'] == 'no_gate'
    too_far = observed([(1.47,-.30), (1.47,.30)])
    assert planner.update(A, too_far, 1.3, pose, entry_far_search=True)['speed'] == 0.


def test_far_entry_never_overrides_ordinary_gate_or_safety_checks():
    pose = (.1, 0., 0.)
    cones = [(1.33,-.30), (1.33,.30)]
    both = observed(cones + [(.65,-.30), (.65,.30)])
    old = Planner(P).update(A, both, 1., pose)
    new = Planner(P).update(A, both, 1., pose, entry_far_search=True)
    assert old['curvature'] == new['curvature']
    assert not new.get('entry_far_gate')
    with patch.object(cone_corridor, '_prefix_observed', return_value=False):
        assert Planner(P).update(A, observed(cones), 1., pose, entry_far_search=True)['speed'] == 0.
    blocked = observed(cones + [(.40, 0.)])
    assert Planner(P).update(A, blocked, 1., pose, entry_far_search=True)['speed'] == 0.


def test_normal_pass_never_pays_for_dense_search():
    z = plan_m8(A, observed(parallel()), P)
    assert z['state'] == 'tracking'
    assert z['search_stage'] == 'coarse'
    assert z['candidate_count'] <= 11 + 2*8


def test_alignment_turns_early_but_releases_once_aligned_and_is_symmetric():
    for sign in (-1, 1):
        gate = {'gate': np.array([.75, sign*.18]),
                'normal': np.array([np.cos(.8), sign*np.sin(.8)]), 'cost': 0.}
        points = np.array([[2., 1.]])
        with patch.object(cone_corridor, 'clusters', return_value=(points, [])), \
                patch.object(cone_corridor, '_gates', return_value=[gate]):
            with patch.object(cone_corridor, 'GATE_ALIGNMENT_WEIGHT', 0.):
                no_alignment = plan_m8(A, np.full(250, 4.), P)
            early = plan_m8(A, np.full(250, 4.), P)
            gate['gate'] = np.array([.75, 0.])
            gate['normal'] = np.array([1., 0.])
            aligned = plan_m8(A, np.full(250, 4.), P)
        assert sign*early['curvature'] > sign*no_alignment['curvature']
        assert abs(aligned['curvature']) < 1e-8


def test_alignment_never_forces_maximum_or_bypasses_missing_rays():
    gate = {'gate': np.array([.65, -.3]), 'normal': np.array([.6, -.8]), 'cost': 0.}
    points = np.array([[1.5, .8]])
    with patch.object(cone_corridor, 'clusters', return_value=(points, [])), \
            patch.object(cone_corridor, '_gates', return_value=[gate]), \
            patch.object(cone_corridor, '_prefix_observed',
                         side_effect=lambda a, r, p, k, **kw: -1.4 < k < -.8):
        z = plan_m8(A, np.full(250, 4.), P)
    assert z['state'] == 'tracking'
    assert -1.4 < z['curvature'] < -.8
    assert abs(z['curvature']) < curvature_limit(P)
    with patch.object(cone_corridor, 'clusters', return_value=(points, [])), \
            patch.object(cone_corridor, '_gates', return_value=[gate]), \
            patch.object(cone_corridor, '_prefix_observed', return_value=False):
        assert plan_m8(A, np.full(250, 4.), P)['speed'] == 0.

if __name__ == '__main__':
    unittest.main()
