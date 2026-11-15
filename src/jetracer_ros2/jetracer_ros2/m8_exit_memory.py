"""Bounded odom-frame passage history, used ONLY for M8 exit readiness.

Remember measured gate endpoints, not commands or empty space. Losing a cone,
expiring history, or resetting odometry can never prove that we have passed it.
All motion still needs the existing live LiDAR collision-checked command.
"""
import math

import numpy as np

from jetracer_ros2.cone_corridor import _prefix_observed, _rotation


class ConeExitMemory:
    MAX_GATES = 16
    MAX_AGE = 12.

    def __init__(self, params):
        self.p = params
        self.reset()

    def reset(self):
        self.gates = []
        self.last_pose = None
        self.last_stamp = None
        self.overflow = False

    def update(self, stamp, odom_pose, result, scan):
        status = dict(cleared=False, gates=0, rear_clearance=None, reason='no_history')
        if (odom_pose is None or result is None or scan is None
                or not math.isfinite(stamp) or not np.isfinite(odom_pose).all()):
            self.reset()
            return {**status, 'reason': 'missing_pose_or_scan'}
        pose = np.array(odom_pose, dtype=float)
        rot = _rotation(pose[2])
        pose[:2] += rot @ np.array([self.p.rear_from_odom_x, 0.])
        if self.last_pose is not None:
            delta = (pose[:2]-self.last_pose[:2]) @ _rotation(self.last_pose[2])
            turn = math.atan2(math.sin(pose[2]-self.last_pose[2]),
                              math.cos(pose[2]-self.last_pose[2]))
            if (not 0 < stamp-self.last_stamp <= .50 or np.linalg.norm(delta) > .12
                    or delta[0] < -.015 or abs(turn) > .35):
                self.reset()
        self.last_pose, self.last_stamp = pose.copy(), stamp

        road = result.get('road')
        groups = result.get('groups', [])
        if (result.get('state') == 'tracking' and road is not None
                and not road.get('predicted') and groups):
            centers = np.array([np.median(g, axis=0) for g in groups])
            ids = [int(np.argmin(np.linalg.norm(centers-end, axis=1))) for end in road['ends']]
            if ids[0] != ids[1] and all(np.linalg.norm(centers[i]-end) < .04
                                      for i, end in zip(ids, road['ends'])):
                ends = road['ends'] @ rot.T + pose[:2]
                normal = road['normal'] @ rot.T
                cloud = np.vstack([groups[i] for i in ids]) @ rot.T + pose[:2]
                # One match per measured pair; opposite normals are not the same passage.
                matches = [(float(np.max(np.linalg.norm(g['ends']-ends, axis=1))), g)
                           for g in self.gates if g['normal'] @ normal > .9]
                distance, match = min(matches, key=lambda z: z[0], default=(np.inf, None))
                if distance <= .12:
                    match['hits'] += 1
                    match['stamp'] = stamp
                    match['ends'] = .8*match['ends'] + .2*ends
                    # Keep the full observed extent: partial views must not shrink it.
                    match['front_bound'] = max(match['front_bound'], float((cloud @ match['normal']).max()))
                elif len(self.gates) < self.MAX_GATES:
                    self.gates.append(dict(ends=ends, normal=normal, hits=1, stamp=stamp,
                                           front_bound=float((cloud @ normal).max())))
                else:
                    self.overflow = True

        confirmed = [g for g in self.gates if g['hits'] >= 2]
        status['gates'] = len(confirmed)
        if not confirmed or self.overflow:
            return {**status, 'reason': 'history_overflow' if self.overflow else 'no_history'}
        # Never drop old gates and call an empty map "clear".
        if any(stamp-g['stamp'] > self.MAX_AGE and not g.get('passed') for g in confirmed):
            return {**status, 'reason': 'history_expired'}
        p = self.p
        body = np.array([[-p.rear, -p.half_width], [-p.rear, p.half_width],
                         [p.front, -p.half_width], [p.front, p.half_width]]) @ rot.T + pose[:2]
        margins = []
        for g in confirmed:
            margin = float((body @ g['normal']).min() -
                           g['front_bound'] - p.cone_padding - p.clearance)
            margins.append(margin)
            g['passed'] = margin > 0
        status['rear_clearance'] = min(margins)
        if status['rear_clearance'] <= 0:
            return {**status, 'reason': 'body_not_past_gates'}
        # A still-visible possible next cone vetoes map-only exit. This is an
        # exit-context check, NOT a replacement or relaxation of collision checks.
        if any(np.any((g[:, 0] > -p.rear) & (g[:, 0] < 1.15) & (abs(g[:, 1]) < .65))
               for g in groups):
            return {**status, 'reason': 'forward_cones_remain'}
        if not _prefix_observed(*scan, p, 0., horizon=.40):
            return {**status, 'reason': 'forward_view_unknown'}
        return {**status, 'cleared': True, 'reason': 'body_past_observed_gates'}
