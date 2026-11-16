"""M8-only road-boundary identity, separate from the race lane controller.

An ego-left yellow line is not necessarily a LEFT road boundary: the car can
still be outside the exit lane. Transverse white tape must end at that yellow
line, on the lane-interior side. Odom associates the observed boundary across
frames; remembered geometry alone never authorizes motion. Coordinates below
are (forward from rear axle, left), metres, unless explicitly image pixels.
"""
import math

import cv2
import numpy as np


def _rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def tape_contours(image):
    """Small image ROI, shared once for all visible yellow candidates."""
    h, w = image.shape[:2]
    # An oblique tape may start above the old 49% row while its yellow
    # junction is below it. Do not cut that connected component in half.
    top = round(.42*h)
    # HSV saturation tolerates a clipped blue channel on the white tape;
    # HLS saturation near white otherwise splits a visibly continuous tape.
    hsv = cv2.cvtColor(image[top:], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 145), (180, 90, 255))
    # Highlights can merge the yellow boundary into the white mask. Remove
    # narrow longitudinal branches before measuring the transverse tape.
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 9), np.uint8))
    contours = []
    # Keep the original mask as well: opening must not erase a thin/oblique
    # real contact. Conflicting roles from the two views are rejected below.
    for candidate in (mask, opened):
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
        found, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    out = []
    for contour in contours:
        (_, _), (a, b), angle = cv2.minAreaRect(contour)
        if a < b:
            angle += 90
        angle = (angle+90)%180-90
        length, thick = max(a, b), min(a, b)
        if (length < .15*w or not 2 <= thick <= .095*h or length < 5*thick
                or abs(angle) > 35 or cv2.contourArea(contour) < .19*length*thick):
            continue
        xy = contour[:, 0].astype(float)
        xy[:, 1] += top
        if xy[:, 1].min() < top+2 or xy[:, 1].max() >= h-3:
            continue
        out.append(xy)
    return out


def contact_role(pixels, contours, width):
    """Tape terminates on ONE side of a near-to-far yellow polyline.

    Do not label from screen half, first-seen order, or an extrapolated tape.
    A white line crossing/continuing on both sides supplies no boundary role.
    """
    if len(pixels) < 4 or not np.isfinite(pixels).all():
        return None
    roles = set()
    for xy in contours:
        distances = np.linalg.norm(xy[:, None, :]-pixels[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        if distances[i, j] > .019*width:
            continue
        tangent = pixels[min(j+3, len(pixels)-1)]-pixels[max(0, j-3)]
        norm = np.linalg.norm(tangent)
        if norm < 3:
            continue
        right_normal = np.array([-tangent[1], tangent[0]])/norm
        signed = (xy-pixels[j]) @ right_normal
        # Contact must be near the END of the white tape. A longitudinal
        # outer white border has too little transverse extent to pass.
        lo, hi = np.quantile(signed, [.05, .95])
        tol = .025*width
        if lo < -.12*width and hi <= tol:
            roles.add('right')  # Interior lies to the yellow line's left.
        elif hi > .12*width and lo >= -tol:
            roles.add('left')
    return next(iter(roles)) if len(roles) == 1 else None


class ExitLaneResolver:
    """At most two tracks / 40 samples each, no point-cloud map or EKF."""
    def __init__(self, forward_offset=.184, rear_from_odom=-.10):
        self.forward_offset, self.rear_from_odom = forward_offset, rear_from_odom
        self.reset()

    def reset(self):
        self.tracks = {}
        self.last = None
        self.agreement_hits = 0

    def update(self, image, result, geometry, pose, stamp):
        out = dict(valid=False, aligned=False, path=None, angle=0., roles=[],
                   reason='no_exit_boundary_evidence')
        if image is None or geometry is None or pose is None or not np.isfinite([*pose, stamp]).all():
            self.reset()
            return {**out, 'reason': 'no_fresh_odom_or_image'}
        pose = np.asarray(pose, float).copy()
        rot = _rotation(pose[2])
        pose[:2] += rot @ np.array([self.rear_from_odom, 0.])
        if self.last is not None:
            old_t, old_pose = self.last
            dt = stamp-old_t
            delta = (pose[:2]-old_pose[:2]) @ _rotation(old_pose[2])
            turn = math.atan2(math.sin(pose[2]-old_pose[2]), math.cos(pose[2]-old_pose[2]))
            if dt <= 0:
                return {**out, 'reason': 'repeated_image'}
            if dt > .5 or np.linalg.norm(delta) > .15 or delta[0] < -.015 or abs(turn) > .35:
                self.reset()
        self.last = stamp, pose.copy()
        self.tracks = {role: t for role, t in self.tracks.items()
                       if stamp-t['anchor_time'] <= 4. and
                       np.linalg.norm(pose[:2]-t['anchor_pose']) <= 1.0}
        contours = tape_contours(image)
        candidates = []
        for side in ('left', 'right'):
            line = result.get(side)
            if line is None or len(line.xs) < 8:
                continue
            pts = np.column_stack((line.fs+self.forward_offset, -line.xs))
            if not np.isfinite(pts).all() or np.ptp(pts[:, 0]) < .15:
                continue
            world = pts @ rot.T+pose[:2]
            role = contact_role(geometry.ground_to_image(line.xs, line.fs), contours, image.shape[1])
            matches = []
            overlaps = {}
            for label, track in self.tracks.items():
                d = np.linalg.norm(world[:, None, :]-track['points'][None, :, :], axis=2).min(axis=1)
                overlaps[label] = np.count_nonzero(d < .08) >= max(8, len(d)//3)
                if overlaps[label] and track['hits'] >= 3:
                    matches.append(label)
            if role is not None and matches and role not in matches:
                self.agreement_hits = 0
                return {**out, 'reason': 'tape_odom_role_conflict'}
            direct = role is not None
            if role is None:
                role = matches[0] if len(matches) == 1 else None
            if role is not None:
                candidates.append((role, pts, world, direct, overlaps.get(role, False)))
        # Two current detections cannot both claim the same boundary.
        roles = [c[0] for c in candidates]
        if len(set(roles)) != len(roles):
            self.agreement_hits = 0
            return {**out, 'reason': 'ambiguous_boundary_association'}
        paths = []
        width = float(result.get('lane_width', .52))
        width = width if .42 <= width <= .62 else .52
        for role, pts, world, direct, overlaps in candidates:
            track = self.tracks.get(role)
            continuous = track is not None and overlaps and stamp-track['seen'] <= .25
            hits = track['hits']+1 if continuous else 1
            if direct or track is not None:
                self.tracks[role] = dict(points=world[np.linspace(0, len(world)-1, min(40, len(world))).astype(int)],
                    hits=hits, seen=stamp,
                    anchor_time=stamp if direct else track['anchor_time'],
                    anchor_pose=pose[:2].copy() if direct else track['anchor_pose'])
            if hits < 3:
                continue
            delta = np.gradient(pts, axis=0)
            normal = np.column_stack((-delta[:, 1], delta[:, 0]))
            normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-6)
            center = pts + (1 if role == 'right' else -1)*width/2*normal
            center = center[(center[:, 0] >= .30) & (center[:, 0] <= 1.5)]
            # A noisy offset at a corner may fold a few far points. Use only
            # a contiguous forward run, never sort a folded path into a fake road.
            runs = np.split(center, np.flatnonzero(np.diff(center[:, 0]) <= 0)+1)
            center = max(runs, key=len)
            if len(center) >= 8 and np.ptp(center[:, 0]) >= .15:
                paths.append(center)
        if not paths:
            self.agreement_hits = 0
            return {**out, 'roles': roles}
        if len(paths) == 2:
            near = max(p[0, 0] for p in paths)
            far = min(p[-1, 0] for p in paths)
            if far-near < .15:
                self.agreement_hits = 0
                return {**out, 'reason': 'no_shared_lane_span'}
            x = np.linspace(near, far, 40)
            ys = [np.interp(x, p[:, 0], p[:, 1]) for p in paths]
            if np.max(np.abs(ys[0]-ys[1])) > .12:
                self.agreement_hits = 0
                return {**out, 'reason': 'boundary_width_conflict'}
            path = np.column_stack((x, np.mean(ys, axis=0)))
        else:
            path = paths[0]
        # Geometry only. The existing LiDAR connector makes the final command.
        distance = np.linalg.norm(path, axis=1)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        eligible = np.flatnonzero((distance >= .7) & (arc >= .15))
        target = path[eligible[0] if len(eligible) else -1]
        k = 2*target[1]/max(float(target @ target), 1e-6)
        from jetracer_ros2.utils.ackermann_utils import WHEELBASE, TRACK
        angle = math.copysign(min(.42, math.atan2(WHEELBASE,
                    max(1/max(abs(k), 1e-9)-TRACK/2, 1e-3))), k) if abs(k) > 1e-9 else 0.
        # Hand back to the untouched race controller ONLY when its current
        # path actually agrees; otherwise M8 keeps the corrected connector.
        agree = False
        ordinary = result.get('center')
        if ordinary is not None and result.get('valid'):
            r, f = ordinary
            x = np.asarray(f)+self.forward_offset
            keep = (path[:, 0] >= x.min()) & (path[:, 0] <= x.max())
            if np.count_nonzero(keep) >= 8:
                err = np.abs(np.interp(path[keep, 0], x, -np.asarray(r))-path[keep, 1])
                agree = bool(np.quantile(err, .9) <= .08)
        self.agreement_hits = self.agreement_hits+1 if agree else 0
        return dict(valid=True, aligned=self.agreement_hits >= 3, path=path,
                    angle=float(angle), roles=roles, reason='tape_odom_verified')
