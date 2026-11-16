"""M7-only, ROS-independent time alignment, road hypotheses and depth input.

All geometry uses metres: odom x/y and rear-axle base x forward/y left/z up.
Camera line offsets are normal distances from the camera ground projection.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from . import three_lane as T


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PoseHistory:
    """Interpolate acquisition times; never silently replace them by latest pose."""
    def __init__(self, size=300, extrapolation=.035, max_gap=.10):
        self.samples = deque(maxlen=size)
        self.extrapolation, self.max_gap = extrapolation, max_gap
        self.rejected = 0

    def add(self, stamp, pose):
        if not np.isfinite((stamp, *pose)).all() or stamp <= 0:
            return False
        if self.samples and stamp <= self.samples[-1][0]:
            self.rejected += 1
            return False
        x, y, yaw = pose
        if self.samples:
            yaw = self.samples[-1][3] + wrap(yaw-self.samples[-1][3])
        self.samples.append((float(stamp), float(x), float(y), float(yaw)))
        return True

    def at(self, stamp):
        if not self.samples or not math.isfinite(stamp):
            return None
        a = np.asarray(self.samples)
        if stamp < a[0, 0] or stamp > a[-1, 0] + self.extrapolation:
            return None
        if stamp >= a[-1, 0]:
            # Bounded hold, not unbounded extrapolation through missing odom.
            return tuple(a[-1, 1:])
        i = int(np.searchsorted(a[:, 0], stamp, side='right'))
        left, right = a[i-1], a[i]
        if right[0]-left[0] > self.max_gap:
            return None
        w = (stamp-left[0])/(right[0]-left[0])
        return tuple(left[1:] + w*(right[1:]-left[1:]))

    def rays(self, stamp, increment, count, scan_time):
        """Opt-in only: caller must verify driver array order/time semantics."""
        duration = increment*max(0, count-1)
        if count < 1 or increment <= 0 or duration > scan_time*1.05 or scan_time <= 0:
            return None
        if not self.samples:
            return None
        a = np.asarray(self.samples)
        times = stamp+np.arange(count)*increment
        if times[0]<a[0,0] or times[-1]>a[-1,0]+self.extrapolation:
            return None
        right = np.clip(np.searchsorted(a[:,0],times,side='right'),1,len(a)-1)
        if len(a)>1 and np.any((a[right,0]-a[right-1,0]>self.max_gap)&(times<a[-1,0])):
            return None
        return np.column_stack([np.interp(times,a[:,0],a[:,j]) for j in (1,2,3)])


def fresh(stamp, ros_now, received, monotonic_now, limit):
    """Both clock domains must agree; queued old messages do not become fresh."""
    return (all(math.isfinite(v) for v in (stamp, ros_now, received, monotonic_now))
            and stamp > 0 and -.035 <= ros_now-stamp <= limit
            and 0 <= monotonic_now-received <= limit)


def move_observation(obs, image_pose, current_pose, camera_offset=.184):
    """Express a fresh image's road observation at the current control pose."""
    if obs is None:
        return None
    out = dict(obs)
    p,q = np.asarray(image_pose),np.asarray(current_pose)
    a = p[:2]+camera_offset*np.array([math.cos(p[2]),math.sin(p[2])])
    b = q[:2]+camera_offset*np.array([math.cos(q[2]),math.sin(q[2])])
    yaw = p[2]+obs['th']
    delta = b-a
    lateral = -delta[0]*math.sin(yaw)+delta[1]*math.cos(yaw)
    along = delta[0]*math.cos(yaw)+delta[1]*math.sin(yaw)
    out['th'] = wrap(yaw-q[2])
    def encode(u):
        return u if abs(u)<=T.FAR_U else math.copysign(T.FAR_U+(abs(u)-T.FAR_U)*T.FAR_SHRINK,u)
    key = {u:encode(T._shrink(u)-lateral) for u,_,_ in obs.get('lines',())}
    out['lines'] = [(key[u],n,r) for u,n,r in obs.get('lines',())]
    out['short'] = {key[u] for u in obs.get('short',()) if u in key}
    out['ystart'] = {key[u]:distance-along for u,distance in (obs.get('ystart') or {}).items() if u in key}
    # Island fitting needs the original colour pixels in the current camera
    # ground frame, not stale pixels mixed with a new road heading.
    if obs.get('yellow_xy') is not None:
        xy = np.asarray(obs['yellow_xy'])
        world = a+np.column_stack((xy[:,1]*math.cos(p[2])+xy[:,0]*math.sin(p[2]),
                                  xy[:,1]*math.sin(p[2])-xy[:,0]*math.cos(p[2])))
        d = world-b
        out['yellow_xy'] = np.column_stack((d[:,0]*math.sin(q[2])-d[:,1]*math.cos(q[2]),
                                           d[:,0]*math.cos(q[2])+d[:,1]*math.sin(q[2])))
    return out


@dataclass(frozen=True)
class RoadHypothesis:
    c: float
    heading: float
    support: int
    residual: float
    confidence: float
    matches: tuple


def road_hypotheses(obs, prior=0., beside_island=False, min_points=30, road_half=.80):
    """Joint colour/spacing assignments, with one observation per boundary.

    A prior is a weak tie-breaker, never a reason to discard a consistent
    yellow+white pair on the adjacent lane. Ambiguous single lines stay weak.
    """
    if not obs or not math.isfinite(obs.get('th', math.nan)) or abs(obs['th']) > math.radians(65):
        return []
    lines = []
    for u, n, ratio in obs.get('lines', ()):
        if not np.isfinite((u, n, ratio)).all() or n < min_points or abs(u) > 1.1:
            continue
        if u in obs.get('short', ()):
            continue
        if ratio >= .7:
            if (obs.get('ystart') or {}).get(u, 0.) > .60:
                continue
            spots = (-road_half, road_half) + ((-.27, .27) if beside_island else ())
        elif ratio <= .3:
            spots = (-.27, .27)
        else:
            continue
        lines.append((T._shrink(u), min(n, 200), spots, ratio))
    hypotheses = []
    for u, _, spots, _ in lines:
        for boundary in spots:
            c0 = boundary-u
            if abs(c0) > .79:
                continue
            matched = {}
            for j, (v, weight, choices, ratio) in enumerate(lines):
                target = min(choices, key=lambda b: abs(c0+v-b))
                error = abs(c0+v-target)
                if error <= .10 and (target not in matched or error < matched[target][0]):
                    matched[target] = (error, target-v, weight, j)
            if not matched:
                continue
            pairs = list(matched.values())
            c = float(np.average([p[1] for p in pairs], weights=[p[2] for p in pairs]))
            residual = float(np.average([abs(c-p[1]) for p in pairs], weights=[p[2] for p in pairs]))
            score = len(pairs)*2. + sum(p[2] for p in pairs)/400. - residual*12.
            score -= min(abs(c-prior), .6)*.35
            if any(abs(c-h[1]) < .06 for h in hypotheses):
                continue
            hypotheses.append((score, c, residual, tuple((float(b), int(p[3])) for b,p in matched.items())))
    hypotheses.sort(reverse=True)
    result = []
    for i, (score, c, residual, matches) in enumerate(hypotheses):
        other = max((h[0] for h in hypotheses if abs(h[1]-c) > .2), default=-math.inf)
        unique = score-other > .65
        confidence = min(1., len(matches)/2.) * max(0., 1.-residual/.10)
        if not unique:
            confidence *= .35
        result.append(RoadHypothesis(c, float(obs['th']), len(matches), residual, confidence, matches))
    return result


class RoadFilter:
    def __init__(self):
        self.last_stamp = None
        self.pending = None
        self.pending_since = None
        self.last = None
        self.reason = 'no joint lane observation'

    def update(self, obs, c, heading, stamp, beside_island=False, road_half=.80):
        if obs is None:
            self.reason = 'waiting for a new road observation'
            return c,heading,False,False
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return c, heading, False, False
        if self.last_stamp is not None and stamp-self.last_stamp>.5:
            self.pending = self.pending_since = None
        dt = .1 if self.last_stamp is None else min(.25, stamp-self.last_stamp)
        self.last_stamp = stamp
        options = road_hypotheses(obs, c, beside_island,road_half=road_half)
        self.last = options[0] if options else None
        if not options:
            self.pending = self.pending_since = None
            self.reason = 'no consistent road geometry'
            return c, heading, False, False
        best = options[0]
        dh = wrap(best.heading-heading)
        # One line can maintain a previously anchored pose, but cannot choose
        # a new lane or justify a large re-localisation. Prefer agreement with
        # the predicted boundary, not the strongest isolated colour blob.
        close = [h for h in options if h.support==1 and abs(h.c-c)<.08
                 and abs(wrap(h.heading-heading))<math.radians(8)]
        if best.support==1 and close:
            weak = min(close,key=lambda h:abs(h.c-c))
            alpha = 1.-math.exp(-dt/.5)
            self.reason = 'single boundary maintaining an anchored pose'
            self.last = weak
            return c+alpha*(weak.c-c),heading+alpha*wrap(weak.heading-heading),True,False
        if best.confidence < .55 or best.support < 2 or abs(dh) > math.radians(25):
            self.pending = self.pending_since = None
            self.reason = 'ambiguous lines or implausible heading'
            return c, heading, False, False
        large = abs(best.c-c) > .20 or abs(dh) > math.radians(10)
        if large:
            consistent = (self.pending is not None and abs(best.c-self.pending[0]) < .08
                          and abs(wrap(best.heading-self.pending[1])) < math.radians(5))
            self.pending_since = self.pending_since if consistent else stamp
            self.pending = (best.c, best.heading)
            if stamp-self.pending_since < .25:
                self.reason = 'confirming adjacent-lane correction; stop to re-anchor'
                return c, heading, False, True
        else:
            self.pending = self.pending_since = None
        alpha = 1.-math.exp(-dt/.22)
        # A large confirmed correction is applied only by the stopped supervisor.
        nc = best.c if large else c + alpha*(best.c-c)
        nh = best.heading if large else heading + alpha*dh
        self.reason = 'joint white/yellow road geometry'
        return nc, nh, True, large


class DepthProjector:
    """Optional occupied-only depth layer; no fabricated free-space clearing.

    T_base_optical must be measured for THIS robot, relative to the same rear
    axle used by odom. No nominal/default mounting transform is supplied.
    """
    def __init__(self, transform, stride=4, near=.3, far=3., min_height=.04, max_height=.65):
        self.transform = np.asarray(transform, dtype=float)
        if self.transform.shape != (4,4) or not np.isfinite(self.transform).all():
            raise ValueError('depth transform must be a finite 4x4 matrix')
        r = self.transform[:3,:3]
        if not np.allclose(r.T@r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1., atol=1e-4):
            raise ValueError('depth transform must contain a proper rotation')
        if not np.allclose(self.transform[3], [0,0,0,1]) or stride < 1:
            raise ValueError('invalid homogeneous depth transform or stride')
        self.stride, self.near, self.far = int(stride), near, far
        self.min_height, self.max_height = min_height, max_height

    def project(self, depth, k, pose, scale=1.):
        depth, k = np.asarray(depth), np.asarray(k, dtype=float).reshape(3,3)
        if depth.ndim != 2 or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0:
            raise ValueError('valid depth image and its own CameraInfo are required')
        if not np.isfinite(pose).all() or scale <= 0 or not math.isfinite(scale):
            raise ValueError('invalid depth pose or unit scale')
        v,u = np.mgrid[0:depth.shape[0]:self.stride, 0:depth.shape[1]:self.stride]
        z = depth[::self.stride,::self.stride].astype(float)*scale
        keep = np.isfinite(z) & (z >= self.near) & (z <= self.far)
        z,u,v = z[keep],u[keep],v[keep]
        optical = np.column_stack(((u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z))
        base = optical@self.transform[:3,:3].T + self.transform[:3,3]
        base = base[(base[:,2] >= self.min_height) & (base[:,2] <= self.max_height)]
        c,s = math.cos(pose[2]), math.sin(pose[2])
        return np.column_stack((pose[0]+base[:,0]*c-base[:,1]*s,
                                pose[1]+base[:,0]*s+base[:,1]*c))
