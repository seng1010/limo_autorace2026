"""M8 local gate pursuit. Same deterministic core for ROS and offline replay.

Rear axle: x forward, y left; steering is the inner wheel angle. No ROS I/O.
No full-row fitting, global map, learned cone detector or blind command hold.
"""
from dataclasses import dataclass
import math
import numpy as np
from jetracer_ros2.utils.ackermann_utils import TRACK, WHEELBASE

LOCAL_SWEEP_M = .20
PREVIEW_SWEEP_M = .38
PREVIEW_COMFORT_M = .07
PREVIEW_WEIGHT = 3.0
COARSE_CURVATURE_SAMPLES = 11
REFINED_CURVATURE_SAMPLES = 81
TARGET_CURVATURE_WEIGHT = .08
GATE_ALIGNMENT_WEIGHT = .45
GATE_MAX_X_M = 1.15
ENTRY_GATE_MAX_X_M = 1.40
ENTRY_SEARCH_MAX_M = .40
ENTRY_SEARCH_MAX_SEC = 8.
# Commanded distance stops earlier in the manager to leave room for transport/braking.
RECOVERY_STEP_M = .10

@dataclass(frozen=True)
class Params:
    lidar_x: float = .203
    lidar_y: float = 0.
    lidar_yaw: float = 0.
    half_width: float = .12
    front: float = .29
    rear: float = .09
    cone_padding: float = .015
    clearance: float = .020
    minimum_radius: float = .50
    max_steer_rad: float = .42
    speed: float = .10
    target_hold_sec: float = .50
    target_hold_distance: float = .10
    rear_from_odom_x: float = -.10

def clusters(a, ranges, p):
    phi = a[0] + np.arange(len(ranges)) * a[2] + p.lidar_yaw
    valid = np.isfinite(ranges) & (ranges >= a[5]) & (ranges <= min(a[6], 2.5))
    ix = np.flatnonzero(valid)
    r = ranges[ix]
    points = np.column_stack([r*np.cos(phi[ix]) + p.lidar_x, r*np.sin(phi[ix]) + p.lidar_y])
    if len(points) < 2:
        return points, []
    # Preserve beam order. A gap of one beam may be a hole in a cone.
    jump = np.linalg.norm(np.diff(points, axis=0), axis=1)
    gap = np.diff(ix)
    split = np.flatnonzero((gap > 2) | (jump > .055 + abs(a[2])*r[1:]*gap*1.6)) + 1
    candidates = []
    for q in np.split(points, split):
        extent = np.linalg.norm(np.ptp(q, axis=0))
        if 2 <= len(q) <= 36 and .012 <= extent <= .30:
            candidates.append(q)
    return points, candidates


def rollout(k, length):
    s = np.linspace(0, length, max(25, int(np.ceil(length / .025)) + 1))
    t = k*s
    if abs(k) < 1e-8:
        return np.column_stack([s, np.zeros_like(s), t])
    return np.column_stack([np.sin(t)/k, (1-np.cos(t))/k, t])


def swept_clearance(points, arc, p):
    if not len(points):
        return 0.
    c, s = np.cos(arc[:, 2, None]), np.sin(arc[:, 2, None])
    dx, dy = (points.T[None, :, :] - arc[:, :2, None]).transpose(1, 0, 2)
    x, y = dx*c+dy*s, -dx*s+dy*c
    d = np.hypot(np.maximum(np.maximum(x-p.front, -p.rear-x), 0),
                 np.maximum(abs(y)-p.half_width, 0))
    return float(d.min()) - p.cone_padding


def curvature_limit(p):
    return min(1/p.minimum_radius, 1/(WHEELBASE/np.tan(p.max_steer_rad) + TRACK/2))


def _preview_penalty(points, k, p):
    """Prefer room beyond the authorized arc; never use this to permit motion."""
    if not len(points):
        return 0.
    # Only score observed obstacles on the extension, not free-space evidence.
    # Bounded cost prevents distant geometry from dominating gate selection.
    arc = rollout(k, PREVIEW_SWEEP_M)
    steps = np.linspace(0., PREVIEW_SWEEP_M, len(arc))
    gap = swept_clearance(points, arc[steps > LOCAL_SWEEP_M], p)
    return PREVIEW_WEIGHT * float(np.clip(PREVIEW_COMFORT_M-gap, 0.,
                                          PREVIEW_COMFORT_M+p.cone_padding))


def _gate_alignment(gate):
    """Gate normal is the observed passage direction, not a turn-side latch."""
    normal = gate.get('normal')
    if normal is None:
        return 0., 0.
    angle = math.atan2(normal[1], normal[0])
    # Far pairs are less certain. Alignment gains influence as we approach,
    # without an extra permission state or a maximum-steering hold timer.
    weight = GATE_ALIGNMENT_WEIGHT * min(1., .75/max(np.linalg.norm(gate['gate']), .01))
    return angle, weight


def _alignment_cost(angle, weight, yaw):
    error = math.atan2(math.sin(angle-yaw), math.cos(angle-yaw))
    # Score improvement relative to NOW so selecting a different gate does
    # not win just because its initial direction happened to be straight.
    return weight * (error*error-angle*angle)


def checked_exit_lane(command, points, p, *, scan=None, approach=False):
    """Check lane curvature against raw points; only the M2 approach is short."""
    v, w = command
    if len(points) < 4 or not np.isfinite((v, w)).all() or v < p.speed:
        return 0.0, 0.0
    k = w/v
    horizon = LOCAL_SWEEP_M if approach else .30
    if abs(k) > curvature_limit(p) or swept_clearance(points, rollout(k, horizon), p) < p.clearance:
        return 0.0, 0.0
    if scan is not None and not _prefix_observed(*scan, p, k, horizon=horizon if approach else .12):
        return 0.0, 0.0
    return p.speed, k*p.speed


def approach_clearance_low(command, points, p):
    """Use observed obstacles to request guarded ownership, never to grant motion."""
    v, w = command
    if v < p.speed or not np.isfinite((v, w)).all() or not len(points):
        return False
    # Look beyond the independently checked 20 cm command at race lane speed.
    horizon = min(.35, LOCAL_SWEEP_M + .45*v)
    return swept_clearance(points, rollout(w/v, horizon), p) < p.clearance + .02


def _prefix_observed(a, ranges, p, k, horizon=.12):
    """Require finite clear-space evidence across the commanded front sector."""
    yaw = k*horizon
    x,y = (horizon,0.) if abs(k)<1e-8 else (np.sin(yaw)/k,(1-np.cos(yaw))/k)
    return _footprint_observed(a, ranges, p, x, y, yaw)


def _footprint_observed(a, ranges, p, x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    side = np.array([-p.half_width, p.half_width])
    fx = x + p.front*c - side*s - p.lidar_x
    fy = y + p.front*s + side*c - p.lidar_y
    if np.any(fx <= 0.0):
        return False
    limits = np.sort(np.arctan2(fy, fx))
    angles = a[0] + np.arange(len(ranges))*a[2] + p.lidar_yaw
    edges = np.linspace(*limits, 4)
    # A missing/invalid beam is not evidence of usable free space. This is
    # coarse coverage, in addition to—not instead of—the raw footprint sweep.
    distance = max(a[5], float(np.max(np.hypot(fx, fy))) + p.clearance)
    observed = np.isfinite(ranges) & (ranges >= distance) & (ranges <= a[6])
    return all(np.any(observed & (angles >= lo) & (angles <= hi))
               for lo, hi in zip(edges[:-1], edges[1:]))


def _forward_recovery(scan, points, p, target, nominal_k, *, horizon,
                      min_progress, lane_path=None, lane_angle=None, blocked_horizon=None):
    """Return first curvature only; the second leg is evidence, never a command."""
    limit = curvature_limit(p)
    if (len(points) < 4 or abs(nominal_k) > limit
            or swept_clearance(points, np.zeros((1,3)), p) < p.clearance):
        return None
    if swept_clearance(points, rollout(nominal_k, blocked_horizon or horizon), p) >= p.clearance:
        return None  # Not a clearance-blocked target.
    distance = np.linalg.norm(target)
    direction = float(np.sign(nominal_k))
    first_choices = [0., -.5*direction*limit, -direction*limit] if abs(nominal_k) > .03 else [
        0., -.5*limit, .5*limit, -limit, limit]
    best = None
    for k1 in first_choices:
        first = rollout(k1, max(LOCAL_SWEEP_M, horizon))
        gap1 = swept_clearance(points, first, p)
        if (gap1 < p.clearance
                or not _footprint_observed(*scan, p, *first[-1])):
            continue
        for distance1 in (.06, RECOVERY_STEP_M):
            # The stop reserve means the realized first leg need not be exactly 10 cm.
            offset = rollout(k1, distance1)[-1]
            rot = _rotation(offset[2])
            x, y = (target-offset[:2]) @ rot
            if x <= .08:
                break
            desired = float(np.clip(2*y/(x*x+y*y),-limit,limit))
            # Rejoining the same geometry is required, not the old steering sign
            # measured from the pre-recovery pose.
            options = np.unique([desired, nominal_k, *np.linspace(-limit, limit, 15)])
            candidates = []
            for k2 in options:
                local = rollout(k2, horizon)
                second = np.column_stack((local[:,:2] @ rot.T + offset[:2], local[:,2] + offset[2]))
                end = second[-1]
                progress = distance - np.linalg.norm(target-end[:2])
                if progress < min_progress:
                    continue
                if lane_path is not None and (np.linalg.norm(lane_path-end[:2], axis=1).min() > .42
                        or abs(end[2]-lane_angle) > math.radians(44)):
                    continue
                gap2 = swept_clearance(points, second, p)
                if (gap2 >= p.clearance and _footprint_observed(*scan, p, *second[len(second)//2])
                        and _footprint_observed(*scan, p, *end)):
                    candidates.append((progress + .35*min(gap2, .10), gap2))
            if not candidates:
                break
            if distance1 == .06:
                near_score = max(candidates)[0]
            else:
                score = min(near_score, max(candidates)[0]) + .35*min(gap1, .10) - .015*abs(k1)
                if best is None or score > best[0]:
                    best = score, k1
    return None if best is None else float(best[1])


def lane_forward_recovery(scan, points, p, center, command, *, approach=False):
    v, w = command
    path = np.asarray(center, dtype=float)
    if (not np.isfinite(command).all() or v < p.speed or path.ndim != 2
            or path.shape[1] != 2 or not np.isfinite(path).all()):
        return None
    path = path[(path[:,0] >= .30) & (path[:,0] <= 1.30)]
    if len(path) < 8:
        return None
    delta = path[min(15,len(path)-1)] - path[0]
    angle = math.atan2(delta[1], delta[0])
    if np.linalg.norm(delta) < .05 or abs(angle) > math.radians(65):
        return None
    target = path[np.argmin(np.linalg.norm(path, axis=1))]
    return _forward_recovery(scan, points, p, target, w/v,
        horizon=LOCAL_SWEEP_M if approach else .40, min_progress=.07 if approach else .14,
        lane_path=path, lane_angle=angle, blocked_horizon=LOCAL_SWEEP_M if approach else .30)


def safe_lane_arc(scan, points, p, measured_center, nominal_command):
    """Find an executable nearby yaw rate that makes progress toward the live lane."""
    if measured_center is None or not np.isfinite(nominal_command).all():
        return 0., 0.
    path = np.asarray(measured_center, float)
    if path.ndim != 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        return 0., 0.
    path = path[(path[:,0] >= .30) & (path[:,0] <= 1.30)]
    if len(path) < 8:
        return 0.,0.
    kmax = curvature_limit(p)
    axis = path[:5].mean(axis=0)
    # Test progress to actually observed camera geometry, not to the last
    # remembered center. For this connector a lateral correction alone is
    # not enough if the endpoint's heading disagrees strongly with the lane.
    target = path[np.argmin(np.linalg.norm(path,axis=1))]
    dxdy = path[min(15,len(path)-1)]-path[0]
    angle = math.atan2(dxdy[1], dxdy[0])
    if axis[0] < .26 or abs(angle) > math.radians(65):
        return 0.,0.
    candidates = np.linspace(-kmax, kmax, 15)
    direction = np.sign(nominal_command[1])
    score = None
    for k in candidates:
        if direction and k*direction < -.03:
            continue
        arc = rollout(k, .40)
        dist = np.linalg.norm(path-arc[-1,:2],axis=1).min()
        progress = np.linalg.norm(target)-np.linalg.norm(target-arc[-1,:2])
        if progress < .14 or dist > .42 or abs(arc[-1,2]-angle) > math.radians(44):
            continue
        # A few or no close returns can mean open space, not scan loss.
        # Require finite supporting rays in the command direction at both
        # horizons before interpreting the lack of near returns as clear.
        clear = swept_clearance(points,arc,p) if len(points) else float('inf')
        if (clear < p.clearance or not _prefix_observed(*scan,p,k)
                or not _prefix_observed(*scan,p,k,horizon=.40)):
            continue
        val = -dist-.15*abs(arc[-1,2]-angle)+.35*min(clear,.10)
        if score is None or val > score[0]:
            score = val, k
    return (p.speed, p.speed*score[1]) if score is not None else (0.,0.)


def _choose(result, best):
    safe, _, clear, k, arc = best
    result.update(arc=arc, min_clearance=clear)
    if not safe:
        result['state'] = 'path_blocked'
        return result
    steer = np.sign(k)*np.arctan(WHEELBASE / (1/max(abs(k), 1e-9) - TRACK/2))
    result.update(state='tracking', steer=float(steer), curvature=float(k))
    return result


class ResumeGate:
    """Stop immediately, resume only after consecutive verified scans."""
    def __init__(self, required):
        self.required = required
        self.count = 0
        self.mouth = None

    def update(self, safe, mouth=None):
        # While confirming, require two scans to agree on the local corridor.
        # Do not delay a stop on the strength of past observations.
        mouth = np.asarray(mouth) if safe and mouth is not None else None
        if (safe and self.count < self.required and mouth is not None
                and self.mouth is not None and np.linalg.norm(mouth-self.mouth) > .18):
            self.count = 0
        self.count = min(self.count + 1, self.required) if safe else 0
        self.mouth = mouth
        return safe and self.count >= self.required



def _rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class EntrySearchWindow:
    """One odom-bounded window after cone ownership; planner resets do not reopen it."""
    def __init__(self):
        self.set_phase('off')

    def set_phase(self, phase):
        self.open = phase == 'cone'
        self.started = None
        self.last_xy = None
        self.travel = 0.

    def allows(self, stamp, pose):
        if not self.open:
            return False
        if self.started is None:
            self.started = stamp
        if not 0. <= stamp-self.started < ENTRY_SEARCH_MAX_SEC:
            self.open = False
            return False
        if pose is None or not np.isfinite(pose).all():
            # Once armed, a gap in odometry could hide travel; do not rearm.
            if self.last_xy is not None:
                self.open = False
            return False
        xy = np.asarray(pose[:2])
        if self.last_xy is not None:
            step = math.hypot(*(xy-self.last_xy))
            self.travel += step
            if self.travel >= ENTRY_SEARCH_MAX_M or step > .12:
                self.open = False
                return False
        self.last_xy = xy
        return True


def _gates(groups, points, p, *, far_gates=None):
    """Observed cross-track pairs, not independent left/right line fits."""
    ids = [i for i,g in enumerate(groups) if -.08 < np.median(g[:,0]) < 1.50]
    # Bound pairing work, never discard raw returns from collision checks.
    ids = sorted(ids,key=lambda i:np.linalg.norm(np.median(groups[i],axis=0)))[:24]
    centers = {i: np.median(groups[i],axis=0) for i in ids}
    margin = p.half_width+p.cone_padding+p.clearance
    max_x = GATE_MAX_X_M if far_gates is None else ENTRY_GATE_MAX_X_M
    choices = []
    for ai,i in enumerate(ids):
        for j in ids[ai+1:]:
            l, r = centers[i], centers[j]
            if l[1] < r[1]:
                l, r, i1, j1 = r, l, j, i
            else:
                i1, j1 = i, j
            across = l-r
            distance = np.linalg.norm(across)
            if not 2*margin <= distance <= 1.05:
                continue
            across /= distance
            forward = np.array([across[1], -across[0]])
            if forward[0] < .50:  # Reject almost longitudinal pairs from one row.
                continue
            left = float((groups[i1]@across).min())
            right = float((groups[j1]@across).max())
            if left-right < 2*margin:
                continue
            station = float(((l+r)/2)@forward)
            middle = (left+right)/2
            gate = station*forward+middle*across
            x,y = gate
            if not .12 <= x <= max_x or abs(y) > .95*x:
                continue
            # Never connect two cones over a third obstacle in the opening.
            inside = (abs(points@forward-station) < .10) & (abs(points@across-middle) < margin)
            if inside.any():
                continue
            angle = math.atan2(y,x)
            normal_angle=math.atan2(forward[1],forward[0])
            # A circular approach reaches the gate at approximately 2*bearing.
            # Penalize diagonal cross-row aliases, not the turn itself.
            cost=(.60*abs(np.linalg.norm(gate)-.55)+.15*abs(angle)
                  +.35*abs(normal_angle-2*angle)+.45*abs(distance-.62))
            # Advance toward the next gate once the bumper reaches this one.
            cost += .22 if x < p.front+.05 else 0.
            candidate = dict(gate=gate, normal=forward, ends=np.array([l,r]), cost=cost)
            if x <= GATE_MAX_X_M:
                choices.append(candidate)
            else:
                far_gates.append(candidate)
    return sorted(choices,key=lambda g:g['cost'])[:8]


class Planner:
    """Keep a measured goal briefly in odom, but recheck today's raw scan.

    History is a coordinate-corrected target, not permission to reuse a command.
    Without fresh odometry only fresh pairs can be used. Prediction never renews
    its own age. Resets and reverse/manual repositioning clear the history.
    """
    def __init__(self, config=None):
        self.p = config or Params()
        self.reset()

    def reset(self):
        self.memory = None
        self.previous_curvature = None
        self.last_pose = None
        self.last_time = None

    def _pose(self, pose):
        if pose is None or len(pose) != 3 or not np.isfinite(pose).all():
            return None
        pose = np.array(pose,dtype=float)
        pose[:2] += _rotation(pose[2])@np.array([self.p.rear_from_odom_x,0.])
        return pose

    def update(self, a, ranges, stamp, odom_pose=None, *, entry_far_search=False, recovery=False):
        p = self.p
        pose = self._pose(odom_pose)
        if not math.isfinite(stamp) or (self.last_time is not None and
                not 0 < stamp-self.last_time <= .30):
            self.reset()
        if pose is not None and self.last_pose is not None:
            delta = (pose[:2]-self.last_pose[:2])@_rotation(self.last_pose[2])
            yaw_delta = math.atan2(math.sin(pose[2]-self.last_pose[2]),math.cos(pose[2]-self.last_pose[2]))
            if np.linalg.norm(delta) > .12 or abs(yaw_delta) > .35 or delta[0] < -.015:
                self.reset()
        self.last_time, self.last_pose = stamp, pose
        points, groups = clusters(a,ranges,p)
        result = dict(points=points, groups=groups, road=None, arc=None, state='no_gate',
                      min_clearance=None, steer=None, speed=0., mode=None)
        measured=np.isfinite(ranges)&(ranges>=a[5])&(ranges<=a[6])
        if np.count_nonzero(measured) < 4:
            return {**result,'state':'insufficient_scan'}
        far_gates = []
        gates = (_gates(groups, points, p, far_gates=far_gates) if entry_far_search
                 else _gates(groups, points, p))
        predicted = None
        if self.memory is not None and pose is not None:
            m = self.memory
            age, moved = stamp-m['stamp'], np.linalg.norm(pose[:2]-m['pose'][:2])
            if 0 <= age <= p.target_hold_sec and moved <= p.target_hold_distance:
                rot = _rotation(pose[2])
                target = (m['world_gate']-pose[:2])@rot
                if target[0] > .12:
                    predicted = dict(gate=target,normal=m['world_normal']@rot,
                                     ends=(m['world_ends']-pose[:2])@rot,cost=.12,predicted=True)
            else:
                self.memory = None
        if predicted is not None:
            # Association is a score, not an extra hard permission gate.
            for g in gates:
                g['cost'] += .20*min(np.linalg.norm(g['gate']-predicted['gate']),.5)
            if not gates:
                gates = [predicted]
                memory_points = (self.memory['world_points']-pose[:2])@_rotation(pose[2])
                points = np.vstack((points,memory_points))
                result['points'] = points
        if len(points) and swept_clearance(points,np.zeros((1,3)),p) < p.clearance:
            return {**result,'state':'footprint_blocked'}
        using_far = not gates and bool(far_gates)
        if using_far:
            gates = sorted(far_gates, key=lambda g:g['cost'])[:8]
            result['entry_far_gate'] = True
        if not gates:
            return result
        result['road'] = gates[0]
        limit = curvature_limit(p)
        best = None
        checked = {}
        preview = {}
        # The rear axle travels at most PREVIEW_SWEEP_M; points beyond this
        # enclosing radius cannot contribute to the bounded clearance cost.
        radius = (PREVIEW_SWEEP_M + math.hypot(max(p.front, p.rear), p.half_width)
                  + p.cone_padding + PREVIEW_COMFORT_M)
        preview_points = points[np.sum(points*points, axis=1) <= radius*radius]

        def search(grid):
            for g in gates:
                x,y = g['gate']
                desired = float(np.clip(2*y/(x*x+y*y),-limit,limit))
                angle, weight = _gate_alignment(g)
                # Optimum of the two local quadratic terms; includes an
                # intermediate steering target, not just coarse/max angles.
                aligned = ((TARGET_CURVATURE_WEIGHT*desired + weight*LOCAL_SWEEP_M*angle)
                           / (TARGET_CURVATURE_WEIGHT + weight*LOCAL_SWEEP_M**2))
                ks = np.unique(np.clip(np.r_[grid, desired, aligned],-limit,limit))
                for k in ks:
                    consider(g, k, desired, angle, weight)

        def consider(g, k, desired, angle, weight):
            nonlocal best
            key=round(float(k),10)
            if key not in checked:
                arc=rollout(k,LOCAL_SWEEP_M)
                clearance=swept_clearance(points,arc,p)
                safe=(clearance>=p.clearance and
                      _prefix_observed(a,ranges,p,k,horizon=LOCAL_SWEEP_M))
                checked[key]=arc,clearance,safe
            arc,clearance,safe=checked[key]
            if not safe:
                return
            progress = np.linalg.norm(g['gate'])-np.linalg.norm(g['gate']-arc[-1,:2])
            if progress < .07:
                return
            if key not in preview:
                preview[key] = _preview_penalty(preview_points, k, p)
            cost = (g['cost'] + TARGET_CURVATURE_WEIGHT*(k-desired)**2 + preview[key]
                    + _alignment_cost(angle, weight, arc[-1,2]))
            if self.previous_curvature is not None:
                cost += .025*abs(k-self.previous_curvature)
            if best is None or cost < best[0]:
                best = cost,g,clearance,k,arc
        # Only forward 20 cm arcs. Cached tests are shared across gates and
        # both passes. Refine ONLY when the cheap pass has no executable arc.
        search(np.linspace(-limit,limit,COARSE_CURVATURE_SAMPLES))
        result['search_stage'] = 'coarse'
        if best is None:
            search(np.linspace(-limit,limit,REFINED_CURVATURE_SAMPLES))
            result['search_stage'] = 'refined'
        result['candidate_count'] = len(checked)
        if best is None:
            if not using_far:
                self.previous_curvature = None
            if recovery and pose is not None and not gates[0].get('predicted'):
                target = gates[0]['gate']
                x, y = target
                desired = float(np.clip(2*y/(x*x+y*y), -limit, limit))
                result['recovery_curvature'] = _forward_recovery(
                    (a, ranges), points, p, target, desired, horizon=LOCAL_SWEEP_M, min_progress=.07)
            return {**result,'state':'path_blocked'}
        _,g,clear,k,arc = best
        # An entry-only candidate may not leak into normal temporal tracking.
        if not using_far:
            self.previous_curvature = k
        if pose is not None and not using_far and not g.get('predicted'):
            rot = _rotation(pose[2])
            self.memory = dict(stamp=stamp,pose=pose.copy(),
                world_gate=g['gate']@rot.T+pose[:2],world_normal=g['normal']@rot.T,
                world_ends=g['ends']@rot.T+pose[:2],
                world_points=points[np.linalg.norm(points,axis=1)<1.5]@rot.T+pose[:2])
        result.update(road=g,mode='tracked_gate' if g.get('predicted') else 'local_gate',speed=p.speed)
        return _choose(result,(True,-best[0],clear,k,arc))


def plan_m8(a,ranges,p,previous_curvature=None):
    """Stateless compatibility entry point; live node uses Planner.update."""
    planner = Planner(p)
    planner.previous_curvature = previous_curvature
    return planner.update(a,ranges,0.)
