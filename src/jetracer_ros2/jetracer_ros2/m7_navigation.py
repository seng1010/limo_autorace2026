"""M7 visibility mapping and receding-horizon, footprint-checked commands.

The map is in one odometry frame, not in the camera-corrected road frame.
Only rays before the first return are free. Missing/occluded cells are unknown.
Plans may look beyond visibility, but motion is limited to the observed prefix.
No ROS imports: the same implementation is used by the node and simulator.
"""
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

try:
    from jetracer_ros2 import three_lane as T
    from jetracer_ros2 import three_lane_simple as S
except ImportError:
    import three_lane as T
    import three_lane_simple as S

CONTROLLER_VERSION = 'm7-mapped-20261116-entry-recovery'


class EntryLineFitter(T.LineFitter):
    """The flaring yellow entrance must not outvote the two white dashes."""
    def masks(self, image, tune=None):
        yellow, white = super().masks(image, tune)
        self.yellow = np.zeros_like(yellow)
        return self.yellow, white


def central_white_pair(obs, dark_fraction, heading_limit=math.radians(12), min_points=40):
    """Require a dark road and two independent white boundaries around us.

    Never turn a far-away/sideways road fit into an entry by clamping its pose.
    This gate is for the central entrance and stopped central-lane recovery.
    """
    if not obs or dark_fraction < .55 or obs.get('n',0)<120 or abs(obs['th'])>heading_limit:
        return None
    white = [(T._shrink(u),n) for u,n,ratio in obs['lines'] if ratio<=.3 and n>=min_points]
    pairs = []
    for left,nl in white:
        for right,nr in white:
            width = left-right
            c = -(left+right)/2
            if right<-.06 and left>.06 and .44<=width<=.68 and abs(c)<=.18:
                pairs.append((min(nl,nr)-100*abs(width-.54),c))
    if not pairs:
        return None
    return max(pairs)[1],float(obs['th'])


@dataclass(frozen=True)
class NavigationConfig:
    resolution: float = 0.05
    scan_range: float = 3.3
    free_age: float = 15.0
    body_length: float = 0.32
    body_width: float = 0.22
    body_center: float = 0.10       # rear axle -> body centre
    margin: float = 0.035
    road_half: float = 0.80
    speed: float = 0.18
    reverse_speed: float = 0.08
    max_steer: float = math.radians(24.0)
    horizon: float = 2.8
    step: float = 0.05
    reaction_sec: float = 0.50
    deceleration: float = 0.35
    stop_buffer: float = 0.06
    wait_before_reverse: float = 1.5
    reverse_distance: float = 0.25
    reverse_timeout: float = 4.0
    reverse_attempts: int = 3
    blind_distance: float = 1.2


class VisibilityMap:
    """Sparse occupied/free cells. Absence of evidence remains unknown.

    Occupied evidence persists while occluded; two different clear scans are
    required to remove a hit. Scan IDs prevent repeated camera callbacks from
    fabricating independent lidar observations.
    """
    def __init__(self, config=NavigationConfig()):
        self.cfg = config
        self.free = {}
        self.hits = {}
        self.clears = {}
        self.last_scan_id = None
        self.points = np.empty((0, 2))

    def keys(self, points):
        return np.floor(np.asarray(points) / self.cfg.resolution).astype(np.int64)

    def observe(self, scan, pose, now):
        if scan is None or pose is None:
            return False
        ranges = np.asarray(scan[0], dtype=float)
        if ranges.size == 0 or scan[2] == 0 or not np.isfinite(scan[1:3]).all():
            return False
        sensor_max = scan[6] if len(scan)>6 else self.cfg.scan_range
        if not math.isfinite(scan[3]) or not math.isfinite(sensor_max) or sensor_max<=max(.05,scan[3]):
            return False
        valid = ((np.isfinite(ranges) & (ranges >= max(0.05, scan[3])) & (ranges<=sensor_max))
                 | np.isposinf(ranges))
        if np.count_nonzero(valid) < max(3, ranges.size // 10):
            return False
        scan_id = scan[5] if len(scan) > 5 else None
        if scan_id is not None and scan_id == self.last_scan_id:
            return True
        pose = scan[4] if len(scan) > 4 and scan[4] is not None else pose
        if not np.isfinite(pose).all():
            return False
        x, y, yaw = pose
        offset = S.CAM_FROM_ODOM + S.LIDAR_AHEAD
        origin = np.array([x + offset * math.cos(yaw), y + offset * math.sin(yaw)])
        angles = scan[1] + np.arange(ranges.size)[valid] * scan[2] + yaw
        measured = ranges[valid]
        max_range = min(self.cfg.scan_range,sensor_max)
        limits = np.minimum(measured, max_range)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        hit = np.isfinite(measured) & (measured <= max_range)
        ends = origin + directions[hit] * limits[hit, None]
        occupied = {tuple(k) for k in self.keys(ends)}
        distances = np.arange(0.0, self.cfg.scan_range, self.cfg.resolution * 0.5)
        # Leave the endpoint cell (and its quantisation uncertainty) un-cleared.
        stop = limits - np.where(hit, self.cfg.resolution * 1.5, 0.0)
        mask = (distances[None, :] < stop[:, None]) & (distances[None,:]>=max(0.,scan[3]))
        rays = origin + directions[:, None, :] * distances[None, :, None]
        cleared = {tuple(k) for k in np.unique(self.keys(rays[mask]),axis=0)} - occupied
        for k in cleared:
            self.free[k] = now
            if k in self.hits:
                self.clears[k] = self.clears.get(k, 0) + 1
                if self.clears[k] >= 2:
                    self.hits.pop(k, None)
                    self.clears.pop(k, None)
        for k in occupied:
            self.hits[k] = now
            self.free.pop(k, None)
            self.clears.pop(k, None)
        # Bound memory to the local course; do not erase occluded nearby hits.
        radius = 8.0 / self.cfg.resolution
        centre = self.keys(origin)
        self.hits = {k: t for k, t in self.hits.items()
                     if max(abs(k[0] - centre[0]), abs(k[1] - centre[1])) < radius}
        self.free = {k: t for k, t in self.free.items()
                     if now - t <= self.cfg.free_age and
                     max(abs(k[0] - centre[0]), abs(k[1] - centre[1])) < radius}
        self.points = ((np.asarray(list(self.hits), dtype=float) + 0.5) * self.cfg.resolution
                       if self.hits else np.empty((0, 2)))
        self.last_scan_id = scan_id
        return True

    def mark_footprint(self, pose, now):
        """The robot's own occupied footprint is known traversable, not its rear."""
        x, y, yaw = pose
        xs = np.arange(-0.05, 0.251, self.cfg.resolution / 2)
        ys = np.arange(-0.10, 0.101, self.cfg.resolution / 2)
        a, b = np.meshgrid(xs, ys)
        points = np.column_stack((x + a.ravel() * math.cos(yaw) - b.ravel() * math.sin(yaw),
                                  y + a.ravel() * math.sin(yaw) + b.ravel() * math.cos(yaw)))
        for k in map(tuple, self.keys(points)):
            if k not in self.hits:
                self.free[k] = now

    def known(self, points, now):
        shape = np.asarray(points).shape[:-1]
        keys = self.keys(np.asarray(points).reshape(-1, 2))
        return np.array([k not in self.hits and now - self.free.get(k, -1e9) <= self.cfg.free_age
                         for k in map(tuple, keys)]).reshape(shape)


class FootprintPlanner:
    def __init__(self, config=NavigationConfig()):
        self.cfg = config
        self.target = 1
        self.previous_steer = 0.0
        self.candidates = []
        self.selected = None
        self.merge_distance = math.inf
        self.merge_look = .50

    def rollout(self, s, c, th, first, second=None, delay=0.0, reverse=False):
        """Rear-axle bicycle rollout including steering actuator lag."""
        cfg = self.cfg
        psi = -th
        x, y = s - S.CAM_FROM_ODOM * math.cos(psi), c - S.CAM_FROM_ODOM * math.sin(psi)
        steer, command, switched = self.previous_steer, 0.0, False
        settled_at = None
        path = []
        for i in range(int(cfg.horizon / cfg.step)):
            distance = (i + 1) * cfg.step
            cam_y = y + S.CAM_FROM_ODOM * math.sin(psi)
            if settled_at is None and abs(cam_y - T.LANE_CENTER[first]) < 0.09 and abs(psi) < math.radians(12):
                settled_at = distance
            if second is not None and settled_at is not None and distance - settled_at >= delay:
                switched = True
            target = second if switched else first
            merging = x + S.CAM_FROM_ODOM*math.cos(psi) >= s + self.merge_distance
            if merging and not reverse:
                target = 1
            desired = (0.0 if reverse else T.steer_to_lane(cam_y, -psi, target, self.merge_look if merging else .30))
            desired = float(np.clip(T.limit_heading(desired, -psi, math.radians(40)),
                                    -cfg.max_steer, cfg.max_steer))
            if i == 0:
                command = desired
            steer += (desired - steer) * 0.65
            ds = -cfg.step if reverse else cfg.step
            curvature = (math.copysign(1.0 / (T.WHEELBASE / math.tan(abs(steer)) + T.TRACK / 2), steer)
                         if abs(steer) > 1e-5 else 0.0)
            dpsi = ds * curvature
            x += ds * math.cos(psi + dpsi / 2)
            y += ds * math.sin(psi + dpsi / 2)
            psi += dpsi
            path.append((x, y, psi))
        return np.asarray(path), command

    def assess(self, path, core, now, reverse=False):
        cfg = self.cfg
        # Convert a road-frame plan to the fixed odom frame. Camera corrections
        # therefore move the road model, never smear stored lidar obstacles.
        yaw = core.pose[2]
        road_yaw = yaw + core.th
        cs, sn = math.cos(road_yaw), math.sin(road_yaw)
        cam = np.array([core.pose[0] + S.CAM_FROM_ODOM * math.cos(yaw),
                        core.pose[1] + S.CAM_FROM_ODOM * math.sin(yaw)])
        delta = path[:, :2] - [core.s, core.c]
        world = cam + np.column_stack((delta[:, 0] * cs - delta[:, 1] * sn,
                                       delta[:, 0] * sn + delta[:, 1] * cs))
        angles = path[:, 2] + road_yaw
        c, s = np.cos(angles), np.sin(angles)
        margin = cfg.margin + cfg.resolution * math.sqrt(2) / 2
        collision = np.zeros(len(path), dtype=bool)
        points = core.visibility.points
        if len(points):
            # Exclude far points before constructing pose-by-point arrays.
            points = points[np.linalg.norm(points - cam, axis=1) < cfg.horizon + 0.5]
            dx, dy = points[:, 0] - world[:, 0, None], points[:, 1] - world[:, 1, None]
            bx = dx * c[:, None] + dy * s[:, None] - cfg.body_center
            by = -dx * s[:, None] + dy * c[:, None]
            collision = np.any((np.abs(bx) <= cfg.body_length / 2 + margin) &
                               (np.abs(by) <= cfg.body_width / 2 + margin), axis=1)
        # The entrance widens into three lanes. Check all body corners, including
        # the trailing corners; a camera point past the line is not sufficient.
        corners = []
        for fx,fy in ((cfg.body_center-cfg.body_length/2-cfg.margin,-cfg.body_width/2-cfg.margin),
                      (cfg.body_center+cfg.body_length/2+cfg.margin,-cfg.body_width/2-cfg.margin),
                      (cfg.body_center+cfg.body_length/2+cfg.margin,cfg.body_width/2+cfg.margin),
                      (cfg.body_center-cfg.body_length/2-cfg.margin,cfg.body_width/2+cfg.margin)):
            rx = path[:, 0] + fx * np.cos(path[:, 2]) - fy * np.sin(path[:, 2])
            ry = path[:, 1] + fx * np.sin(path[:, 2]) + fy * np.cos(path[:, 2])
            half = np.minimum(cfg.road_half, 0.275 + np.maximum(0.0, rx - 0.05))
            half = np.minimum(half,np.maximum(.27,.80-1.06*np.maximum(0.,rx-core.island_at-1.79)))
            collision |= np.abs(ry) > half
            if core.goal in (0,2):
                wrong = ry < .27 if core.goal==0 else ry > -.27
                collision |= (rx>=core.island_at) & (rx<core.island_at+1.0) & wrong
            corners.append(np.column_stack((rx,ry)))
        # Full polygon SAT, including the tapered island ends. Checking only
        # corners misses an island tip penetrating an edge of the robot body.
        body = np.stack(corners,axis=1)
        i = core.island_at
        island = np.array([(i,0.),(i+.21,.27),(i+S.ISLAND_LEN-.21,.27),
                           (i+S.ISLAND_LEN,0.),(i+S.ISLAND_LEN-.21,-.27),(i+.21,-.27)])
        separated = np.zeros(len(path),dtype=bool)
        for edge in np.roll(island,-1,axis=0)-island:
            axis = np.array([-edge[1],edge[0]])
            bp,ip = body@axis,island@axis
            separated |= (bp.max(axis=1)<ip.min()) | (bp.min(axis=1)>ip.max())
        for edge in (body[:,1]-body[:,0],body[:,2]-body[:,1]):
            axes = np.column_stack((-edge[:,1],edge[:,0]))
            bp = np.einsum('nki,ni->nk',body,axes)
            ip = np.einsum('ki,ni->nk',island,axes)
            separated |= (bp.max(axis=1)<ip.min(axis=1)) | (bp.min(axis=1)>ip.max(axis=1))
        collision |= ~separated
        # Check newly swept leading edge and both side edges. The current body
        # was already traversed; its unseen rear is not a reason to block forward
        # exploration. Reversing checks the rear leading edge instead.
        front = cfg.body_center + (-1 if reverse else 1) * (cfg.body_length / 2 + cfg.margin)
        lateral = np.linspace(-cfg.body_width / 2 - cfg.margin,
                              cfg.body_width / 2 + cfg.margin, 9)
        local = [(front, y) for y in lateral]
        local += [(x, y) for x in np.linspace(cfg.body_center-cfg.body_length/2,
                                             cfg.body_center+cfg.body_length/2, 9)
                  for y in (-cfg.body_width / 2, cfg.body_width / 2)]
        local = np.asarray(local)
        sample = world[:, None, :] + np.stack((local[:, 0] * c[:, None] - local[:, 1] * s[:, None],
                                                local[:, 0] * s[:, None] + local[:, 1] * c[:, None]), axis=2)
        observed = core.visibility.known(sample, now)
        # Overlap with the current body envelope is not newly swept space.
        # Do not label its unobserved surroundings free in the persistent map.
        delta0 = sample - np.asarray(core.pose[:2])
        bx0 = delta0[..., 0] * math.cos(yaw) + delta0[..., 1] * math.sin(yaw)
        by0 = -delta0[..., 0] * math.sin(yaw) + delta0[..., 1] * math.cos(yaw)
        observed |= ((np.abs(bx0 - cfg.body_center) <= cfg.body_length / 2 + cfg.margin)
                     & (np.abs(by0) <= cfg.body_width / 2 + cfg.margin))
        coverage = np.mean(observed, axis=1)
        unknown = ~np.all(observed, axis=1)
        unsafe = collision | unknown
        first = lambda m: float(np.flatnonzero(m)[0] * cfg.step) if m.any() else len(path) * cfg.step
        return first(unsafe), first(collision), float(np.mean(coverage)), collision, unknown

    def plan(self, core, now, goal):
        cfg = self.cfg
        candidates = []
        merging = core.s > core.island_at+.6
        if merging:
            # The narrowing road needs more than one fixed merge curve. Odom
            # scale error must not strand a vehicle beside the converging edge.
            side = core.goal if core.goal in (0,2) else (0 if core.c>0 else 2)
            specs = [(side,None,0.,offset,look) for offset in (-.15,0.,.15) for look in (.30,.40,.50,.60)]
        else:
            specs = []
            for first in range(3):
                choices = [(None,0.)]+[(j,d) for j in range(3) if j!=first for d in (.15,.65)]
                specs.extend((first,second,delay,0.,.50) for second,delay in choices)
        for first,second,delay,offset,look in specs:
                self.merge_distance = core.island_at+S.MERGE_TURN+offset-core.s
                self.merge_look = look
                path, steer = self.rollout(core.s, core.c, core.th, first, second, delay)
                certified, collision, coverage, _, _ = self.assess(path, core, now)
                # Unknown space is not rewarded as clear. Lookahead considers
                # obstacle blockage after the first lane change, with hysteresis.
                score = min(certified, 1.5) * 3 + min(collision, cfg.horizon) * 0.35 + coverage * 0.2
                score -= 0.08 * abs(first - self.target)
                if second is not None:
                    score -= 0.04
                if merging:
                    score -= .01*abs(look-.5)+.01*abs(offset)
                elif goal in (0, 2) and core.island_at - core.s < 2.0:
                    score -= 0.60 * abs((second if second is not None else first) - goal)
                elif goal is None and core.island_at-core.s > S.LEAVE_DIST:
                    # Prefer a safe view from between the two white dashed lines
                    # while the sign is unknown. A blocked centre still loses.
                    score -= 0.20 * abs(first-1)
                candidates.append(dict(first=first, second=second, path=path, steer=steer,
                                       certified=certified, collision=collision, score=score,
                                       merge_offset=offset,merge_look=look,
                                       command_lane=1 if self.merge_distance<=0 else first))
        best = max(candidates, key=lambda a: a['score'])
        self.candidates = candidates
        self.selected = best
        self.target = best['command_lane']
        self.merge_distance = core.island_at+S.MERGE_TURN+best['merge_offset']-core.s
        self.merge_look = best['merge_look']
        d = max(0.0, best['certified'] - cfg.stop_buffer)
        # v * reaction + v^2/(2a) <= observed stopping corridor.
        v = min(cfg.speed, -cfg.deceleration * cfg.reaction_sec +
                math.sqrt((cfg.deceleration * cfg.reaction_sec) ** 2 + 2 * cfg.deceleration * d))
        if best['certified'] < 0.15:
            v = 0.0
        return v, best['steer'], best


class MappedThreeLane(S.SimpleThreeLane):
    def __init__(self, config=NavigationConfig()):
        super().__init__()
        self.config = config
        self.visibility = VisibilityMap(config)
        self.planner = FootprintPlanner(config)
        self.blind_travel = 0.0
        self.reverse_since = None
        self.reverse_start = 0.0
        self.stop_since = None
        self.diagnostics = {}
        self.goal = None
        self.recovery_status = ''
        self.relocalizations = 0

    def stopped_diagnostics(self, state, reason, pose_valid=True):
        # A stopped frame must not retain the last moving frame's pose/path.
        self.diagnostics = dict(state=state,reason=reason,s=self.s,c=self.c,heading=self.th,
                                lane=int(np.argmin(np.abs(np.asarray(T.LANE_CENTER)-self.c))),
                                target=self.lane,goal=self.goal,blind_m=self.blind_travel,
                                line_accepted=False,pose_valid=pose_valid,observed_m=None,collision_m=None,
                                next_lane=None,path=[],obstacles=[],reverse_count=self.reverse_count,
                                occupied_cells=len(self.visibility.hits),free_cells=len(self.visibility.free),
                                recovery=self.recovery_status,relocalizations=self.relocalizations)

    def _camera_island(self, obs):
        before = self.island_s
        super()._camera_island(obs)
        if self.island_s is not None and not S.START_TO_ISLAND-.7<=self.island_s<=S.START_TO_ISLAND+.5:
            self.island_s = before
            self._isl_seen = []

    def obstacle_report(self):
        """Visible connected surfaces, not an assertion that all 7 are known.

        One car can yield multiple clusters and touching cars can merge. Report
        every overlapped lane and near/far bounds rather than inventing a car ID.
        """
        pending = set(self.visibility.hits)
        clusters = []
        yaw = self.pose[2]
        road_yaw = yaw+self.th
        cam = np.asarray(self.pose[:2]) + S.CAM_FROM_ODOM*np.array([math.cos(yaw),math.sin(yaw)])
        while pending:
            todo = [pending.pop()]
            group = []
            while todo:
                key = todo.pop()
                group.append(key)
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        neighbour = (key[0]+dx,key[1]+dy)
                        if neighbour in pending:
                            pending.remove(neighbour)
                            todo.append(neighbour)
            if len(group)<2:
                continue
            xy = (np.asarray(group)+.5)*self.config.resolution-cam
            along = xy[:,0]*math.cos(road_yaw)+xy[:,1]*math.sin(road_yaw)
            lateral = self.c-xy[:,0]*math.sin(road_yaw)+xy[:,1]*math.cos(road_yaw)
            if along.max()<-.5:
                continue
            lo,hi = float(lateral.min()-.025),float(lateral.max()+.025)
            lanes = [j for j,c in enumerate(T.LANE_CENTER) if lo<=c+.2675 and hi>=c-.2675]
            clusters.append(dict(near_m=round(float(along.min()),3),
                                 far_m=round(float(along.max()),3),lanes=lanes,
                                 lateral_min=round(lo,3),lateral_max=round(hi,3),cells=len(group)))
        return sorted(clusters,key=lambda c:c['near_m'])

    def step_obs(self, obs, pose, scan, goal=None, now=None):
        now = time.monotonic() if now is None else now
        line_accepted = False
        def stopped(state, reason):
            self.state, self.reason = state, reason
            self.reversing = False
            self.stopped_diagnostics(state,reason,state not in ('NO_ODOM','ODOM_JUMP'))
            self.diagnostics['line_accepted'] = bool(line_accepted)
            return dict(v=0.0, steer=0.0, state=state, reason=reason, wall_text='')
        if pose is None or not np.isfinite(pose).all():
            return stopped('NO_ODOM', 'waiting for odometry; M7 retained')
        if self.pose is not None and (math.hypot(pose[0] - self.pose[0], pose[1] - self.pose[1]) > 0.5
                                     or abs(pose[2] - self.pose[2]) > 1.0):
            return stopped('ODOM_JUMP', 'odometry discontinuity; re-localisation required')
        previous = self.travel
        self._advance(pose)
        self.blind_travel += abs(self.travel - previous)
        self._camera_heading(obs)
        in_merge = self.island_at+1.1 < self.s < self.island_at+S.MERGE_END+.2
        heading_ok = bool(obs and math.isfinite(obs.get('th',math.nan))
                          and abs(obs['th']-self.th)<=S.TH_ACCEPT)
        line_accepted = self._camera_lines(obs) if heading_ok and not in_merge else False
        if line_accepted:
            self._camera_island(obs)
        if line_accepted and obs.get('n', 0) >= S.TH_MIN_POINTS:
            self.blind_travel = 0.0
        if not self.visibility.observe(scan, pose, now):
            return stopped('NO_SCAN', 'lidar unavailable/invalid; M7 retained')
        self.visibility.mark_footprint(pose, now)
        merge_reacquire = self.island_at+1.1 < self.s < self.island_at+S.MERGE_END+.6
        if self.blind_travel > (2.0 if merge_reacquire else self.config.blind_distance):
            return stopped('LOCALISE', 'lane observations needed; M7 retained')
        self.goal = goal
        speed, steer, best = self.planner.plan(self, now, goal)
        reading = goal is None and abs(self.c)<T.SIGN_CENTER_HALF and 0<self.island_at-self.s<S.READ_SLOW_BELOW
        if reading:
            speed = min(speed,S.V_READ)
        self.lane = self.planner.target
        self.changing = abs(self.c - T.LANE_CENTER[self.lane]) > S.ARRIVE_TOL or abs(self.th) > math.radians(10)
        self.ahead = [max((a['certified'] for a in self.planner.candidates if a['command_lane']==j),default=0.) for j in range(3)]
        if self.reversing:
            # Every reverse tick needs a fresh footprint check and a time bound.
            path, _ = self.planner.rollout(self.s, self.c, self.th, self.lane, reverse=True)
            room = self.planner.assess(path, self, now, reverse=True)[0]
            if (self.reverse_left <= 0.0 or now - self.reverse_since >= self.config.reverse_timeout
                    or room < 0.15):
                self.reversing = False
                self.stop_since = now
                speed, steer = 0.0, 0.0
            else:
                speed, steer = -self.config.reverse_speed, 0.0
        elif speed <= 0.0:
            if self.stop_since is None:
                self.stop_since = now
            if (now - self.stop_since >= self.config.wait_before_reverse
                    and self.reverse_count < self.config.reverse_attempts):
                path, _ = self.planner.rollout(self.s, self.c, self.th, self.lane, reverse=True)
                room = self.planner.assess(path, self, now, reverse=True)[0]
                if room >= self.config.reverse_distance + self.config.stop_buffer:
                    self.reversing, self.reverse_since = True, now
                    self.reverse_left = self.config.reverse_distance
                    self.reverse_count += 1
                    speed, steer = -self.config.reverse_speed, 0.0
        else:
            self.stop_since = None
        self.planner.previous_steer = steer
        self.state = 'REVERSE' if self.reversing else ('CHANGE' if speed > 0 and self.changing else
                                                     'KEEP' if speed > 0 else 'BLOCKED')
        self.reason = f"observed={best['certified']:.2f}m collision={best['collision']:.2f}m next={best['second']}"
        self.diagnostics = dict(state=self.state, lane=int(np.argmin(np.abs(np.asarray(T.LANE_CENTER) - self.c))),
                                target=self.lane, s=self.s, c=self.c, heading=self.th,
                                observed_m=best['certified'], collision_m=best['collision'],
                                next_lane=best['second'], occupied_cells=len(self.visibility.hits),
                                free_cells=len(self.visibility.free), reverse_count=self.reverse_count,
                                line_accepted=bool(line_accepted), blind_m=self.blind_travel,
                                goal=goal, reading_sign=reading,
                                pose_valid=True,recovery=self.recovery_status,relocalizations=self.relocalizations,
                                obstacles=self.obstacle_report(), path=best['path'].round(3).tolist(),
                                reason=self.reason)
        return dict(v=speed, steer=steer, state=self.state, reason=self.reason, wall_text='')


class MappedSupervisor(S.Lane3SimpleSupervisor):
    """Same M7/M2 handoff contract; only explicit exit evidence completes M7."""
    smooth_output = False  # The command has already been checked with actuator lag.
    entry_frames = 5

    def __init__(self, forced=False, goal=None):
        super().__init__(forced,goal)
        self.entry_fitter = EntryLineFitter()
        self.entry_reason = 'waiting for two white dashed boundaries on the road'
        self._entry_seen = None
        self._entry_count = 0
        self._recovery_seen = None
        self._recovery_since = None
        self._recovery_count = 0
        self._last_v = 0.

    def make_core(self):
        return MappedThreeLane()

    def set_mission(self, name):
        previous = self.mode
        super().set_mission(name)
        if previous=='off' and self.mode=='armed':
            self._entry_seen,self._entry_count = None,0
            self._recovery_seen,self._recovery_since,self._recovery_count = None,None,0
            self._last_v = 0.

    @staticmethod
    def road_fraction(image):
        if image is None or image.size==0:
            return 0.
        h,w = image.shape[:2]
        roi = image[int(h*.54):int(h*.92),int(w*.25):int(w*.75)]
        return float((cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)[:,:,2]<100).mean())

    def _try_enter(self, image, pose, tune):
        dark = self.road_fraction(image)
        obs = self.entry_fitter.fit(image,0.,tune) if image is not None and dark>=.55 else None
        candidate = central_white_pair(obs,dark)
        if candidate is None or pose is None:
            self._entry_seen,self._entry_count = None,0
            self.entry_reason = ('dark road surface not confirmed' if dark<.55 else
                                 'two central white dashes/aligned heading not confirmed')
            return False
        c,th = candidate
        if self._entry_seen is not None:
            pc,pth,pp = self._entry_seen
            consistent = (abs(c-pc)<.06 and abs(th-pth)<math.radians(4)
                          and math.hypot(pose[0]-pp[0],pose[1]-pp[1])<.10
                          and abs(pose[2]-pp[2])<math.radians(8))
            self._entry_count = self._entry_count+1 if consistent else 1
        else:
            self._entry_count = 1
        self._entry_seen = (c,th,pose)
        self.entry_reason = f'confirming central entrance {self._entry_count}/{self.entry_frames}'
        if self._entry_count<self.entry_frames:
            return False
        d = S.dash_start_ahead(self.entry_fitter,obs)
        self.core = self.make_core()
        self.core.start(c,th,pose,S.START_TO_DASH-d if d is not None else S.START_TO_DASH)
        self.mode = 'on'
        self.message = f'[M7 mapped] entry confirmed: c={c:+.2f}, heading={math.degrees(th):+.1f}, s={self.core.s:.2f}'
        return True

    def _try_relocalise(self, image, pose, tune, now):
        core = self.core
        if pose is None or abs(self._last_v)>1e-6 or core.state not in ('LOCALISE','BLOCKED'):
            self._recovery_since,self._recovery_seen,self._recovery_count = None,None,0
            return
        dark = self.road_fraction(image)
        obs = self.entry_fitter.fit(image,None,tune) if image is not None and dark>=.55 else None
        candidate = central_white_pair(obs,dark,math.radians(40),min_points=30)
        if candidate is None:
            self._recovery_since,self._recovery_seen,self._recovery_count = None,None,0
            core.recovery_status = 'waiting for a confirmed central white pair'
            return
        c,th = candidate
        if abs(math.atan2(math.sin(th-core.th),math.cos(th-core.th)))>math.radians(20) or abs(c-core.c)>.45:
            self._recovery_since,self._recovery_seen,self._recovery_count = None,None,0
            core.recovery_status = 'pose inconsistent after relocation; restart at the entrance'
            return
        if self._recovery_seen is not None:
            pc,pth,pp = self._recovery_seen
            consistent = (abs(c-pc)<.04 and abs(th-pth)<math.radians(3)
                          and math.hypot(pose[0]-pp[0],pose[1]-pp[1])<.01
                          and abs(pose[2]-pp[2])<math.radians(2))
        else:
            consistent = False
        self._recovery_count = self._recovery_count+1 if consistent else 1
        self._recovery_since = self._recovery_since if consistent else now
        self._recovery_seen = (c,th,pose)
        core.recovery_status = f'confirming stopped pose {self._recovery_count}/{self.entry_frames}'
        if self._recovery_count<self.entry_frames or now-self._recovery_since<.5:
            return
        # Keep longitudinal position and fixed-frame obstacle evidence. A white
        # pair cannot tell us where along the road we are or justify clearing it.
        core._advance(pose)
        core.c,core.th,core.blind_travel = c,th,0.
        core.planner.previous_steer = 0.
        core.planner.target = int(np.argmin(np.abs(np.asarray(T.LANE_CENTER)-c)))
        core.relocalizations += 1
        core.recovery_status = 'central pose re-acquired; longitudinal/map evidence retained'
        self._recovery_since,self._recovery_seen,self._recovery_count = None,None,0

    def step(self, image, pose, scan, both_yellow, tune=None, now=None):
        now = time.monotonic() if now is None else now
        if self.mode=='on':
            self._try_relocalise(image,pose,tune,now)
        out = super().step(image,pose,scan,both_yellow,tune,now)
        if self.mode=='armed':
            self.core.stopped_diagnostics('WAIT_ENTRY',self.entry_reason,pose is not None)
            if self.forced:
                out = dict(v=0.,steer=0.,state='LANE3',reason=self.entry_reason,note='M7 WAIT_ENTRY')
        if out is not None:
            self._last_v = out['v']
        return out

    def step_obs(self, obs, pose, scan, both_yellow, now=None):
        if pose is None:
            self.core.stopped_diagnostics('NO_ODOM','odometry unavailable; M7 retained',False)
        return super().step_obs(obs,pose,scan,both_yellow,now)
