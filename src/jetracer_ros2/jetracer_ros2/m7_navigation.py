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

from .m7_grid import FreeGrid, SurfaceTracks
from .m7_sensors import RoadFilter, road_hypotheses
from .m7_signs import SignEvidence
from .m7_planning import RoutePlanner, TravelHistory, integrate

try:
    from jetracer_ros2 import three_lane as T
    from jetracer_ros2 import three_lane_simple as S
except ImportError:
    import three_lane as T
    import three_lane_simple as S

CONTROLLER_VERSION = 'm7-measurement-20261117'


class EntryLineFitter(T.LineFitter):
    """The flaring yellow entrance must not outvote the two white dashes."""
    def masks(self, image, tune=None):
        yellow, white = super().masks(image, tune)
        self.yellow = np.zeros_like(yellow)
        return self.yellow, white


class RoadLineFitter(T.LineFitter):
    """Broaden heading search when the narrow fit has no joint road support."""
    def fit(self,image,th_prior=None,tune=None):
        obs = super().fit(image,th_prior,tune)
        candidates = road_hypotheses(obs)
        if th_prior is not None and (not candidates or candidates[0].confidence<.55):
            wide = super().fit(image,None,tune)
            alternatives = road_hypotheses(wide)
            if alternatives and (not candidates or alternatives[0].confidence>candidates[0].confidence):
                return wide
        return obs


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
    resolution: float = 0.02
    scan_range: float = 3.3
    free_age: float = 15.0
    body_length: float = 0.32
    body_width: float = 0.22
    body_center: float = 0.10       # rear axle -> body centre
    margin: float = 0.035
    road_half: float = 0.80
    speed: float = 0.40
    acceleration: float = 0.25
    lateral_acceleration: float = 0.30
    sign_stop_before: float = 0.95
    sign_center_half: float = 0.25
    sign_heading_max: float = math.radians(12.)
    sign_min_confidence: float = 0.65
    sign_min_height: float = 33.0  # pixels normalised to a 480px-high image
    obstacle_group_distance: float = 0.50
    reverse_speed: float = 0.08
    max_steer: float = math.radians(24.0)
    horizon: float = 2.8
    step: float = 0.025
    reaction_sec: float = 0.50
    deceleration: float = 0.35
    stop_buffer: float = 0.06
    wait_before_reverse: float = 1.5
    reverse_timeout: float = 10.0
    reverse_attempts: int = 3
    blind_distance: float = 1.2
    discovery_radius: float = .15
    review_rest_sec: float = .5
    review_departure_m: float = .18
    reverse_free_age: float = .3
    reverse_max_distance: float = .65
    history_age: float = 20.
    planner: str = 'hybrid'
    search_budget: float = .080
    search_nodes: int = 650
    replan_interval: float = .35
    path_refresh_distance: float = .65
    tracking_lookahead: float = .20
    lidar_x: float = S.CAM_FROM_ODOM + S.LIDAR_AHEAD
    lidar_y: float = 0.
    lidar_yaw: float = 0.
    raytrace_min_range: float = .02  # T-mini physical near limit; not driver hit-filter range_min


class VisibilityMap:
    """Sparse occupied/free cells. Absence of evidence remains unknown.

    Occupied evidence persists while occluded; two different clear scans are
    required to remove a hit. Scan IDs prevent repeated camera callbacks from
    fabricating independent lidar observations.
    """
    def __init__(self, config=NavigationConfig()):
        self.cfg = config
        self.free = FreeGrid(config.resolution)
        self.hits = {}
        self.clears = {}
        self.last_scan_id = None
        self.points = np.empty((0, 2))
        self.point_padding = np.empty(0)
        self.geometry = {}
        self.sequence = 0
        self.added = np.empty((0, 2))
        self.novel = np.empty((0, 2))
        self.depth_hits = {}
        self.depth_id = None
        self.revision = 0
        self.tracks = SurfaceTracks()

    def keys(self, points):
        return np.floor(np.asarray(points) / self.cfg.resolution).astype(np.int64)

    def observe(self, scan, pose, now):
        self.added = self.novel = np.empty((0, 2))
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
        cfg = self.cfg
        origin = np.array([x + cfg.lidar_x*math.cos(yaw)-cfg.lidar_y*math.sin(yaw),
                           y + cfg.lidar_x*math.sin(yaw)+cfg.lidar_y*math.cos(yaw)])
        self.free.recenter(self.keys(origin))
        poses = scan[7] if len(scan)>7 else None
        if poses is not None:
            poses = np.asarray(poses, dtype=float)
            if poses.shape != (ranges.size,3) or not np.isfinite(poses).all():
                return False
            poses = poses[valid]
            origins = poses[:,:2]+np.column_stack((cfg.lidar_x*np.cos(poses[:,2])-cfg.lidar_y*np.sin(poses[:,2]),
                                                   cfg.lidar_x*np.sin(poses[:,2])+cfg.lidar_y*np.cos(poses[:,2])))
            yaw = poses[:,2]
        else:
            origins = np.repeat(origin[None,:], int(valid.sum()), axis=0)
        angles = scan[1] + np.arange(ranges.size)[valid] * scan[2] + yaw + cfg.lidar_yaw
        measured = ranges[valid]
        max_range = min(self.cfg.scan_range,sensor_max)
        limits = np.minimum(measured, max_range)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        hit = np.isfinite(measured) & (measured <= max_range)
        ends = origins[hit] + directions[hit] * limits[hit, None]
        occupied = {tuple(k) for k in self.keys(ends)}
        geometry_changed = self._retain_geometry(ends)
        added = occupied - self.hits.keys()
        self.added = (np.asarray(list(added),dtype=float).reshape(-1,2)+.5)*self.cfg.resolution
        # Nearby new cells are often another face of the same object or lidar
        # quantisation. Still put every cell in collision checking below.
        self.novel = self.added
        if len(self.added) and len(self.points):
            distance = np.linalg.norm(self.added[:,None,:]-self.points[None,:,:],axis=2)
            self.novel = self.added[distance.min(axis=1)>self.cfg.discovery_radius]
        # Rasterise rays once per scan, in C; no Python dictionary for every
        # ray sample. Unknown angular sectors are never interpolated as free.
        stop = limits - np.where(hit, self.cfg.resolution * 1.5, 0.0)
        near = self.cfg.raytrace_min_range
        valid_rays = stop > near
        starts = self.keys(origins+directions*near)-self.free.origin
        ends_free = self.keys(origins+directions*stop[:,None])-self.free.origin
        mask = np.zeros_like(self.free.data, dtype=np.uint8)
        for a,b in zip(starts[valid_rays], ends_free[valid_rays]):
            cv2.line(mask, (int(a[1]),int(a[0])), (int(b[1]),int(b[0])), 1, 1)
        # Close raster holes between adjacent valid beams only. Never bridge a
        # zero/NaN sector or an occlusion edge; use the nearer endpoint on both
        # sides. This assumes obstacles wider than a single native angular gap.
        indices = np.flatnonzero(valid)
        if abs(scan[2]) <= math.radians(1.5):
            for j in np.flatnonzero((np.diff(indices)==1) & (np.abs(np.diff(limits))<.12)):
                if not (valid_rays[j] and valid_rays[j+1]):
                    continue
                distance = min(stop[j],stop[j+1])
                far = self.keys(origins[j:j+2]+directions[j:j+2]*distance)-self.free.origin
                polygon = np.array([starts[j],far[0],far[1],starts[j+1]],np.int32)[:,::-1].copy()
                cv2.fillConvexPoly(mask,polygon,1)
        self.free.data[mask != 0] = now
        removed = False
        for k in list(self.hits):
            if k not in occupied and self.free.get(k) == now:
                self.clears[k] = self.clears.get(k, 0) + 1
                if self.clears[k] >= 2:
                    self.hits.pop(k, None)
                    self.clears.pop(k, None)
                    removed = True
        for k in occupied:
            self.hits[k] = now
            self.free.pop(k, None)
            self.clears.pop(k, None)
        # Bound memory to the local course; do not erase occluded nearby hits.
        radius = 3.8 / self.cfg.resolution
        centre = self.keys(origin)
        previous_keys = set(self.hits) | set(self.depth_hits)
        self.hits = {k: t for k, t in self.hits.items()
                     if max(abs(k[0] - centre[0]), abs(k[1] - centre[1])) < radius}
        self.depth_hits = {k:t for k,t in self.depth_hits.items()
                           if max(abs(k[0]-centre[0]),abs(k[1]-centre[1])) < radius}
        self.clears = {k:n for k,n in self.clears.items() if k in self.hits}
        removed |= previous_keys != (set(self.hits) | set(self.depth_hits))
        # Even a clearing ray must not certify cells still held occupied by
        # the two-scan rule or the independent depth layer.
        self.free.set_many(list(set(self.hits)|set(self.depth_hits)), -math.inf)
        self.free.data[now-self.free.data > self.cfg.free_age] = -math.inf
        self.revision += int(bool(added) or removed or geometry_changed)
        self._refresh_points()
        self.last_scan_id = scan_id
        self.sequence += 1
        return True

    def _refresh_points(self):
        keys = set(self.hits) | set(self.depth_hits)
        self.geometry = {key:bounds for key,bounds in self.geometry.items() if key in keys}
        points,padding = [],[]
        for key in keys:
            if key in self.geometry:
                lo,hi = self.geometry[key]
                # Bound retained raw returns, rather than the entire storage
                # cell. Half-diagonal inflation also covers rectangle/edge
                # intersections that a corners-only test could miss.
                points.append((lo+hi)/2)
                padding.append(float(np.linalg.norm(hi-lo)/2))
            else:
                points.append((np.asarray(key)+.5)*self.cfg.resolution)
                padding.append(self.cfg.resolution*math.sqrt(2)/2)
        self.points = np.asarray(points,dtype=float).reshape(-1,2)
        self.point_padding = np.asarray(padding)

    def _retain_geometry(self, points):
        changed = False
        for key,point in zip(map(tuple,self.keys(points)),points):
            previous = self.geometry.get(key)
            if previous is None:
                self.geometry[key] = (point.copy(),point.copy())
                changed = True
            else:
                lo,hi = previous
                if np.any(point<lo) or np.any(point>hi):
                    self.geometry[key] = (np.minimum(lo,point),np.maximum(hi,point))
                    changed = True
        return changed

    def observe_depth(self, points, ident, now):
        if ident == self.depth_id:
            return False
        self.added = self.novel = np.empty((0,2))
        points = np.asarray(points, dtype=float).reshape(-1,2)
        points = points[np.isfinite(points).all(axis=1)]
        keys = set(map(tuple,self.keys(points)))
        added = keys - (set(self.hits) | set(self.depth_hits))
        changed = self._retain_geometry(points)
        for key in keys:
            self.depth_hits[key] = now
            self.free.pop(key, None)
        self.depth_id = ident
        if added or changed:
            self.revision += 1
        if added:
            self.added = (np.asarray(list(added))+.5)*self.cfg.resolution
            self.novel = self.added
        self._refresh_points()
        return True

    def mark_footprint(self, pose, now):
        """The robot's own occupied footprint is known traversable, not its rear."""
        x, y, yaw = pose
        half = self.cfg.body_length/2-.01
        xs = np.arange(self.cfg.body_center-half,self.cfg.body_center+half+.001,self.cfg.resolution/2)
        ys = np.arange(-self.cfg.body_width/2+.01,self.cfg.body_width/2-.009,self.cfg.resolution/2)
        a, b = np.meshgrid(xs, ys)
        points = np.column_stack((x + a.ravel() * math.cos(yaw) - b.ravel() * math.sin(yaw),
                                  y + a.ravel() * math.sin(yaw) + b.ravel() * math.cos(yaw)))
        keys = np.unique(self.keys(points),axis=0)
        keep = [k for k in keys if tuple(k) not in self.hits and tuple(k) not in self.depth_hits]
        if keep:
            self.free.set_many(keep,now)

    def known(self, points, now, max_age=None):
        shape = np.asarray(points).shape[:-1]
        keys = self.keys(np.asarray(points).reshape(-1, 2))
        age = self.cfg.free_age if max_age is None else max_age
        known = now-self.free.get_many(keys) <= age
        return known.reshape(shape)


class FootprintPlanner:
    def __init__(self, config=NavigationConfig()):
        self.cfg = config
        self.target = 1
        self.previous_steer = 0.0
        self.candidates = []
        self.selected = None
        self.merge_distance = math.inf
        self.merge_look = .50

    def rollout(self, s, c, th, first, second=None, delay=0.0, reverse=False, read_offset=0.):
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
            first_y = T.LANE_CENTER[first] + (read_offset if first == 1 else 0.)
            if settled_at is None and abs(cam_y - first_y) < 0.09 and abs(psi) < math.radians(12):
                settled_at = distance
            if second is not None and settled_at is not None and distance - settled_at >= delay:
                switched = True
            target = second if switched else first
            merging = x + S.CAM_FROM_ODOM*math.cos(psi) >= s + self.merge_distance
            if merging and not reverse:
                target = 1
            offset = read_offset if target == 1 and not merging else 0.
            desired = (0.0 if reverse else T.steer_to_lane(cam_y-offset, -psi, target, self.merge_look if merging else .30))
            desired = max(-cfg.max_steer,min(cfg.max_steer,T.limit_heading(desired,-psi,math.radians(40))))
            if i == 0:
                command = desired
            # Spatial samples must reflect the time available at this speed.
            steer += (desired - steer) * (1.-math.exp(-cfg.step/max(.04,cfg.speed)/.15))
            ds = -cfg.step if reverse else cfg.step
            curvature = (math.copysign(1.0 / (T.WHEELBASE / math.tan(abs(steer)) + T.TRACK / 2), steer)
                         if abs(steer) > 1e-5 else 0.0)
            dpsi = ds * curvature
            x += ds * math.cos(psi + dpsi / 2)
            y += ds * math.sin(psi + dpsi / 2)
            psi += dpsi
            path.append((x, y, psi))
        return np.asarray(path), command

    @staticmethod
    def world_path(path, core):
        # The road and stored obstacle points retain one fixed map frame.
        # Camera corrections update the vehicle's map<-odom transform only.
        road_yaw = core.road_yaw
        cs, sn = math.cos(road_yaw), math.sin(road_yaw)
        world = core.road_origin + np.column_stack((path[:,0]*cs-path[:,1]*sn,
                                                    path[:,0]*sn+path[:,1]*cs))
        angles = path[:, 2] + road_yaw
        return world, angles

    def touches(self, points, world, angles):
        cfg = self.cfg
        if not len(points) or not len(world):
            return np.zeros(len(world),dtype=bool)
        c,s = np.cos(angles),np.sin(angles)
        margin = cfg.margin+cfg.resolution*math.sqrt(2)/2
        dx,dy = points[:,0]-world[:,0,None],points[:,1]-world[:,1,None]
        bx = dx*c[:,None]+dy*s[:,None]-cfg.body_center
        by = -dx*s[:,None]+dy*c[:,None]
        return np.any((np.abs(bx)<=cfg.body_length/2+margin)&
                      (np.abs(by)<=cfg.body_width/2+margin),axis=1)

    def collision_mask(self, path, core):
        cfg = self.cfg
        world,angles = self.world_path(path,core)
        yaw = core.pose[2]
        cam = np.asarray(core.pose[:2])+S.CAM_FROM_ODOM*np.array([math.cos(yaw),math.sin(yaw)])
        c, s = np.cos(angles), np.sin(angles)
        collision = np.zeros(len(path), dtype=bool)
        points = core.visibility.points
        if len(points):
            # Exclude far points before constructing pose-by-point arrays.
            keep = np.linalg.norm(points - cam, axis=1) < cfg.horizon + 0.5
            points = points[keep]
            margin = cfg.margin + core.visibility.point_padding[keep]
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
            elif getattr(core, 'require_sign', False):
                # Do not commit to either fork before a trustworthy sign vote.
                collision |= rx > core.island_at-cfg.sign_stop_before
            corners.append(np.column_stack((rx,ry)))
        # Full polygon SAT, including the tapered island ends. Checking only
        # corners misses an island tip penetrating an edge of the robot body.
        body = np.stack(corners,axis=1)
        # Body edges can cut the concave entrance corner even if all four
        # vertices are inside. Sample edges without relaxing the yellow line.
        for weight in (.25,.5,.75):
            edge_points = body*(1.-weight)+np.roll(body,-1,axis=1)*weight
            rx,ry = edge_points[:,:,0],edge_points[:,:,1]
            half = np.minimum(cfg.road_half,.275+np.maximum(0.,rx-.05))
            half = np.minimum(half,np.maximum(.27,.80-1.06*np.maximum(0.,rx-core.island_at-1.79)))
            collision |= np.any(np.abs(ry)>half,axis=1)
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
        return collision

    def assess(self, path, core, now, reverse=False, max_age=None):
        cfg = self.cfg
        world,angles = self.world_path(path,core)
        yaw = core.pose[2]
        c,s = np.cos(angles),np.sin(angles)
        collision = self.collision_mask(path, core)
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
        observed = core.visibility.known(sample, now, max_age if max_age is not None else
                                         cfg.reverse_free_age if reverse else None)
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
        rear = np.array([core.s-S.CAM_FROM_ODOM*math.cos(core.th),
                         core.c+S.CAM_FROM_ODOM*math.sin(core.th)])
        distances = np.cumsum(np.linalg.norm(np.diff(np.vstack((rear,path[:,:2])),axis=0),axis=1))
        def first(mask):
            if not len(path):
                return 0.
            if not mask.any():
                return float(distances[-1])
            index = int(np.flatnonzero(mask)[0])
            return float(distances[index-1]) if index else 0.
        return first(unsafe), first(collision), float(np.mean(coverage)), collision, unknown

    def seed_paths(self, core, goal):
        """Generate kinematic curves; the caller owns feasibility/ranking."""
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
        specs = [(first,second,delay,offset,look,0.) for first,second,delay,offset,look in specs]
        if not merging and goal is None:
            # A readable central position need not be exactly the lane centre.
            specs += [(1,None,0.,0.,.5,y) for y in (-.18,.18)]
            specs += [(side,1,.15,0.,.5,y) for side in (0,2) for y in (-.18,.18)]
        for first,second,delay,offset,look,read_offset in specs:
            self.merge_distance = core.island_at+S.MERGE_TURN+offset-core.s
            self.merge_look = look
            path,steer = self.rollout(core.s,core.c,core.th,first,second,delay,read_offset=read_offset)
            yield dict(first=first,second=second,delay=delay,path=path,steer=steer,
                       merge_offset=offset,merge_look=look,
                       command_lane=1 if self.merge_distance<=0 else first)

    def plan(self, core, now, goal):
        cfg = self.cfg
        candidates = []
        merging = core.s > core.island_at+.6
        for seed in self.seed_paths(core,goal):
                first,second=seed['first'],seed['second']
                path=seed['path'];look,offset=seed['merge_look'],seed['merge_offset']
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
                candidates.append(dict(seed,certified=certified,collision=collision,score=score))
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
        self.planner = RoutePlanner(FootprintPlanner(config))
        self.road_filter = RoadFilter()
        self._camera_now = None
        self._last_command_speed = 0.
        self.history = TravelHistory(config.history_age)
        self.reverse_world = np.empty((0,3))
        self.reverse_source = ''
        self.blind_travel = 0.0
        self.reverse_since = None
        self.stop_since = None
        self.diagnostics = {}
        self.goal = None
        self.require_sign = False  # Enabled by the M7 supervisor, not a bare planning benchmark.
        self.road_origin = np.zeros(2)
        self.road_yaw = 0.
        self.odom_pose = None
        self.measured_speed = 0.
        self._motion_time = None
        self._speed_time = None
        self._reverse_probe_time = -math.inf
        self._reverse_search_time = -math.inf
        self.reverse_episode_count = 0
        self.reverse_progress_s = -math.inf
        self.recovery_status = ''
        self.relocalizations = 0
        self.review_active = False
        self.review_count = 0
        self.review_reason = ''
        self.review_points = []
        self._rest_anchor = None
        self._rest_last = None
        self._rest_since = None
        self._rest_scans = 0
        self._rest_seq = -1
        self._plan_key = None
        self._plan_seq = -1
        self._plan_count = 0
        self._command_world = np.empty((0,2))
        self._command_angles = np.empty(0)

    def start(self, c, th, pose, s=S.START_TO_DASH):
        super().start(c,th,pose,s)
        self.odom_pose = tuple(pose)
        self.road_yaw = pose[2]+th
        co,si = math.cos(self.road_yaw),math.sin(self.road_yaw)
        camera = np.asarray(pose[:2])+S.CAM_FROM_ODOM*np.array([math.cos(pose[2]),math.sin(pose[2])])
        self.road_origin = camera-np.array([s*co-c*si,s*si+c*co])

    def map_pose(self, raw):
        """Current map<-odom correction; the road and stored hits never move."""
        raw = np.asarray(raw,dtype=float)
        if self.odom_pose is None or self.pose is None:
            return raw.copy()
        angle = self.pose[2]-self.odom_pose[2]
        d = raw[...,:2]-self.odom_pose[:2]
        xy = np.stack((d[...,0]*math.cos(angle)-d[...,1]*math.sin(angle),
                       d[...,0]*math.sin(angle)+d[...,1]*math.cos(angle)),axis=-1)+self.pose[:2]
        return np.concatenate((xy,(raw[...,2]+angle)[...,None]),axis=-1)

    def _advance(self, pose):
        mapped = self.map_pose(pose)
        super()._advance(tuple(mapped))
        self.odom_pose = tuple(pose)

    def correct_road_pose(self, c, th):
        """Correct the vehicle on the fixed road, rather than moving the map."""
        self.c,self.th = float(c),float(th)
        yaw = self.road_yaw-th
        co,si = math.cos(self.road_yaw),math.sin(self.road_yaw)
        camera = self.road_origin+np.array([self.s*co-c*si,self.s*si+c*co])
        rear = camera-S.CAM_FROM_ODOM*np.array([math.cos(yaw),math.sin(yaw)])
        self.pose = (float(rear[0]),float(rear[1]),yaw)

    def _camera_heading(self, obs):
        # Heading and lateral correction are one joint observation in M7.
        pass

    def _camera_lines(self, obs):
        now = self._camera_now
        if now is None:
            now = (self.road_filter.last_stamp or 0.)+.1
        observed_at = obs.get('_observed_at',now) if obs else now
        road_half = max(.27,.80-1.06*max(0.,self.s-self.island_at-1.79))
        c,th,accepted,large = self.road_filter.update(
            obs,self.c,self.th,observed_at,self.island_at-.3<self.s<self.island_at+S.ISLAND_LEN,
            road_half=road_half)
        if large and self._camera_now is not None:
            if not self.review_active:
                self.begin_review('road pose correction; stop before re-anchoring',now)
            if (self._rest_anchor is None or self._rest_since is None
                    or now-self._rest_since < self.config.review_rest_sec or self._rest_scans<2
                    or math.dist(self.pose[:2],self._rest_anchor[:2])>.005):
                return False
        if accepted:
            self.correct_road_pose(c,th)
        return accepted

    def begin_review(self, reason, now, points=()):
        entered = not self.review_active or self.reversing
        self.review_active = True
        self.review_count += 1
        self.review_reason = reason
        self.review_points = np.asarray(points).reshape(-1,2).round(3).tolist()
        if entered:
            self._rest_anchor = None
            self.stop_since = now
        self._last_command_speed = 0.
        self._plan_key,self._plan_count = None,0
        self.reversing = False
        self._command_world = np.empty((0,2))
        self._command_angles = np.empty(0)
        self.planner.invalidate()

    def prepare_reverse(self, now):
        from .m7_recovery import reverse_for_departure
        if now-self._reverse_search_time < self.config.replan_interval:
            return False
        self._reverse_search_time = now
        choice = reverse_for_departure(self,now)
        if choice is None:
            self.recovery_status = 'no observed reverse improves the forward route'
            return False
        self.reverse_world = choice['path']
        self.reverse_source = choice['source']
        self.reverse_evidence_age = choice['age']
        self.reverse_left = choice['distance']
        self.reversing,self.reverse_since = True,now
        self._reverse_probe_time = now
        self.reverse_count += 1
        self.reverse_episode_count += 1
        self.reverse_progress_s = max(self.reverse_progress_s,self.s)
        self.recovery_status = f"reverse {self.reverse_left:.2f}m to open forward {choice['forward_m']:.2f}m"
        return True

    def reverse_command(self, now):
        cfg = self.config
        if now-self._reverse_probe_time>=.10:
            from .m7_recovery import forward_room
            self._reverse_probe_time = now
            if forward_room(self,self.pose,now)>0:
                self.recovery_status = 'forward route available; end reverse now'
                return None
        if not len(self.reverse_world) or self.reverse_left<=.005 or now-self.reverse_since>=cfg.reverse_timeout:
            return None
        at = int(np.argmin(np.linalg.norm(self.reverse_world[:,:2]-self.pose[:2],axis=1)))
        self.reverse_world = self.reverse_world[at:]
        path = self.planner.road_path(self.reverse_world,self)
        age = getattr(self,'reverse_evidence_age',cfg.reverse_free_age)
        room = self.planner.assess(path,self,now,reverse=True,max_age=age)[0]
        if room<cfg.stop_buffer+.04 or len(path)<2:
            return None
        steer,_ = self.planner.track(path,self,now,reverse=True)
        distance = max(0.,min(room-cfg.stop_buffer,self.reverse_left))
        a,r = cfg.deceleration,cfg.reaction_sec
        speed = min(cfg.reverse_speed,-a*r+math.sqrt((a*r)**2+2*a*distance))
        stopping = speed*r+speed*speed/(2*a)+cfg.stop_buffer
        command,_ = integrate(self.planner.start_pose(self),self.planner.previous_steer,steer,
                              -stopping-cfg.step,cfg.step,speed)
        if self.planner.assess(command,self,now,reverse=True,max_age=age)[0]+1e-6<stopping:
            return None
        self.planner.previous_steer = steer
        self._last_command_speed = 0.
        self.state,self.reason = 'REVERSE',f'{self.reverse_source}: remaining {room:.2f}m'
        self.stopped_diagnostics(self.state,self.reason)
        self.diagnostics.update(path=path.round(3).tolist(),reverse_source=self.reverse_source,
                                reverse_budget_m=self.reverse_left,observed_m=room,obstacles=self.obstacle_report())
        return dict(v=-speed,steer=steer,state=self.state,reason=self.reason,wall_text='')

    def review_stationary(self, pose, now):
        # A zero command is not proof of rest: include coast and callback gaps.
        if (self._rest_anchor is None or self._rest_last is None or
                not 0 < now-self._rest_last <= .25 or
                math.hypot(pose[0]-self._rest_anchor[0],pose[1]-self._rest_anchor[1])>.005 or
                abs(pose[2]-self._rest_anchor[2])>math.radians(1)):
            self._rest_anchor,self._rest_since,self._rest_scans = pose,now,0
            self._rest_seq = self.visibility.sequence
            self._plan_key,self._plan_count = None,0
        elif self._rest_seq != self.visibility.sequence:
            self._rest_scans += 1
            self._rest_seq = self.visibility.sequence
        self._rest_last = now
        return now-self._rest_since>=self.config.review_rest_sec and self._rest_scans>=2

    def discovery(self):
        m = self.visibility
        if m.sequence<=1:
            return np.empty((0,2))  # First scan establishes the stationary entry map.
        novel = m.novel
        if len(novel):
            yaw = self.pose[2]+self.th
            cam = np.asarray(self.pose[:2])+S.CAM_FROM_ODOM*np.array([
                math.cos(self.pose[2]),math.sin(self.pose[2])])
            delta = novel-cam
            along = delta[:,0]*math.cos(yaw)+delta[:,1]*math.sin(yaw)
            lateral = self.c-delta[:,0]*math.sin(yaw)+delta[:,1]*math.cos(yaw)
            novel = novel[(along>-.10)&(along<self.config.horizon)&
                          (np.abs(lateral)<self.config.road_half+self.config.margin)]
        # Even an extension of a known object must stop us if it invalidates
        # the previously cleared command corridor. Use its fixed odom frame.
        if self.planner.touches(m.added,self._command_world,self._command_angles).any():
            return m.added
        return novel

    def review_diagnostics(self):
        return dict(review_active=self.review_active,review_count=self.review_count,
                    review_reason=self.review_reason,new_obstacle_points=self.review_points,
                    stationary_scans=self._rest_scans,plan_confirmations=self._plan_count)

    def stopped_diagnostics(self, state, reason, pose_valid=True):
        # A stopped frame must not retain the last moving frame's pose/path.
        self.diagnostics = dict(state=state,reason=reason,s=self.s,c=self.c,heading=self.th,
                                lane=int(np.argmin(np.abs(np.asarray(T.LANE_CENTER)-self.c))),
                                target=self.lane,goal=self.goal,blind_m=self.blind_travel,
                                line_accepted=False,pose_valid=pose_valid,observed_m=None,collision_m=None,
                                next_lane=None,path=[],obstacles=self.obstacle_report() if pose_valid and self.pose is not None else [],reverse_count=self.reverse_count,
                                occupied_cells=len(self.visibility.hits),free_cells=len(self.visibility.free),
                                recovery=self.recovery_status,relocalizations=self.relocalizations,
                                **self.review_diagnostics())
        self.diagnostics.update(road_candidate_c=self.road_filter.last.c if self.road_filter.last else None,
                                road_confidence=self.road_filter.last.confidence if self.road_filter.last else 0.,
                                road_reason=self.road_filter.reason,plan_count=self.planner.plan_count,
                                plan_ms=self.planner.search_ms,planner=self.planner.source,
                                lane_number=self.diagnostics['lane']+1)

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
        clusters = []
        yaw = self.pose[2]
        road_yaw = yaw+self.th
        cam = np.asarray(self.pose[:2]) + S.CAM_FROM_ODOM*np.array([math.cos(yaw),math.sin(yaw)])
        self.visibility.tracks.update(self.visibility.points,self.visibility.revision,self._camera_now or 0.)
        tracks = self.visibility.tracks.groups(self.config.obstacle_group_distance)
        for track in tracks:
            xy = track['points']-cam
            along = xy[:,0]*math.cos(road_yaw)+xy[:,1]*math.sin(road_yaw)
            lateral = self.c-xy[:,0]*math.sin(road_yaw)+xy[:,1]*math.cos(road_yaw)
            if along.max()<-.5:
                continue
            lo,hi = float(lateral.min()-self.config.resolution/2),float(lateral.max()+self.config.resolution/2)
            lanes = [j for j,c in enumerate(T.LANE_CENTER) if lo<=c+.2675 and hi>=c-.2675]
            stamps = [max(self.visibility.hits.get(tuple(key),-math.inf),
                          self.visibility.depth_hits.get(tuple(key),-math.inf))
                      for key in self.visibility.keys(track['points'])]
            age = max(0.,(self._camera_now or 0.)-max(stamps,default=-math.inf))
            clusters.append(dict(near_m=round(float(along.min()),3),
                                 far_m=round(float(along.max()),3),lanes=lanes,
                                 lane_numbers=[j+1 for j in lanes],track_id=track['id'],partial=True,
                                 surface_ids=track['surface_ids'],group_distance_m=self.config.obstacle_group_distance,
                                 observation_age_s=round(age,3),evidence='fresh' if age<=.35 else 'retained',
                                 lateral_min=round(lo,3),lateral_max=round(hi,3),cells=len(xy)))
        return sorted(clusters,key=lambda c:c['near_m'])

    def step_obs(self, obs, pose, scan, goal=None, now=None):
        now = time.monotonic() if now is None else now
        line_accepted = False
        def stopped(state, reason):
            self.state, self.reason = state, reason
            self.reversing = False
            self.planner.previous_steer = 0.
            self._last_command_speed = 0.
            if state in ('NO_ODOM','ODOM_JUMP','NO_SCAN'):
                self._rest_anchor = None
            self.stopped_diagnostics(state,reason,state not in ('NO_ODOM','ODOM_JUMP'))
            self.diagnostics['line_accepted'] = bool(line_accepted)
            return dict(v=0.0, steer=0.0, state=state, reason=reason, wall_text='')
        if pose is None or not np.isfinite(pose).all():
            return stopped('NO_ODOM', 'waiting for odometry; M7 retained')
        if self.odom_pose is not None and (math.hypot(pose[0] - self.odom_pose[0], pose[1] - self.odom_pose[1]) > 0.5
                                          or abs(pose[2] - self.odom_pose[2]) > 1.0):
            self.history.clear()
            self.planner.invalidate()
            return stopped('ODOM_JUMP', 'odometry discontinuity; re-localisation required')
        previous = self.travel
        if self.odom_pose is not None and self._motion_time is not None and now>self._motion_time:
            self.measured_speed = math.dist(pose[:2],self.odom_pose[:2])/(now-self._motion_time)
        self._motion_time = now
        self._advance(pose)
        # Limit oscillation at one blockage, not the number of different
        # obstacles at which a short turning manoeuvre may be needed.
        if self.s > self.reverse_progress_s + .5:
            self.reverse_episode_count = 0
        self.blind_travel += abs(self.travel - previous)
        self._camera_now = now
        line_accepted = self._camera_lines(obs)
        if line_accepted:
            self._camera_island(obs)
        if line_accepted:
            self.blind_travel = 0.0
        mapped_scan = scan
        if scan is not None:
            mapped_scan = list(scan)
            if len(mapped_scan)>4 and mapped_scan[4] is not None:
                mapped_scan[4] = tuple(self.map_pose(mapped_scan[4]))
            if len(mapped_scan)>7 and mapped_scan[7] is not None:
                mapped_scan[7] = self.map_pose(mapped_scan[7])
        if not self.visibility.observe(mapped_scan, self.pose, now):
            return stopped('NO_SCAN', 'lidar unavailable/invalid; M7 retained')
        depth = getattr(self,'pending_depth',None)
        self.pending_depth = None
        if depth is not None:
            added,novel = self.visibility.added.copy(),self.visibility.novel.copy()
            raw = np.column_stack((depth[0],np.zeros(len(depth[0]))))
            if self.visibility.observe_depth(self.map_pose(raw)[:,:2],depth[1],now):
                self.visibility.added = np.vstack((added,self.visibility.added))
                self.visibility.novel = np.vstack((novel,self.visibility.novel))
        self.visibility.mark_footprint(self.pose, now)
        self.history.observe(self.pose,now,self.reversing)
        discoveries = self.discovery()
        if len(discoveries):
            self.begin_review('new occupied space; stop before changing the route',now,discoveries)
        merge_reacquire = self.island_at+1.1 < self.s < self.island_at+S.MERGE_END+.6
        if self.blind_travel > (2.0 if merge_reacquire else self.config.blind_distance):
            if not self.review_active:
                self.begin_review('road observations lost; stop to re-anchor',now)
            # Stationary evidence must accumulate here as well. Otherwise a
            # large camera correction waits for rest while LOCALISE returns
            # before recording that rest, making recovery impossible.
            self.review_stationary(self.pose,now)
            return stopped('LOCALISE', 'lane observations needed; M7 retained')
        self.goal = goal
        # Return the braking command before running the candidate search.
        if self.review_active and not self.reversing and not self.review_stationary(self.pose,now):
            return stopped('OBSERVE_STOP','waiting for odom rest and new lidar scans')
        if self.reversing:
            command = self.reverse_command(now)
            if command is not None:
                return command
            self.begin_review('reverse ended; stop and re-observe before departure',now)
            return stopped('OBSERVE_STOP',self.review_reason)
        speed, steer, best = self.planner.plan(self, now, goal)
        reading = goal is None and abs(self.c)<T.SIGN_CENTER_HALF and 0<self.island_at-self.s<S.READ_SLOW_BELOW
        if reading:
            speed = min(speed,S.V_READ)
        self.lane = self.planner.target
        self.changing = abs(self.c - T.LANE_CENTER[self.lane]) > S.ARRIVE_TOL or abs(self.th) > math.radians(10)
        self.ahead = [max((a['certified'] for a in self.planner.candidates if a['command_lane']==j),default=0.) for j in range(3)]
        if not self.review_active and not self.reversing and speed<=0:
            self.begin_review('forward corridor unavailable; stop before recovery',now)
            return stopped('OBSERVE_STOP',self.review_reason)
        if self.review_active and not self.reversing:
            if speed>0 and best['certified']>=self.config.review_departure_m:
                # Consecutive independent scans must validate the same departure
                # direction, not identical floating-point curve parameters.
                key = (best['command_lane'],self.goal)
                if self._plan_seq != self.visibility.sequence:
                    self._plan_count = self._plan_count+1 if key==self._plan_key else 1
                    self._plan_key,self._plan_seq = key,self.visibility.sequence
                if self._plan_count<2:
                    return stopped('REPLAN','confirming a departure corridor in two scans')
                self.review_active = False
            else:
                self._plan_key,self._plan_count = None,0
                speed = 0.
        if speed <= 0.0:
            if (self.require_sign and goal is None and abs(self.c)<=self.config.sign_center_half
                    and abs(self.th)<=self.config.sign_heading_max
                    and self.island_at-self.s<self.config.sign_stop_before+.35):
                return stopped('WAIT_SIGN','central viewpoint reached; waiting for a confirmed A/B sign')
            if self.stop_since is None:
                self.stop_since = now
            if (now - self.stop_since >= self.config.wait_before_reverse
                    and self.reverse_episode_count < self.config.reverse_attempts):
                if self.prepare_reverse(now):
                    command = self.reverse_command(now)
                    if command is not None:
                        return command
                    self.begin_review('reverse corridor changed before departure',now)
                    return stopped('OBSERVE_STOP',self.review_reason)
        else:
            self.stop_since = None
        self.planner.previous_steer = steer
        self._last_command_speed = speed
        if speed>0:
            count = max(0,int(min(best['certified']-self.config.stop_buffer,1.5)/self.config.step))
            self._command_world,self._command_angles = self.planner.world_path(best['path'][:count],self)
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
                                reason=self.reason,**self.review_diagnostics())
        self.diagnostics.update(lane_number=self.diagnostics['lane']+1,
                                road_confidence=self.road_filter.last.confidence if self.road_filter.last else 0.,
                                road_reason=self.road_filter.reason,planner=self.planner.source,
                                plan_count=self.planner.plan_count,plan_ms=self.planner.search_ms,
                                search_expanded=self.planner.expanded)
        return dict(v=speed, steer=steer, state=self.state, reason=self.reason, wall_text='')


class MappedSupervisor(S.Lane3SimpleSupervisor):
    """Same M7/M2 handoff contract; only explicit exit evidence completes M7."""
    smooth_output = False  # The command has already been checked with actuator lag.
    entry_frames = 5

    def __init__(self, forced=False, goal=None, config=None, entry_s=None):
        self.nav_config = config or NavigationConfig()
        self.entry_s = S.START_TO_DASH if entry_s is None else float(entry_s)
        super().__init__(forced,goal)
        self.fitter = RoadLineFitter()
        self.entry_fitter = EntryLineFitter()
        self.entry_reason = 'waiting for two white dashed boundaries on the road'
        self._entry_seen = None
        self._entry_count = 0
        self._recovery_seen = None
        self._recovery_since = None
        self._recovery_count = 0
        self._last_v = 0.
        self.sign_evidence = SignEvidence()

    def make_core(self):
        core = MappedThreeLane(self.nav_config)
        core.require_sign = True
        return core

    def set_mission(self, name):
        previous = self.mode
        super().set_mission(name)
        if previous=='off' and self.mode=='armed':
            self.sign_evidence = SignEvidence()
            self._entry_seen,self._entry_count = None,0
            self._recovery_seen,self._recovery_since,self._recovery_count = None,None,0
            self._last_v = 0.

    def see_signs(self, detections, stamp=None, capture_pose=None, now=None,
                  image_height=480, monotonic_now=None):
        if self.fixed_goal or self.mode!='on' or self.sign_goal is not None:
            return
        if stamp is None or capture_pose is None or now is None:
            self.sign_evidence.reason = 'camera acquisition timestamp and odom required'
            return
        core=self.core
        p=core.map_pose(capture_pose)
        camera=p[:2]+S.CAM_FROM_ODOM*np.array([math.cos(p[2]),math.sin(p[2])])
        d=camera-core.road_origin
        c=-d[0]*math.sin(core.road_yaw)+d[1]*math.cos(core.road_yaw)
        heading=math.atan2(math.sin(core.road_yaw-p[2]),math.cos(core.road_yaw-p[2]))
        road=core.road_filter.last
        ready=bool(road and road.confidence>=.55 and road.support>=2 and core.blind_travel<.25)
        if monotonic_now is not None:
            ready &= core.road_filter.last_stamp is not None and monotonic_now-core.road_filter.last_stamp<=.35
        goal=self.sign_evidence.observe(detections,stamp,now,float(c),heading,self.nav_config,
                                        image_height,ready)
        if goal is not None:
            self.sign_goal=goal
            core.planner.invalidate()
            self.message='[M7] '+self.sign_evidence.reason

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
        if candidate is None and image is not None and dark>=.55:
            obs = self.fitter.fit(image,None,tune)
            hypotheses = road_hypotheses(obs)
            if hypotheses and hypotheses[0].confidence>=.65 and abs(hypotheses[0].heading)<math.radians(25):
                candidate = hypotheses[0].c,hypotheses[0].heading
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
        self.core = self.make_core()
        # A visible dash can be the second/third dash after occlusion. It is not
        # an absolute longitudinal landmark. Entry s is an explicit run setting.
        self.core.start(c,th,pose,self.entry_s)
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
        if candidate is None and image is not None and dark>=.55:
            obs = self.fitter.fit(image,core.th,tune)
            hypotheses = road_hypotheses(obs,core.c,core.s>core.island_at-.3)
            if hypotheses and hypotheses[0].confidence>=.6:
                candidate = hypotheses[0].c,hypotheses[0].heading
        if candidate is None:
            self._recovery_since,self._recovery_seen,self._recovery_count = None,None,0
            core.recovery_status = 'waiting for consistent white/yellow road boundaries'
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
        core.correct_road_pose(c,th)
        core.blind_travel = 0.
        core.planner.previous_steer = 0.
        core.planner.invalidate()
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
            # The parent returns early without calling core.step_obs.
            self.core._rest_anchor = None
            self.core.reversing = False
            self.core.planner.previous_steer = 0.
            self.core.stopped_diagnostics('NO_ODOM','odometry unavailable; M7 retained',False)
        return super().step_obs(obs,pose,scan,both_yellow,now)
