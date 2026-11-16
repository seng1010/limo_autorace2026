"""Bounded M7 search, persistent odom-frame routes and regulated tracking.

This is a small ROS-independent implementation, NOT the Nav2 Smac plugin.
Hybrid search integrates the vehicle model from continuous poses. The optional
lattice connects exact quantised endpoints with curvature-checked primitives.
Both use the same full-footprint evaluator and execution visibility limits.
"""
import heapq
import itertools
import math
import time

import numpy as np

from . import three_lane as T
from . import three_lane_simple as S
from .m7_sensors import wrap


def curvature(steer):
    return (math.copysign(1./(T.WHEELBASE/math.tan(abs(steer))+T.TRACK/2),steer)
            if abs(steer)>1e-7 else 0.)


def steering(kappa):
    if abs(kappa)<1e-7:
        return 0.
    return math.copysign(math.atan(T.WHEELBASE/max(.01,1./abs(kappa)-T.TRACK/2)),kappa)


def integrate(pose, previous, command, distance, step, speed=.18, steering_rate=None):
    x,y,theta = pose
    points = []
    count = max(1,int(math.ceil(abs(distance)/step)))
    ds = distance/count
    alpha = 1.-math.exp(-abs(ds)/max(.04,abs(speed))/.15)
    applied = previous
    for _ in range(count):
        if steering_rate is None:
            applied = command
        else:
            change = steering_rate*abs(ds)/max(.04,abs(speed))
            applied += float(np.clip(command-applied,-change,change))
        previous += (applied-previous)*alpha
        dtheta = ds*curvature(previous)
        x += ds*math.cos(theta+dtheta/2)
        y += ds*math.sin(theta+dtheta/2)
        theta += dtheta
        points.append((x,y,theta))
    return np.asarray(points), previous


def curve_steering(path, reverse=False):
    """Recover the Ackermann steering profile of a generated pose curve once."""
    if len(path)<2:
        return np.zeros(len(path))
    length=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)
    kappa=np.diff(np.unwrap(path[:,2]))/np.maximum(length,1e-6)
    if reverse:
        kappa=-kappa
    angles=np.array([steering(k) for k in kappa])
    return np.r_[angles,angles[-1]]


class TravelHistory:
    """Bounded executed poses; reverse candidates retrace this curve, not a line."""
    def __init__(self, max_age=20., max_distance=2.):
        self.max_age,self.max_distance = max_age,max_distance
        self.samples = []

    def clear(self):
        self.samples = []

    def observe(self, pose, now, reversing=False):
        pose = np.asarray(pose,dtype=float)
        if self.samples:
            previous = self.samples[-1][1]
            d = float(np.linalg.norm(pose[:2]-previous[:2]))
            if d>.5 or abs(wrap(pose[2]-previous[2]))>1.:
                self.clear()
            elif reversing:
                distances = [np.linalg.norm(pose[:2]-p[:2]) for _,p in self.samples]
                at = int(np.argmin(distances))
                self.samples = self.samples[:at+1]
                return
            elif d<.012:
                return  # Waiting does not refresh historical evidence age.
        self.samples.append((now,pose.copy()))
        self.samples = [(t,p) for t,p in self.samples if now-t<=self.max_age]
        while len(self.samples)>2:
            length = np.linalg.norm(np.diff(np.asarray([p[:2] for _,p in self.samples]),axis=0),axis=1).sum()
            if length<=self.max_distance:
                break
            self.samples.pop(0)

    def reverse_path(self, pose, now, distance, step):
        history = [p for t,p in self.samples if now-t<=self.max_age]
        if len(history)<2 or np.linalg.norm(history[-1][:2]-np.asarray(pose[:2]))>.06:
            return np.empty((0,3))
        vertices = np.vstack((pose,np.asarray(history[::-1])))
        lengths = np.linalg.norm(np.diff(vertices[:,:2],axis=0),axis=1)
        keep = np.r_[True,lengths>1e-5]
        vertices = vertices[keep]
        if len(vertices)<2:
            return np.empty((0,3))
        arcs = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(vertices[:,:2],axis=0),axis=1))]
        end = min(distance,float(arcs[-1]))
        samples = np.arange(step,end+1e-8,step)
        angles = np.unwrap(vertices[:,2])
        return np.column_stack((np.interp(samples,arcs,vertices[:,0]),
                                np.interp(samples,arcs,vertices[:,1]),np.interp(samples,arcs,angles)))


class LatticePrimitives:
    """Quintic curves join grid states with zero endpoint curvature.

    The lattice is anchored at the actual starting pose, so there is no unsafe
    snap-to-grid connector. Endpoints are multiples of resolution in that frame.
    """
    def __init__(self, max_curvature, resolution=.05, headings=32, step=.025):
        self.resolution,self.headings = resolution,headings
        self.angle = 2*math.pi/headings
        self.by_heading = {}
        for h in range(headings):
            theta = h*self.angle
            items = []
            for dh in (-2,-1,0,1,2):
                for length in ((.20,.35) if dh==0 else (.30,.40,.50,.60)):
                    end_theta = theta+dh*self.angle
                    midpoint = theta+dh*self.angle/2
                    target = np.round(length*np.array([math.cos(midpoint),math.sin(midpoint)])/resolution)*resolution
                    d = np.linalg.norm(target)
                    if d < .1:
                        continue
                    v0 = d*np.array([math.cos(theta),math.sin(theta)])
                    v1 = d*np.array([math.cos(end_theta),math.sin(end_theta)])
                    a3,a4,a5 = 10*target-6*v0-4*v1,-15*target+8*v0+7*v1,6*target-3*v0-3*v1
                    t = np.linspace(0.,1.,max(8,int(math.ceil(d/step))+1))[:,None]
                    xy = t*v0+t**3*a3+t**4*a4+t**5*a5
                    vel = v0+3*t**2*a3+4*t**3*a4+5*t**4*a5
                    acc = 6*t*a3+12*t**2*a4+20*t**3*a5
                    norm = np.linalg.norm(vel,axis=1)
                    k = (vel[:,0]*acc[:,1]-vel[:,1]*acc[:,0])/np.maximum(norm,1e-9)**3
                    if norm.min()<.05 or np.max(np.abs(k))>max_curvature:
                        continue
                    angle = np.unwrap(np.arctan2(vel[:,1],vel[:,0]))
                    path = np.column_stack((xy,angle))[1:]
                    items.append((path,(h+dh)%headings))
            self.by_heading[h] = items


class RoutePlanner:
    def __init__(self, evaluator):
        self.checker = evaluator
        self.cfg = evaluator.cfg
        self.candidates = []
        self.selected = None
        self.world = np.empty((0,3))
        self.index = 0
        self.last_search = -math.inf
        self.plan_count = 0
        self.search_ms = 0.
        self.expanded = 0
        self.source = 'none'
        self.plan_goal = None
        self.invalidated = True
        self.lattice = (LatticePrimitives(curvature(self.cfg.max_steer),step=self.cfg.step)
                        if self.cfg.planner=='lattice' else None)
        self.last_track = None
        self.steering_profile = np.empty(0)
        self.feedforward_steer = 0.

    def __getattr__(self, name):
        checker = self.__dict__.get('checker')
        if checker is None:
            raise AttributeError(name)
        return getattr(checker,name)

    @property
    def target(self):
        return self.checker.target

    @target.setter
    def target(self, value):
        self.checker.target = value

    @property
    def previous_steer(self):
        return self.checker.previous_steer

    @previous_steer.setter
    def previous_steer(self, value):
        self.checker.previous_steer = value

    def invalidate(self):
        self.invalidated = True

    @staticmethod
    def start_pose(core):
        return np.array([core.s-S.CAM_FROM_ODOM*math.cos(core.th),
                         core.c+S.CAM_FROM_ODOM*math.sin(core.th),-core.th])

    @staticmethod
    def road_path(world, core):
        yaw = core.road_yaw
        delta = world[:,:2]-core.road_origin
        return np.column_stack((delta[:,0]*math.cos(yaw)+delta[:,1]*math.sin(yaw),
                                -delta[:,0]*math.sin(yaw)+delta[:,1]*math.cos(yaw),
                                world[:,2]-yaw))

    def remaining(self, core):
        if not len(self.world):
            return np.empty((0,3)),math.inf
        end = min(len(self.world),self.index+30)
        d = np.linalg.norm(self.world[self.index:end,:2]-core.pose[:2],axis=1)
        at = int(np.argmin(d))+self.index
        self.index = max(self.index,at)
        return self.road_path(self.world[self.index:],core),float(d.min())

    def _candidate(self, path, core, now, source):
        if not len(path):
            return None
        certified,collision,coverage,blocked,unknown = self.checker.assess(path,core,now)
        if blocked.any():
            return None  # Never win a score by ending in a known obstacle.
        last = path[-1]
        look = path[min(int(.80/self.cfg.step),len(path)-1)]
        lane = int(np.argmin(np.abs(np.asarray(T.LANE_CENTER)-(look[1]+S.CAM_FROM_ODOM*math.sin(look[2])))))
        # Preference only after feasibility. Path length/progress dominate.
        progress = last[0]-self.start_pose(core)[0]
        score = progress + .20*min(certified,1.5)-.03*abs(lane-self.target)
        if core.goal in (0,2) and core.island_at-core.s<2.0:
            score -= .25*abs(lane-core.goal)
        elif core.goal is None and core.island_at-core.s>S.LEAVE_DIST:
            score -= .15*abs(lane-1)
        if core.goal is None and getattr(core,'require_sign',False):
            camera_y = path[:,1]+S.CAM_FROM_ODOM*np.sin(path[:,2])
            camera_s = path[:,0]+S.CAM_FROM_ODOM*np.cos(path[:,2])
            reading = ((camera_s>core.island_at-2.) & (camera_s<core.island_at-.8)
                       & (np.abs(path[:,2])<=self.cfg.sign_heading_max))
            if reading.any():
                # Encourage a usable sign viewpoint, allowing a safe straddle.
                error = float(np.min(np.abs(camera_y[reading])))
                score += .8*max(0.,1.-error/self.cfg.sign_center_half)
        return dict(first=lane,second=None,delay=0.,path=path,steer=0.,certified=certified,
                    collision=collision,score=score,coverage=coverage,merge_offset=0.,merge_look=.5,
                    command_lane=lane,source=source,unknown=unknown)

    def _search(self, core, now, started):
        cfg = self.cfg
        start = self.start_pose(core)
        goal_x = start[0]+cfg.horizon*.90
        desired_y = (T.LANE_CENTER[core.goal] if core.goal in (0,2) and core.island_at-core.s<2.
                     else 0.)
        if core.s>core.island_at+S.MERGE_TURN:
            desired_y = 0.
        serial = itertools.count()
        nodes = [(start,self.previous_steer,-1,np.empty((0,3)),0.,0)]
        queue = [(0.,next(serial),0)]
        visited = {}
        best_index,best_progress = 0,0.
        finished = None
        self.expanded = 0
        c0,s0 = math.cos(start[2]),math.sin(start[2])
        while queue and self.expanded < cfg.search_nodes and time.perf_counter()-started<cfg.search_budget:
            _,_,index = heapq.heappop(queue)
            pose,prev,_,_,g,h = nodes[index]
            self.expanded += 1
            progress = pose[0]-start[0]
            merit = progress-.12*abs(pose[1]-desired_y)-.15*abs(wrap(pose[2]))
            if merit>best_progress:
                best_index,best_progress = index,merit
            if pose[0]>=goal_x and abs(wrap(pose[2]))<math.radians(32):
                finished = index
                break
            if self.lattice is None:
                options = []
                for command in np.linspace(-cfg.max_steer,cfg.max_steer,5):
                    path,steer = integrate(pose,prev,float(command),.20,cfg.step,cfg.speed)
                    options.append((path,steer,0))
            else:
                options = []
                for local,nh in self.lattice.by_heading[h]:
                    path = np.column_stack((pose[0]+local[:,0]*c0-local[:,1]*s0,
                                            pose[1]+local[:,0]*s0+local[:,1]*c0,
                                            local[:,2]+start[2]))
                    options.append((path,0.,nh))
            if not options:
                continue
            lengths = [len(p) for p,_,_ in options]
            blocked = self.checker.collision_mask(np.concatenate([p for p,_,_ in options]),core)
            offset = 0
            for (path,steer,nh),length in zip(options,lengths):
                bad = blocked[offset:offset+length].any()
                offset += length
                end = path[-1]
                if bad or abs(wrap(end[2]))>math.radians(65) or end[0]<start[0]-.05:
                    continue
                segment_length = float(np.linalg.norm(np.diff(np.vstack((pose[:2],path[:,:2])),axis=0),axis=1).sum())
                ng = g+segment_length+.035*abs(steer-prev)+.015*abs(steer)
                if self.lattice is None:
                    key = (round(end[0]/.045),round(end[1]/.04),round(wrap(end[2])/math.radians(7.5)),round(steer/.15))
                else:
                    delta = end[:2]-start[:2]
                    key = (round((delta[0]*c0+delta[1]*s0)/self.lattice.resolution),
                           round((-delta[0]*s0+delta[1]*c0)/self.lattice.resolution),nh)
                if visited.get(key,math.inf)<=ng:
                    continue
                visited[key] = ng
                node_index = len(nodes)
                nodes.append((end,steer,index,path,ng,nh))
                distance = max(0.,goal_x-end[0])
                heuristic = distance + .22*abs(end[1]-desired_y)+.06*abs(wrap(end[2]))
                heapq.heappush(queue,(ng+1.6*heuristic,next(serial),node_index))
        index = finished if finished is not None else best_index
        parts = []
        while index>0:
            _,_,parent,path,_,_ = nodes[index]
            parts.append(path)
            index = parent
        if not parts:
            return None
        candidate = self._candidate(np.concatenate(parts[::-1]),core,now,cfg.planner)
        if candidate is not None:
            candidate['search_complete'] = finished is not None
        return candidate

    def _make_route(self, core, now):
        started = time.perf_counter()
        candidates = []
        # Existing lane-change primitives are fast warm starts and preserve
        # the tested fork/merge geometry. Reject every colliding complete path.
        seeds=list(self.checker.seed_paths(core,core.goal))
        for old in seeds:
            path = old['path']
            if core.goal is None and getattr(core,'require_sign',False):
                # The unread-sign plane is the current goal, not an obstacle
                # which invalidates every otherwise useful complete lane curve.
                front = (path[:,0]+self.cfg.body_center*np.cos(path[:,2])
                         +(self.cfg.body_length/2+self.cfg.margin)*np.abs(np.cos(path[:,2]))
                         +(self.cfg.body_width/2+self.cfg.margin)*np.abs(np.sin(path[:,2])))
                beyond = np.flatnonzero(front>core.island_at-self.cfg.sign_stop_before)
                if len(beyond):
                    path = path[:beyond[0]]
            candidate = self._candidate(path,core,now,'lane_seed')
            if candidate is not None:
                candidates.append(candidate)
        # Reserve the search budget for search itself, not warm-start checks.
        search_start = time.perf_counter()
        if not candidates:
            candidate = self._search(core,now,search_start)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            # A distant obstacle/unknown fork must not erase a usable near
            # corridor. Only use this fallback after full-route search fails.
            for old in seeds:
                blocked = self.checker.collision_mask(old['path'],core)
                stop = int(np.flatnonzero(blocked)[0]) if blocked.any() else len(blocked)
                if stop<2:
                    continue
                candidate = self._candidate(old['path'][:stop],core,now,'safe_prefix')
                if candidate is not None and candidate['certified']>=self.cfg.review_departure_m:
                    candidates.append(candidate)
        self.plan_count += 1
        self.last_search = now
        self.search_ms = (time.perf_counter()-started)*1000
        self.candidates = candidates
        best = max(candidates,key=lambda c:c['score'],default=None)
        self.plan_goal = core.goal
        self.invalidated = False
        if best is None:
            self.world = np.empty((0,3))
            self.steering_profile = np.empty(0)
            self.source = 'blocked'
            return
        world,angles = self.checker.world_path(best['path'],core)
        self.world = np.column_stack((world,angles))
        self.steering_profile = curve_steering(best['path'])
        self.index = 0
        self.source = best['source']

    def track(self, path, core, now, reverse=False):
        if len(path)<2:
            return 0.,0.
        start = self.start_pose(core)
        d = np.linalg.norm(path[:,:2]-start[:2],axis=1)
        # At this vehicle's low speed, a longer lookahead cuts the final merge
        # bend and can leave its front corner outside the taper. Track the
        # locally certified curve more closely in either direction.
        look = self.cfg.tracking_lookahead
        target_index=min(int(np.searchsorted(np.maximum.accumulate(d),look)),len(path)-1)
        target = path[target_index]
        delta = target[:2]-start[:2]
        lateral = -delta[0]*math.sin(start[2])+delta[1]*math.cos(start[2])
        k = 2*lateral/max(.01,float(delta@delta))
        # Follow the steering which generated the curve; pure-pursuit geometry
        # supplies only the correction for displacement from that curve.
        profile=(curve_steering(path,True) if reverse else self.steering_profile[self.index:])
        if len(profile) and target_index>0:
            reference=target[:2]-path[0,:2]
            reference_lateral=-reference[0]*math.sin(path[0,2])+reference[1]*math.cos(path[0,2])
            reference_k=2*reference_lateral/max(.01,float(reference@reference))
            self.feedforward_steer=float(profile[0])
            # Retain a small feed-forward term during curvature transitions.
            # The model's speed/actuator response is approximate; measured
            # cross-track and heading feedback must dominate on tight bends.
            k+=.15*(curvature(self.feedforward_steer)-reference_k)
        angle = float(np.clip(steering(k),-self.cfg.max_steer,self.cfg.max_steer))
        dt = .05 if self.last_track is None else max(.001,min(.15,now-self.last_track))
        self.last_track = now
        angle = float(np.clip(angle,self.previous_steer-1.4*dt,self.previous_steer+1.4*dt))
        return angle,abs(k)

    def plan(self, core, now, goal):
        path,error = self.remaining(core)
        length = float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum()) if len(path)>1 else 0.
        needed = self.invalidated or goal!=self.plan_goal or error>.12 or length<self.cfg.path_refresh_distance
        if needed and now-self.last_search>=self.cfg.replan_interval:
            self._make_route(core,now)
            path,error = self.remaining(core)
        best = self._candidate(path,core,now,self.source)
        if best is None:
            self.invalidated = True
            best = dict(first=self.target,second=None,delay=0.,path=np.empty((0,3)),steer=0.,certified=0.,
                        collision=0.,score=-math.inf,coverage=0.,merge_offset=0.,merge_look=.5,command_lane=self.target)
            self.selected = best
            return 0.,0.,best
        self.selected = best
        self.target = best['command_lane']
        if not self.candidates:
            self.candidates = [best]
        angle,kappa = self.track(path,core,now)
        distance = max(0.,best['certified']-self.cfg.stop_buffer)
        a,reaction = self.cfg.deceleration,self.cfg.reaction_sec
        v = min(self.cfg.speed,-a*reaction+math.sqrt((a*reaction)**2+2*a*distance))
        v = min(v, self.cfg.speed/(1.+.35*kappa))
        # Curvature and steering headroom regulate speed, not just a flat cap.
        limit = curvature(self.cfg.max_steer)
        headroom = max(0.,1.-kappa/max(limit,1e-6))
        v = min(v,self.cfg.speed*(.35+.65*headroom),
                math.sqrt(self.cfg.lateral_acceleration/max(kappa,1e-6)))
        dt = .05 if core._speed_time is None else max(0.,min(.1,now-core._speed_time))
        v = min(v,core._last_command_speed+self.cfg.acceleration*dt)
        core._speed_time = now
        if best['certified']<.12 or error>.20:
            v = 0.
        # Check what the tracker will actually command, including steering lag,
        # rather than assuming its swept rectangle equals the reference path.
        for _ in range(3):
            if v<=0:
                break
            braking_speed = max(v,core.measured_speed)
            stop_distance = braking_speed*reaction+braking_speed*braking_speed/(2*a)+self.cfg.stop_buffer
            command,_ = integrate(self.start_pose(core),self.previous_steer,angle,
                                  stop_distance+self.cfg.step,self.cfg.step,v)
            certified = self.checker.assess(command,core,now)[0]
            if certified+1e-6>=stop_distance:
                break
            room = max(0.,certified-self.cfg.stop_buffer)
            v = min(v*.8,-a*reaction+math.sqrt((a*reaction)**2+2*a*room))
            if v<.03:
                v = 0.
        else:
            v = 0.  # No command passed its own braking-footprint check.
        if v<=0:
            self.invalidated = True
        best['steer'] = angle
        return v,angle,best
