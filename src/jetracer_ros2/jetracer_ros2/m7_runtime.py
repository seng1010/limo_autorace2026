"""M7 runtime: short sensor callbacks, bounded worker and independent stop gate.

Only `simple_lane ... l3mapped` uses this runtime. Other mission entry points
keep their existing implementation. ROS imports are confined to run_ros.
"""
from collections import deque
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from .m7_navigation import MappedSupervisor, NavigationConfig, CONTROLLER_VERSION
from .m7_planning import integrate
from .m7_sensors import DepthProjector, PoseHistory, fresh, move_observation
from . import three_lane_simple as S


def jsonable(value):
    if isinstance(value,dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value,(np.integer,np.floating,np.bool_)):
        value = value.item()
    if isinstance(value,float) and not math.isfinite(value):
        return 'nan' if math.isnan(value) else 'inf' if value>0 else '-inf'
    return value


def load_config(path=None):
    config = json.loads(Path(path).read_text()) if path else {}
    runtime_keys = {'navigation','camera_hz','control_hz','depth','deskew','scan_timing_verified','entry_s'}
    if set(config)-runtime_keys:
        raise ValueError(f'unknown M7 runtime settings: {sorted(set(config)-runtime_keys)}')
    allowed = {f.name for f in fields(NavigationConfig)}
    unknown = set(config.get('navigation',{}))-allowed
    if unknown:
        raise ValueError(f'unknown M7 navigation settings: {sorted(unknown)}')
    nav = NavigationConfig(**config.get('navigation',{}))
    if nav.planner not in ('hybrid','lattice') or not .01<=nav.resolution<=.05:
        raise ValueError('planner must be hybrid/lattice and resolution must be 0.01..0.05m')
    if nav.margin<.02 or not 0<nav.speed<=.4 or nav.reaction_sec<=0 or nav.deceleration<=0:
        raise ValueError('invalid footprint or stopping model')
    if not all(not isinstance(v,float) or math.isfinite(v) for v in asdict(nav).values()):
        raise ValueError('M7 settings must be finite')
    if nav.search_budget<=0 or nav.search_nodes<1 or nav.replan_interval<=0 or nav.step<=0 or nav.tracking_lookahead<=0:
        raise ValueError('invalid search budget or sampling step')
    if min(nav.body_length,nav.body_width,nav.max_steer,nav.reverse_speed,nav.free_age,nav.history_age)<=0:
        raise ValueError('vehicle dimensions, evidence ages and motion limits must be positive')
    if nav.scan_range<=nav.raytrace_min_range or nav.raytrace_min_range<0:
        raise ValueError('invalid physical lidar raytracing range')
    if (min(nav.acceleration,nav.lateral_acceleration,nav.sign_stop_before,nav.sign_min_height,
            nav.obstacle_group_distance)<=0 or not 0<nav.sign_center_half<=.30
            or not 0<nav.sign_min_confidence<=1 or not 0<nav.sign_heading_max<=math.radians(20)):
        raise ValueError('invalid speed, sign viewpoint or obstacle grouping constraints')
    if not math.isfinite(float(config.get('entry_s',S.START_TO_DASH))):
        raise ValueError('entry_s must be finite')
    if config.get('deskew',False) and not config.get('scan_timing_verified',False):
        raise ValueError('deskew requires verified driver angle order and first-ray timestamps')
    depth = config.get('depth',{})
    if depth.get('enabled',False):
        if not depth.get('calibration_verified',False):
            raise ValueError('depth requires verified extrinsics, intrinsic calibration and depth units')
        if not all(depth.get(k) for k in ('topic','camera_info_topic','frame_id','T_base_optical')):
            raise ValueError('depth requires topics, frame_id and a measured T_base_optical transform')
        DepthProjector(depth['T_base_optical'],stride=depth.get('stride',4))
        if any(not math.isfinite(depth.get(k,v)) or depth.get(k,v)<=0
               for k,v in (('scale_16u',.001),('scale_32f',1.))):
            raise ValueError('depth unit scales must be finite and positive')
    camera_hz = float(config.get('camera_hz',12.))
    control_hz = float(config.get('control_hz',20.))
    if not 5<=camera_hz<=30 or not 10<=control_hz<=30:
        raise ValueError('camera_hz must be 5..30 and control_hz 10..30')
    return nav,dict(config,camera_hz=camera_hz,control_hz=control_hz)


class LatestInputs:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = {}
        self.poses = PoseHistory()
        self.generation = 0
        self.events = deque(maxlen=32)

    def put(self, name, data, stamp, received, frame_id=''):
        with self.lock:
            previous = self.values.get(name)
            if previous and stamp<=previous['stamp']:
                return False
            self.values[name] = dict(data=data,stamp=stamp,received=received,frame_id=frame_id)
        return True

    def odom(self, pose, stamp, received):
        with self.lock:
            if not self.poses.add(stamp,pose):
                return False
            self.values['odom'] = dict(data=tuple(self.poses.samples[-1][1:]),stamp=stamp,received=received)
        return True

    def event(self, kind, data=None):
        with self.lock:
            self.generation += 1
            self.events.append((self.generation,kind,data))
            return self.generation

    def snapshot(self):
        with self.lock:
            values = dict(self.values)
            values['poses'] = tuple(self.poses.samples)
            values['events'] = tuple(self.events)
            return values


class AsyncRecorder:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else None
        self.queue = queue.Queue(maxsize=32)
        self.dropped = 0
        self.error = None
        self.thread = None
        if self.directory:
            self.directory.mkdir(parents=True,exist_ok=False)
            self.thread = threading.Thread(target=self._run,name='m7-record',daemon=True)
            self.thread.start()

    def submit(self, kind, data, image=None):
        if not self.directory:
            return
        try:
            self.queue.put_nowait((kind,data,image))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        streams = {}
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                kind,data,image = item
                if kind not in streams:
                    streams[kind] = (self.directory/(kind+'.jsonl')).open('a',encoding='utf-8')
                streams[kind].write(json.dumps(jsonable(data),ensure_ascii=False)+'\n')
                streams[kind].flush()
                if image is not None:
                    cv2.imwrite(str(self.directory/f"{int(data['frame']):05d}.jpg"),image,
                                [cv2.IMWRITE_JPEG_QUALITY,85])
        except Exception as exc:
            self.error = str(exc)
        finally:
            for stream in streams.values():
                stream.close()

    def close(self):
        if self.thread:
            # Shutdown may wait briefly for disk; control never waits for it.
            try:
                self.queue.put(None,timeout=1.)
            except queue.Full:
                return
            self.thread.join(timeout=3.)


def packet_fresh(packet, ros_now, now, limit):
    return bool(packet and fresh(packet['stamp'],ros_now,packet['received'],now,limit))


class ResourceMeter:
    """Low-rate CPU observations; process-core percent is not whole-host CPU."""
    def __init__(self):
        self.wall = time.monotonic()
        self.process = time.process_time()
        self.host = None
        self.latest = {}

    def sample(self, now):
        if now-self.wall<1.:
            return self.latest
        process = time.process_time()
        elapsed = now-self.wall
        count = os.cpu_count() or 1
        host_percent = None
        try:
            lines = Path('/proc/stat').read_text().splitlines()
            cpu = next(line for line in lines if line.startswith('cpu ')).split()[1:]
            ticks = [int(v) for v in cpu[:8]]  # guest fields already counted in user/nice
            total,idle = sum(ticks),ticks[3]+ticks[4]
            count = sum(line.startswith('cpu') and len(line)>3 and line[3].isdigit() for line in lines) or count
            if self.host and total>self.host[0]:
                host_percent = 100*(1-(idle-self.host[1])/(total-self.host[0]))
            self.host = (total,idle)
        except (OSError,ValueError,StopIteration):
            pass
        core_percent = 100*(process-self.process)/elapsed
        self.latest = dict(process_cpu_core_percent=core_percent,
                           process_cpu_host_percent=core_percent/count,host_cpu_percent=host_percent,
                           online_cpus=count)
        self.wall,self.process = now,process
        return self.latest


class SafetyGate:
    """Runs in the ROS executor even while the worker is planning/encoding.

    Raw valid returns can stop the vehicle without waiting for tracking. Full
    visibility and road-boundary checks remain in the worker's map controller.
    """
    def __init__(self, config, deskew=False):
        self.cfg = config
        self.deskew = deskew

    def check(self, result, inputs, ros_now, now):
        if result is None or now-result['started']>.20:
            return 'controller deadline missed'
        if not packet_fresh(inputs.get('image'),ros_now,now,.35):
            return 'camera acquisition stale'
        if not result.get('active',False):
            return ''
        if not packet_fresh(inputs.get('odom'),ros_now,now,.15):
            return 'odometry acquisition stale'
        if not packet_fresh(inputs.get('scan'),ros_now,now,.35):
            return 'lidar acquisition stale'
        v,angle = result['v'],result['steer']
        if not v:
            return ''
        scan = inputs['scan']['data']
        ranges = np.asarray(scan['ranges'])
        valid = np.isfinite(ranges)&(ranges>=max(.02,scan['range_min']))&(ranges<=scan['range_max'])
        if np.count_nonzero(valid|np.isposinf(ranges))<max(3,len(ranges)//10):
            return 'lidar contains no usable rays'
        poses = PoseHistory()
        poses.samples = deque(inputs['poses'],maxlen=300)
        scan_pose = poses.at(inputs['scan']['stamp'])
        if scan_pose is None:
            return 'lidar/odom time alignment unavailable'
        pose = inputs['odom']['data']
        measured = 0.
        if len(poses.samples)>=2:
            a,b=list(poses.samples)[-2:]
            if 0<b[0]-a[0]<=.15:
                measured=math.dist(a[1:3],b[1:3])/(b[0]-a[0])
        if self.deskew:
            all_poses = poses.rays(inputs['scan']['stamp'],scan['time_increment'],len(ranges),scan['scan_time'])
            if all_poses is None:
                return 'per-ray odometry unavailable'
            ray_poses = all_poses[valid]
        else:
            ray_poses = np.repeat(np.asarray(scan_pose)[None,:],int(valid.sum()),axis=0)
        a = scan['angle_min']+np.flatnonzero(valid)*scan['angle_increment']+ray_poses[:,2]+self.cfg.lidar_yaw
        origin = ray_poses[:,:2]+np.column_stack((
            self.cfg.lidar_x*np.cos(ray_poses[:,2])-self.cfg.lidar_y*np.sin(ray_poses[:,2]),
            self.cfg.lidar_x*np.sin(ray_poses[:,2])+self.cfg.lidar_y*np.cos(ray_poses[:,2])))
        points = origin+np.column_stack((np.cos(a),np.sin(a)))*ranges[valid,None]
        braking_speed=max(abs(v),measured)
        distance = braking_speed*self.cfg.reaction_sec+braking_speed**2/(2*self.cfg.deceleration)+self.cfg.stop_buffer
        path,_ = integrate(pose,result.get('previous_steer',angle),angle,
                           math.copysign(distance+self.cfg.step,v),self.cfg.step,abs(v))
        # Include the present body: a close point must not disappear between
        # the current pose and the first predicted integration sample.
        path = np.vstack((pose,path))
        delta = points[None,:,:]-path[:,None,:2]
        c,s = np.cos(path[:,2,None]),np.sin(path[:,2,None])
        x = delta[:,:,0]*c+delta[:,:,1]*s-self.cfg.body_center
        y = -delta[:,:,0]*s+delta[:,:,1]*c
        if np.any((np.abs(x)<=self.cfg.body_length/2+self.cfg.margin)&
                  (np.abs(y)<=self.cfg.body_width/2+self.cfg.margin)):
            return 'new scan intersects braking footprint'
        return ''


class M7Worker:
    """Single owner of controller state; callback threads only replace inputs."""
    def __init__(self, inputs, nav, options, tune, legacy, forced, goal, recorder=None):
        self.inputs,self.cfg,self.options,self.legacy = inputs,nav,options,legacy
        self.sup = MappedSupervisor(forced,goal,nav,options.get('entry_s'))
        self.tune = dict(tune)
        self.engine = legacy.LaneEngine()
        self.recorder = recorder or AsyncRecorder()
        self.started = time.monotonic()
        self.last_image_stamp = self.last_depth_stamp = None
        self.last_vision = -math.inf
        self.image = None
        self.frame = 0
        self.maps = None
        self.maps_key = None
        self.processed_event = 0
        self.hold = False
        self.latest = None
        self.stop_event = threading.Event()
        self.thread = None
        self.ros_clock = None
        depth = options.get('depth',{})
        self.depth = DepthProjector(depth['T_base_optical'],stride=int(depth.get('stride',4))) if depth.get('enabled') else None

    def start(self):
        self.thread = threading.Thread(target=self._loop,name='m7-control',daemon=True)
        self.thread.start()

    def _loop(self):
        period = 1./self.options['control_hz']
        while not self.stop_event.is_set():
            before = time.monotonic()
            data = self.inputs.snapshot()
            clock = data.get('clock')
            if clock:
                ros_now = self.ros_clock() if self.ros_clock is not None else clock['data']+before-clock['received']
                try:
                    self.latest = self.tick(data,ros_now,before)
                except Exception as exc:
                    if self.sup.mode=='on':
                        self.sup.core.begin_review('controller exception; re-observe before restart',before)
                    self.latest = dict(v=0.,steer=0.,active=True,started=before,state='ERROR',
                                       reason=f'{type(exc).__name__}: {exc}',event=self.processed_event)
            self.stop_event.wait(max(0.,period-(time.monotonic()-before)))

    def _image(self, packet, camera):
        raw = cv2.imdecode(np.frombuffer(packet['data'],np.uint8),cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError('invalid compressed camera image')
        h,w = raw.shape[:2]
        if camera is None:
            raise ValueError('M7 requires CameraInfo for geometric lane matching')
        info = camera['data']
        if (info['width'],info['height'])!=(w,h):
            raise ValueError('RGB CameraInfo dimensions do not match image')
        key = (w,h,tuple(info['k']),tuple(info['d']))
        if key!=self.maps_key:
            k = np.asarray(info['k'],float).reshape(3,3)
            self.maps = cv2.initUndistortRectifyMap(k,np.asarray(info['d']),None,k,(w,h),cv2.CV_16SC2)
            self.maps_key = key
        return cv2.resize(cv2.remap(raw,*self.maps,cv2.INTER_LINEAR),(640,480))

    def tick(self, data, ros_now, now):
        began = time.perf_counter()
        sign_packets=[]
        for generation,kind,value in data.get('events',()):
            if generation<=self.processed_event:
                continue
            if kind=='mission':
                self.sup.set_mission(value)
            elif kind=='detections':
                sign_packets.append(value)
            elif kind=='tune':
                self.tune = dict(value)
            elif kind=='hold':
                self.hold = True
            elif kind=='resume':
                self.hold = False
                self.sup.core.begin_review('operator resumed; verify rest and route',now)
            elif kind=='brake':
                self.sup.core.begin_review(str(value),now)
            self.processed_event = generation
        active = self.sup.mode in ('armed','on')
        result = dict(v=0.,steer=0.,started=now,active=active,event=self.processed_event,
                      state='WAIT',reason='waiting for current sensors',phase=self.sup.mode)
        def sensor_stop(state,reason):
            if active and self.sup.mode=='on':
                core = self.sup.core
                if not core.review_active:
                    core.begin_review(reason,now)
                core._rest_anchor = None
                core.reversing = False
                core.planner.invalidate()
            return dict(result,state=state,reason=reason)
        if self.hold:
            return dict(result,state='HOLD',reason='operator hold')
        if not packet_fresh(data.get('image'),ros_now,now,.35):
            return sensor_stop('NO_IMAGE','camera acquisition stale')
        poses = PoseHistory()
        poses.samples = deque(data.get('poses',()),maxlen=300)
        pose = data['odom']['data'] if packet_fresh(data.get('odom'),ros_now,now,.15) else None
        packet = data['image']
        image_pose = poses.at(packet['stamp']) if pose is not None else None
        new_image = packet['stamp']!=self.last_image_stamp and now-self.last_vision>=1./self.options['camera_hz']
        obs = None
        if new_image:
            self.image = self._image(packet,data.get('camera'))
            self.last_image_stamp,self.last_vision = packet['stamp'],now
            self.frame += 1
        if not active:
            if new_image:
                perception = self.engine.process(self.image,self.tune,now,pose=image_pose)
                v,steer,state,reason = self.engine.command(perception,self.tune,now)
                self._legacy_result = dict(result,v=v,steer=steer,state=state,reason=reason)
            # Cached command retains its ORIGINAL computation age.
            return getattr(self,'_legacy_result',result)
        if pose is None or image_pose is None:
            return sensor_stop('NO_ODOM','odom age or image/odom interpolation failed')
        if self.sup._pose is not None:
            distance = math.dist(pose[:2],self.sup._pose[:2])
            if distance<.5:
                self.sup.arm_travel += distance
        self.sup._pose = pose
        if self.sup.mode=='armed':
            ready = self.sup.forced or self.sup.arm_travel>=S.ARM_AFTER_M
            if not ready:
                if new_image:
                    perception = self.engine.process(self.image,self.tune,now,pose=image_pose)
                    v,steer,state,reason = self.engine.command(perception,self.tune,now)
                    self._legacy_result = dict(result,v=v,steer=steer,state=state,reason=reason,active=False)
                return getattr(self,'_legacy_result',result)
            if not new_image or not self.sup._try_enter(self.image,image_pose,self.tune):
                return dict(result,state='WAIT_ENTRY',reason=self.sup.entry_reason)
        scan = None
        if packet_fresh(data.get('scan'),ros_now,now,.35):
            p = data['scan']; raw = p['data']; scan_pose = poses.at(p['stamp'])
            if scan_pose is not None:
                ray_poses = None
                if self.options.get('deskew'):
                    ray_poses = poses.rays(p['stamp'],raw['time_increment'],len(raw['ranges']),raw['scan_time'])
                    if ray_poses is None:
                        return sensor_stop('SCAN_TIME','per-ray odom unavailable')
                scan = (raw['ranges'],raw['angle_min'],raw['angle_increment'],raw['range_min'],
                        scan_pose,p['stamp'],raw['range_max'],ray_poses)
        if new_image:
            prior = self.sup.core.th+self.sup.core.odom_pose[2]-image_pose[2]
            obs = self.sup.fitter.fit(self.image,prior,self.tune)
            obs = move_observation(obs,image_pose,pose)
            if obs is not None:
                obs['_observed_at'] = now-max(0.,ros_now-packet['stamp'])
        depth_points = None
        depth_packet = data.get('depth')
        if self.depth and packet_fresh(depth_packet,ros_now,now,.30) and depth_packet['stamp']!=self.last_depth_stamp:
            info = data.get('depth_camera')
            depth_pose = poses.at(depth_packet['stamp'])
            expected = self.options['depth']['frame_id']
            if info and depth_pose is not None and depth_packet['frame_id']==expected and info['frame_id']==expected:
                depth_image,scale = depth_packet['data']
                if depth_image.shape==(info['data']['height'],info['data']['width']):
                    depth_points = self.depth.project(depth_image,info['data']['k'],depth_pose,scale)
                    self.last_depth_stamp = depth_packet['stamp']
        # Depth and lidar updates enter the same core tick, so scan.observe
        # cannot erase the new-depth discovery before its stop decision.
        self.sup.core.pending_depth = ((depth_points,depth_packet['stamp']) if depth_points is not None else None)
        previous_steer = self.sup.core.planner.previous_steer
        out = self.sup.step_obs(obs,pose,scan,False,now)
        for packet_sign in sign_packets:
            if not isinstance(packet_sign,dict):
                self.sup.sign_evidence.reason='legacy detections missing capture metadata'
                continue
            stamp=packet_sign.get('image_stamp_ns')
            if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or stamp<=0:
                self.sup.sign_evidence.reason='camera acquisition timestamp required'
                continue
            stamp=float(stamp)*1e-9
            capture=poses.at(stamp)
            if capture is not None:
                self.sup.see_signs(packet_sign.get('detections',[]),stamp,capture,ros_now,
                                  packet_sign.get('image_height',480),now)
        if out is None:
            return dict(result,state='HANDOFF',reason='M7 complete; next image uses normal lane controller')
        result.update(v=out['v'],steer=out['steer'],state=self.sup.core.state,reason=out['reason'],
                      phase=self.sup.mode,previous_steer=previous_steer,pose=pose,frame=self.frame,
                      diagnostics=dict(self.sup.core.diagnostics),controller_version=CONTROLLER_VERSION,
                      image_stamp=packet['stamp'],scan_stamp=data.get('scan',{}).get('stamp'),
                      depth_enabled=self.depth is not None,depth_used=depth_points is not None,
                      deskew=bool(self.options.get('deskew')),image=self.image if new_image else None)
        result['diagnostics'].update(sign_reason=self.sup.sign_evidence.reason,
                                     map_pose=list(self.sup.core.pose),odom_pose=list(pose),
                                     road_origin=self.sup.core.road_origin.tolist(),road_yaw=self.sup.core.road_yaw)
        result['total_ms'] = (time.perf_counter()-began)*1000
        result['record_dropped'] = self.recorder.dropped
        result['record_error'] = self.recorder.error
        row = {k:v for k,v in result.items() if k!='image'}
        row.update(t=now-self.started,**self.sup.core.diagnostics)
        self.recorder.submit('m7',row,self.image.copy() if new_image else None)
        return result

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.)


def run_ros(mode, topic, record, legacy, config_path=None, headless=False):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, CameraInfo, LaserScan, Image
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String

    nav,options = load_config(config_path)
    cv2.setNumThreads(1)
    rclpy.init()
    class M7Node(Node):
        def __init__(self):
            super().__init__('simple_lane')
            self.inputs = LatestInputs()
            self.drive = mode=='drive'
            self.pub = self.create_publisher(Twist,topic,10) if self.drive else None
            self.debug = self.create_publisher(String,'/simple_lane/debug/m7',10)
            self.started = time.monotonic()
            self.quit = False
            self.tune = legacy.load_tune()
            self.gui = None if headless else legacy.Gui(self.tune)
            self.last_display = self.last_print = 0.
            self.last_control = None
            self.meter = ResourceMeter()
            self.last_frame = -1
            self.selected_odom = None
            self.brake_latch = 0
            directory = str(Path(legacy.LOG_DIR)/(time.strftime('%m%d_%H%M%S')+f'_{mode}_m7')) if record else None
            self.recorder = AsyncRecorder(directory)
            if directory:
                (Path(directory)/'tune_start.json').write_text(json.dumps(self.tune,indent=2))
                (Path(directory)/'run_start.json').write_text(json.dumps(dict(controller_version=CONTROLLER_VERSION,
                    navigation=asdict(nav),runtime=options,mode=mode,output_topic=topic,forced=legacy.LANE3,
                    goal=legacy.LANE3_GOAL),indent=2))
                print(f'[기록] {directory}')
            self.worker = M7Worker(self.inputs,nav,options,self.tune,legacy,legacy.LANE3,legacy.LANE3_GOAL,self.recorder)
            self.worker.ros_clock = lambda:self.get_clock().now().nanoseconds*1e-9
            self.gate = SafetyGate(nav,bool(options.get('deskew')))
            def stamp(msg):
                return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            self.stamp = stamp
            self.create_subscription(CompressedImage,legacy.IMAGE_TOPIC,
                lambda m:self.inputs.put('image',bytes(m.data),stamp(m),time.monotonic(),m.header.frame_id),qos_profile_sensor_data)
            self.create_subscription(CameraInfo,legacy.CAMERA_INFO_TOPIC,lambda m:self.camera('camera',m),qos_profile_sensor_data)
            self.create_subscription(LaserScan,legacy.SCAN_TOPIC,self.scan,qos_profile_sensor_data)
            for name in (legacy.ODOM_TOPIC,'/limo/odom' if legacy.ODOM_TOPIC=='/odom' else '/odom'):
                self.create_subscription(Odometry,name,lambda m,n=name:self.odom(m,n),qos_profile_sensor_data)
            self.create_subscription(String,'mission/active',lambda m:self.inputs.event('mission',m.data.strip()),10)
            self.create_subscription(String,'perception/detections',self.detections,10)
            self.create_subscription(String,'mission/obstacle_avoid/recovery',self.recovery,10)
            depth = options.get('depth',{})
            if depth.get('enabled'):
                self.create_subscription(Image,depth['topic'],self.depth,qos_profile_sensor_data)
                self.create_subscription(CameraInfo,depth['camera_info_topic'],lambda m:self.camera('depth_camera',m),qos_profile_sensor_data)
            self.create_timer(.05,self.control)
            self.worker.start()
            print(f'[M7] {CONTROLLER_VERSION}; {nav.planner}; depth={bool(depth.get("enabled"))}; output={topic if self.drive else "view only"}')

        def camera(self,name,msg):
            data = dict(k=list(msg.k),d=list(msg.d),width=msg.width,height=msg.height)
            self.inputs.put(name,data,self.stamp(msg),time.monotonic(),msg.header.frame_id)

        def scan(self,msg):
            received = time.monotonic()
            data = dict(ranges=np.asarray(msg.ranges,dtype=np.float32),angle_min=msg.angle_min,
                        angle_increment=msg.angle_increment,range_min=msg.range_min,range_max=msg.range_max,
                        time_increment=msg.time_increment,scan_time=msg.scan_time)
            if self.inputs.put('scan',data,self.stamp(msg),received,msg.header.frame_id):
                self.recorder.submit('m7_scans',dict(data,stamp=self.stamp(msg),t=received-self.started,frame_id=msg.header.frame_id))

        def odom(self,msg,name):
            if self.selected_odom is None:
                self.selected_odom = name
            if name!=self.selected_odom:
                return  # Never mix frames under a persistent obstacle map.
            q = msg.pose.pose.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            self.inputs.odom((msg.pose.pose.position.x,msg.pose.pose.position.y,yaw),self.stamp(msg),time.monotonic())

        def depth(self,msg):
            if msg.encoding not in ('16UC1','32FC1'):
                return
            dtype = np.dtype(('>' if msg.is_bigendian else '<')+('u2' if msg.encoding=='16UC1' else 'f4'))
            if msg.step<msg.width*dtype.itemsize or len(msg.data)<msg.height*msg.step:
                return
            array = np.ndarray((msg.height,msg.width),dtype=dtype,buffer=bytes(msg.data),strides=(msg.step,dtype.itemsize))
            scale = options['depth'].get('scale_16u',.001) if msg.encoding=='16UC1' else options['depth'].get('scale_32f',1.)
            self.inputs.put('depth',(array,scale),
                            self.stamp(msg),time.monotonic(),msg.header.frame_id)

        def detections(self,msg):
            try:
                self.inputs.event('detections',json.loads(msg.data))
            except (ValueError,AttributeError):
                pass

        def recovery(self,msg):
            action = msg.data.strip().lower()
            if action in ('hold','resume'):
                generation = self.inputs.event(action)
                if action=='hold':
                    self.brake_latch = generation
                    self.send(0.,0.)

        def send(self,v,steer):
            if self.pub is not None:
                msg = Twist(); msg.linear.x=float(v); msg.angular.z=float(legacy.inner_angle_to_omega(steer,v))
                self.pub.publish(msg)

        def control(self):
            now = time.monotonic(); ros_now = self.get_clock().now().nanoseconds*1e-9
            control_interval = None if self.last_control is None else (now-self.last_control)*1000
            self.last_control = now
            self.inputs.put('clock',ros_now,now,now)
            data = self.inputs.snapshot(); result = self.worker.latest
            reason = self.gate.check(result,data,ros_now,now)
            if self.brake_latch and result and result['event']>=self.brake_latch and result['v']==0:
                self.brake_latch = 0
            if reason and result and result.get('active') and result['v'] and not self.brake_latch:
                self.brake_latch = self.inputs.event('brake',reason)
            stop = reason or self.brake_latch or now-self.started<legacy.RESET_SEC
            self.send(0. if stop else result['v'],0. if stop else result['steer'])
            if result:
                status = {k:v for k,v in result.items() if k!='image'}
                status.update(gate_reason=reason,published_v=(0. if stop else result['v']) if self.drive else None,
                              output_enabled=self.drive,odom_source=self.selected_odom,
                              control_interval_ms=control_interval,worker_age_ms=(now-result['started'])*1000,
                              sensor_age_ms={name:(ros_now-data[name]['stamp'])*1000
                                             for name in ('image','scan','odom') if name in data},
                              **self.meter.sample(now))
                msg=String(); msg.data=json.dumps(jsonable(status),ensure_ascii=False); self.debug.publish(msg)
                self.recorder.submit('m7_commands',dict(status,stamp=ros_now,t=now-self.started))
                if now-self.last_print>.5:
                    print(f"[{result['state']}] proposed={result['v']:.2f} published={status['published_v']} {reason or result['reason']}")
                    self.last_print = now
                if self.gui and now-self.last_display>.10:
                    image = self.worker.image
                    if image is not None:
                        canvas=image.copy()
                        cv2.putText(canvas,f"M7 {result['state']}  v={result['v']:.2f}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
                        if self.gui.show(canvas)=='q':
                            self.quit = True
                        self.tune = self.gui.read(self.tune)
                        self.inputs.event('tune',self.tune)
                    self.last_display = now
    node = M7Node()
    try:
        while rclpy.ok() and not node.quit:
            rclpy.spin_once(node,timeout_sec=.02)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.send(0.,0.)
        except Exception:
            pass  # SIGINT may have already shut down the ROS context.
        node.worker.close()
        node.recorder.close()
        legacy.save_tune(node.tune)
        if node.gui:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
