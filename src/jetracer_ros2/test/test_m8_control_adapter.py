"""Execute the M8 authority adapter with fake ROS transport, not just text checks."""
import importlib
import sys
import types
from types import SimpleNamespace as NS

import pytest


class Twist:
    def __init__(self,**kw):
        self.linear=NS(x=0.,y=0.,z=0.)
        self.angular=NS(x=0.,y=0.,z=0.)


class Stamped:
    def __init__(self,**kw):
        self.header=NS(stamp=NS(sec=0,nanosec=0),frame_id='')
        self.twist=Twist()
        self.point=NS(x=0.,y=0.,z=0.)
        self.__dict__.update(kw)


class Pub:
    def __init__(self): self.messages=[]
    def publish(self,msg): self.messages.append(msg)
    def get_subscription_count(self): return 1


class Owner:
    def __init__(self):
        self.now=10_000_000_000
        self._active='M8'
        self._odom=NS(s=0.)
        self._odom_ns=self.now
        self._odom_speed=0.
        self._blocked=False
        self._section_limit=None
        self._section_limit_ns=0
        self.params={'odom_timeout_sec':.5,'normal_speed':.25,'auto_start_mission_nodes':False}
        self.output=[]
        self.subscriptions={}
    def declare_parameter(self,k,v): self.params.setdefault(k,v)
    def get_parameter(self,k): return NS(value=self.params[k])
    def create_publisher(self,*a,**kw): return Pub()
    def create_subscription(self,t,topic,cb,*a,**kw): self.subscriptions[topic]=cb
    def get_clock(self): return NS(now=lambda:NS(nanoseconds=self.now))
    def get_logger(self): return NS(info=lambda *a:None,warn=lambda *a:None)
    def _publish(self,msg): self.output.append((msg.linear.x,msg.angular.z))
    def _debug(self,*a): pass
    def _start_process(self,*a): pass
    def _control_tick(self): self.m8.tick(self.now)
    def _advance(self,*a):
        self._active='M8' if self._active=='M2' else 'M1'
        self.m8.enter(self._active)
    def _transition_to(self, target, reason):
        self._active = target
        self.m8.enter(target)


@pytest.fixture
def owner(monkeypatch):
    modules={
        'geometry_msgs.msg':dict(PointStamped=Stamped,Twist=Twist,TwistStamped=Stamped),
        'std_msgs.msg':dict(Bool=NS,Float32=NS,Header=NS,String=NS),
        'rclpy.qos':dict(DurabilityPolicy=NS(TRANSIENT_LOCAL=1),QoSProfile=lambda **kw:kw)}
    for name,attrs in modules.items():
        mod=types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,mod)
    sys.modules.pop('jetracer_ros2.m8_control',None)
    module=importlib.import_module('jetracer_ros2.m8_control')
    n=Owner()
    n.m8=module.M8Control(n)
    n.m8.enter('M8')
    yield n
    sys.modules.pop('jetracer_ros2.m8_control',None)


def proposal(n,v=.10,w=-.13,age=.1):
    msg=Stamped()
    t=n.now-int(age*1e9)
    msg.header.stamp=NS(sec=t//10**9,nanosec=t%10**9)
    msg.twist.linear.x,msg.twist.angular.z=v,w
    n.m8._store_cone_cmd(msg)


def test_safe_cone_is_immediately_forwarded_without_lane_overwrite(owner):
    proposal(owner)
    assert owner.output[-1]==(.10,-.13)
    owner._control_tick()
    assert owner.output[-1]==(.10,-.13)


def recovery_proposal(n, phase='cone', v=.10, w=.09, age=.1):
    msg=Stamped()
    t=n.now-int(age*1e9)
    msg.header.stamp=NS(sec=t//10**9,nanosec=t%10**9)
    msg.header.frame_id=phase
    msg.twist.linear.x,msg.twist.angular.z=v,w
    n.m8._store_recovery(msg)


def recovery_tick(n, dt=.10, move=0.):
    n.now+=int(dt*1e9)
    n._odom_ns=n.now
    n._odom.s+=move


def test_recovery_confirms_distinct_scans_and_stops_at_odom_limit(owner):
    recovery_proposal(owner)
    owner.m8.tick(owner.now)
    assert owner.output[-1]==(0.,0.)
    recovery_tick(owner)
    recovery_proposal(owner)
    assert owner.output[-1]==(.10,.09)
    recovery_tick(owner,move=.066)
    recovery_proposal(owner)
    assert owner.output[-1]==(0.,0.)
    recovery_tick(owner)
    recovery_proposal(owner)
    assert owner.output[-1]==(0.,0.)
    # Only independently authorized ordinary travel can rearm.
    proposal(owner)
    recovery_tick(owner,dt=.05,move=.151)
    proposal(owner)
    proposal(owner,v=0.)
    recovery_tick(owner,dt=.11)
    recovery_proposal(owner)
    assert owner.output[-1]==(0.,0.)
    recovery_tick(owner)
    recovery_proposal(owner)
    assert owner.output[-1]==(.10,.09)


def test_late_recovery_evidence_survives_gap_but_no_stale_motion(owner):
    # Bag 020242: source scans ~100 ms apart, offers arrive ~200 ms old.
    recovery_proposal(owner, age=.20)
    recovery_tick(owner, dt=.06)
    owner.m8.tick(owner.now)
    assert owner.output[-1] == (0., 0.)  # Previous command has expired.
    recovery_tick(owner, dt=.04)
    recovery_proposal(owner, age=.20)
    assert owner.output[-1] == (.10, .09)
    original = owner.m8._recovery_budget.started
    recovery_tick(owner, dt=.06, move=.004)
    owner.m8.tick(owner.now)
    assert owner.output[-1] == (0., 0.)
    assert owner.m8._recovery_budget.started == original
    recovery_tick(owner, dt=.04)
    recovery_proposal(owner, age=.20)
    assert owner.output[-1] == (.10, .09)
    assert owner.m8._recovery_budget.started == original
    recovery_tick(owner, dt=.56)
    recovery_proposal(owner, age=.20)
    assert owner.output[-1] == (0., 0.)
    assert owner.m8._recovery_budget.spent


@pytest.mark.parametrize('interruption', ['zero', 'invalid', 'odom', 'phase', 'long_gap', 'estop'])
def test_recovery_confirmation_does_not_survive_evidence_invalidation(owner, interruption):
    recovery_proposal(owner, age=.20)
    recovery_tick(owner, dt=.06)
    if interruption == 'zero':
        recovery_proposal(owner, v=0., age=0.)
    elif interruption == 'invalid':
        recovery_proposal(owner, w=.3, age=0.)
    elif interruption == 'odom':
        owner._odom_ns -= 200_000_000
        owner.m8.tick(owner.now)
    elif interruption == 'phase':
        owner.m8._set_m8_phase('exit_lane')
        owner.m8._set_m8_phase('cone')
    elif interruption == 'long_gap':
        recovery_tick(owner, dt=.5)
        owner.m8.tick(owner.now)
    else:
        owner.m8.event('ESTOP')
        owner.m8.event('ESTOP_CLEAR')
    recovery_tick(owner, dt=.04)
    recovery_proposal(owner, age=.20)
    assert owner.output[-1] == (0., 0.)


def test_active_recovery_cancellation_cannot_resume_as_transport_gap(owner):
    recovery_proposal(owner)
    recovery_tick(owner)
    recovery_proposal(owner)
    assert owner.output[-1] == (.10,.09)
    recovery_proposal(owner, v=0., age=0.)
    for _ in range(3):
        recovery_tick(owner, dt=.04)
        recovery_proposal(owner, age=0.)
        assert owner.output[-1] == (0.,0.)


@pytest.mark.parametrize('stop_by',['timeout','stale','estop','reverse','expired_odom','invalid_offer'])
def test_active_recovery_fails_closed(owner,stop_by):
    recovery_proposal(owner)
    recovery_tick(owner)
    recovery_proposal(owner)
    assert owner.output[-1]==(.10,.09)
    if stop_by=='timeout':
        recovery_tick(owner,dt=.65)
        recovery_proposal(owner)
    elif stop_by=='stale':
        recovery_tick(owner,dt=.16)
        owner.m8.tick(owner.now)
    elif stop_by=='estop':
        owner.m8.event('ESTOP')
    elif stop_by=='reverse':
        owner._odom_speed=-.05
        recovery_proposal(owner)
    elif stop_by=='expired_odom':
        owner._odom_ns-=120_000_000
        owner.m8.tick(owner.now)
    else:
        recovery_proposal(owner,v=-.10)
    assert owner.output[-1]==(0.,0.)


def test_guarded_approach_recovery_cannot_arm_or_transfer_ownership(owner):
    owner._active='M2'
    owner.m8.enter('M2')
    for _ in range(2):
        recovery_tick(owner)
        recovery_proposal(owner,phase='approach')
        assert not owner.m8.tick(owner.now)
    owner.m8.event('ARM_M8')
    assert owner.m8.tick(owner.now)
    for _ in range(2):
        recovery_tick(owner)
        recovery_proposal(owner,phase='approach')
        assert owner.m8.tick(owner.now)
    assert owner.output[-1]==(.10,.09) and owner._active=='M2'


def test_exit_recovery_does_not_count_as_verified_lane(owner):
    owner.m8._m8_exit.reset(.10)
    owner.m8._m8_exit.line_s=.9
    owner._odom.s=1.30
    owner.m8._set_m8_phase('exit_lane')
    for _ in range(2):
        recovery_tick(owner)
        recovery_proposal(owner,phase='exit_lane')
    assert owner.output[-1]==(.10,.09)
    assert owner.m8._m8_exit.lane_start_s is None
    assert owner._active=='M8'


def test_stale_and_estop_stop_but_clear_and_fresh_proposal_resume(owner):
    proposal(owner)
    owner.m8.event('ESTOP')
    proposal(owner)
    assert owner.output[-1]==(0.,0.)
    owner.m8.event('ESTOP_CLEAR')
    proposal(owner)
    assert owner.output[-1]==(.10,-.13)
    proposal(owner,age=.26)
    assert owner.output[-1]==(0.,0.)
    proposal(owner,v=.09)
    assert owner.output[-1]==(0.,0.)


def test_approach_keeps_race_until_armed_then_cone_owns(owner):
    owner._active='M2'
    owner.m8.enter('M2')
    assert owner.m8.tick(owner.now) is False
    assert owner.output==[]
    assert owner.m8.event('ARM_M8')
    owner.m8.tick(owner.now)
    assert owner.output[-1]==(0.,0.)
    proposal(owner)
    assert owner._active=='M8'
    assert owner.output[-1]==(.10,-.13)


def test_visual_watch_leaves_race_in_control_until_a_near_line_is_confirmed(owner):
    owner.params['m8_entry_arm_near_line_m'] = .45
    owner.m8 = type(owner.m8)(owner)
    owner._active = 'M2'
    owner.m8.enter('M2')

    def observe(dist, delta=0.):
        owner.now += 50_000_000
        owner._odom_ns = owner.now
        owner._odom.s += delta
        owner.m8._m8_visual_line_callback(NS(data=dist))
        return owner.m8.tick(owner.now)

    assert owner.m8._m8_phase_sent == 'watch_entry'
    assert not owner.m8._exit_check_enabled
    proposal(owner)  # Even a valid cone proposal cannot take over here.
    for dist in (.55, .55, .55, .44, -1., .44):
        assert observe(dist) is False
    assert owner.output == []
    assert observe(.43, .01) is False
    assert observe(.42, .01) is True
    assert owner.m8._m8_phase_sent == 'approach'
    assert owner.m8._m8_entry.line_s == pytest.approx(.44)
    assert owner.m8._exit_check_enabled
    assert owner.output[-1] == (0., 0.)  # No verified approach command yet.
    proposal(owner)
    assert owner._active == 'M8'
    assert owner.m8._m8_exit.entry_line == pytest.approx(.44)
    assert owner.output[-1] == (.10, -.13)


def test_no_motion_launch_can_keep_immediate_approach_guard(owner):
    owner.params.update(m8_entry_arm_after_m=0., m8_entry_arm_near_line_m=.45,
                        m8_entry_precheck_m=.55)
    owner.m8 = type(owner.m8)(owner)
    owner._active = 'M2'
    owner.m8.enter('M2')
    assert owner.m8._m8_phase_sent == 'off'
    assert owner.m8.tick(owner.now) is True
    assert owner.m8._m8_phase_sent == 'approach'
    assert owner.output[-1] == (0., 0.)


def _confirm_preflight(owner):
    owner.params.update(m8_entry_precheck_m=.55, m8_entry_arm_near_line_m=.45)
    owner.m8 = type(owner.m8)(owner)
    owner._active = 'M2'
    owner.m8.enter('M2')
    for i in range(3):
        owner.now += 50_000_000
        owner._odom_ns = owner.now
        owner._odom.s = .01*i
        owner.m8._m8_visual_line_callback(NS(data=.54-.01*i))
        assert not owner.m8.tick(owner.now)
    assert owner.m8._m8_phase_sent == 'preflight'
    assert owner.m8._exit_check_enabled
    assert owner.m8._m8_entry.started_s is None


def _checked_lane(owner):
    lane = Stamped()
    lane.header.stamp = NS(sec=owner.now//10**9, nanosec=owner.now%10**9)
    lane.twist.linear.x, lane.twist.angular.z = .10, .05
    owner.m8._store_exit_lane(lane)
    return lane.header


def test_preflight_keeps_race_then_odom_reuses_recent_checked_command(owner):
    _confirm_preflight(owner)
    _checked_lane(owner)
    assert not owner.m8.tick(owner.now)
    # No new camera line observation is needed after confirmation.
    owner._odom.s = .10
    assert owner.m8.tick(owner.now)
    assert owner.m8._m8_phase_sent == 'approach'
    assert owner.output[-1] == (.10, .05)


def test_low_lidar_clearance_advances_guard_without_using_older_commands(owner):
    _confirm_preflight(owner)
    old = _checked_lane(owner)
    owner.now += 300_000_000
    owner._odom_ns = owner.now
    owner.m8._store_approach_risk(old)
    assert not owner.m8.tick(owner.now)  # Stale alert.
    _checked_lane(owner)
    owner.now += 10_000_000
    owner.m8._store_approach_risk(NS(stamp=NS(sec=owner.now//10**9,
                                              nanosec=owner.now%10**9)))
    assert owner.m8.tick(owner.now)
    assert owner.m8._m8_phase_sent == 'approach'
    assert owner.m8._m8_entry.line_s-owner._odom.s > .45
    assert owner.output[-1] == (0., 0.)  # Pre-alert command is not reused.


def test_lost_odometry_after_preflight_stops_instead_of_bypassing_control(owner):
    _confirm_preflight(owner)
    owner.now += 600_000_000
    assert owner.m8.tick(owner.now)
    assert owner.output[-1] == (0., 0.)


def test_reverse_does_not_accumulate_entry_observations(owner):
    owner.params.update(m8_entry_precheck_m=.55, m8_entry_arm_near_line_m=.45)
    owner.m8 = type(owner.m8)(owner)
    owner._active = 'M2'
    owner.m8.enter('M2')
    owner._odom_speed = -.50
    for _ in range(3):
        owner.now += 30_000_000
        owner._odom_ns = owner.now
        owner.m8._m8_visual_line_callback(NS(data=.42))
        proposal(owner, age=0.)
        assert not owner.m8.tick(owner.now)
    assert owner.m8._m8_phase_sent == 'watch_entry'
    assert owner.m8._m8_entry.line_s is None


def test_reverse_blocks_handoff_and_waits_for_post_reverse_proposal(owner):
    _confirm_preflight(owner)
    owner._odom_speed = -.50
    owner._odom.s = .10  # The near-distance trigger is already satisfied.
    proposal(owner, age=0.)
    assert owner.m8.tick(owner.now)
    assert owner._active == 'M2'
    assert owner.output[-1] == (0., 0.)
    owner.now += 50_000_000
    owner._odom_ns = owner.now
    owner._odom_speed = 0.
    assert owner.m8.tick(owner.now)
    assert owner._active == 'M2'
    assert owner.output[-1] == (0., 0.)
    proposal(owner, age=0.)
    assert owner._active == 'M8'
    assert owner.output[-1] == (.10, -.13)


def test_no_other_mission_control_is_consumed(owner):
    for mission in ('M1','M3','M4','M5','M6','M7'):
        owner._active=mission
        owner.m8.enter(mission)
        assert owner.m8.tick(owner.now) is False


def partial_watch(owner):
    owner.params.update(m8_entry_precheck_m=.55,m8_entry_arm_near_line_m=.45)
    owner.m8=type(owner.m8)(owner)
    owner._active='M2'
    owner.m8.enter('M2')


def partial_cue(owner,distance=.4,age=0.,frame='front_axle'):
    msg=Stamped()
    t=owner.now-int(age*1e9)
    msg.header.stamp=NS(sec=t//10**9,nanosec=t%10**9)
    msg.header.frame_id=frame
    msg.point.x=distance
    owner.m8._m8_partial_callback(msg)


def confirm_partial(owner):
    for _ in range(3):
        owner.now+=40_000_000
        owner._odom_ns=owner.now
        partial_cue(owner)


def test_partial_alone_keeps_race_then_fresh_safe_cone_transfers(owner):
    partial_watch(owner)
    confirm_partial(owner)
    assert owner.m8.tick(owner.now) is False
    assert owner.output==[] and owner.m8._m8_entry.line_s is None
    proposal(owner,age=0.)
    assert owner.m8.tick(owner.now)
    assert owner._active=='M8' and owner.output[-1]==(.1,-.13)
    assert owner.m8._m8_exit.entry_line==pytest.approx(.4)


def test_early_partial_transfers_only_with_fresh_checked_cone(owner):
    partial_watch(owner)
    for _ in range(3):
        owner.now += 40_000_000
        owner._odom_ns = owner.now
        partial_cue(owner, distance=.65)
    # Early observation alone does not alter the race lane command.
    assert owner.m8.tick(owner.now) is False
    assert owner.output == [] and owner._active == 'M2'
    proposal(owner, age=0.)
    assert owner.m8.tick(owner.now)
    assert owner._active == 'M8' and owner.output[-1] == (.1, -.13)
    assert owner.m8._m8_exit.entry_line == pytest.approx(.65)


@pytest.mark.parametrize('age,speed',[(.26,.1),(.1,0.),(.1,.09)])
def test_partial_does_not_authorize_stale_stopped_or_unexecutable_cone(owner,age,speed):
    partial_watch(owner)
    confirm_partial(owner)
    proposal(owner,age=age,v=speed)
    assert not owner.m8.tick(owner.now)
    assert owner._active=='M2' and owner.output==[]


@pytest.mark.parametrize('age,frame',[(.3,'front_axle'),(-.1,'front_axle'),(0.,'camera')])
def test_partial_rejects_stale_future_or_wrong_frame(owner,age,frame):
    partial_watch(owner)
    for _ in range(3):
        owner.now+=40_000_000
        owner._odom_ns=owner.now
        partial_cue(owner,age=age,frame=frame)
    proposal(owner,age=0.)
    assert not owner.m8.tick(owner.now)


def test_reverse_erases_partial_cue_and_requires_new_images(owner):
    partial_watch(owner)
    confirm_partial(owner)
    owner._odom_speed=-.1
    proposal(owner,age=0.)
    assert not owner.m8.tick(owner.now)
    owner.now+=50_000_000
    owner._odom_ns=owner.now
    owner._odom_speed=0.
    assert not owner.m8.tick(owner.now)
    proposal(owner,age=0.)
    assert not owner.m8.tick(owner.now)
    confirm_partial(owner)
    proposal(owner,age=0.)
    assert owner.m8.tick(owner.now) and owner._active=='M8'


def test_partial_ignores_blocked_other_missions_and_preserves_full_preflight(owner):
    partial_watch(owner)
    owner._blocked=True
    confirm_partial(owner)
    assert owner.m8._m8_partial.confirmed is None
    owner._blocked=False
    for mission in ('M1','M3','M4','M5','M6','M7','M8'):
        owner._active=mission
        confirm_partial(owner)
        assert owner.m8._m8_partial.confirmed is None
    _confirm_preflight(owner)
    anchor=owner.m8._m8_entry.line_s
    confirm_partial(owner)
    proposal(owner,age=0.)
    assert not owner.m8.tick(owner.now)
    assert owner.m8._m8_phase_sent=='preflight'
    assert owner.m8._m8_entry.line_s==anchor


def test_queued_reverse_images_cannot_confirm_after_reverse_ends(owner):
    partial_watch(owner)
    owner._odom_speed=-.1
    owner.m8.tick(owner.now)
    owner.now+=50_000_000
    owner._odom_ns=owner.now
    owner._odom_speed=0.
    owner.m8.tick(owner.now)
    for age in (.15,.11,.07):
        partial_cue(owner,age=age)
    proposal(owner,age=0.)
    assert not owner.m8.tick(owner.now)
    assert owner.m8._m8_partial.confirmed is None


def test_expired_partial_cue_does_not_authorize_later_cone_proposal(owner):
    partial_watch(owner)
    confirm_partial(owner)
    owner.now+=900_000_000
    owner._odom_ns=owner.now
    proposal(owner,age=0.)
    assert not owner.m8.tick(owner.now)
    assert owner.m8._m8_entry.line_s is None


def test_partial_observer_is_entry_only_stamped_and_requires_distance_calibration(owner):
    from jetracer_ros2.m8_observers import M8StopLineObserver
    from test_m8_partial_entry import tape
    model=NS(calibrated=True,front_distance=lambda row:.4)
    obs=M8StopLineObserver(owner,model,NS(roi_top_ratio=.58))
    header=NS(stamp=NS(sec=10,nanosec=0))
    im=tape()
    obs.update(im,header)
    assert not obs.partial.messages
    obs._phase(NS(data='watch_entry'))
    obs.update(im,header)
    cue=obs.partial.messages[-1]
    assert cue.header.stamp==header.stamp and cue.header.frame_id=='front_axle'
    assert cue.point.x==.4 and obs.entry.messages[-1].data==-1.
    model.calibrated=False
    obs.update(im,header)
    assert obs.partial.messages[-1].point.x==-1.
    count=len(obs.partial.messages)
    for phase in ('cone','exit_lane','off'):
        obs._phase(NS(data=phase))
        obs.update(im,header)
    assert len(obs.partial.messages)==count
    obs._phase(NS(data='watch_entry'))
    owner.now+=300_000_000
    obs.update(im,header)
    assert len(obs.partial.messages)==count


def test_partial_needs_fresh_odometry_and_no_motion_still_holds(owner):
    partial_watch(owner)
    confirm_partial(owner)
    owner.now+=600_000_000
    proposal(owner,age=0.)
    assert not owner.m8.tick(owner.now)
    assert owner._active=='M2'
    owner.params['m8_entry_arm_after_m']=0.
    owner.m8=type(owner.m8)(owner)
    owner.m8.enter('M2')
    confirm_partial(owner)
    proposal(owner,v=0.,age=0.)
    assert owner.m8.tick(owner.now)
    assert owner.output[-1]==(0.,0.) and owner._active=='M2'


def confirm_exit(owner):
    owner.m8._m8_exit.reset(0.)
    owner._odom.s=.8
    for t in (0,.06,.12):
        owner.now=10_000_000_000+int(t*1e9)
        owner._odom_ns=owner.now
        msg=Stamped()
        msg.header.frame_id='front_axle'
        msg.header.stamp=NS(sec=owner.now//10**9,nanosec=owner.now%10**9)
        msg.point.x=.3
        owner.m8._m8_exit_line_callback(msg)
    assert not owner.m8._m8_exit.owns_lane
    assert owner.m8._m8_phase_sent == 'exit_prepare'


def exit_proposal(owner, v=.1, age=0.):
    lane=Stamped()
    t=owner.now-int(age*1e9)
    lane.header.stamp=NS(sec=t//10**9,nanosec=t%10**9)
    lane.twist.linear.x=v
    lane.twist.angular.z=.04 if v else 0.
    owner.m8._store_exit_lane(lane)


def exit_evidence(owner, aligned=True, age=0.):
    evidence=Stamped()
    stamp=owner.now-int(age*1e9)
    evidence.header.frame_id='rear_axle'
    evidence.header.stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9)
    evidence.point.x=1.;evidence.point.y=float(aligned)
    owner.m8._store_exit_evidence(evidence)


def test_observer_changes_only_m8_exit_proposals_not_race_result(owner,monkeypatch):
    import numpy as np
    from test_m8_exit_lane import scene
    from jetracer_ros2.m8_observers import M8LaneObserver
    class Polygon(Stamped):
        def __init__(self):
            super().__init__();self.polygon=NS(points=[])
    monkeypatch.setattr(sys.modules['geometry_msgs.msg'],'PolygonStamped',Polygon,raising=False)
    monkeypatch.setattr(sys.modules['geometry_msgs.msg'],'Point32',NS,raising=False)
    observer=M8LaneObserver(owner,.184)
    image,r,g=scene();r['raw_angle']=-.1
    original=r['center'][0].copy()
    command=(.2,-.1,'TRACK','test')
    observer.update(NS(sec=10,nanosec=0),r,command,image=image,pose=(0.,0.,0.),geometry=g)
    assert observer.command.messages[-1].twist.linear.x==.2
    observer._phase(NS(data='exit_prepare'))
    for i in range(5):
        observer.update(NS(sec=10,nanosec=i*33_000_000),r,command,
                        image=image,pose=(0.,0.,0.),geometry=g)
    assert observer.evidence.messages[-1].point.x==1.
    assert observer.evidence.messages[-1].point.y==0.
    assert observer.command.messages[-1].twist.linear.x==0.
    assert all(p.y>.30 for p in observer.path.messages[-1].polygon.points)
    assert np.array_equal(r['center'][0],original)
    observer._phase(NS(data='off'))
    observer.update(NS(sec=11,nanosec=0),r,command,image=image,pose=(0.,0.,0.),geometry=g)
    assert observer.command.messages[-1].twist.linear.x==.2


@pytest.mark.parametrize('phase',['exit_prepare','exit_lane'])
def test_observer_forwards_only_current_aligned_lane_command(owner,monkeypatch,phase):
    from test_m8_exit_lane import scene, run
    from jetracer_ros2.m8_exit_lane import ExitLaneResolver
    from jetracer_ros2.m8_observers import M8LaneObserver
    from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega
    class Polygon(Stamped):
        def __init__(self):
            super().__init__();self.polygon=NS(points=[])
    monkeypatch.setattr(sys.modules['geometry_msgs.msg'],'PolygonStamped',Polygon,raising=False)
    monkeypatch.setattr(sys.modules['geometry_msgs.msg'],'Point32',NS,raising=False)
    image,r,g=scene(ordinary_correct=True)
    angle=run(ExitLaneResolver(),image,r,g)['angle'];r['raw_angle']=angle
    observer=M8LaneObserver(owner,.184);observer._phase(NS(data=phase))
    for i in range(180):  # Visible for >4 s even without another tape contact.
        observer.update(NS(sec=10+i//20,nanosec=(i%20)*50_000_000),r,(.2,angle,'TRACK'),
                        image=image,pose=(0.,0.,0.),geometry=g)
        if i==4:image[:]=0
    command=observer.command.messages[-1]
    assert observer.evidence.messages[-1].point.y==1.
    assert command.twist.linear.x==.2
    assert command.twist.angular.z==pytest.approx(inner_angle_to_omega(angle,.2))
    assert command.header.stamp==observer.evidence.messages[-1].header.stamp
    # A stale held command is not a fresh observation, even with a fitted path.
    observer.update(NS(sec=19,nanosec=0),r,(.2,angle,'HOLD'),image=image,pose=(0.,0.,0.),geometry=g)
    assert observer.command.messages[-1].twist.linear.x==0.
    # A live path with an unrelated steering command also cannot authorize it.
    observer.update(NS(sec=19,nanosec=50_000_000),r,(.2,angle-1.,'TRACK'),
                    image=image,pose=(0.,0.,0.),geometry=g)
    assert observer.command.messages[-1].twist.linear.x==0.
    observer.update(NS(sec=19,nanosec=100_000_000),None,(0.,0.,'LOST'))
    assert observer.evidence.messages[-1].point.x==0.
    assert observer.command.messages[-1].twist.linear.x==0.


def test_exit_line_keeps_cone_until_a_fresh_checked_lane(owner):
    confirm_exit(owner)
    proposal(owner)
    assert owner.output[-1]==(.10,-.13)
    exit_proposal(owner, v=0.)
    assert owner.output[-1]==(.10,-.13)
    exit_proposal(owner, age=.30)
    assert owner.output[-1]==(.10,-.13)
    exit_proposal(owner)
    assert owner.output[-1]==(.10,-.13)  # A connector alone cannot displace usable cones.
    exit_evidence(owner)
    exit_proposal(owner)
    assert owner.output[-1]==(.1,.04)
    assert owner.m8._m8_phase_sent == 'exit_lane'
    assert owner.m8._m8_exit.owns_lane
    # Once committed, a cone callback cannot overwrite the lane trajectory.
    proposal(owner)
    assert owner.output[-1]==(.1,.04)
    recovery_tick(owner, dt=.30)
    owner.m8.tick(owner.now)
    assert owner.output[-1]==(0.,0.)
    proposal(owner)
    assert owner.output[-1]==(.10,-.13)  # Loss of exit evidence returns to fresh checked cones.


def progress(owner, clear=True, age=0., frame='rear_axle'):
    msg=Stamped()
    t=owner.now-int(age*1e9)
    msg.header.stamp=NS(sec=t//10**9,nanosec=t%10**9)
    msg.header.frame_id=frame
    msg.point.x=float(clear)
    msg.point.y=2.
    msg.point.z=.03
    owner.m8._store_exit_progress(msg)


def test_lane_alone_cannot_leave_cones_but_measured_passage_can(owner):
    proposal(owner)
    exit_proposal(owner)
    owner.m8.tick(owner.now)
    assert owner.output[-1]==(.10,-.13)
    progress(owner, age=.30)
    assert owner.m8._m8_phase_sent == 'cone'
    progress(owner, frame='odom')
    assert owner.m8._m8_phase_sent == 'cone'
    progress(owner)
    assert owner.m8._m8_phase_sent == 'exit_prepare'
    assert owner.output[-1]==(.10,-.13)
    proposal(owner,v=0.)  # No cone command: use the role-verified connector, not blind hold.
    exit_proposal(owner)
    assert owner.output[-1]==(.1,.04)
    recovery_tick(owner, dt=.1, move=.16)
    progress(owner)
    exit_proposal(owner)
    assert owner._active=='M8'  # Ordinary lane still needs verified agreement.
    evidence=Stamped()
    evidence.header.frame_id='rear_axle'
    evidence.header.stamp=NS(sec=owner.now//10**9,nanosec=owner.now%10**9)
    evidence.point.x=evidence.point.y=1.
    owner.m8._store_exit_evidence(evidence)
    owner.m8.tick(owner.now)
    assert owner._active=='M1'


def test_expired_or_retracted_map_evidence_cannot_start_handoff(owner):
    progress(owner)
    recovery_tick(owner, dt=.30)
    proposal(owner)
    exit_proposal(owner)
    assert not owner.m8._m8_exit.owns_lane
    assert owner.output[-1]==(.10,-.13)
    exit_evidence(owner)
    progress(owner)
    assert owner.m8._m8_exit.owns_lane


def test_estop_during_exit_preparation_blocks_both_sources(owner):
    confirm_exit(owner)
    proposal(owner)
    owner.m8.event('ESTOP')
    exit_proposal(owner)
    assert owner.output[-1]==(0.,0.)
    assert not owner.m8._m8_exit.owns_lane


def test_exit_observer_remains_available_after_map_based_preparation(owner, monkeypatch):
    import numpy as np
    from jetracer_ros2 import m8_observers as obs
    monkeypatch.setattr(obs,'detect_m8_exit_stop_line',lambda im:NS(row_ratio=.8))
    observer=obs.M8StopLineObserver(owner,NS(calibrated=True,front_distance=lambda row:.3),
                                   NS(roi_top_ratio=.58))
    for phase in ('exit_prepare','exit_lane'):
        observer._phase(NS(data=phase))
        observer.update(np.zeros((480,640,3),np.uint8),NS(stamp=NS(sec=10,nanosec=0)))
    assert len(observer.exit.messages)==2
    assert not observer.entry.messages and not observer.partial.messages
