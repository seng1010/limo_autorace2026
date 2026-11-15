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


def test_exit_line_transfers_only_to_fresh_checked_lane_and_waits_otherwise(owner):
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
    assert owner.m8._m8_exit.owns_lane
    proposal(owner)
    assert owner.output[-1]==(0.,0.)
    lane=Stamped()
    lane.header.stamp=NS(sec=owner.now//10**9,nanosec=owner.now%10**9)
    lane.twist.linear.x=.1
    lane.twist.angular.z=.04
    owner.m8._store_exit_lane(lane)
    assert owner.output[-1]==(.1,.04)
