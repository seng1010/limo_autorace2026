"""Real manager callbacks with in-memory ROS transport; never sends to hardware.

M7_DONE and M2_DONE are explicit external events in the sequence test. This does
not claim that the current 3-lane node produces them. Geometry/perception and
physical parking accuracy require a separate on-vehicle test.
"""
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


class Twist:
    def __init__(self):
        self.linear = NS(x=0., y=0., z=0.)
        self.angular = NS(x=0., y=0., z=0.)


class Stamped:
    def __init__(self):
        self.header = NS(stamp=NS(sec=0, nanosec=0), frame_id='')
        self.twist = Twist()
        self.point = NS(x=0., y=0., z=0.)


class Pub:
    def __init__(self):
        self.messages = []
    def publish(self, msg):
        self.messages.append(msg)
    def get_subscription_count(self):
        return 1


class Node:
    overrides = {}
    def __init__(self, name):
        self.params = dict(self.overrides)
        self.now = 10_000_000_000
        self.pubs, self.subs, self.started = {}, {}, []
    def declare_parameter(self, key, value):
        self.params.setdefault(key, value)
    def get_parameter(self, key):
        return NS(value=self.params[key])
    def create_publisher(self, kind, topic, *args, **kwargs):
        return self.pubs.setdefault(topic, Pub())
    def create_subscription(self, kind, topic, callback, *args, **kwargs):
        self.subs[topic] = callback
    def create_timer(self, *args):
        return None
    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=self.now))
    def get_logger(self):
        return NS(**{k: lambda *args: None for k in ('info', 'warn', 'warning', 'error')})


@pytest.fixture
def modules(monkeypatch):
    fields = {
        'rclpy': dict(ok=lambda: True),
        'rclpy.node': dict(Node=Node),
        'rclpy.executors': dict(ExternalShutdownException=type('Shutdown', (Exception,), {})),
        'rclpy.qos': dict(DurabilityPolicy=NS(TRANSIENT_LOCAL=1),
                         QoSProfile=lambda **kwargs: kwargs),
        'geometry_msgs.msg': dict(Twist=Twist, PointStamped=Stamped, TwistStamped=Stamped),
        'nav_msgs.msg': dict(Odometry=NS),
        'std_msgs.msg': dict(Bool=NS, String=NS, Float32=NS, Header=NS),
        'ament_index_python.packages': dict(get_package_share_directory=lambda _: ''),
    }
    for name, attrs in fields.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    names = ['m8_control', 'mission_manager_node', 'parking_line_detection']
    for name in names:
        sys.modules.pop('jetracer_ros2.' + name, None)
    manager = importlib.import_module('jetracer_ros2.mission_manager_node')
    parking = importlib.import_module('jetracer_ros2.parking_line_detection')
    monkeypatch.setattr(manager.MissionManagerNode, '_start_process',
                        lambda self, name: self.started.append(name))
    monkeypatch.setattr(manager.MissionManagerNode, '_stop_process', lambda *args, **kw: None)
    Node.overrides = {}
    yield NS(manager=manager, parking=parking)
    for name in names:
        sys.modules.pop('jetracer_ros2.' + name, None)
    Node.overrides = {}


def owner(modules, mission='M5', **overrides):
    Node.overrides = dict(start_mission=mission, **overrides)
    n = modules.manager.MissionManagerNode()
    n._odom_ns = n.now
    return n


def event(n, value):
    n._mission_event_callback(NS(data=value))


def lane(n, v=.25, w=.10):
    msg = Twist()
    msg.linear.x, msg.angular.z = v, w
    n._store_cmd('lane', msg)


def proposal(n, v=.10, w=-.10):
    msg = Stamped()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(n.now, 10**9)
    msg.twist.linear.x, msg.twist.angular.z = v, w
    return msg


def step(n, dt=.05, *, line=None, speed=0., light=None, refresh_odom=True):
    n.now += int(dt * 1e9)
    if refresh_odom:
        n._odom_ns, n._odom_speed = n.now, speed
    if line is not None:
        n._stop_line_callback(NS(data=line))
    if light is not None:
        n._traffic_callback(NS(data=light))
    lane(n)
    n._control_tick()


@pytest.mark.parametrize('mission', ['M5', 'M6'])
def test_pre_m7_lane_commands_do_not_require_manager_odom(modules, mission):
    n = owner(modules, mission)
    n._odom_ns = 0
    step(n, refresh_odom=False)
    assert n.cmd_pub.messages[-1].linear.x == .25
    assert n.cmd_pub.messages[-1].angular.z == .10
    assert n.m8._m8_phase_sent == 'off'
    n._speed_limit_callback(NS(data=.13))
    step(n, refresh_odom=False)
    cmd = n.cmd_pub.messages[-1]
    assert cmd.linear.x == .13
    assert cmd.angular.z / cmd.linear.x == pytest.approx(.10 / .25)
    assert n._active == mission


@pytest.mark.parametrize('mission', ['M5', 'M6', 'M3', 'M7', 'M2', 'M1', 'M4'])
def test_cone_proposals_cannot_override_unarmed_lane_missions(modules, mission):
    n = owner(modules, mission, m8_entry_arm_after_m=-1.,
              m8_entry_arm_near_line_m=.45, m8_entry_precheck_m=.55)
    lane(n, .22, -.40)
    before = len(n.cmd_pub.messages)
    n.m8._store_cone_cmd(proposal(n, .10, .15))  # deliberately opposite steering
    n.m8._store_exit_lane(proposal(n, .10, .12))
    assert len(n.cmd_pub.messages) == before  # observers do not own final output
    n._control_tick()
    cmd = n.cmd_pub.messages[-1]
    assert (cmd.linear.x, cmd.angular.z) == (.22, -.40)
    assert n._active == mission


def test_m8_cone_steering_is_not_added_to_or_overwritten_by_lane(modules):
    n = owner(modules, 'M8')
    lane(n, .22, -.40)
    n.m8._store_cone_cmd(proposal(n, .10, .15))
    for v, w in ((.22, -.40), (.25, .40), (0., 0.)):
        lane(n, v, w)
        n._control_tick()
        cmd = n.cmd_pub.messages[-1]
        assert (cmd.linear.x, cmd.angular.z) == (.10, .15)
    n.m8._store_cone_cmd(proposal(n, 0., 0.))
    lane(n, .22, -.40)
    n._control_tick()
    cmd = n.cmd_pub.messages[-1]
    assert (cmd.linear.x, cmd.angular.z) == (0., 0.)  # no blind lane fallback in cones
    n._advance('test: leave M8')  # exit geometry is covered by the full handoff simulation
    assert n._active == 'M1'
    lane(n, .22, -.40)
    n.m8._store_cone_cmd(proposal(n, .10, .15))
    n._control_tick()
    cmd = n.cmd_pub.messages[-1]
    assert (cmd.linear.x, cmd.angular.z) == (.22, -.40)


@pytest.mark.parametrize('mission', ['M5', 'M3', 'M7', 'M2', 'M8', 'M1', 'M4'])
def test_estop_blocks_events_callbacks_and_output_until_explicit_clear(modules, mission):
    n = owner(modules, mission)
    event(n, 'ESTOP')
    event(n, mission + '_DONE')
    event(n, 'ARM_M8')
    n._obstacle_callback(NS(data='clear'))
    n.m8._store_cone_cmd(proposal(n))  # immediate callback path must also stop
    step(n)
    assert n._active == mission and n._operator_stop
    assert n.cmd_pub.messages[-1].linear.x == 0.
    event(n, 'ESTOP_CLEAR')
    assert not n._operator_stop and all(t == 0 for t in n._last_cmd_ns.values())
    assert n.m8._cone_cmd is None
    n._control_tick()
    assert n.cmd_pub.messages[-1].linear.x == 0.


def test_clear_does_not_clear_a_mission_specific_stop(modules):
    n = owner(modules, 'M7')
    n._blocked = True
    event(n, 'ESTOP')
    event(n, 'ESTOP_CLEAR')
    step(n)
    assert n._blocked and n.cmd_pub.messages[-1].linear.x == 0.


def test_red_does_not_depart_after_old_thirty_second_timeout(modules):
    n = owner(modules, 'M1', m1_trigger='light_or_line', m1_skip_lines=0)
    for _ in range(45):
        step(n, 1., line=.005, light='red')
    assert n._m1.state == 'hold' and n._active == 'M1'
    assert n.cmd_pub.messages[-1].linear.x == 0.
    step(n, light='green')
    assert n._m1.state == 'go' and n.cmd_pub.messages[-1].linear.x > 0.


def test_stale_green_cannot_authorize_arrival(modules):
    n = owner(modules, 'M1', m1_skip_lines=0)
    n._traffic_callback(NS(data='green'))
    step(n, 1., line=.40, speed=.20)
    step(n, line=.40, speed=.20)  # existing line counter confirms on two updates
    assert n._fresh_traffic(n.now) == 'unknown'
    assert n._m1.state == 'stopping'
    n._traffic_ns = n.now + 1
    assert n._fresh_traffic(n.now) == 'unknown'


def test_m1_missing_odom_never_confirms_stop_or_completes_wait(modules):
    n = owner(modules, 'M1', m1_trigger='light_or_line', m1_skip_lines=0)
    for _ in range(10):
        step(n, .1, line=.005, light='red')
    controller = n._m1
    assert controller.state == 'hold'
    n.now += 600_000_000  # beyond the configured 0.5-second freshness window
    for _ in range(50):
        step(n, .2, line=.005, light='green', refresh_odom=False)
    assert n._active == 'M1' and controller.state == 'stopping'
    assert n._measured_speed(n.now) is None
    assert n.cmd_pub.messages[-1].linear.x == 0.
    step(n, line=.005, light='red')
    assert controller.state == 'stopping'  # renewed 0.3-second stop confirmation


@pytest.mark.parametrize('odom_pattern', ['continuous', 'single_gap', 'repeated_gaps', 'absent'])
def test_m3_main_dwell_is_not_restarted_by_odom_gaps(modules, odom_pattern):
    """M3 retains main's command-speed fallback; it is NOT measured-stop proof."""
    n = owner(modules, 'M3', m3_trigger='line')
    start = n.now
    if odom_pattern == 'absent':
        n._odom_ns = 0
    first_hold = None
    for i in range(1, 101):
        elapsed = i * .05
        refresh = (odom_pattern == 'continuous'
                   or odom_pattern == 'single_gap' and not 3.0 <= elapsed < 3.8
                   or odom_pattern == 'repeated_gaps' and i % 20 == 0)
        step(n, .05, line=.005, refresh_odom=refresh)
        if n._m3.state == 'hold':
            if first_hold is None:
                first_hold = n._m3.hold_start
            assert n._m3.hold_start == first_hold
        if n._m3.state == 'go':
            break
        assert n.cmd_pub.messages[-1].linear.x == 0.
    assert first_hold is not None and n._m3.state == 'go'
    assert (n.now - start) / 1e9 == pytest.approx(3.9)
    assert n.now / 1e9 - first_hold >= n._m3.config.hold_sec
    assert n.cmd_pub.messages[-1].linear.x == .25
    assert n.cmd_pub.messages[-1].angular.z == .10
    assert n._active == 'M3'  # go is not completion: still needs measured distance
    n._odom.s += .51
    step(n)
    assert n._active == 'M7' and n.cmd_pub.messages[-1].linear.x == 0.
    step(n)
    assert n.cmd_pub.messages[-1].linear.x == .25


def test_m3_fallback_is_local_and_fresh_measured_motion_wins(modules):
    n = owner(modules, 'M3', m3_trigger='line')
    for _ in range(10):
        step(n, .1, line=.005, speed=.03)
    assert n._m3.state == 'stopping'  # zero command does not override fresh motion
    for _ in range(100):
        step(n, .05, line=.005, speed=0.)
        if n._m3.state == 'go':
            break
    assert n._m3.state == 'go'
    n._odom.s += .51
    step(n)
    assert n._active == 'M7'
    step(n, .6, refresh_odom=False)
    assert n._measured_speed(n.now) is None  # no global command-speed fallback


def test_m3_missing_odom_does_not_stop_ordinary_approach(modules):
    n = owner(modules, 'M3', m3_trigger='line')
    n._odom_ns = 0
    for _ in range(10):
        step(n, .1, line=-1., refresh_odom=False)
        assert n._m3.state == 'approach'
        assert n.cmd_pub.messages[-1].linear.x == .25
        assert n.cmd_pub.messages[-1].angular.z == .10


def test_m3_explicit_estop_still_restarts_stop_confirmation(modules):
    n = owner(modules, 'M3', m3_trigger='line')
    for _ in range(20):
        step(n, .1, line=.005)
    assert n._m3.state == 'hold'
    event(n, 'ESTOP')
    for _ in range(50):
        step(n, .1, line=.005, refresh_odom=False)
    assert n._m3.state == 'stopping' and n.cmd_pub.messages[-1].linear.x == 0.
    event(n, 'ESTOP_CLEAR')
    step(n, .1, line=.005)
    assert n._m3.state == 'stopping'
    for _ in range(40):
        step(n, .1, line=.005)
    assert n._m3.state == 'go'


def test_invalid_or_future_inputs_do_not_create_motion(modules):
    n = owner(modules)
    lane(n, float('nan'))
    n._control_tick()
    assert n.cmd_pub.messages[-1].linear.x == 0.
    lane(n)
    n._last_cmd_ns['lane'] = n.now + 1
    n._control_tick()
    assert n.cmd_pub.messages[-1].linear.x == 0.
    n._odom_callback(NS(pose=NS(pose=NS(position=NS(x=float('nan'), y=0.))),
                        twist=NS(twist=NS(linear=NS(x=0.)))))
    assert not n._odom_fresh(n.now)


def test_m3_completion_does_not_overwrite_transition_stop_in_same_tick(modules):
    n = owner(modules, 'M3')
    n._m3.state, n._m3.go_start_s = 'go', 0.
    n._odom.s = .60
    step(n)
    assert n._active == 'M7'
    assert n.cmd_pub.messages[-1].linear.x == 0.
    step(n)
    assert n.cmd_pub.messages[-1].linear.x == .25


def test_parking_approach_keeps_lane_curvature_and_waits_for_measured_stop(modules):
    n = owner(modules, 'M4')
    n._yolo_callback(NS(data='parking_sign'))
    n._parking_state_callback(NS(data='complete'))  # stale pre-start status ignored
    step(n, line=.20, speed=.15)
    cmd = n.cmd_pub.messages[-1]
    assert 0 < cmd.linear.x < .25
    assert cmd.angular.z / cmd.linear.x == pytest.approx(.10 / .25)
    assert not n.parking_start_pub.messages[-1].data and not n._blocked
    for _ in range(10):
        step(n, line=.005, speed=.03)
    assert n._m4_state == 'stopping'
    for _ in range(12):
        step(n, line=.005, speed=0.)
    assert n._m4_state == 'parking' and n.parking_start_pub.messages[-1].data
    assert n.cmd_pub.messages[-1].linear.x == 0.  # no fresh parking command yet
    n._store_cmd('parking', proposal(n, .50).twist)
    n._control_tick()
    assert n.cmd_pub.messages[-1].linear.x == .50  # tuned sequence is not lane-capped
    n._parking_state_callback(NS(data='complete'))
    step(n)
    assert n._blocked and n._m4_state == 'complete'
    assert n.cmd_pub.messages[-1].linear.x == 0.


def test_lost_parking_line_or_odom_cannot_start_parking(modules):
    n = owner(modules, 'M4')
    step(n, line=.20)
    for _ in range(15):
        step(n, .1, line=-1.)
    assert n._m4_state == 'stopping' and not n._m4_stop_latched
    assert n.cmd_pub.messages[-1].linear.x == 0.
    n.now += 600_000_000
    for _ in range(15):
        step(n, .1, line=.005, refresh_odom=False)
    assert not n.parking_start_pub.messages[-1].data


def test_m8_exit_starts_fresh_traffic_and_m4_starts_lane_approach(modules):
    n = owner(modules, 'M8')
    n._traffic_callback(NS(data='green'))  # not M1 yet: ignore
    n.m8._m8_entry.arm(0.)
    n.m8._m8_exit.reset(0.)
    n.m8._m8_exit.line_s = 1.
    n._odom.s = 1.1
    n.m8._prepare_exit(n.now)
    n.m8._store_exit_lane(proposal(n))
    n.now += 200_000_000
    n._odom_ns, n._odom.s = n.now, 1.31
    evidence = proposal(n)
    evidence.header.frame_id = 'rear_axle'
    evidence.point.x = evidence.point.y = 1.
    n.m8._store_exit_evidence(evidence)
    n.m8._store_exit_lane(proposal(n))
    assert n._active == 'M1' and n._traffic == 'unknown'
    assert 'mission_traffic_light_node' in n.started
    n._m1.state, n._m1.go_start_s = 'go', n._odom.s - .6
    n._store_cmd('parking', proposal(n, .50).twist)
    step(n)
    assert n._active == 'M4' and n._last_cmd_ns['parking'] == 0
    assert 'parking_line_detection_managed' in n.started
    assert n.cmd_pub.messages[-1].linear.x == 0.
    step(n)
    assert n._m4_state == 'approach' and n.cmd_pub.messages[-1].linear.x == .25


def test_normal_course_event_order_and_m8_evidence_gate(modules):
    n = owner(modules)
    expected = ('M5', 'M6', 'M3', 'M7', 'M2', 'M8', 'M1', 'M4')
    assert modules.manager.MISSION_ORDER == expected
    for mission in expected[:4]:
        assert n._active == mission
        event(n, mission + '_DONE')  # producer contract, not simulated visual success
    assert n._active == 'M2'
    event(n, 'M2_DONE')
    step(n)
    assert n._active == 'M2'  # no cone proposal -> cannot advance merely on DONE
    n.m8._store_cone_cmd(proposal(n))
    step(n)
    assert n._active == 'M8'
    event(n, 'M8_DONE')
    assert n._active == 'M8'  # physical exit evidence still required


def test_shipped_traffic_timeout_does_not_restore_thirty_second_bypass():
    params = (Path(__file__).parents[1] / 'params/params.yaml').read_text()
    assert 'm1_max_wait_sec: 0.0' in params


def test_full_run_m8_profile_keeps_lane_then_uses_existing_preflight(modules):
    n = owner(modules, 'M2', m8_entry_arm_after_m=-1.,
              m8_entry_arm_near_line_m=.45, m8_entry_precheck_m=.55)
    assert n.m8._m8_phase_sent == 'watch_entry'
    step(n)
    assert n.cmd_pub.messages[-1].linear.x == .25
    for _ in range(3):
        n.now += 50_000_000
        n._odom_ns = n.now
        n.m8._m8_visual_line_callback(NS(data=.54))
    step(n)
    assert n.m8._m8_phase_sent == 'preflight' and n._active == 'M2'
    assert n.cmd_pub.messages[-1].linear.x == .25  # precheck must not replace lane
    n._odom.s += .12
    step(n)
    assert n.m8._m8_phase_sent == 'approach' and n._active == 'M2'
    assert n.cmd_pub.messages[-1].linear.x == 0.  # no checked proposal yet
    n.m8._store_cone_cmd(proposal(n))
    step(n)
    assert n._active == 'M8' and n.cmd_pub.messages[-1].linear.x == .10
    params = (Path(__file__).parents[1] / 'params/params.yaml').read_text()
    assert 'm8_entry_arm_near_line_m: 0.45' in params
    assert 'm8_entry_precheck_m: 0.55' in params


def test_manager_and_parking_exchange_real_callbacks_to_completion(modules, monkeypatch):
    n = owner(modules, 'M4')
    clock = [n.now / 1e9]
    monkeypatch.setattr(modules.parking.time, 'monotonic', lambda: clock[0])
    Node.overrides = {'managed_mode': True}
    p = modules.parking.ParkingMissionNode()
    offsets = [0, 0, 0, 0]
    def forward(pub, callback, slot):
        for message in pub.messages[offsets[slot]:]:
            callback(message)
        offsets[slot] = len(pub.messages)
    def exchange():
        forward(n.active_pub, p._active_cb, 0)
        forward(n.parking_start_pub, p._start_cb, 1)
        p._tick()
        forward(p.cmd_pub, lambda msg: n._store_cmd('parking', msg), 2)
        forward(p.state_pub, n._parking_state_callback, 3)
    for i in range(130):
        step(n, line=.005 if i >= 10 else .20, speed=0. if i >= 15 else .15)
        clock[0] = n.now / 1e9
        exchange()
        if i < 15:
            assert p._state == 'WAIT_START'
    assert p._state == 'DONE' and n._m4_state == 'complete' and n._blocked
    assert any(m.linear.x == .50 for m in n.cmd_pub.messages)
    assert n.cmd_pub.messages[-1].linear.x == 0.


@pytest.mark.parametrize('failure', ['start_timeout', 'mission_timeout', 'other_mission'])
def test_managed_parking_cannot_run_without_current_authority(modules, monkeypatch, failure):
    clock = [0.]
    monkeypatch.setattr(modules.parking.time, 'monotonic', lambda: clock[0])
    Node.overrides = {'managed_mode': True}
    p = modules.parking.ParkingMissionNode()
    p._state, p._state_start = 'PARKING', 0.
    p._start_cb(NS(data=True))
    p._active_cb(NS(data='M4'))
    if failure == 'start_timeout':
        clock[0] = .6
    elif failure == 'mission_timeout':
        clock[0] = 1.6
        p._start_cb(NS(data=True))
    else:
        p._active_cb(NS(data='M1'))
    p._tick()
    assert p.cmd_pub.messages[-1].linear.x == 0.
    assert p._phase_index == 0


def test_managed_parking_interface_preserves_phases_and_pauses_elapsed_time(modules, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(modules.parking.time, 'monotonic', lambda: clock[0])
    Node.overrides = {'managed_mode': True}
    p = modules.parking.ParkingMissionNode()
    original = list(p.parking_phases)
    assert original == [('forward', 1.7, .50, 0.), ('turn', 1.8, .50, -.45),
                        ('enter', .1, .50, 0.)]
    assert '/cmd_vel' not in p.pubs and 'mission/parking/cmd_vel' in p.pubs
    p.go()
    p._tick()
    assert p._state == 'WAIT_START'
    p._active_cb(NS(data='M4'))
    p._start_cb(NS(data=True))
    p._tick()
    assert p._state == 'STOP_WAIT'  # never own APPROACH in managed mode
    for _ in range(25):
        clock[0] += .05
        p._active_cb(NS(data='M4'))
        p._start_cb(NS(data=True))
        p._tick()
    assert p._state == 'PARKING'
    elapsed = clock[0] - p._state_start
    p._start_cb(NS(data=False))
    clock[0] += 10.
    p._tick()
    assert p.cmd_pub.messages[-1].linear.x == 0.
    p._active_cb(NS(data='M4'))
    p._start_cb(NS(data=True))
    p._tick()
    assert clock[0] - p._state_start == pytest.approx(elapsed)
    assert p._phase_index == 0 and p.parking_phases == original
    for _ in range(80):
        clock[0] += .05
        p._active_cb(NS(data='M4'))
        p._start_cb(NS(data=True))
        p._tick()
    assert p._state == 'DONE' and p.state_pub.messages[-1].data == 'complete'


def test_standalone_parking_still_owns_its_original_approach(modules, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(modules.parking.time, 'monotonic', lambda: clock[0])
    p = modules.parking.ParkingMissionNode()
    assert '/cmd_vel' in p.pubs
    p.go()
    p._tick()
    assert p._state == 'APPROACH'
    p._tick()
    assert p.cmd_pub.messages[-1].linear.x == .25


def test_managed_entrypoint_is_registered_and_preserves_tuning_arguments(modules, monkeypatch):
    calls = []
    monkeypatch.setattr(modules.parking, 'main', lambda args: calls.append(args))
    modules.parking.managed_main(['parking_line_detection_managed', '--ros-args',
                                 '-p', 'forward_sec:=2.2'])
    args = calls[-1]
    assert args.count('--ros-args') == 1
    assert 'forward_sec:=2.2' in args and args[-1] == 'managed_mode:=true'
    setup = (Path(__file__).parents[1] / 'setup.py').read_text()
    assert ('parking_line_detection_managed = '
            'jetracer_ros2.parking_line_detection:managed_main') in setup


@pytest.mark.parametrize('start', ['M2', 'M7'])
def test_curve_approach_through_m8_traffic_light_and_final_parking(modules, monkeypatch, start):
    """Continuous callback simulation: no DONE events or forced controller states.

    Inputs emulate observed line distances, checked cone/lane proposals, light
    states and odom. It tests authority/lifecycle, not perception or physics.
    """
    n = owner(modules, start, normal_speed=.30, m8_entry_arm_after_m=-1.,
              m8_entry_arm_near_line_m=.45, m8_entry_precheck_m=.55,
              m1_skip_lines=1, m1_stop_offset_m=.03)
    if start == 'M7':
        import json
        # No M7_DONE injection: use the same fresh route-progress interface
        # as the lane reporter; its sign/plan/movement producer is tested in
        # test_mission_handoffs.py. Here exercise direct M7 -> M8 -> M1 -> M4.
        n._lane3_progress_callback(NS(data=json.dumps(dict(
            stamp_ns=n.now, fork_side=None, route_complete=True))))
    # Before the cone entrance, preserve the curved lane command.
    lane(n, .25, -.12)
    n._control_tick()
    assert n.cmd_pub.messages[-1].angular.z == -.12
    for _ in range(3):
        n.now += 50_000_000
        n._odom_ns = n.now
        n.m8._m8_visual_line_callback(NS(data=.54))
    assert n.m8._m8_phase_sent == 'preflight'
    n._odom.s = .12
    step(n)
    n.m8._store_cone_cmd(proposal(n))
    assert n._active == 'M8'

    # Reach the exit line; a checked exit lane plus odom crossing completes M8.
    n._odom.s = 1.5
    for _ in range(3):
        step(n)
        n.m8._store_cone_cmd(proposal(n))
        line_msg = proposal(n)
        line_msg.header.frame_id, line_msg.point.x = 'front_axle', .40
        n.m8._m8_exit_line_callback(line_msg)
    assert n.m8._m8_phase_sent == 'exit_prepare'
    evidence = proposal(n)
    evidence.header.frame_id = 'rear_axle'
    evidence.point.x = evidence.point.y = 1.
    n.m8._store_exit_evidence(evidence)
    n.m8._store_exit_lane(proposal(n))
    n.now += 100_000_000
    n._odom_ns, n._odom.s = n.now, 2.21
    evidence = proposal(n)
    evidence.header.frame_id = 'rear_axle'
    evidence.point.x = evidence.point.y = 1.
    n.m8._store_exit_evidence(evidence)
    n.m8._store_exit_lane(proposal(n))
    assert n._active == 'M1'

    # The configured first marking is skipped, then the traffic stop line holds.
    n._odom.s = 2.4
    step(n, line=-1., speed=.20, light='green')  # begin M1 well before the first marking
    n._odom.s = 4.5  # regression: the old 2m bypass skipped M1 here
    for _ in range(3):
        step(n, line=.20, speed=.15, light='red')
    assert n._m1.lines_seen == 1 and n._m1.state == 'approach'
    n._odom.s = 5.3
    for _ in range(3):
        step(n, line=.40, speed=.15, light='red')
    assert n._m1.lines_seen == 2 and n._m1.state == 'stopping'
    n._odom.s = 5.68
    for _ in range(25):
        step(n, line=.02, light='red')
    assert n._m1.state == 'hold' and n.cmd_pub.messages[-1].linear.x == 0.
    step(n, line=.02, light='green')
    assert n._m1.state == 'go' and n.cmd_pub.messages[-1].linear.x > 0.
    n._odom.s += .6
    step(n, line=-1., light='green')
    assert n._active == 'M4'

    # The third course line is the FAR edge of the crosswalk, not parking.
    for _ in range(3):
        step(n, line=.20, speed=.15)
    assert n._parking_exit_s == pytest.approx(n._odom.s+.20)
    assert n._m4_state == 'approach' and not n.parking_start_pub.messages[-1].data
    n._odom.s += .35
    step(n, line=-1., speed=.15)
    n._odom.s = 7.0  # Fourth line is at s=7.20, beyond the crosswalk end.

    # Wire the managed parking node to the actual manager transport callbacks.
    clock = [n.now / 1e9]
    monkeypatch.setattr(modules.parking.time, 'monotonic', lambda: clock[0])
    Node.overrides = {'managed_mode': True}
    parking = modules.parking.ParkingMissionNode()
    parking._active_cb(NS(data='M4'))
    state_offset = 0
    for i in range(140):
        n._odom.s = 7.0 if i < 10 else 7.195
        step(n, line=.20 if i < 10 else .005, speed=.15 if i < 15 else 0.)
        clock[0] = n.now / 1e9
        parking._active_cb(NS(data=n._active))
        parking._start_cb(n.parking_start_pub.messages[-1])
        parking._tick()
        n._store_cmd('parking', parking.cmd_pub.messages[-1])
        for message in parking.state_pub.messages[state_offset:]:
            n._parking_state_callback(message)
        state_offset = len(parking.state_pub.messages)
    assert n._m4_state == 'complete' and n._blocked
    assert n.cmd_pub.messages[-1].linear.x == 0.
    transitions = []
    for msg in n.active_pub.messages:
        if not transitions or transitions[-1] != msg.data:
            transitions.append(msg.data)
    assert transitions == [start, 'M8', 'M1', 'M4']


def test_crosswalk_end_seen_during_m1_is_not_recounted_as_parking_in_m4(modules):
    n=owner(modules,'M1')
    n._m1.state='go';n._m1.go_start_s=0.
    n._m1._line_detector.last_line_s=.03
    n._odom.s=.4
    for _ in range(3):step(n,line=.35,speed=.15,light='green')
    assert n._parking_exit_s==pytest.approx(.75)
    n._odom.s=.6
    step(n,line=.15,speed=.15,light='green')
    assert n._active=='M4'
    n._odom.s=.75
    for _ in range(20):step(n,line=0.,speed=0.)
    assert n._m4_state=='approach' and not n.parking_start_pub.messages[-1].data
    n._odom.s=.88
    step(n,line=-1.,speed=.15)
    n._odom.s=1.10
    for _ in range(3):step(n,line=.30,speed=.15)
    assert n._parking_target_s==pytest.approx(1.40) and n._m4_state=='stopping'
    n._odom.s=1.395
    for _ in range(12):step(n,line=.005,speed=0.)
    assert n._m4_state=='parking' and n.parking_start_pub.messages[-1].data


def test_parking_gate_requires_distinct_messages_and_does_not_count_reverse(modules):
    n=owner(modules,'M1')
    n._m1.state='go';n._m1.go_start_s=0.;n._m1._line_detector.last_line_s=.03
    n._odom.s=.4
    step(n,line=.35,speed=.15,light='green')
    for _ in range(8):step(n,speed=.15,light='green')
    assert n._parking_exit_s is None
    for _ in range(3):step(n,line=.35,speed=-.10,light='green')
    assert n._parking_exit_s is None
    step(n,line=.35,speed=.15,light='green')
    assert n._parking_exit_s==pytest.approx(.75)


@pytest.mark.parametrize('old_limit', [0., 2.])
@pytest.mark.parametrize('arrival_light', ['red', 'green', 'unknown'])
def test_m1_two_lines_beyond_two_metres_always_stop_before_fresh_green(
        modules, old_limit, arrival_light):
    n = owner(modules, 'M1', m1_max_approach_distance_m=old_limit)
    step(n, line=-1., speed=.20, light=arrival_light)
    n._odom.s = 2.01
    for _ in range(3):
        step(n, line=.12, speed=.15, light=arrival_light)
    assert n._active == 'M1' and n._m1.lines_seen == 1
    assert n._m1.state == 'approach'
    assert n.cmd_pub.messages[-1].linear.x == .15
    assert n.cmd_pub.messages[-1].angular.z == pytest.approx(.10*.15/.25)
    assert 'parking_line_detection_managed' not in n.started
    n._odom.s = 2.3
    step(n, line=-1., speed=.15, light=arrival_light)
    n._odom.s = 3.2
    for _ in range(3):
        step(n, line=.20, speed=.15, light=arrival_light)
    assert n._m1.lines_seen == 2 and n._m1.state == 'stopping'
    for _ in range(10):
        step(n, line=.005, speed=.04, light=arrival_light)
    assert n._active == 'M1' and n._m1.state == 'stopping'
    assert n.cmd_pub.messages[-1].linear.x == 0.
    # An old green must not authorize departure after the measured stop.
    n._traffic_callback(NS(data='green'))
    step(n, .6, line=.005, speed=0.)
    for _ in range(10):
        step(n, line=.005, speed=0.)
    assert n._m1.state == 'hold' and n.cmd_pub.messages[-1].linear.x == 0.
    step(n, line=.005, light='green')
    assert n._m1.state == 'go' and n._active == 'M1'
    n._odom.s += .6
    step(n, line=-1., light='green')
    assert n._active == 'M4'


def test_m1_green_on_arrival_waits_for_full_measured_stop(modules):
    n = owner(modules, 'M1', m1_skip_lines=0)
    for _ in range(8):
        step(n, line=.005, speed=.02, light='green')
    assert n._m1.state == 'stopping' and n.cmd_pub.messages[-1].linear.x == 0.
    for _ in range(5):
        step(n, line=.005, speed=0., light='green')
    assert n._m1.state == 'stopping'
    for _ in range(3):
        step(n, line=.005, speed=0., light='green')
    assert n._m1.state == 'go' and n.cmd_pub.messages[-1].linear.x > 0.


def test_m1_skip_result_is_never_a_parking_start(modules):
    n = owner(modules, 'M1')
    decision = NS(done=True, skipped=True, state='skipped', speed_limit=None)
    limit = n._report(decision, .12, .15, 'done', 'skip')
    assert limit == 0. and n._active == 'M1'
    assert 'parking_line_detection_managed' not in n.started


def test_m2_prewarms_cone_observation_without_transferring_control(modules):
    n = owner(modules, 'M2', m8_entry_arm_near_line_m=.45, m8_entry_precheck_m=.55)
    assert 'mission_cone_avoid_node' in n.started
    # A warm planner may already see a gate. It must not steal an early curve.
    for _ in range(60):
        n.now += 50_000_000
        n._odom_ns = n.now
        n.m8._store_cone_cmd(proposal(n))
        lane(n, .22, -.12)
        n._control_tick()
        assert n._active == 'M2' and n.m8._m8_phase_sent == 'watch_entry'
        assert n.cmd_pub.messages[-1].linear.x == .22
        assert n.cmd_pub.messages[-1].angular.z == -.12
    # Fresh partial tape plus the already-running planner can hand over before
    # the tape's 0.8s lifetime elapses. No widening of its evidence window.
    for _ in range(3):
        n.now += 50_000_000
        n._odom_ns = n.now
        cue = proposal(n)
        cue.header.frame_id, cue.point.x = 'front_axle', .49
        n.m8._m8_partial_callback(cue)
        n.m8._store_cone_cmd(proposal(n))
        n._control_tick()
    assert n._active == 'M8' and n.cmd_pub.messages[-1].linear.x == .10


def test_external_launch_ownership_does_not_start_duplicate_cone_node(modules):
    n = owner(modules, 'M2', auto_start_mission_nodes=False)
    assert not n.started
    step(n)
    assert n._active == 'M2' and n.cmd_pub.messages[-1].linear.x > 0.


def test_m4_half_second_dwell_resets_on_movement_and_odom_loss(modules):
    n = owner(modules, 'M4')
    for _ in range(9):
        step(n, line=.005, speed=0.)
    assert n._m4_state == 'stopping' and not n.parking_start_pub.messages[-1].data
    step(n, line=.005, speed=.03)
    for _ in range(9):
        step(n, line=.005, speed=0.)
    assert not n.parking_start_pub.messages[-1].data
    step(n, .6, line=.005, refresh_odom=False)
    for _ in range(9):
        step(n, line=.005, speed=0.)
    assert not n.parking_start_pub.messages[-1].data
    for _ in range(3):
        step(n, line=.005, speed=0.)
    assert n._m4_state == 'parking' and n.parking_start_pub.messages[-1].data


def test_shipped_m1_two_line_stop_and_parking_dwell_profile():
    params = (Path(__file__).parents[1] / 'params/params.yaml').read_text()
    for setting in ("m1_trigger: 'light_or_line'", 'm1_stop_on_green_arrival: true',
                    'm1_max_approach_distance_m: 0.0', 'm1_skip_lines: 1',
                    'm1_slow_speed: 0.15', 'm4_full_stop_sec: 0.5'):
        assert setting in params
