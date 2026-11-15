"""M8-only authority adapter. The shared manager remains the sole /cmd_vel owner.

Integration hooks: construct, event, enter, tick. Other missions stay in the
race manager; cone state strings cannot latch/overwrite an operator ESTOP.
"""
import json
import math
from geometry_msgs.msg import PointStamped, Twist, TwistStamped
from std_msgs.msg import Bool, Float32, Header, String
from rclpy.qos import DurabilityPolicy, QoSProfile
from jetracer_ros2.m8_entry import (
    M8_APPROACH_RISK_TOPIC, M8_COMMAND_TIMEOUT, M8_CONTROL_TIMING_TOPIC,
    M8_EXIT_CHECK_TOPIC, M8_EXIT_COMMAND_TOPIC,
    M8_EXIT_LINE_TOPIC, M8_PHASE_TOPIC, M8_PARTIAL_ENTRY_TOPIC, M8_RECOVERY_COMMAND_TOPIC,
    MIN_M8_SPEED, M8EntryGate, M8ExitGate, M8PartialEntryGate, M8RecoveryBudget,
    m8_entry_policy, m8_phase, select_m8_command)

M8_REVERSE_SPEED = -.02


class M8Control:
    def __init__(self,node):
        self.node=node
        defaults = {
            'm8_entry_arm_after_m': -1.0,
            'm8_entry_arm_near_line_m': -1.0,
            'm8_entry_precheck_m': -1.0,
            'm8_entry_pass_margin_m': 0.02,
            'm8_entry_search_limit_m': 0.50,
            'm8_entry_approach_speed': 0.10,
            'm8_exit_min_gap_m': 0.60,
            'm8_exit_pass_margin_m': 0.30,
            'm8_exit_settle_m': 0.15,
        }
        for name,value in defaults.items():
            node.declare_parameter(name,value)
        p=lambda name: node.get_parameter(name).value
        self._cone_cmd = None
        self._cone_stamp_ns = 0
        self._cone_rx_ns = 0
        self._exit_lane_cmd = None
        self._exit_lane_stamp_ns = 0
        self._exit_lane_rx_ns = 0
        self._recovery_offer = None
        self._recovery_budget = M8RecoveryBudget()
        self._exit_check_enabled = False
        self._m8_phase_sent = None
        self._m8_exit_handoff_ns = 0
        self._m8_entry = M8EntryGate(p('m8_entry_pass_margin_m'), p('m8_entry_search_limit_m'))
        self._m8_partial = M8PartialEntryGate()
        self._m8_exit = M8ExitGate(p('m8_exit_min_gap_m'), p('m8_exit_pass_margin_m'),
                                   p('m8_exit_settle_m'), handoff_on_detection=True)
        if not (.30 <= self._m8_exit.min_gap <= 20.0 and
                .20 <= self._m8_exit.pass_margin <= .60 and
                .10 <= self._m8_exit.settle_distance <= .30):
            raise ValueError('Invalid M8 exit parameters')
        self._m8_arm_after = p('m8_entry_arm_after_m')
        self._m8_near_line = p('m8_entry_arm_near_line_m')
        self._m8_precheck = p('m8_entry_precheck_m')
        self._m8_risk_ns = 0
        self._m8_reversing = False
        self._m8_post_reverse_ns = 0
        self._m8_approach_speed = p('m8_entry_approach_speed')
        if not (math.isfinite(self._m8_arm_after) and 0 <= self._m8_entry.pass_margin <= .30
                and (self._m8_near_line == -1.0 or .15 <= self._m8_near_line <= .60)
                and (self._m8_precheck == -1.0 or
                     .15 <= self._m8_near_line <= self._m8_precheck <= .60)
                and 0 < self._m8_entry.search_limit <= 1.0 and 0 < self._m8_approach_speed <= .10):
            raise ValueError('Invalid M8 entry parameters')
        self._m2_start_s = None
        self._m8_arm_requested = False
        self._m8_last_output = ''
        self._m8_entry_state = 'off'
        self.m8_timing_pub = self.node.create_publisher(String, M8_CONTROL_TIMING_TOPIC, 10)
        self.exit_check_pub = self.node.create_publisher(
            Bool, M8_EXIT_CHECK_TOPIC,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.exit_check_pub.publish(Bool(data=False))
        self.m8_phase_pub = self.node.create_publisher(
            String, M8_PHASE_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._m8_operator_stop = False
        self.node.create_subscription(TwistStamped, 'mission/cone/safe_cmd_vel', self._store_cone_cmd, 1)
        self.node.create_subscription(TwistStamped, M8_EXIT_COMMAND_TOPIC, self._store_exit_lane, 1)
        self.node.create_subscription(TwistStamped, M8_RECOVERY_COMMAND_TOPIC, self._store_recovery, 1)
        self.node.create_subscription(Header, M8_APPROACH_RISK_TOPIC, self._store_approach_risk, 1)
        self.node.create_subscription(Float32, 'mission/stop_line_visual_distance', self._m8_visual_line_callback, 1)
        self.node.create_subscription(PointStamped, M8_EXIT_LINE_TOPIC, self._m8_exit_line_callback, 1)
        self.node.create_subscription(PointStamped, M8_PARTIAL_ENTRY_TOPIC, self._m8_partial_callback, 1)

    def _m8_partial_callback(self, msg):
        n = self.node
        if (n._active != 'M2' or self._m8_phase_sent != 'watch_entry'
                or self._m8_near_line <= 0 or n._blocked):
            return
        now = n.get_clock().now().nanoseconds
        stamp = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        if (not self._odom_fresh(now) or not math.isfinite(n._odom_speed)
                or n._odom_speed < M8_REVERSE_SPEED or self._m8_reversing):
            self._m8_partial.reset()
            return
        if (msg.header.frame_id != 'front_axle' or stamp <= 0 or stamp < self._m8_post_reverse_ns
                or not 0 <= (now-stamp)/1e9 <= M8_COMMAND_TIMEOUT):
            return
        confirmed = self._m8_partial.observe(n._odom.s, msg.point.x, stamp/1e9)
        if confirmed and n.get_parameter('auto_start_mission_nodes').value:
            # Start observation only: race still owns control until a checked
            # cone command arrives. _start_process already avoids duplicates.
            n._start_process('mission_cone_avoid_node')

    def _m8_visual_line_callback(self, msg):
        # Odom extrapolation from stop_line_node is not an independent observation.
        now = self.node.get_clock().now().nanoseconds
        if self._odom_fresh(now):
            if self.node._active == 'M2':
                if self.node._odom_speed < M8_REVERSE_SPEED:
                    self._m8_reversing = True
                    self._m8_entry.observations.clear()
                    self._m8_partial.reset()
                    return
                s = self.node._odom.s
                if self._m8_phase_sent == 'watch_entry':
                    precheck = self._m8_precheck > 0
                    if not self.node._blocked and self._m8_entry.observe_near(
                            s, msg.data, now/1e9,
                            self._m8_precheck if precheck else self._m8_near_line):
                        if precheck:
                            self._set_m8_phase('preflight')
                            self._set_exit_check(True)
                            if self.node.get_parameter('auto_start_mission_nodes').value:
                                self.node._start_process('mission_cone_avoid_node')
                            self.node.get_logger().info('M8 preflight: checking LiDAR; race lane still owns control')
                        else:
                            self._arm_m8()
                else:
                    self._m8_entry.observe(s, msg.data, now/1e9)
            elif self.node._active == 'M8' and self._m8_exit.entry_line is None:
                # Only the strict entry detector may establish the first anchor.
                armed_s = self._m8_entry.started_s
                if armed_s is not None and 0 <= self.node._odom.s-armed_s <= .90:
                    self._m8_entry.observe(self.node._odom.s, msg.data, now/1e9)
                    if self._m8_entry.line_s is not None:
                        self._m8_exit.reset(self._m8_entry.line_s)


    def _m8_exit_line_callback(self, msg):
        if self.node._active != 'M8' or self._m8_exit.owns_lane:
            return
        now = self.node.get_clock().now().nanoseconds
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if (msg.header.frame_id != 'front_axle' or stamp <= 0
                or not 0 <= (now-stamp)/1e9 <= M8_COMMAND_TIMEOUT or not self._odom_fresh(now)):
            return
        self._m8_exit.observe(self.node._odom.s, msg.point.x, stamp/1e9)
        if self._m8_exit.owns_lane:
            self._m8_exit_handoff_ns = now
            self._exit_lane_cmd = None
            self._exit_lane_stamp_ns = self._exit_lane_rx_ns = 0
            self._set_m8_phase(m8_phase(True, True))
            self._set_exit_check(True)
            self._publish_m8_control(now, trigger='exit_line')


    def _store_cone_cmd(self, msg):
        self._cone_rx_ns = self.node.get_clock().now().nanoseconds
        self._cone_stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self._cone_cmd = msg.twist.linear.x, msg.twist.angular.z
        if self.node._active == 'M2' and self._m8_entry.started_s is not None and self._cone_cmd[0] > 0:
            self.node._control_tick()
        elif self.node._active == 'M8' and not self._m8_exit.owns_lane:
            self._publish_m8_control(self._cone_rx_ns, trigger='cone_rx')


    def _store_exit_lane(self, msg):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if self._m8_exit.owns_lane and stamp < self._m8_exit_handoff_ns:
            return
        self._exit_lane_rx_ns = self.node.get_clock().now().nanoseconds
        self._exit_lane_stamp_ns = stamp
        self._exit_lane_cmd = msg.twist.linear.x, msg.twist.angular.z
        if self.node._active == 'M8' and self._m8_exit.owns_lane:
            self._publish_m8_control(self._exit_lane_rx_ns, trigger='exit_rx')

    def _store_recovery(self, msg):
        phase = msg.header.frame_id
        if phase != self._m8_phase_sent or phase not in ('approach', 'cone', 'exit_lane'):
            return
        now = self.node.get_clock().now().nanoseconds
        stamp = msg.header.stamp.sec*10**9 + msg.header.stamp.nanosec
        command, reason = select_m8_command((msg.twist.linear.x, msg.twist.angular.z), (now-stamp)/1e9)
        self._recovery_offer = (phase, stamp, command) if command[0] and abs(command[1]) <= .20 else None
        if self._recovery_offer is None and reason != 'stale_cone':
            self._recovery_budget.cancel()
        if self.node._active == 'M8':
            self._publish_m8_control(now, trigger='recovery_rx')

    def _recovery_command(self, now, normal, *, eligible=True, after_stamp=0):
        n = self.node
        phase = self._m8_phase_sent
        offer = None
        if not eligible or n._blocked or self._m8_operator_stop:
            self._recovery_budget.cancel()
        if eligible and not (n._blocked or self._m8_operator_stop) and self._recovery_offer:
            p, stamp, cmd = self._recovery_offer
            if p == phase and stamp >= after_stamp and 0 <= (now-stamp)/1e9 <= M8_COMMAND_TIMEOUT:
                offer = stamp, cmd
        return self._recovery_budget.step(now, phase, n._odom.s, n._odom_speed,
            self._odom_fresh(now) and now-n._odom_ns <= 100_000_000, normal, offer)

    def _store_approach_risk(self, msg):
        if self.node._active != 'M2' or self._m8_phase_sent != 'preflight':
            return
        if self._m8_reversing or self.node._odom_speed < M8_REVERSE_SPEED:
            return
        stamp = msg.stamp.sec * 10**9 + msg.stamp.nanosec
        now = self.node.get_clock().now().nanoseconds
        if stamp > 0 and 0 <= (now-stamp)/1e9 <= M8_COMMAND_TIMEOUT:
            self._m8_risk_ns = max(self._m8_risk_ns, stamp)

    def _set_exit_check(self, enabled):
        if self._exit_check_enabled != enabled:
            self._exit_check_enabled = enabled
            if not enabled:
                self._exit_lane_cmd = None
                self._exit_lane_stamp_ns = self._exit_lane_rx_ns = 0
            self.exit_check_pub.publish(Bool(data=enabled))


    def _set_m8_phase(self, phase):
        if self._m8_phase_sent != phase:
            self._recovery_offer = None
            self._recovery_budget.cancel()
            self._m8_phase_sent = phase
            self.m8_phase_pub.publish(String(data=phase))


    def _odom_fresh(self, now):
        return bool(math.isfinite(self.node._odom.s) and self.node._odom_ns and
                    0 <= (now-self.node._odom_ns)/1e9 <= self.node.get_parameter('odom_timeout_sec').value)


    def _arm_m8(self, reason='explicit or odom arm'):
        if self._m8_entry.started_s is not None:
            return
        self._m8_entry.arm(self.node._odom.s)
        self._set_exit_check(True)
        self._set_m8_phase('approach')
        self.node.get_logger().info(f'M8 entry armed ({reason}): verified cone may take over before the line')
        if self.node.get_parameter('auto_start_mission_nodes').value:
            self.node._start_process('mission_cone_avoid_node')


    def _m8_handoff_state(self, now):
        odom_ok = self._odom_fresh(now)
        if (odom_ok and self._m8_phase_sent == 'watch_entry'
                and self._m8_entry.line_s is None and self._m8_near_line > 0):
            anchor = self._m8_partial.anchor(self.node._odom.s, now/1e9)
            cone, _ = select_m8_command(self._cone_cmd,
                (now-self._cone_stamp_ns)/1e9 if self._cone_stamp_ns else float('inf'))
            if (anchor is not None and cone[0] > 0 and math.isfinite(self.node._odom_speed)
                    and self._cone_stamp_ns >= self._m8_post_reverse_ns):
                # The partial projection is an approximate first-line anchor
                # for the unchanged distinct-exit check, not a 45 cm trigger.
                self._m8_entry.line_s = anchor
                self._arm_m8('partial entry tape + verified cone passage')
        if self._m8_phase_sent == 'preflight':
            if not odom_ok:
                return 'wait_odom'
            if self._m8_risk_ns:
                self._arm_m8('LiDAR clearance low')
            elif self._m8_entry.line_s-self.node._odom.s <= self._m8_near_line:
                self._arm_m8('odom distance to entry line')
        auto_arm = (self._m8_arm_after >= 0 and self._m2_start_s is not None and
                    self.node._odom.s-self._m2_start_s >= self._m8_arm_after)
        if odom_ok and (self._m8_arm_requested or auto_arm or self._m8_arm_after == 0):
            self._arm_m8()
        state = self._m8_entry.decide(self.node._odom.s, odom_ok)
        if state == 'off' and (self._m8_arm_requested or self._m8_arm_after == 0):
            state = 'wait_odom'
        if state != self._m8_entry_state:
            line = self._m8_entry.line_s
            left = 'unknown' if line is None else f'{line+self._m8_entry.pass_margin-self.node._odom.s:+.3f} m'
            self.node.get_logger().info(f'M8 entry: {state} (distance to pass={left})')
            self._m8_entry_state = state
        return state


    def _publish_m8_control(self, now, trigger='timer'):
        age = (now-self._cone_stamp_ns)/1e9 if self._cone_stamp_ns > 0 else float('inf')
        (v, omega), reason = select_m8_command(self._cone_cmd, age, self._m8_operator_stop)
        source = 'cone' if v > 0 else 'stop'
        lane_age = ((now-self._exit_lane_stamp_ns)/1e9
                    if self._exit_lane_stamp_ns > 0 else float('inf'))
        (lv, lw), lane_reason = select_m8_command(
            self._exit_lane_cmd, lane_age, self._m8_operator_stop)
        exit_state = self._m8_exit.decide(self.node._odom.s, self._odom_fresh(now), lv > 0)
        if exit_state == 'complete':
            self.node._advance('M8 exit line passed; LiDAR-checked lane guidance completed')
            return
        if exit_state == 'exit_lane':
            v, omega, source, reason = lv, lw, 'lane', 'verified_exit_lane'
        elif exit_state in ('wait_exit_lane', 'wait_odom'):
            v, omega, source = 0.0, 0.0, 'stop'
            reason = ('operator_stop' if self._m8_operator_stop else
                      'exit_wait_odom' if exit_state == 'wait_odom' else f'exit_lane:{lane_reason}')
        exit_input = self._m8_exit.owns_lane
        stamp_ns = self._exit_lane_stamp_ns if exit_input else self._cone_stamp_ns
        rx_ns = self._exit_lane_rx_ns if exit_input else self._cone_rx_ns
        published_ns = self.node.get_clock().now().nanoseconds
        if v > 0 and not 0 <= (published_ns-stamp_ns)/1e9 <= M8_COMMAND_TIMEOUT:
            v, omega, source = 0.0, 0.0, 'stop'
            reason = 'exit_lane:stale_cone' if exit_input else 'stale_cone'
        recovery = self._recovery_command(published_ns, v > 0, eligible=exit_state != 'wait_odom',
            after_stamp=self._m8_exit_handoff_ns if exit_input else 0)
        if not v and recovery[0]:
            v, omega = recovery
            _, stamp_ns, _ = self._recovery_offer
            rx_ns = 0
            source, reason = 'recovery', 'forward_recovery'
        command = Twist()
        command.linear.x, command.angular.z = v, omega
        self.node._publish(command)
        if self.m8_timing_pub.get_subscription_count():
            self.m8_timing_pub.publish(String(data=json.dumps(dict(
                input='exit_lane' if exit_input else 'cone', trigger=trigger,
                input_stamp_ns=stamp_ns, received_ns=rx_ns, published_ns=published_ns,
                output_age_ms=(published_ns-stamp_ns)/1e6 if stamp_ns else None,
                rx_to_output_ms=(published_ns-rx_ns)/1e6 if rx_ns else None,
                source=source, reason=reason, speed=v), separators=(',', ':'))))
        if self._m8_last_output != (reason, exit_state):
            self._m8_last_output = reason, exit_state
            self.node.get_logger().info(f'M8 source={source} reason={reason} exit={exit_state}')
        pv, pw = self._cone_cmd if self._cone_cmd is not None else (0.0, 0.0)
        self.node._debug(f'source={source} reason={reason} entry={self._m8_entry_state} exit={exit_state} '
                    f'proposal_v={pv:.3f} proposal_omega={pw:+.3f} '
                    f'v={v:.3f} omega={omega:+.3f}')


    def event(self,event):
        if self.node._active == 'M2' and event in ('NEXT','M2_DONE','ARM_M8'):
            self._m8_arm_requested=True
            return True
        if self.node._active != 'M8':
            return False
        if event in ('NEXT','M8_DONE'):
            self.node.get_logger().warn('M8 waits for the distinct exit line and observed lane')
            return True
        if event in ('ESTOP','ESTOP_CLEAR'):
            self._m8_operator_stop = event == 'ESTOP'
            self._publish_m8_control(self.node.get_clock().now().nanoseconds,trigger='estop')
            return True
        return False

    def enter(self,mission):
        if mission == 'M2':
            self._recovery_budget.reset()
            self._m8_entry.reset()
            self._m8_partial.reset()
            self._m8_entry_state='off'
            self._m8_arm_requested=False
            self._m8_risk_ns=0
            self._m8_reversing=False
            self._m8_post_reverse_ns=0
            self._m2_start_s=None
        if mission == 'M8':
            self._set_exit_check(False)
            self._m8_exit.reset(self._m8_entry.line_s)
            self._m8_last_output=''
            self._m8_exit_handoff_ns=0
        else:
            self._set_exit_check(False)
        watch_entry = mission == 'M2' and self._m8_near_line > 0 and self._m8_arm_after != 0
        self._set_m8_phase('watch_entry' if watch_entry else
                           m8_phase(mission=='M8', self._m8_exit.owns_lane))

    def tick(self,now):
        n=self.node
        if self._odom_fresh(now):
            if n._active=='M2' and self._m2_start_s is None:
                self._m2_start_s=n._odom.s
            elif n._active=='M8' and self._m8_entry.started_s is None:
                self._m8_entry.arm(n._odom.s)
        if n._active=='M8':
            self._publish_m8_control(now)
            return True
        if n._active!='M2':
            return False
        if n._blocked:
            self._recovery_command(now, False, eligible=False)
            return False  # The original manager applies the operator stop.
        if self._odom_fresh(now) and math.isfinite(n._odom_speed):
            reversing = n._odom_speed < M8_REVERSE_SPEED
            if self._m8_reversing and not reversing:
                self._m8_post_reverse_ns = now
            self._m8_reversing = reversing
        if self._m8_reversing:
            self._recovery_command(now, False, eligible=False)
            self._m8_entry.observations.clear()
            self._m8_partial.reset()
            if self._m8_phase_sent == 'preflight':
                self._m8_risk_ns = 0
            if (self._m8_phase_sent in ('preflight', 'approach')
                    or self._m8_arm_requested or self._m8_arm_after == 0):
                n._publish(Twist())
                n._debug('source=stop entry=reverse_hold v=0.000')
                return True
            return False  # Do not take over ordinary race control while backing.
        gate=self._m8_handoff_state(now)
        if gate=='off':
            return False  # All ordinary lane/mission behavior stays in race.
        cone,_=select_m8_command(self._cone_cmd,
            (now-self._cone_stamp_ns)/1e9 if self._cone_stamp_ns else float('inf'))
        lane,lane_reason=select_m8_command(self._exit_lane_cmd,
            (now-self._exit_lane_stamp_ns)/1e9 if self._exit_lane_stamp_ns else float('inf'))
        fresh_after = max(self._m8_risk_ns, self._m8_post_reverse_ns)
        if not math.isfinite(n._odom_speed) or self._cone_stamp_ns < fresh_after:
            cone = (0., 0.)
        if not math.isfinite(n._odom_speed) or self._exit_lane_stamp_ns < fresh_after:
            lane, lane_reason = (0., 0.), 'entry_wait_fresh_command'
        limit=min(self._m8_approach_speed,n.get_parameter('normal_speed').value)
        if n._section_limit is not None and 0 <= (now-n._section_limit_ns)/1e9 <= 1.:
            limit=min(limit,n._section_limit)
        action=m8_entry_policy(gate,self._m8_entry.started_s is not None,
                               self._odom_fresh(now),cone[0]>0,lane[0]>0 and limit>=MIN_M8_SPEED)
        if action=='cone':
            self._recovery_command(now, True, eligible=False)
            n._advance('M8 local cone gate acquired')
            self._publish_m8_control(now,trigger='entry')
        else:
            v,w=lane if action=='guarded_lane' else (0.,0.)
            recovery = self._recovery_command(now, v > 0,
                eligible=gate == 'approach' and limit >= MIN_M8_SPEED, after_stamp=fresh_after)
            source = 'lane' if v else 'stop'
            if not v and recovery[0]:
                v, w = recovery
                source, lane_reason = 'recovery', 'forward_recovery'
            cmd=Twist()
            cmd.linear.x,cmd.angular.z=v,w
            n._publish(cmd)
            n._debug(f'source={source} entry={gate} reason={lane_reason} v={v:.3f}')
        return True
