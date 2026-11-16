"""M8 command authorization and separate entry/exit stop-line gates."""
import math

MIN_M8_SPEED = .10
# This is still the tested upper bound of the current cone planner.
MAX_M8_SPEED = .10
M8_COMMAND_TIMEOUT = .25
# Spend at most 6.5 cm / 0.65 s of the 10 cm recovery budget under command.
# Leave 3.5 cm nominal reserve for observation, scheduling and deceleration.
M8_RECOVERY_CUTOFF_M = .065
M8_RECOVERY_TIMEOUT_NS = 650_000_000
OBSERVED_LANE_TOPIC = 'mission/lane/observed_cmd_vel'
M8_LANE_PATH_TOPIC = 'mission/lane/observed_center'
M8_LANE_HINT_TOPIC = 'mission/lane/geometry_hint'
M8_EXIT_LANE_EVIDENCE_TOPIC = 'mission/cone/exit_lane_evidence'
M8_LANE_PATH_FRAME = 'rear_axle'
M8_PHASE_TOPIC = 'mission/cone/phase'
M8_EXIT_LINE_TOPIC = 'mission/cone/exit_line'
M8_PARTIAL_ENTRY_TOPIC = 'mission/cone/partial_entry_line'
M8_EXIT_COMMAND_TOPIC = 'mission/cone/safe_exit_cmd_vel'
M8_EXIT_PROGRESS_TOPIC = 'mission/cone/exit_progress'
M8_RECOVERY_COMMAND_TOPIC = 'mission/cone/recovery_cmd_vel'
M8_EXIT_CHECK_TOPIC = 'mission/cone/exit_check_enabled'
M8_APPROACH_RISK_TOPIC = 'mission/cone/approach_risk'
M8_PLANNER_TIMING_TOPIC = 'mission/cone/timing'
M8_CONTROL_TIMING_TOPIC = 'mission/cone/control_timing'


def m8_phase(owns_m8, owns_lane):
    if not owns_m8:
        return 'off'
    return 'exit_lane' if owns_lane else 'cone'


def m8_entry_policy(gate_state, armed, odom_fresh, cone_safe, lane_safe):
    """A verified cone path, not passage of the first line, grants command ownership."""
    if not armed:
        return 'lane' if gate_state == 'off' else 'stop'
    if not odom_fresh:
        return 'stop'
    if cone_safe:
        return 'cone'
    return 'guarded_lane' if gate_state == 'approach' and lane_safe else 'stop'


def _observe_line(observations, s, distance, now):
    if distance is None or not .04 <= distance <= .60 or not all(map(math.isfinite, (s, now))):
        observations.clear()
        return None
    line = s + distance
    if observations:
        dt = now - observations[-1][0]
        if dt <= 0:
            return None
        if dt > .25 or abs(line - observations[0][1]) > .08:
            observations.clear()
    observations.append((now, line))
    if len(observations) >= 3:
        anchor = sorted(x[1] for x in observations)[1]
        observations.clear()
        return anchor
    return None


def select_m8_command(command, age, operator_stop=False):
    """The M8 owner may select the stamped cone proposal or a zero Twist, never lane."""
    stop = (0.0, 0.0)
    if operator_stop:
        return stop, 'operator_stop'
    if command is None:
        return stop, 'waiting_for_cone'
    if not 0 <= age <= M8_COMMAND_TIMEOUT:
        return stop, 'stale_cone'
    v, omega = command
    if not (math.isfinite(v) and math.isfinite(omega)) or v < 0 or v > MAX_M8_SPEED:
        return stop, 'invalid_cone'
    if v == 0:
        return stop, 'cone_stop'
    if v < MIN_M8_SPEED:
        return stop, 'below_min_speed'
    return command, 'cone_tracking'


class M8RecoveryBudget:
    """Bound only the recovery command stream; ordinary commands always win."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.pending = None
        self.started = None
        self.spent = False
        self.normal_start_s = None
        self.newer_than = 0

    def cancel(self):
        """An explicit invalid/zero proposal is not a harmless transport gap."""
        self.pending = None
        if self.started is not None:
            self.started = None
            self.spent = True

    def step(self, now, phase, s, speed, fresh_odom, normal, offer):
        stop = (0., 0.)
        if self.started is not None:
            start_ns, start_s, start_phase, w = self.started
            if (not fresh_odom or normal or phase != start_phase
                    or not -.005 <= s-start_s < M8_RECOVERY_CUTOFF_M
                    or not 0 <= now-start_ns < M8_RECOVERY_TIMEOUT_NS
                    or not -.02 <= speed <= .13
                    or (offer and abs(offer[1][1]-w) > .03)):
                self.started = None
                self.spent = True
            else:
                # Missing/expired input stops motion NOW, but does not start
                # a new budget. Resume only on fresh evidence in the original
                # 0.65 s / 6.5 cm window; neither clock nor distance is reset.
                return offer[1] if offer else stop
        if self.spent:
            if normal and fresh_odom and speed >= -.02:
                if self.normal_start_s is None:
                    self.normal_start_s = s
                elif s-self.normal_start_s >= .15:
                    self.reset()
                    self.newer_than = now
            else:
                self.normal_start_s = None
            return stop
        if normal or not fresh_odom or not -.02 <= speed <= .03:
            self.pending = None
            return stop
        if not offer:
            # Retain EVIDENCE only across a brief expiration between scans.
            # No command is returned without a currently fresh offer. The
            # second distinct source stamp must still be within 250 ms of
            # the first; the extra window accounts for its allowed latency.
            if self.pending is not None and (self.pending[1] != phase
                    or not 0 <= now-self.pending[0] <= 500_000_000):
                self.pending = None
            return stop
        stamp, command = offer
        if stamp <= self.newer_than:
            return stop
        if self.pending is not None:
            last_stamp, last_phase, last_w = self.pending
            if (last_phase == phase and abs(command[1]-last_w) <= .03
                    and 0 < stamp-last_stamp <= 250_000_000):
                self.started = now, s, phase, command[1]
                self.pending = None
                return command
        self.pending = stamp, phase, command[1]
        return stop


class M8EntryGate:
    def __init__(self, pass_margin, search_limit):
        self.pass_margin = pass_margin
        self.search_limit = search_limit
        self.reset()

    def reset(self):
        self.started_s = None
        self.line_s = None
        self.observations = []
        self.passed = False

    def arm(self, s):
        if self.started_s is None and math.isfinite(s):
            self.started_s = s

    def observe(self, s, distance, now):
        """Call once per new distance message, not on every manager tick."""
        if self.started_s is None or self.line_s is not None:
            return
        self.line_s = _observe_line(self.observations, s, distance, now)

    def observe_near(self, s, distance, now, max_distance):
        """Confirm a nearby entrance while ordinary lane following still owns control."""
        if self.started_s is not None or self.line_s is not None:
            return False
        distance = distance if distance is not None and distance <= max_distance else None
        line_s = _observe_line(self.observations, s, distance, now)
        if line_s is None:
            return False
        self.line_s = line_s
        return True

    def decide(self, s, odom_fresh):
        # A later retreat, odom dropout or cone loss never gives lane control back.
        if self.passed:
            return 'passed'
        if self.started_s is None:
            return 'off'
        if not odom_fresh or not math.isfinite(s):
            return 'wait_odom'
        if self.line_s is None:
            return 'line_not_found' if s - self.started_s >= self.search_limit else 'approach'
        if s < self.line_s + self.pass_margin:
            return 'approach'
        self.passed = True
        return 'passed'


class M8PartialEntryGate:
    """Short-lived entrance evidence, separate from the metric entry gate."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.observations = []
        self.last_stamp = None
        self.confirmed = None

    def observe(self, s, distance, stamp):
        if not math.isfinite(stamp) or (self.last_stamp is not None and stamp <= self.last_stamp):
            return False
        self.last_stamp = stamp
        line = _observe_line(self.observations, s, distance, stamp)
        if line is not None:
            self.confirmed = (line, s, stamp)
            return True
        return False

    def anchor(self, s, now):
        if self.confirmed is None or not all(map(math.isfinite, (s, now))):
            return None
        line, at_s, stamp = self.confirmed
        # An unseen cue cannot authorize later unrelated cones. Permit the
        # small overlap needed for the next 10 Hz validated LiDAR proposal.
        if not (0 <= now-stamp <= .8 and 0 <= s-at_s <= .20):
            self.confirmed = None
            return None
        return line


class M8ExitGate:
    """Exit line OR measured cone passage arms a fresh checked lane handover."""

    def __init__(self, min_gap, pass_margin, settle_distance):
        self.min_gap = min_gap
        self.pass_margin = pass_margin
        self.settle_distance = settle_distance
        self.reset()

    def reset(self, entry_line=None):
        self.entry_line = entry_line
        self.line_s = None
        self.observations = []
        self.passed = False
        self.lane_start_s = None
        self.lane_owned = False

    @property
    def owns_lane(self):
        return self.lane_owned

    def observe(self, s, distance, now):
        if self.entry_line is None or self.line_s is not None:
            return
        if not (math.isfinite(s) and math.isfinite(now)):
            self.observations.clear()
            return
        if self.observations and now <= self.observations[-1][0]:
            return
        self.observations[:] = [(t, q) for t, q in self.observations if now-t <= .25]
        if distance is None or not .04 <= distance <= .85 or s < self.entry_line + .25:
            return
        anchor = s + distance
        if anchor < self.entry_line + self.min_gap:
            return
        if self.observations and abs(anchor-self.observations[0][1]) > .08:
            self.observations.clear()
        self.observations.append((now, anchor))
        if len(self.observations) == 3:
            self.line_s = sorted(q for _, q in self.observations)[1]
            self.observations.clear()

    def decide(self, s, odom_fresh, lane_safe, cones_clear=False, *, lane_aligned=True):
        if not odom_fresh or not math.isfinite(s):
            return 'wait_odom'
        if self.entry_line is None and not cones_clear and not self.owns_lane:
            return 'no_entry_anchor'
        if self.line_s is None and not cones_clear and not self.owns_lane:
            return 'search_exit'
        crossed = self.line_s is not None and s >= self.line_s + self.pass_margin
        # Exit context alone never suppresses a still-usable cone command.
        if not lane_safe:
            return 'wait_exit_lane' if self.owns_lane else 'prepare_exit'
        if self.passed and not crossed and not cones_clear:
            return 'wait_exit_lane'
        self.lane_owned = True
        self.passed |= crossed
        if self.lane_start_s is None or s < self.lane_start_s:
            self.lane_start_s = s
        return ('complete' if lane_aligned and (crossed or cones_clear) and s-self.lane_start_s >= self.settle_distance
                else 'exit_lane')
