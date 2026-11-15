"""M8 command authorization and separate entry/exit stop-line gates."""
import math

MIN_M8_SPEED = .10
# This is still the tested upper bound of the current cone planner.
MAX_M8_SPEED = .10
M8_COMMAND_TIMEOUT = .25
OBSERVED_LANE_TOPIC = 'mission/lane/observed_cmd_vel'
M8_LANE_PATH_TOPIC = 'mission/lane/observed_center'
M8_LANE_HINT_TOPIC = 'mission/lane/geometry_hint'
M8_LANE_PATH_FRAME = 'rear_axle'
M8_PHASE_TOPIC = 'mission/cone/phase'
M8_EXIT_LINE_TOPIC = 'mission/cone/exit_line'
M8_EXIT_COMMAND_TOPIC = 'mission/cone/safe_exit_cmd_vel'
M8_EXIT_CHECK_TOPIC = 'mission/cone/exit_check_enabled'
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


class M8ExitGate:
    """Require a distinct exit line and a verified lane handover, never a fallback."""

    def __init__(self, min_gap, pass_margin, settle_distance, *, handoff_on_detection=False):
        self.min_gap = min_gap
        self.pass_margin = pass_margin
        self.settle_distance = settle_distance
        self.handoff_on_detection = handoff_on_detection
        self.reset()

    def reset(self, entry_line=None):
        self.entry_line = entry_line
        self.line_s = None
        self.observations = []
        self.passed = False
        self.lane_start_s = None

    @property
    def owns_lane(self):
        return self.passed or (self.handoff_on_detection and self.line_s is not None)

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

    def decide(self, s, odom_fresh, lane_safe):
        if not odom_fresh or not math.isfinite(s):
            return 'wait_odom'
        if self.entry_line is None:
            return 'no_entry_anchor'
        if self.line_s is None:
            return 'search_exit'
        crossed = s >= self.line_s + self.pass_margin
        if not crossed and not self.owns_lane:
            return 'approach_exit'
        if (self.passed and not crossed) or not lane_safe:
            self.passed |= crossed
            return 'wait_exit_lane'
        self.passed |= crossed
        if self.lane_start_s is None:
            self.lane_start_s = s
        return ('complete' if crossed and s-self.lane_start_s >= self.settle_distance
                else 'exit_lane')
