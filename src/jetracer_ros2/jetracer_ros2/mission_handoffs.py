"""Observation-only M7 handoff reporting; never computes a driving command."""
import math

LANE3_PROGRESS_TOPIC = 'mission/three_lane/progress'


def lane3_progress(supervisor):
    """Observed sign + actual selected fork, separate from correctness scoring.

    Adapts the three existing controller layouts without changing their plans.
    Fixed goals and GapCore's timed default-side guess are NOT sign detections.
    """
    complete = bool(getattr(supervisor, 'route_complete', False))
    goal = getattr(getattr(supervisor, 'vote', None), 'goal', None)
    if supervisor.mode not in ('on', 'fork') or goal not in (0, 2):
        return None, complete
    if hasattr(supervisor, 'sign_goal') and hasattr(supervisor, 'core'):
        from .three_lane_simple import LEAVE_DIST, MERGE_TURN
        core = supervisor.core
        side = core.lane
        selected = (side in (0, 2)
                    and -MERGE_TURN < core.island_at-core.s <= LEAVE_DIST)
    elif hasattr(supervisor, 'tl'):
        from .three_lane import ISLAND_URGENT
        out = supervisor.last
        side = out.target if out is not None else None
        selected = (out is not None and out.ok and side in (0, 2)
                    and out.island is not None and 0 <= out.island <= ISLAND_URGENT)
    elif hasattr(supervisor, 'core'):
        core = supervisor.core
        side = (0 if core.tgt > core.isl[1] else 2) if core.isl is not None and core.tgt is not None else None
        selected = (core.isl is not None and core.tgt is not None
                    and -1.0 <= core.isl[0]-core.sP <= 2.0
                    and abs(core.tgt-core.isl[1]) > .05)
    else:
        selected = False
    return (side if selected else None), complete


class ForkProgressReporter:
    """Repeat evidence until authority changes; require actual forward odometry.

    One millimetre is a movement/noise check, not a new course-distance gate.
    Stopped/reverse commands and pose jumps cannot declare a fork start.
    """
    def __init__(self):
        self.mission = None
        self.previous = None
        self.fork_side = None

    def set_mission(self, mission):
        if mission != self.mission and mission != 'M2':
            self.previous = self.fork_side = None
        self.mission = mission

    def update(self, supervisor, pose, speed, stamp_ns):
        if self.mission not in ('M7', 'M2') or supervisor is None:
            return None
        side, complete = lane3_progress(supervisor)
        if (pose is None or not all(math.isfinite(x) for x in pose)
                or not math.isfinite(speed) or speed <= 0):
            self.previous = None
            return (dict(stamp_ns=stamp_ns, fork_side=None, route_complete=True)
                    if complete else None)
        if self.previous is not None:
            old_side, old_pose, old_stamp = self.previous
            dx, dy = pose[0]-old_pose[0], pose[1]-old_pose[1]
            forward = dx*math.cos(old_pose[2]) + dy*math.sin(old_pose[2])
            if (side is not None and side == old_side
                    and 0 < stamp_ns-old_stamp <= 500_000_000
                    and .001 <= forward and math.hypot(dx, dy) < .10):
                self.fork_side = side
        self.previous = side, pose, stamp_ns
        return dict(stamp_ns=stamp_ns, fork_side=self.fork_side, route_complete=complete,
                    sign_side=getattr(supervisor.vote, 'goal', None))
