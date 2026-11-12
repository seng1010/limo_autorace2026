"""Time-based slow-zone latch and lane-following speed selection."""


class ZigzagTracker:
    """Slow immediately on evidence; retain a confirmed zone across short gaps."""

    def __init__(self):
        self.first_seen = None
        self.last_seen = None
        self.confirmed = False
        self.active = False

    def reset(self):
        self.first_seen = None
        self.last_seen = None
        self.confirmed = False
        self.active = False

    def update(self, detected, now_sec, confirm_sec=2.0, gap_sec=0.35,
               hold_sec=1.5):
        gap_sec = max(0.0, float(gap_sec))
        hold_sec = max(gap_sec, float(hold_sec))
        confirm_sec = max(0.0, float(confirm_sec))
        if self.last_seen is not None and now_sec < self.last_seen:
            self.reset()  # ROS clock reset or simulation restart.
        if detected:
            if self.last_seen is None or now_sec - self.last_seen > gap_sec:
                self.first_seen = now_sec
                self.confirmed = False
            self.last_seen = now_sec
            self.active = True
            if now_sec - self.first_seen >= confirm_sec:
                self.confirmed = True
        elif self.last_seen is not None:
            lifetime = hold_sec if self.confirmed else gap_sec
            if now_sec - self.last_seen > lifetime:
                self.reset()
        return self.active


def select_speed(zigzag, curve, straight_speed, curve_speed, zigzag_speed):
    """Zigzag takes priority over a coincident curved boundary."""
    return max(0.0, float(zigzag_speed if zigzag else
                          curve_speed if curve else straight_speed))


def yellow_zigzag_evidence(result):
    """Ignore zigzags inferred solely from white dashes or the mat edge."""
    return bool(result.valid and result.zigzag and any(
        label.startswith('yellow/') and '/dashed/' not in label
        for label in (result.left_type, result.right_type)))
