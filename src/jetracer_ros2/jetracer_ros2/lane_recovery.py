"""Bounded propagation of a previously validated ground path using odometry.

The path uses lane coordinates (lateral right, forward) in metres. ROS odom
uses (forward, left) in the robot frame. No image or ROS dependency is needed
here, so the geometry and stop limits can be tested without a vehicle.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


def brief_hold_allowed(has_valid_command: bool, age_sec: float, limit_sec: float) -> bool:
    """A short camera-gap hold is independent of optional odom recovery."""
    return (has_valid_command and math.isfinite(age_sec) and math.isfinite(limit_sec)
            and 0.0 <= age_sec <= max(0.0, limit_sec))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class PathSnapshot:
    world_points: Tuple[Tuple[float, float], ...]
    length_m: float
    start_ns: int
    start_odometer_m: float


@dataclass(frozen=True)
class RecoveryTarget:
    lateral_m: float
    forward_m: float
    progress_m: float
    remaining_m: float


def snapshot_path(points, pose: Pose2D, now_ns: int, odometer_m: float) -> Optional[PathSnapshot]:
    """Store only finite, forward-ordered *validated* path samples."""
    if (len(points) < 2 or not all(math.isfinite(v) for v in
                                    (pose.x, pose.y, pose.yaw, odometer_m))):
        return None
    world = []
    length = 0.0
    previous_lateral, previous_forward = 0.0, 0.0
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    for lateral, forward in points:
        if (not math.isfinite(lateral) or not math.isfinite(forward)
                or forward <= previous_forward):
            return None
        length += math.hypot(lateral - previous_lateral, forward - previous_forward)
        world.append((pose.x + c * forward + s * lateral,
                      pose.y + s * forward - c * lateral))
        previous_lateral, previous_forward = lateral, forward
    return PathSnapshot(tuple(world), length, now_ns, odometer_m)


def recovery_target(snapshot: PathSnapshot, pose: Pose2D, odometer_m: float,
                    now_ns: int, max_age_sec: float, max_distance_m: float,
                    lookahead_m: float, min_forward_m: float) -> Optional[RecoveryTarget]:
    """Never extrapolate beyond the stored path or its earliest stop limit."""
    if (not all(math.isfinite(v) for v in
                (pose.x, pose.y, pose.yaw, odometer_m, max_age_sec,
                 max_distance_m, lookahead_m, min_forward_m))
            or max_age_sec <= 0 or max_distance_m <= 0 or lookahead_m <= 0
            or min_forward_m <= 0 or now_ns < snapshot.start_ns):
        return None
    progress = odometer_m - snapshot.start_odometer_m
    limit = min(max_distance_m, 0.5 * snapshot.length_m)
    if (progress < 0 or progress >= limit
            or (now_ns - snapshot.start_ns) / 1e9 >= max_age_sec):
        return None
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    local = []
    for wx, wy in snapshot.world_points:
        dx, dy = wx - pose.x, wy - pose.y
        forward = c * dx + s * dy
        lateral = s * dx - c * dy
        if forward >= min_forward_m:
            local.append((lateral, forward))
    if not local:
        return None
    # Pick a measured point at or just beyond lookahead, never a polynomial
    # extension or a point behind the rear axle.
    lateral, forward = next((p for p in local if p[1] >= lookahead_m), local[-1])
    return RecoveryTarget(lateral, forward, progress, limit - progress)


def right_boundary_clearance(target: RecoveryTarget, right_boundary, minimum_m: float):
    """Check a target against the *observed* right boundary at the same range.

    Missing/short observations are not extrapolated into a pass.
    """
    if (not math.isfinite(minimum_m) or minimum_m <= 0 or len(right_boundary) < 2
            or not all(math.isfinite(v) for point in right_boundary for v in point)):
        return False
    for near, far in zip(right_boundary, right_boundary[1:]):
        if near[1] <= target.forward_m <= far[1] and far[1] > near[1]:
            ratio = (target.forward_m - near[1]) / (far[1] - near[1])
            boundary_lateral = near[0] + ratio * (far[0] - near[0])
            return boundary_lateral - target.lateral_m >= minimum_m
    return False
