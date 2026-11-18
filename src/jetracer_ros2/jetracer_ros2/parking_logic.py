"""Pure LiDAR helpers for the M4 parking mission."""

import math
from typing import NamedTuple

import numpy as np


class ScanSectorStatistics(NamedTuple):
    p10: float
    p30: float
    p50: float
    p70: float
    p90: float
    valid_points: int
    finite_points: int
    no_return_points: int


def _scan_sector_values(scan, angle_min_deg, angle_max_deg, minimum_points,
                        front_angle_deg):
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    if (ranges.size == 0
            or not math.isfinite(scan.angle_increment)
            or not math.isfinite(front_angle_deg)):
        return None

    angles = (scan.angle_min
              + np.arange(ranges.size, dtype=np.float32) * scan.angle_increment
              - math.radians(front_angle_deg))
    angles = np.degrees((angles + math.pi) % (2.0 * math.pi) - math.pi)
    in_sector = (angles >= angle_min_deg) & (angles <= angle_max_deg)
    sector_ranges = ranges[in_sector]
    if sector_ranges.size == 0:
        return None

    no_return = np.isposinf(sector_ranges)
    valid = (np.isfinite(sector_ranges)
             & (sector_ranges >= scan.range_min)
             & (sector_ranges <= scan.range_max))
    valid |= no_return
    if np.count_nonzero(valid) < minimum_points:
        return None

    values = sector_ranges[valid].copy()
    values[np.isposinf(values)] = scan.range_max
    return values, int(np.count_nonzero(valid & ~no_return)), int(np.count_nonzero(no_return))


def scan_sector_percentile(scan, angle_min_deg, angle_max_deg, percentile=30.0,
                           minimum_points=3, front_angle_deg=0.0):
    """Return a robust range estimate for an angle sector, or None if unusable."""
    measurements = _scan_sector_values(
        scan, angle_min_deg, angle_max_deg, minimum_points, front_angle_deg)
    if measurements is None:
        return None
    values, _, _ = measurements
    return float(np.percentile(values, percentile))


def scan_sector_statistics(scan, angle_min_deg, angle_max_deg,
                           minimum_points=3, front_angle_deg=0.0):
    """Return range percentiles and finite/no-return counts for a sector."""
    measurements = _scan_sector_values(
        scan, angle_min_deg, angle_max_deg, minimum_points, front_angle_deg)
    if measurements is None:
        return None
    values, finite_points, no_return_points = measurements
    p10, p30, p50, p70, p90 = np.percentile(values, [10, 30, 50, 70, 90])
    return ScanSectorStatistics(
        float(p10), float(p30), float(p50), float(p70), float(p90),
        len(values), finite_points, no_return_points)


def parking_motion_verified(start_pose, end_pose, minimum_distance_m, minimum_right_turn_deg):
    """Check that the maneuver translated and turned right by a measurable amount."""
    if start_pose is None or end_pose is None:
        return False

    distance = math.hypot(end_pose[0] - start_pose[0], end_pose[1] - start_pose[1])
    yaw_delta = math.atan2(
        math.sin(end_pose[2] - start_pose[2]),
        math.cos(end_pose[2] - start_pose[2]))
    return (distance >= minimum_distance_m
            and yaw_delta <= -math.radians(minimum_right_turn_deg))


def parking_phase_reached(phase, start_pose, current_pose, target):
    """Check a straight-line distance or right-turn yaw target from odometry."""
    if start_pose is None or current_pose is None:
        return False
    if phase == 'turn':
        yaw_delta = math.atan2(
            math.sin(current_pose[2] - start_pose[2]),
            math.cos(current_pose[2] - start_pose[2]))
        return -math.degrees(yaw_delta) >= target
    distance = math.hypot(current_pose[0] - start_pose[0],
                          current_pose[1] - start_pose[1])
    return distance >= target


class EmptyBayConfirmation:
    """Require consecutive usable scans to confirm an open bay."""

    def __init__(self, minimum_open_distance, confirm_scans):
        if minimum_open_distance <= 0.0:
            raise ValueError('minimum_open_distance must be positive')
        if confirm_scans < 1:
            raise ValueError('confirm_scans must be at least one')
        self.minimum_open_distance = float(minimum_open_distance)
        self.confirm_scans = int(confirm_scans)
        self.hits = 0

    def reset(self):
        self.hits = 0

    def update(self, distance):
        if distance is not None and math.isfinite(distance) and distance >= self.minimum_open_distance:
            self.hits += 1
        else:
            self.reset()
        return self.hits >= self.confirm_scans


class BaySlotClassifier:
    """정지선(0 m)부터 직진하며 우측 P30과 odom 거리로 빈 칸 번호(1/2/3)를 판정.

    실측(정지선 기준, 0.10 m/s, 우측 -105..-75deg P30):
      1번 빈칸: 정지선부터 열린 구간이 ~0.93 m 까지 이어짐 (차1이 없음)
      2번 빈칸: 0.41~0.47 m 에서 막힘(차1) -> 0.70~0.74 m 에서 다시 열림 (폭 ~0.63 m)
      3번 빈칸: 0.41 m 에서 막힘(차1+차2, 사이 틈 <=0.06 m) -> 1.16 m 에서 열림
    """

    def __init__(self, open_min_range_m=1.0, min_block_m=0.10, min_open_m=0.15,
                 slot1_open_until_m=0.65, slot2_max_open_start_m=0.95,
                 give_up_m=1.80):
        self.open_min_range = float(open_min_range_m)
        self.min_block = float(min_block_m)
        self.min_open = float(min_open_m)
        self.slot1_open_until = float(slot1_open_until_m)
        self.slot2_max_open_start = float(slot2_max_open_start_m)
        self.give_up = float(give_up_m)
        self.reset()

    def reset(self):
        self.blocked_seen = False   # 차1(또는 차1+차2) 막힘 확정 여부
        self.block_start = None     # 막힘 후보 시작 거리
        self.open_start = None      # 막힘 이후 열림 후보 시작 거리
        self.result = None

    def update(self, distance_m, p30):
        """Return 1/2/3 when decided, 0 when nothing found before give_up, else None."""
        if self.result is not None:
            return self.result
        if p30 is not None and math.isfinite(p30):
            is_open = p30 >= self.open_min_range
            if not self.blocked_seen:
                if is_open:
                    self.block_start = None   # 짧은 막힘은 잡음
                    if distance_m >= self.slot1_open_until:
                        self.result = 1
                        return self.result
                else:
                    if self.block_start is None:
                        self.block_start = distance_m
                    if distance_m - self.block_start >= self.min_block:
                        self.blocked_seen = True
            else:
                if is_open:
                    if self.open_start is None:
                        self.open_start = distance_m
                    if distance_m - self.open_start >= self.min_open:
                        self.result = 2 if self.open_start < self.slot2_max_open_start else 3
                        return self.result
                else:
                    self.open_start = None    # 차 사이 짧은 틈(<=0.06 m)은 무시
        if distance_m >= self.give_up:
            self.result = 0
        return self.result
