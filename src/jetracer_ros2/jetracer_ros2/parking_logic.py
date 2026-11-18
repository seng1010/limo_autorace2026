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

    1) 정지선 직후 ignore_until_m(0.45 m)까지는 항상 열려 보이므로 무시
    2) 그 뒤 P30 >= open_min_range_m 가 min_open_m(0.30 m) 이상 연속이면 빈칸 확정
       (차 사이 틈 0.01~0.06 m 는 여기서 걸러짐)
    3) 확정 시점의 정지선 기준 거리로 칸 판별
       실측 확정 지점: 1번 0.75 m / 2번 0.995~1.04 m / 3번 1.46 m
       -> < slot1_max_confirm_m(0.87): 1번, < slot2_max_confirm_m(1.25): 2번, 그 외 3번
    """

    def __init__(self, open_min_range_m=1.0, ignore_until_m=0.45, min_open_m=0.30,
                 slot1_max_confirm_m=0.87, slot2_max_confirm_m=1.25, give_up_m=2.0):
        self.open_min_range = float(open_min_range_m)
        self.ignore_until = float(ignore_until_m)
        self.min_open = float(min_open_m)
        self.slot1_max_confirm = float(slot1_max_confirm_m)
        self.slot2_max_confirm = float(slot2_max_confirm_m)
        self.give_up = float(give_up_m)
        self.reset()

    def reset(self):
        self.open_start = None   # 현재 열린 구간 시작 거리
        self.confirm_at = None   # 빈칸 확정 거리
        self.result = None

    def update(self, distance_m, p30):
        """Return 1/2/3 when decided, 0 when nothing found before give_up, else None."""
        if self.result is not None:
            return self.result
        if (distance_m >= self.ignore_until and p30 is not None and math.isfinite(p30)):
            if p30 >= self.open_min_range:
                if self.open_start is None:
                    self.open_start = distance_m
                if distance_m - self.open_start >= self.min_open:
                    self.confirm_at = distance_m
                    if distance_m < self.slot1_max_confirm:
                        self.result = 1
                    elif distance_m < self.slot2_max_confirm:
                        self.result = 2
                    else:
                        self.result = 3
                    return self.result
            else:
                self.open_start = None
        if distance_m >= self.give_up:
            self.result = 0
        return self.result
